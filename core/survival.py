from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cox_ph_loss(
    risk_scores: torch.Tensor,
    event_times: torch.Tensor,
    events: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood for a mini-batch.

    Args:
        risk_scores: [B], larger means higher risk / shorter survival.
        event_times: [B], observed event or censoring time.
        events: [B], 1 = observed event/death, 0 = censored.

    Notes:
        Cox loss needs comparable patients in the same mini-batch. The stage-1
        dataloader therefore returns a list of WSI bags so B can be >1 while
        each bag is still forwarded separately through TITAN.
    """
    risk_scores = torch.nan_to_num(risk_scores.float().view(-1), nan=0.0, posinf=20.0, neginf=-20.0)
    risk_scores = risk_scores.clamp(min=-20.0, max=20.0)
    event_times = event_times.float().view(-1)
    events = events.float().view(-1)

    if risk_scores.numel() <= 1 or events.sum() <= 0:
        return risk_scores.sum() * 0.0

    # Sort by decreasing time so the cumulative logsumexp is each risk set.
    order = torch.argsort(event_times, descending=True)
    sorted_risk = risk_scores[order]
    sorted_events = events[order]

    log_cumsum_exp = torch.logcumsumexp(sorted_risk, dim=0)
    event_mask = sorted_events > 0.5
    partial = sorted_risk[event_mask] - log_cumsum_exp[event_mask]
    if partial.numel() == 0:
        return risk_scores.sum() * 0.0
    loss = -partial.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite Cox PH loss.")
    return loss

def censored_aware_prompt_loss(
    prompt_logits: torch.Tensor,
    time_bin: torch.Tensor,
    censorship: torch.Tensor,
) -> torch.Tensor:
    """VLSA-style prompt alignment loss with censoring awareness.

    prompt_logits: [B, K] similarity to ordinal prognosis prompts.
    time_bin: [B], task-specific quantile bin.
    censorship: [B], 0 = event observed, 1 = censored.

    Event sample: CE against its observed prognosis bin.
    Censored sample at bin b: -log sum_{k >= b} p(k), because the true event
    could occur in the censoring bin or any later/better prognosis bin.
    """
    logits = torch.nan_to_num(prompt_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    logits = logits.clamp(min=-20.0, max=20.0)
    batch_size, num_bins = logits.shape
    y = time_bin.long().clamp(min=0, max=num_bins - 1).view(-1)
    c = censorship.float().view(-1)

    log_probs = F.log_softmax(logits, dim=1)
    losses = []

    event_mask = c < 0.5
    if event_mask.any():
        losses.append(F.nll_loss(log_probs[event_mask], y[event_mask], reduction="mean"))

    censored_mask = c >= 0.5
    if censored_mask.any():
        idxs = torch.nonzero(censored_mask, as_tuple=False).view(-1)
        censored_losses = []
        for idx in idxs:
            b = int(y[idx].item())
            censored_losses.append(-torch.logsumexp(log_probs[idx, b:], dim=0))
        losses.append(torch.stack(censored_losses).mean())

    if not losses:
        return logits.sum() * 0.0
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite censored-aware prompt loss.")
    return loss


def ordinal_prompt_kl_loss(
    prompt_logits: torch.Tensor,
    time_bin: torch.Tensor,
    censorship: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Soft ordinal KL loss for ordered prognosis prompts.

    For event samples, the target is a soft distribution centered at the event
    bin. For censored samples, bins earlier than the censoring bin are masked
    out, reflecting that the patient survived at least until that bin.
    """
    logits = torch.nan_to_num(prompt_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    logits = logits.clamp(min=-20.0, max=20.0)
    batch_size, num_bins = logits.shape
    y = time_bin.long().clamp(min=0, max=num_bins - 1).view(-1)
    c = censorship.float().view(-1)

    bins = torch.arange(num_bins, device=logits.device, dtype=logits.dtype).view(1, -1)
    centers = y.to(logits.device, dtype=logits.dtype).view(-1, 1)
    target = torch.exp(-torch.abs(bins - centers) / max(float(tau), 1e-6))

    censored_mask = c.to(logits.device) >= 0.5
    if censored_mask.any():
        allowed = bins >= centers
        target = torch.where(censored_mask.view(-1, 1), target * allowed.float(), target)

    target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    log_probs = F.log_softmax(logits, dim=1)
    loss = F.kl_div(log_probs, target, reduction="batchmean")
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite ordinal prompt KL loss.")
    return loss


def concordance_index(event_times, risk_scores, events) -> float:
    """Harrell C-index where higher risk_score means shorter survival.

    event_times: observed time or censoring time.
    events: 1 = observed event/death, 0 = censored.
    """
    times = np.asarray(event_times, dtype=float)
    scores = np.asarray(risk_scores, dtype=float)
    events = np.asarray(events, dtype=int)

    finite = np.isfinite(times) & np.isfinite(scores) & np.isfinite(events)
    times, scores, events = times[finite], scores[finite], events[finite]

    permissible = 0.0
    concordant = 0.0
    ties = 0.0

    n = len(times)
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if times[i] < times[j]:
                permissible += 1.0
                if scores[i] > scores[j]:
                    concordant += 1.0
                elif scores[i] == scores[j]:
                    ties += 1.0

    if permissible == 0:
        return 0.0
    return float((concordant + 0.5 * ties) / permissible)


def _km_censoring_survival(train_times, train_events, eps: float = 1e-12):
    """Kaplan-Meier estimate of censoring survival G(t).

    train_events: 1 = observed event/death, 0 = censored.
    For censoring distribution, censored observations are treated as events.
    """
    times = np.asarray(train_times, dtype=float)
    events = np.asarray(train_events, dtype=int)

    finite = np.isfinite(times) & np.isfinite(events)
    times, events = times[finite], events[finite]
    if len(times) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    censor_events = 1 - events
    unique_times = np.sort(np.unique(times))

    surv_values = []
    surv = 1.0
    for t in unique_times:
        at_risk = np.sum(times >= t)
        n_censor = np.sum((times == t) & (censor_events == 1))
        if at_risk > 0:
            surv *= max(0.0, 1.0 - n_censor / at_risk)
        surv_values.append(max(float(surv), eps))

    return unique_times.astype(float), np.asarray(surv_values, dtype=float)


def _step_value_at(query_times, step_times, step_values, default: float = 1.0):
    query_times = np.asarray(query_times, dtype=float)
    if len(step_times) == 0:
        return np.full_like(query_times, fill_value=default, dtype=float)

    idx = np.searchsorted(step_times, query_times, side="right") - 1
    out = np.full_like(query_times, fill_value=default, dtype=float)
    valid = idx >= 0
    out[valid] = step_values[idx[valid]]
    return out


def concordance_index_ipcw(
    train_event_times,
    train_events,
    test_event_times,
    test_risk_scores,
    test_events,
    tau: Optional[float] = None,
    eps: float = 1e-12,
) -> float:
    """Uno/IPCW C-index where higher risk_score means shorter survival."""
    train_event_times = np.asarray(train_event_times, dtype=float)
    train_events = np.asarray(train_events, dtype=int)
    times = np.asarray(test_event_times, dtype=float)
    scores = np.asarray(test_risk_scores, dtype=float)
    events = np.asarray(test_events, dtype=int)

    train_finite = np.isfinite(train_event_times) & np.isfinite(train_events)
    train_event_times = train_event_times[train_finite]
    train_events = train_events[train_finite]

    finite = np.isfinite(times) & np.isfinite(scores) & np.isfinite(events)
    times, scores, events = times[finite], scores[finite], events[finite]

    if len(train_event_times) == 0 or len(times) == 0:
        return 0.0

    if tau is None:
        tau = float(np.max(train_event_times))

    km_times, km_values = _km_censoring_survival(train_event_times, train_events, eps=eps)
    g_at_times = _step_value_at(times, km_times, km_values, default=1.0)

    weighted_concordant = 0.0
    weighted_tied = 0.0
    weighted_permissible = 0.0

    n = len(times)
    for i in range(n):
        if events[i] != 1 or times[i] > tau:
            continue

        g_i = max(float(g_at_times[i]), eps)
        weight = 1.0 / (g_i * g_i)

        for j in range(n):
            if times[i] < times[j]:
                weighted_permissible += weight
                if scores[i] > scores[j]:
                    weighted_concordant += weight
                elif scores[i] == scores[j]:
                    weighted_tied += weight

    if weighted_permissible <= 0.0:
        return 0.0
    return float((weighted_concordant + 0.5 * weighted_tied) / weighted_permissible)
