from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel

from core.config import add_config_arg, ensure_dir, get_num_prompt_bins, get_task_order, load_config, seed_everything
from data_utils.datasets import Sequential_Generic_MIL_Survival_Dataset
from data_utils.prompts import (
    blca_survival_prompts,
    brca_survival_prompts,
    luad_survival_prompts,
    task_level_prompts,
    ucec_survival_prompts,
)
from core.survival import (
    cox_ph_loss,
    censored_aware_prompt_loss,
    concordance_index,
    concordance_index_ipcw,
    ordinal_prompt_kl_loss,
)
from core.utils import seed_torch

PROMPT_FN_BY_TASK = {
    "blca": blca_survival_prompts,
    "brca": brca_survival_prompts,
    "luad": luad_survival_prompts,
    "ucec": ucec_survival_prompts,
}

EMBED_DIM = 768
TITAN_PS_ARG = 1024


class CustomSequential(nn.Module):
    """Stage-1 task-specific survival model.

    The backbone is TITAN's slide aggregator/vision encoder. The head is a
    task-specific Cox/DeepSurv scalar risk head. Prompt embeddings are NOT used
    as the final classifier; they are auxiliary semantic regularizers.
    """

    def __init__(self, model, survival_head):
        super().__init__()
        self.backbone = model.vision_encoder
        # Keep name `mlp` for compatibility with merging scripts that strip or
        # optionally process non-backbone heads.
        self.mlp = survival_head

    def forward(self, features, coords, ps, return_hidden: bool = False):
        z = self.backbone(features, coords, ps)
        return self.mlp(z, return_hidden=return_hidden)


class TaskSpecificCoxHead(nn.Module):
    """Strong task-specific survival head for offline fine-tuning.

    Output is a scalar Cox risk score, so the head is bin-free. Prompt bins are
    only used by the auxiliary branch through the hidden prognosis feature.
    """

    def __init__(self, in_dim: int = EMBED_DIM, hidden_dim: int = 512, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.risk = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor, return_hidden: bool = False):
        z = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
        hidden = self.net(z)
        risk = self.risk(hidden).squeeze(-1).clamp(min=-20.0, max=20.0)
        if return_hidden:
            return risk, hidden
        return risk


class PromptAuxiliaryHead(nn.Module):
    """Projection from survival hidden state to frozen TITAN prompt space."""

    def __init__(self, hidden_dim: int, prompt_embeddings: torch.Tensor, temperature: float = 0.07):
        super().__init__()
        if prompt_embeddings.ndim != 2:
            raise ValueError(f"prompt_embeddings must be [K, D], got {tuple(prompt_embeddings.shape)}")
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, prompt_embeddings.shape[1]),
        )
        prompt_embeddings = F.normalize(prompt_embeddings.float(), dim=1)
        self.register_buffer("prompt_embeddings", prompt_embeddings)
        self.temperature = float(temperature)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = F.normalize(self.proj(hidden.float()), dim=1)
        return (h @ self.prompt_embeddings.T) / max(self.temperature, 1e-6)


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    warmup_length = max(1, int(warmup_length))
    steps = max(warmup_length + 1, int(steps))

    def _assign_learning_rate(new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr * param_group.get("lr_scale", 1.0)

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = base_lr * (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        _assign_learning_rate(lr)
        return lr

    return _lr_adjuster


class EarlyStopping:
    """Early stop and checkpoint by validation C-index.

    This setting is appropriate when the available held-out split is validation
    only and the primary model-selection target is ranking performance. val_loss
    is still stored and used only as a tie-breaker when val_cindex is unchanged.
    """

    def __init__(
        self,
        patience: int = 5,
        cindex_min_delta: float = 0.0,
        verbose: bool = False,
    ):
        self.patience = int(patience)
        self.cindex_min_delta = float(cindex_min_delta)
        self.verbose = bool(verbose)

        self.counter = 0
        self.early_stop = False

        self.best_cindex = -float("inf")
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_model_weights = None
        self.best_prompt_aux_weights = None

    @staticmethod
    def _copy_state_dict(module: nn.Module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def __call__(self, val_cindex, val_loss, model, prompt_aux, epoch):
        if not np.isfinite(val_cindex):
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience} (non-finite val_cindex)")
            if self.counter >= self.patience:
                self.early_stop = True
            return

        improved_cindex = val_cindex > self.best_cindex + self.cindex_min_delta
        tie_better_loss = (
            abs(val_cindex - self.best_cindex) <= self.cindex_min_delta
            and np.isfinite(val_loss)
            and val_loss < self.best_val_loss
        )

        if improved_cindex or tie_better_loss:
            self.best_cindex = float(val_cindex)
            self.best_val_loss = float(val_loss) if np.isfinite(val_loss) else self.best_val_loss
            self.best_epoch = int(epoch)
            self.counter = 0
            self.best_model_weights = self._copy_state_dict(model)
            self.best_prompt_aux_weights = self._copy_state_dict(prompt_aux)
            return

        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter}/{self.patience} (val_cindex did not improve)")
        if self.counter >= self.patience:
            self.early_stop = True

    def state_dict(self):
        return {
            "early_stop_monitor": "val_cindex",
            "checkpoint_monitor": "val_cindex",
            "patience": self.patience,
            "cindex_min_delta": self.cindex_min_delta,
            "best_cindex": self.best_cindex,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }

def make_grad_scaler():
    if torch.cuda.is_available():
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.amp.GradScaler("cpu", enabled=False)


def _stable_int_hash(value, base_seed: int = 0) -> int:
    """Stable hash for deterministic validation sampling.

    Python's built-in hash() is intentionally randomized between processes, so
    use md5 to make the sampled patch subset reproducible across epochs/runs.
    """
    if isinstance(value, (list, tuple)):
        value = "_".join(str(v) for v in value)
    key = f"{base_seed}_{str(value)}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def sample_patches(features, coords, k, mode: str = "random", seed: int | None = None):
    """Sample patches from one WSI bag.

    mode="random":
        stochastic subset, used for training as bag-level augmentation.
    mode="fixed_random":
        deterministic pseudo-random subset, used for validation/test so the
        same slide sees the same patch subset every epoch.
    mode="first":
        first K patches, deterministic but can be biased if H5 patch order is spatial.
    """
    k = min(int(k), features.shape[0])
    n = int(features.shape[0])

    if mode == "random":
        indices = torch.randperm(n, device=features.device)[:k]
    elif mode == "fixed_random":
        if seed is None:
            seed = 0
        # Generate the permutation on CPU for compatibility across CUDA versions.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) % (2**31 - 1))
        indices = torch.randperm(n, generator=generator, device="cpu")[:k].to(features.device)
    elif mode == "first":
        indices = torch.arange(k, device=features.device)
    else:
        raise ValueError(f"Unknown patch sampling mode: {mode!r}")

    return features[indices, :], coords[indices, :]


def sanitize_features(features: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _to_numpy_list(tensor_or_array):
    if torch.is_tensor(tensor_or_array):
        return tensor_or_array.detach().cpu().numpy().tolist()
    return np.asarray(tensor_or_array).tolist()


def _events_from_censorship(censorship: torch.Tensor | np.ndarray | list) -> np.ndarray:
    c = np.asarray(_to_numpy_list(censorship), dtype=int)
    return (1 - c).astype(int)


def forward_bag_batch(
    model,
    feature_list,
    coord_list,
    device,
    num_patches,
    ps,
    slide_ids=None,
    sampling_mode: str = "random",
    sampling_seed: int = 0,
):
    risks: List[torch.Tensor] = []
    hiddens: List[torch.Tensor] = []

    if slide_ids is None:
        slide_ids = [None] * len(feature_list)

    for bag_idx, (features, coords, slide_id) in enumerate(zip(feature_list, coord_list, slide_ids)):
        features = sanitize_features(features.to(device, non_blocking=True))
        coords = coords.long().to(device, non_blocking=True)

        seed = None
        if sampling_mode == "fixed_random":
            seed = _stable_int_hash(slide_id if slide_id is not None else bag_idx, base_seed=sampling_seed)

        features, coords = sample_patches(
            features,
            coords,
            num_patches,
            mode=sampling_mode,
            seed=seed,
        )
        risk, hidden = model(features, coords, ps, return_hidden=True)
        risks.append(risk.view(-1))
        hiddens.append(hidden)
    return torch.cat(risks, dim=0), torch.cat(hiddens, dim=0)


def compute_total_loss(
    risks: torch.Tensor,
    hiddens: torch.Tensor,
    prompt_aux: PromptAuxiliaryHead,
    survival_time: torch.Tensor,
    censorship: torch.Tensor,
    time_bin: torch.Tensor,
    cfg,
):
    events = 1.0 - censorship.float()
    l_cox = cox_ph_loss(risks, survival_time, events, eps=float(cfg.survival.get("cox_eps", 1e-7)))
    prompt_logits = prompt_aux(hiddens)
    l_prompt = censored_aware_prompt_loss(prompt_logits, time_bin, censorship)
    l_ord = ordinal_prompt_kl_loss(
        prompt_logits,
        time_bin,
        censorship,
        tau=float(cfg.survival.get("ordinal_temperature", 1.0)),
    )
    total = (
        float(cfg.training.get("cox_loss_weight", 1.0)) * l_cox
        + float(cfg.training.get("prompt_loss_weight", 0.1)) * l_prompt
        + float(cfg.training.get("ordinal_loss_weight", 0.05)) * l_ord
    )
    return total, {"cox": l_cox.detach(), "prompt": l_prompt.detach(), "ordinal": l_ord.detach()}


def _dataset_times_events(loader):
    """Return stable train split times/events for IPCW censoring estimation."""
    dataset = loader.dataset
    if not hasattr(dataset, "slide_data") or not hasattr(dataset, "time_col"):
        return None, None
    df = dataset.slide_data
    times = df[dataset.time_col].values.astype(float)
    events = (1 - df["censorship"].values.astype(int)).astype(int)
    return times.tolist(), events.tolist()


def evaluate_full_split(
    loader,
    model,
    prompt_aux,
    device,
    num_patches,
    ps,
    cfg,
    sampling_mode: str = "fixed_random",
    sampling_seed: int = 0,
):
    """Evaluate an entire split and compute Cox/prompt/total loss once.

    Cox partial likelihood depends on the risk set. For validation, computing it
    mini-batch-by-mini-batch makes val_loss noisy and inconsistent, so this
    function first collects all risks/hidden states from the full split, then
    computes cox_ph_loss and auxiliary losses once on the concatenated tensors.
    """
    model.eval()
    prompt_aux.eval()

    all_risks: List[torch.Tensor] = []
    all_hiddens: List[torch.Tensor] = []
    all_times: List[torch.Tensor] = []
    all_censorships: List[torch.Tensor] = []
    all_bins: List[torch.Tensor] = []
    skipped_batches = 0

    with torch.no_grad():
        for features, coords, time_bin, censorship, survival_time, slide_id in loader:
            time_bin = time_bin.long().to(device, non_blocking=True)
            censorship = censorship.float().to(device, non_blocking=True)
            survival_time = survival_time.float().to(device, non_blocking=True)

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    risks, hiddens = forward_bag_batch(
                        model,
                        features,
                        coords,
                        device,
                        num_patches,
                        ps,
                        slide_ids=slide_id,
                        sampling_mode=sampling_mode,
                        sampling_seed=sampling_seed,
                    )
            except (FloatingPointError, RuntimeError) as exc:
                skipped_batches += 1
                print(f"[Warning] Skip eval batch: {exc}")
                continue

            all_risks.append(risks.float())
            all_hiddens.append(hiddens.float())
            all_times.append(survival_time)
            all_censorships.append(censorship)
            all_bins.append(time_bin)

    if not all_risks:
        nan_parts = {"cox": float("nan"), "prompt": float("nan"), "ordinal": float("nan")}
        return {
            "loss": float("nan"),
            "parts": nan_parts,
            "times": [],
            "censorships": [],
            "events": [],
            "risks": [],
            "cindex": 0.0,
            "skipped_batches": skipped_batches,
        }

    risks_all = torch.cat(all_risks, dim=0)
    hiddens_all = torch.cat(all_hiddens, dim=0)
    times_all = torch.cat(all_times, dim=0)
    censorships_all = torch.cat(all_censorships, dim=0)
    bins_all = torch.cat(all_bins, dim=0)

    loss, parts = compute_total_loss(
        risks=risks_all,
        hiddens=hiddens_all,
        prompt_aux=prompt_aux,
        survival_time=times_all,
        censorship=censorships_all,
        time_bin=bins_all,
        cfg=cfg,
    )

    times_np = times_all.detach().cpu().numpy().astype(float).tolist()
    censorships_np = censorships_all.detach().cpu().numpy().astype(int).tolist()
    events_np = (1 - np.asarray(censorships_np, dtype=int)).astype(int).tolist()
    risks_np = risks_all.detach().cpu().numpy().astype(float).tolist()
    cindex = concordance_index(times_np, risks_np, events_np)

    return {
        "loss": float(loss.detach().cpu().item()),
        "parts": {k: float(v.detach().cpu().item()) for k, v in parts.items()},
        "times": times_np,
        "censorships": censorships_np,
        "events": events_np,
        "risks": risks_np,
        "cindex": cindex,
        "skipped_batches": skipped_batches,
    }


def train(train_loader, val_loader, model, prompt_aux, num_epochs, lr, head_lr, weight_decay, device, train_num_patches, cfg):
    named_parameters = list(model.named_parameters()) + [(f"prompt_aux.{n}", p) for n, p in prompt_aux.named_parameters()]

    # Param groups: train TITAN aggregator conservatively, train survival/prompt heads faster.
    backbone_params, head_params, no_decay_params = [], [], []
    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name or "ln" in name.lower() or "norm" in name.lower():
            no_decay_params.append(p)
        elif name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": no_decay_params, "weight_decay": 0.0, "lr_scale": 1.0},
            {"params": backbone_params, "weight_decay": weight_decay, "lr_scale": 1.0},
            {"params": head_params, "weight_decay": weight_decay, "lr_scale": float(head_lr) / max(float(lr), 1e-12)},
        ],
        lr=lr,
    )
    lr_scheduler = cosine_lr(optimizer, lr, int(len(train_loader) * num_epochs * 0.1), len(train_loader) * num_epochs)

    scaler = make_grad_scaler()
    early_stopping = EarlyStopping(
        patience=int(cfg.training.get("early_stopping_patience", 5)),
        cindex_min_delta=float(cfg.training.get("early_stopping_min_delta", cfg.training.get("checkpoint_cindex_min_delta", 0.0))),
        verbose=True,
    )
    step = 0
    ps = torch.tensor(TITAN_PS_ARG).int().to(device)

    for epoch in tqdm(range(num_epochs), desc="train epochs"):
        model.train()
        prompt_aux.train()
        total_train_loss = 0.0
        train_loss_parts = {"cox": 0.0, "prompt": 0.0, "ordinal": 0.0}
        valid_train_batches = 0
        skipped_train_batches = 0
        train_times, train_censorships, train_risks = [], [], []

        for features, coords, time_bin, censorship, survival_time, _slide_id in tqdm(train_loader, leave=False):
            lr_scheduler(step)
            time_bin = time_bin.long().to(device, non_blocking=True)
            censorship = censorship.float().to(device, non_blocking=True)
            survival_time = survival_time.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    risks, hiddens = forward_bag_batch(
                        model,
                        features,
                        coords,
                        device,
                        train_num_patches,
                        ps,
                        slide_ids=_slide_id,
                        sampling_mode="random",
                    )
                    loss, parts = compute_total_loss(risks, hiddens, prompt_aux, survival_time, censorship, time_bin, cfg)
            except (FloatingPointError, RuntimeError) as exc:
                skipped_train_batches += 1
                print(f"[Warning] Skip train batch: {exc}")
                step += 1
                continue

            if not torch.isfinite(loss):
                skipped_train_batches += 1
                print("[Warning] Skip train batch because loss is non-finite.")
                step += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in list(model.parameters()) + list(prompt_aux.parameters()) if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_risks.extend(risks.detach().float().cpu().numpy().tolist())
            train_times.extend(_to_numpy_list(survival_time))
            train_censorships.extend(censorship.detach().cpu().numpy().astype(int).tolist())
            total_train_loss += float(loss.detach().cpu().item())
            for k in train_loss_parts:
                train_loss_parts[k] += float(parts[k].cpu().item())
            valid_train_batches += 1
            step += 1

        avg_train_loss = total_train_loss / max(1, valid_train_batches)
        train_events = (1 - np.asarray(train_censorships, dtype=int)).tolist()
        train_cindex = concordance_index(train_times, train_risks, train_events)
        train_cindex_ipcw = concordance_index_ipcw(train_times, train_events, train_times, train_risks, train_events)

        eval_num_patches = int(cfg.training.get("eval_num_patches", train_num_patches))
        val_sampling_mode = str(cfg.training.get("val_sampling_mode", "fixed_random"))
        val_eval = evaluate_full_split(
            loader=val_loader,
            model=model,
            prompt_aux=prompt_aux,
            device=device,
            num_patches=eval_num_patches,
            ps=ps,
            cfg=cfg,
            sampling_mode=val_sampling_mode,
            sampling_seed=int(cfg.training.seed),
        )

        avg_val_loss = val_eval["loss"]
        val_loss_parts = val_eval["parts"]
        skipped_val_batches = val_eval["skipped_batches"]
        val_times = val_eval["times"]
        val_events = val_eval["events"]
        val_risks = val_eval["risks"]
        val_cindex = val_eval["cindex"]

        ipcw_train_times, ipcw_train_events = _dataset_times_events(train_loader)
        if ipcw_train_times is None:
            ipcw_train_times, ipcw_train_events = train_times, train_events

        val_cindex_ipcw = concordance_index_ipcw(
            ipcw_train_times,
            ipcw_train_events,
            val_times,
            val_risks,
            val_events,
        )

        denom_tr = max(1, valid_train_batches)
        tqdm.write(
            f"epoch {epoch}, loss={avg_train_loss:.4f} "
            f"(cox={train_loss_parts['cox']/denom_tr:.4f}, prompt={train_loss_parts['prompt']/denom_tr:.4f}, ord={train_loss_parts['ordinal']/denom_tr:.4f}), "
            f"val_loss={avg_val_loss:.4f} "
            f"(cox={val_loss_parts['cox']:.4f}, prompt={val_loss_parts['prompt']:.4f}, ord={val_loss_parts['ordinal']:.4f}), "
            f"cindex={train_cindex:.4f}, cindex_ipcw={train_cindex_ipcw:.4f}, "
            f"val_cindex={val_cindex:.4f}, val_cindex_ipcw={val_cindex_ipcw:.4f}, "
            f"skipped_train={skipped_train_batches}, skipped_val={skipped_val_batches}, "
        )

        early_stopping(val_cindex, avg_val_loss, model, prompt_aux, epoch)
        if early_stopping.early_stop:
            break

    if early_stopping.best_model_weights is not None:
        print(
            f"Loading best-val-cindex checkpoint from epoch {early_stopping.best_epoch} "
            f"with val_cindex={early_stopping.best_cindex:.4f}, "
            f"val_loss={early_stopping.best_val_loss:.4f}. "
            f"Early stopping was monitored by val_cindex."
        )
        model.load_state_dict(early_stopping.best_model_weights)
        prompt_aux.load_state_dict(early_stopping.best_prompt_aux_weights)
    model.eval()
    prompt_aux.eval()
    return model, prompt_aux, early_stopping.state_dict()


def build_survival_prompt_embeddings(task: str, num_prompt_bins: int, device) -> torch.Tensor:
    titan_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)
    prompts, templates = PROMPT_FN_BY_TASK[task](num_levels=num_prompt_bins)
    with torch.autocast("cuda", torch.float16, enabled=torch.cuda.is_available()), torch.inference_mode():
        classifier = titan_model.zero_shot_classifier(prompts, templates, device=device)
    # TITAN returns [D, K]. Use [K, D].
    del titan_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return classifier.float().T.contiguous()


def build_task_prompt_embedding(task: str, device) -> torch.Tensor:
    titan_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)
    prompts, templates = task_level_prompts(task)
    with torch.autocast("cuda", torch.float16, enabled=torch.cuda.is_available()), torch.inference_mode():
        classifier = titan_model.zero_shot_classifier(prompts, templates, device=device)
    del titan_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return classifier.float().squeeze(-1).contiguous()


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Stage-1 offline task-specific TITAN survival fine-tuning")
    add_config_arg(parser)
    parser.add_argument("--save_dir", type=str, default=None, help="Override paths.finetuned_checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(cfg.training.seed), device)

    try:
        seed_torch(device, int(cfg.training.seed))
    except Exception:
        pass

    task_order = get_task_order(cfg)[: int(cfg.training.num_tasks)]
    num_tasks = len(task_order)
    num_prompt_bins = get_num_prompt_bins(cfg)
    save_root = ensure_dir(args.save_dir or cfg.paths.finetuned_checkpoints)

    seq_dataset = Sequential_Generic_MIL_Survival_Dataset(cfg=cfg)

    # Cache text embeddings once per run. They are frozen and task-specific.
    prompt_embeddings_by_task: Dict[str, torch.Tensor] = {}
    task_embeddings_by_task: Dict[str, torch.Tensor] = {}
    for task in task_order:
        print(f"Building prompt embeddings for task={task}")
        prompt_embeddings_by_task[task] = build_survival_prompt_embeddings(task, num_prompt_bins, device).cpu()
        task_embeddings_by_task[task] = build_task_prompt_embedding(task, device).cpu()

    for fold_id in range(int(cfg.training.num_folds)):
        fold_dir = ensure_dir(save_root / f"fold_{fold_id}")

        for task_id, task in enumerate(task_order):
            train_loader, val_loader, _ = seq_dataset.get_data_loaders(fold_id, task_id)
            titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)

            survival_head = TaskSpecificCoxHead(
                in_dim=EMBED_DIM,
                hidden_dim=int(cfg.training.get("survival_hidden_dim", 512)),
                dropout=float(cfg.training.get("survival_dropout", 0.25)),
            ).to(device)
            prompt_aux = PromptAuxiliaryHead(
                hidden_dim=int(cfg.training.get("survival_hidden_dim", 512)),
                prompt_embeddings=prompt_embeddings_by_task[task].to(device),
                temperature=float(cfg.survival.get("prompt_temperature", 0.07)),
            ).to(device)
            wrapped = CustomSequential(titan, survival_head).to(device)

            start = time.time()
            wrapped, prompt_aux, best_info = train(
                train_loader=train_loader,
                val_loader=val_loader,
                model=wrapped,
                prompt_aux=prompt_aux,
                num_epochs=int(cfg.training.num_epochs),
                lr=float(cfg.training.lr),
                head_lr=float(cfg.training.get("head_lr", cfg.training.lr)),
                weight_decay=float(cfg.training.weight_decay),
                device=device,
                train_num_patches=int(cfg.training.train_num_patches),
                cfg=cfg,
            )
            elapsed = time.time() - start

            out_path = fold_dir / f"finetune_task_{task_id}.pt"
            torch.save(wrapped.state_dict(), out_path)

            aux_path = fold_dir / f"prompt_aux_task_{task_id}.pt"
            torch.save(prompt_aux.state_dict(), aux_path)

            metadata = {
                "stage": "stage1_offline_task_specific_survival_finetuning",
                "task_id": task_id,
                "task_name": task,
                "num_prompt_bins": num_prompt_bins,
                "prompt_boundaries": train_loader.dataset.boundaries.tolist(),
                "checkpoint": str(out_path),
                "prompt_aux_checkpoint": str(aux_path),
                "task_prompt_embedding_shape": list(task_embeddings_by_task[task].shape),
                "survival_head": "TaskSpecificCoxHead",
                "main_loss": "Cox partial likelihood",
                "prompt_auxiliary": "censored-aware ordinal prognosis prompt loss",
                "early_stopping": best_info,
            }
            meta_path = fold_dir / f"finetune_task_{task_id}_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved {out_path}, {aux_path}, {meta_path} in {elapsed:.2f}s")

            del titan, wrapped, survival_head, prompt_aux
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
