from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from baselines.common import (
    EarlyStopping,
    TITAN_PS_ARG,
    batch_to_events,
    concordance_index,
    evaluate_loader,
    forward_bag_batch,
    make_optimizer,
    sample_patches,
    sanitize_features,
    stable_int_hash,
    to_numpy_list,
    unpack_batch,
)
from core.survival import cox_ph_loss


@dataclass
class ReplayBatch:
    features: List[torch.Tensor]
    coords: List[torch.Tensor]
    times: torch.Tensor
    censorship: torch.Tensor
    slide_ids: List[Any]
    task_ids: torch.Tensor
    risk_old: Optional[torch.Tensor] = None
    z_old: Optional[torch.Tensor] = None


class WSIReplayBuffer:
    """Reservoir replay buffer for WSI-level survival samples.

    It stores one item per WSI/patient: the pre-extracted patch features,
    coordinates, survival labels, task id, and optional distillation targets.

    For DER/DER++, `risk_old` is the scalar Cox risk and `z_old` is the
    slide-level embedding from the model at the moment the sample entered the
    buffer. Keeping `z_old` makes distillation less fragile than using the Cox
    scalar alone.
    """

    def __init__(self, buffer_size: int, seed: int = 0):
        self.buffer_size = int(buffer_size)
        self.rng = np.random.default_rng(int(seed))
        self.num_seen = 0
        self.items: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.items)

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def state_dict(self) -> Dict[str, Any]:
        # Torch can serialize tensors inside this list. RNG state is stored for
        # reproducibility when resuming manually.
        return {
            "buffer_size": self.buffer_size,
            "num_seen": self.num_seen,
            "items": self.items,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.buffer_size = int(state["buffer_size"])
        self.num_seen = int(state["num_seen"])
        self.items = list(state["items"])
        self.rng = np.random.default_rng()
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]

    def _reservoir_insert(self, item: Dict[str, Any]) -> None:
        if self.buffer_size <= 0:
            self.num_seen += 1
            return
        if self.num_seen < self.buffer_size:
            self.items.append(item)
        else:
            j = int(self.rng.integers(0, self.num_seen + 1))
            if j < self.buffer_size:
                self.items[j] = item
        self.num_seen += 1

    def add_batch(
        self,
        feature_list: Sequence[torch.Tensor],
        coord_list: Sequence[torch.Tensor],
        survival_time: torch.Tensor | Sequence[float],
        censorship: torch.Tensor | Sequence[float],
        slide_ids: Sequence[Any],
        task_id: int,
        risks: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        if self.buffer_size <= 0:
            return

        times = survival_time.detach().cpu().view(-1).float() if torch.is_tensor(survival_time) else torch.tensor(survival_time, dtype=torch.float32).view(-1)
        cens = censorship.detach().cpu().view(-1).float() if torch.is_tensor(censorship) else torch.tensor(censorship, dtype=torch.float32).view(-1)
        risk_cpu = None if risks is None else risks.detach().cpu().view(-1).float()
        z_cpu = None if embeddings is None else embeddings.detach().cpu().float()

        if slide_ids is None:
            slide_ids = [None] * len(feature_list)

        for i, (features, coords) in enumerate(zip(feature_list, coord_list)):
            item = {
                "features": features.detach().cpu().clone(),
                "coords": coords.detach().cpu().clone(),
                "time": float(times[i].item()),
                "censorship": float(cens[i].item()),
                "slide_id": slide_ids[i],
                "task_id": int(task_id),
                "risk_old": None if risk_cpu is None else float(risk_cpu[i].item()),
                "z_old": None if z_cpu is None else z_cpu[i].detach().cpu().clone(),
            }
            self._reservoir_insert(item)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        n = min(int(batch_size), len(self.items))
        idxs = self.rng.choice(len(self.items), size=n, replace=False)
        batch = [self.items[int(i)] for i in idxs]

        features = [x["features"] for x in batch]
        coords = [x["coords"] for x in batch]
        times = torch.tensor([x["time"] for x in batch], dtype=torch.float32)
        censorship = torch.tensor([x["censorship"] for x in batch], dtype=torch.float32)
        slide_ids = [x["slide_id"] for x in batch]
        task_ids = torch.tensor([x["task_id"] for x in batch], dtype=torch.long)

        risks = [x.get("risk_old") for x in batch]
        risk_old = torch.tensor(risks, dtype=torch.float32) if all(v is not None for v in risks) else None

        zs = [x.get("z_old") for x in batch]
        z_old = torch.stack(zs, dim=0).float() if all(v is not None for v in zs) else None

        return ReplayBatch(features, coords, times, censorship, slide_ids, task_ids, risk_old, z_old)


def forward_bag_batch_with_z(
    model,
    feature_list: Sequence[torch.Tensor],
    coord_list: Sequence[torch.Tensor],
    device: torch.device,
    num_patches: int,
    ps: torch.Tensor,
    slide_ids: Optional[Sequence[Any]] = None,
    sampling_mode: str = "random",
    sampling_seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward a list of WSI bags and return scalar risks plus embeddings.

    This mirrors `baseline_survival_common.forward_bag_batch` but exposes the
    TITAN slide embedding so DER/DER++/LwF can distill features, not only a
    single Cox risk scalar.
    """
    risks, zs = [], []
    if slide_ids is None:
        slide_ids = [None] * len(feature_list)

    for bag_idx, (features, coords, slide_id) in enumerate(zip(feature_list, coord_list, slide_ids)):
        features = sanitize_features(features.to(device, non_blocking=True))
        coords = coords.long().to(device, non_blocking=True)
        seed = None
        if sampling_mode in {"random", "fixed_random"}:
            seed = stable_int_hash(slide_id if slide_id is not None else bag_idx, base_seed=sampling_seed)
        features, coords = sample_patches(features, coords, num_patches, mode=sampling_mode, seed=seed)

        # The SurvivalModel in baseline_survival_common is TITAN backbone + Cox head.
        # Calling the submodules directly avoids changing the existing file.
        z = model.backbone(features, coords, ps)
        z = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
        risk = model.mlp(z)
        risks.append(risk.view(-1))
        zs.append(z.view(1, -1))

    return torch.cat(risks, dim=0), torch.cat(zs, dim=0)


def freeze_model(model) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def distill_loss(
    risk_now: torch.Tensor,
    z_now: Optional[torch.Tensor],
    risk_old: Optional[torch.Tensor],
    z_old: Optional[torch.Tensor],
    device: torch.device,
    risk_weight: float = 1.0,
    z_weight: float = 1.0,
    normalize_z: bool = True,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=device)
    if risk_old is not None and float(risk_weight) != 0.0:
        loss = loss + float(risk_weight) * F.mse_loss(risk_now.float().view(-1), risk_old.to(device).float().view(-1))
    if z_now is not None and z_old is not None and float(z_weight) != 0.0:
        target = z_old.to(device).float()
        current = z_now.float()
        if normalize_z:
            target = F.normalize(target, dim=1)
            current = F.normalize(current, dim=1)
        loss = loss + float(z_weight) * F.mse_loss(current, target)
    return loss


class EWCHelper:
    """Online EWC helper for Cox survival models.

    After each task, call `update_after_task(...)` to estimate a diagonal Fisher
    approximation by squared gradients of the Cox loss. During later tasks, add
    `ewc_lambda * penalty(model)` to the task loss.
    """

    def __init__(
        self,
        gamma: float = 0.9,
        include_regex: str = ".*",
        exclude_regex: str = "",
        max_batches: int = 0,
    ):
        self.gamma = float(gamma)
        self.include_regex = include_regex or ".*"
        self.exclude_regex = exclude_regex or ""
        self.max_batches = int(max_batches)
        self.fisher: Dict[str, torch.Tensor] = {}
        self.theta_star: Dict[str, torch.Tensor] = {}

    def _selected_params(self, model) -> Iterable[Tuple[str, torch.nn.Parameter]]:
        inc = re.compile(self.include_regex)
        exc = re.compile(self.exclude_regex) if self.exclude_regex else None
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if not inc.search(name):
                continue
            if exc is not None and exc.search(name):
                continue
            yield name, p

    def penalty(self, model, device: torch.device) -> torch.Tensor:
        if not self.fisher or not self.theta_star:
            return torch.tensor(0.0, device=device)
        loss = torch.tensor(0.0, device=device)
        for name, p in self._selected_params(model):
            if name not in self.fisher:
                continue
            f = self.fisher[name].to(device, non_blocking=True)
            old = self.theta_star[name].to(device, non_blocking=True)
            loss = loss + (f * (p - old).pow(2)).sum()
        return loss

    def update_after_task(
        self,
        model,
        train_loader,
        cfg,
        device: torch.device,
        task_id: int,
        num_patches: Optional[int] = None,
    ) -> Dict[str, Any]:
        model.train()
        num_patches = int(num_patches or cfg.training.train_num_patches)
        ps = torch.tensor(TITAN_PS_ARG).int().to(device)
        eps = float(cfg.survival.get("cox_eps", 1e-7))

        new_fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p.detach().cpu()) for name, p in self._selected_params(model)
        }
        n_batches = 0
        skipped = 0

        for batch_idx, batch in enumerate(train_loader):
            if self.max_batches > 0 and batch_idx >= self.max_batches:
                break
            features, coords, _time_bin, censorship, survival_time, slide_id = unpack_batch(batch)
            censorship = censorship.float().to(device, non_blocking=True)
            survival_time = survival_time.float().to(device, non_blocking=True)
            events = batch_to_events(censorship)

            model.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    risks = forward_bag_batch(
                                model,
                                features,
                                coords,
                                device,
                                num_patches,
                                ps,
                                slide_ids=slide_id,
                                sampling_mode="fixed_random",
                                sampling_seed=int(cfg.training.seed) + 300000 + int(task_id),
                                )
                    loss = cox_ph_loss(risks, survival_time, events, eps=eps)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                for name, p in self._selected_params(model):
                    if p.grad is not None:
                        new_fisher[name] += p.grad.detach().cpu().float().pow(2)
                n_batches += 1
            except (FloatingPointError, RuntimeError) as exc:
                skipped += 1
                print(f"[EWC] skip Fisher batch {batch_idx}: {exc}")
                continue

        denom = max(1, n_batches)
        for name in new_fisher:
            new_fisher[name] /= denom

        if not self.fisher:
            self.fisher = new_fisher
        else:
            for name, f in new_fisher.items():
                old = self.fisher.get(name, torch.zeros_like(f))
                self.fisher[name] = self.gamma * old + f

        self.theta_star = {name: p.detach().cpu().clone() for name, p in self._selected_params(model)}
        model.zero_grad(set_to_none=True)
        return {"task_id": int(task_id), "fisher_batches": n_batches, "fisher_skipped": skipped, "num_params": len(self.theta_star)}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma,
            "include_regex": self.include_regex,
            "exclude_regex": self.exclude_regex,
            "max_batches": self.max_batches,
            "fisher": self.fisher,
            "theta_star": self.theta_star,
        }


def train_one_task_cl(
    train_loader,
    val_loader,
    model,
    cfg,
    device: torch.device,
    method: str,
    task_id: int,
    buffer: Optional[WSIReplayBuffer] = None,
    ewc: Optional[EWCHelper] = None,
    old_model=None,
    phase_name: str = "cl",
    buffer_batch_size: int = 8,
    er_beta: float = 1.0,
    der_alpha: float = 0.5,
    der_beta: float = 1.0,
    distill_risk_weight: float = 1.0,
    distill_z_weight: float = 1.0,
    lwf_alpha: float = 0.5,
    lwf_risk_weight: float = 1.0,
    lwf_z_weight: float = 1.0,
    ewc_lambda: float = 1e-4,
) -> Tuple[Any, Dict[str, Any]]:
    method = method.lower()
    assert method in {"seqft", "finetune", "ewc", "lwf", "er", "der", "derpp"}, method

    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    num_epochs = int(cfg.training.num_epochs)
    train_num_patches = int(cfg.training.train_num_patches)
    eval_num_patches = int(cfg.training.get("eval_num_patches", train_num_patches))
    val_sampling_mode = str(cfg.training.get("val_sampling_mode", "fixed_random"))
    ps = torch.tensor(TITAN_PS_ARG).int().to(device)
    eps = float(cfg.survival.get("cox_eps", 1e-7))
    early = EarlyStopping(
        patience=int(cfg.training.get("early_stopping_patience", 5)),
        min_delta=float(cfg.training.get("early_stopping_min_delta", cfg.training.get("checkpoint_cindex_min_delta", 0.0))),
        verbose=True,
    )

    if old_model is not None:
        freeze_model(old_model)

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        total_task_loss, total_replay_loss, total_distill_loss, total_ewc_loss, total_lwf_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        train_times, train_cens, train_risks = [], [], []
        skipped = 0

        for batch in train_loader:
            buffer_risks = None
            buffer_z = None
            
            features, coords, _time_bin, censorship, survival_time, slide_id = unpack_batch(batch)
            censorship = censorship.float().to(device, non_blocking=True)
            survival_time = survival_time.float().to(device, non_blocking=True)
            events = batch_to_events(censorship)

            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    if method in {"der", "derpp", "lwf"}:
                        risks, z = forward_bag_batch_with_z(
                            model, features, coords, device, train_num_patches, ps,
                            slide_ids=slide_id, sampling_mode="random", sampling_seed=int(cfg.training.seed) + epoch,
                        )
                    else:
                        risks = forward_bag_batch(
                            model, features, coords, device, train_num_patches, ps,
                            slide_ids=slide_id, sampling_mode="random", sampling_seed=int(cfg.training.seed) + epoch,
                        )
                        z = None

                    task_loss = cox_ph_loss(risks, survival_time, events, eps=eps)
                    loss = task_loss

                    replay_loss = torch.tensor(0.0, device=device)
                    dist_loss = torch.tensor(0.0, device=device)
                    lwf_loss = torch.tensor(0.0, device=device)
                    ewc_loss = torch.tensor(0.0, device=device)

                    if method in {"er", "der", "derpp"} and buffer is not None and not buffer.is_empty():
                        replay = buffer.sample(buffer_batch_size)
                        bt = replay.times.to(device, non_blocking=True)
                        bc = replay.censorship.to(device, non_blocking=True)
                        be = batch_to_events(bc)

                        buffer_risks = None
                        buffer_z = None
                        
                        if method in {"der", "derpp"}:
                            replay_distill_seed = int(cfg.training.seed) + 200000
                            brisks, bz = forward_bag_batch_with_z(
                                model,
                                replay.features,
                                replay.coords,
                                device,
                                train_num_patches,
                                ps,
                                slide_ids=replay.slide_ids,
                                sampling_mode="fixed_random",
                                sampling_seed=replay_distill_seed,
                                )
                            
                            dist_loss = distill_loss(
                                brisks, bz, replay.risk_old, replay.z_old, device,
                                risk_weight=distill_risk_weight, z_weight=distill_z_weight,
                            )
                            loss = loss + float(der_alpha) * dist_loss
                            if method == "derpp":
                                replay_loss = cox_ph_loss(brisks, bt, be, eps=eps)
                                loss = loss + float(der_beta) * replay_loss
                        elif method == "er":
                            brisks = forward_bag_batch(
                                model, replay.features, replay.coords, device, train_num_patches, ps,
                                slide_ids=replay.slide_ids, sampling_mode="random", sampling_seed=int(cfg.training.seed) + epoch,
                            )
                            replay_loss = cox_ph_loss(brisks, bt, be, eps=eps)
                            loss = loss + float(er_beta) * replay_loss

                    if method == "lwf" and old_model is not None:
                        distill_seed = int(cfg.training.seed) + 100000 + int(task_id)
                        
                        with torch.no_grad():
                            old_risks, old_z = forward_bag_batch_with_z(
                                old_model, features, coords, device, train_num_patches, ps,
                                slide_ids=slide_id, sampling_mode="fixed_random", sampling_seed=distill_seed,
                            )
                        
                        cur_risks_distill, cur_z_distill = forward_bag_batch_with_z(
                            model, features, coords, device, train_num_patches, ps,
                            slide_ids=slide_id,
                            sampling_mode="fixed_random",
                            sampling_seed=distill_seed,
                            )    
                            
                        lwf_loss = distill_loss(
                            cur_risks_distill,
                            cur_z_distill,
                            old_risks,
                            old_z,
                            device,
                            risk_weight=lwf_risk_weight,
                            z_weight=lwf_z_weight,
                            )
                        loss = loss + float(lwf_alpha) * lwf_loss

                    if method == "ewc" and ewc is not None and task_id > 0:
                        ewc_loss = ewc.penalty(model, device)
                        loss = loss + float(ewc_lambda) * ewc_loss

                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # Store current samples after a successful update. For DER/DER++
                # targets are the pre-update predictions from this iteration,
                # matching the standard dark-replay idea.
                if method in {"er", "der", "derpp"} and buffer is not None:
                    buffer.add_batch(
                        features,
                        coords,
                        survival_time.detach().cpu(),
                        censorship.detach().cpu(),
                        slide_id,
                        task_id=task_id,
                        risks=buffer_risks.detach().cpu() if buffer_risks is not None else None,
                        embeddings=buffer_z.detach().cpu() if buffer_z is not None else None,
                    )

            except (FloatingPointError, RuntimeError) as exc:
                skipped += 1
                print(f"[Warning] Skip train batch: {exc}")
                continue

            total_loss += float(loss.detach().cpu().item())
            total_task_loss += float(task_loss.detach().cpu().item())
            total_replay_loss += float(replay_loss.detach().cpu().item())
            total_distill_loss += float(dist_loss.detach().cpu().item())
            total_lwf_loss += float(lwf_loss.detach().cpu().item())
            total_ewc_loss += float(ewc_loss.detach().cpu().item())
            n_batches += 1
            train_risks.extend(risks.detach().float().cpu().numpy().tolist())
            train_times.extend(to_numpy_list(survival_time.detach().cpu()))
            train_cens.extend(np.asarray(to_numpy_list(censorship.detach().cpu()), dtype=int).tolist())

        train_events = (1 - np.asarray(train_cens, dtype=int)).astype(int).tolist()
        train_c = concordance_index(train_times, train_risks, train_events) if train_risks else 0.0
        val_eval = evaluate_loader(
            val_loader, model, device, eval_num_patches, cfg,
            sampling_mode=val_sampling_mode, sampling_seed=int(cfg.training.seed), return_loss=True,
        )
        denom = max(1, n_batches)
        buffer_len = len(buffer) if buffer is not None else 0
        print(
            f"[{phase_name}] epoch {epoch}, loss={total_loss/denom:.4f}, task={total_task_loss/denom:.4f}, "
            f"replay={total_replay_loss/denom:.4f}, distill={total_distill_loss/denom:.4f}, "
            f"lwf={total_lwf_loss/denom:.4f}, ewc={total_ewc_loss/denom:.4f}, "
            f"train_cindex={train_c:.4f}, val_loss={val_eval['loss']:.4f}, val_cindex={val_eval['cindex']:.4f}, "
            f"buffer={buffer_len}, skipped={skipped}"
        )
        early(val_eval["cindex"], val_eval["loss"], model, epoch)
        if early.early_stop:
            break

    model = early.restore(model)
    model.eval()
    info = early.state_dict()
    info.update({"method": method, "task_id": int(task_id), "buffer_size_after_task": len(buffer) if buffer is not None else 0})
    return model, info
