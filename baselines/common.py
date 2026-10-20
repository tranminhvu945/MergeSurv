from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from core.survival import cox_ph_loss, concordance_index, concordance_index_ipcw

EMBED_DIM = 768
TITAN_PS_ARG = 1024


class CoxHead(nn.Module):
    """Shared Cox/DeepSurv scalar-risk head used by baseline methods."""

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return self.risk(self.net(z)).squeeze(-1).clamp(min=-20.0, max=20.0)


class SurvivalModel(nn.Module):
    """TITAN vision encoder + Cox head. Names match your stage-1 checkpoints: backbone.*, mlp.*"""

    def __init__(self, titan_model, head: CoxHead):
        super().__init__()
        self.backbone = titan_model.vision_encoder
        self.mlp = head

    def forward(self, features, coords, ps):
        z = self.backbone(features, coords, ps)
        return self.mlp(z)


def make_model(cfg, device):
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)
    head = CoxHead(
        hidden_dim=int(cfg.training.get("survival_hidden_dim", 512)),
        dropout=float(cfg.training.get("survival_dropout", 0.25)),
    ).to(device)
    return SurvivalModel(titan, head).to(device)


def stable_int_hash(value, base_seed: int = 0) -> int:
    if isinstance(value, (list, tuple)):
        value = "_".join(str(v) for v in value)
    key = f"{base_seed}_{str(value)}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def sample_patches(features: torch.Tensor, coords: torch.Tensor, k: int, mode: str, seed: int | None = None):
    n = int(features.shape[0])
    if mode == "all":
        idx = torch.arange(n, device=features.device)
    else:
        k = min(int(k), n)
        if mode == "random":
            if seed is None:
                idx = torch.randperm(n, device=features.device)[:k]
            else:
                g = torch.Generator(device="cpu")
                g.manual_seed(int(seed) % (2**31 - 1))
                idx = torch.randperm(n, generator=g, device="cpu")[:k].to(features.device)
        elif mode == "fixed_random":
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed or 0) % (2**31 - 1))
            idx = torch.randperm(n, generator=g, device="cpu")[:k].to(features.device)
        elif mode == "first":
            idx = torch.arange(k, device=features.device)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
    return features[idx], coords[idx]


def sanitize_features(features: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)


def unpack_batch(batch):
    if len(batch) == 6:
        features, coords, time_bin, censorship, survival_time, slide_id = batch
    elif len(batch) == 5:
        features, coords, time_bin, censorship, survival_time = batch
        slide_id = [None] * len(survival_time)
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    if torch.is_tensor(features):
        if len(survival_time) != 1:
            raise RuntimeError(
                "features is a single concatenated Tensor but batch has multiple labels. "
                "Use collate_MIL_survival that returns a list of WSI bags, or set batch_size=1."
            )
        features, coords = [features], [coords]
    return features, coords, time_bin, censorship, survival_time, slide_id


def forward_bag_batch(model, feature_list, coord_list, device, num_patches, ps, slide_ids=None, sampling_mode="random", sampling_seed=0):
    risks = []
    if slide_ids is None:
        slide_ids = [None] * len(feature_list)
    for bag_idx, (features, coords, slide_id) in enumerate(zip(feature_list, coord_list, slide_ids)):
        features = sanitize_features(features.to(device, non_blocking=True))
        coords = coords.long().to(device, non_blocking=True)
        seed = None
        if sampling_mode in {"random", "fixed_random"}:
            seed = stable_int_hash(slide_id if slide_id is not None else bag_idx, base_seed=sampling_seed)
        features, coords = sample_patches(features, coords, num_patches, mode=sampling_mode, seed=seed)
        risk = model(features, coords, ps)
        risks.append(risk.view(-1))
    return torch.cat(risks, dim=0)


def batch_to_events(censorship) -> torch.Tensor:
    return 1.0 - censorship.float()


def to_numpy_list(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().tolist()
    return np.asarray(x).tolist()


def dataset_times_events(dataset):
    if hasattr(dataset, "slide_data") and hasattr(dataset, "time_col"):
        df = dataset.slide_data
        times = df[dataset.time_col].values.astype(float).tolist()
        events = (1 - df["censorship"].values.astype(int)).astype(int).tolist()
        return times, events
    return None, None


def evaluate_loader(loader, model, device, num_patches, cfg, sampling_mode="fixed_random", sampling_seed=0, return_loss=True):
    model.eval()
    ps = torch.tensor(TITAN_PS_ARG).int().to(device)
    all_risks, all_times, all_censorships = [], [], []

    with torch.no_grad():
        for batch in loader:
            features, coords, _time_bin, censorship, survival_time, slide_id = unpack_batch(batch)
            censorship = censorship.float().to(device, non_blocking=True)
            survival_time_tensor = survival_time.float().to(device, non_blocking=True) if torch.is_tensor(survival_time) else torch.tensor(survival_time, dtype=torch.float32, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                risks = forward_bag_batch(
                    model, features, coords, device, num_patches, ps,
                    slide_ids=slide_id, sampling_mode=sampling_mode, sampling_seed=sampling_seed,
                )
            all_risks.append(risks.float().detach().cpu())
            all_times.extend(to_numpy_list(survival_time))
            all_censorships.extend(np.asarray(to_numpy_list(censorship.detach().cpu()), dtype=int).tolist())

    if not all_risks:
        return {"loss": float("nan"), "times": [], "events": [], "risks": [], "cindex": 0.0}

    risks_all = torch.cat(all_risks, dim=0)
    times_np = np.asarray(all_times, dtype=float)
    cens_np = np.asarray(all_censorships, dtype=int)
    events_np = (1 - cens_np).astype(int)
    cidx = concordance_index(times_np, risks_all.numpy(), events_np)

    loss_val = float("nan")
    if return_loss:
        with torch.no_grad():
            loss = cox_ph_loss(
                risks_all.to(device),
                torch.tensor(times_np, dtype=torch.float32, device=device),
                torch.tensor(events_np, dtype=torch.float32, device=device),
                eps=float(cfg.survival.get("cox_eps", 1e-7)),
            )
            loss_val = float(loss.detach().cpu().item())

    return {
        "loss": loss_val,
        "times": times_np.tolist(),
        "events": events_np.tolist(),
        "risks": risks_all.numpy().astype(float).tolist(),
        "cindex": cidx,
    }


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.verbose = bool(verbose)
        self.counter = 0
        self.best_cindex = -float("inf")
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.best_state = None
        self.early_stop = False

    def _copy_state(self, model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def __call__(self, val_cindex, val_loss, model, epoch):
        if not np.isfinite(val_cindex):
            self.counter += 1
        else:
            improved = val_cindex > self.best_cindex + self.min_delta
            tie_better = abs(val_cindex - self.best_cindex) <= self.min_delta and np.isfinite(val_loss) and val_loss < self.best_loss
            if improved or tie_better:
                self.best_cindex = float(val_cindex)
                self.best_loss = float(val_loss) if np.isfinite(val_loss) else self.best_loss
                self.best_epoch = int(epoch)
                self.best_state = self._copy_state(model)
                self.counter = 0
                return
            self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter}/{self.patience} (val_cindex did not improve)")
        if self.counter >= self.patience:
            self.early_stop = True

    def restore(self, model):
        if self.best_state is not None:
            print(f"Loading best checkpoint from epoch {self.best_epoch} with val_cindex={self.best_cindex:.4f}")
            model.load_state_dict(self.best_state)
        return model

    def state_dict(self):
        return {
            "monitor": "val_cindex",
            "best_cindex": self.best_cindex,
            "best_val_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }


def make_optimizer(model, cfg):
    lr = float(cfg.training.lr)
    head_lr = float(cfg.training.get("head_lr", lr))
    weight_decay = float(cfg.training.weight_decay)
    backbone_params, head_params, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name or "norm" in name.lower() or "ln" in name.lower():
            no_decay.append(p)
        elif name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": no_decay, "weight_decay": 0.0, "lr": lr},
            {"params": backbone_params, "weight_decay": weight_decay, "lr": lr},
            {"params": head_params, "weight_decay": weight_decay, "lr": head_lr},
        ]
    )


def train_one_phase(train_loader, val_loader, model, cfg, device, phase_name="train"):
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    num_epochs = int(cfg.training.num_epochs)
    train_num_patches = int(cfg.training.train_num_patches)
    eval_num_patches = int(cfg.training.get("eval_num_patches", train_num_patches))
    val_sampling_mode = str(cfg.training.get("val_sampling_mode", "fixed_random"))
    ps = torch.tensor(TITAN_PS_ARG).int().to(device)
    early = EarlyStopping(
        patience=int(cfg.training.get("early_stopping_patience", 5)),
        min_delta=float(cfg.training.get("early_stopping_min_delta", cfg.training.get("checkpoint_cindex_min_delta", 0.0))),
        verbose=True,
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        train_times, train_cens, train_risks = [], [], []
        skipped = 0
        for batch in train_loader:
            features, coords, _time_bin, censorship, survival_time, slide_id = unpack_batch(batch)
            censorship = censorship.float().to(device, non_blocking=True)
            survival_time = survival_time.float().to(device, non_blocking=True)
            events = batch_to_events(censorship)
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    risks = forward_bag_batch(
                        model, features, coords, device, train_num_patches, ps,
                        slide_ids=slide_id, sampling_mode="random", sampling_seed=int(cfg.training.seed),
                    )
                    loss = cox_ph_loss(risks, survival_time, events, eps=float(cfg.survival.get("cox_eps", 1e-7)))
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            except (FloatingPointError, RuntimeError) as exc:
                skipped += 1
                print(f"[Warning] Skip train batch: {exc}")
                continue
            total_loss += float(loss.detach().cpu().item())
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
        print(
            f"[{phase_name}] epoch {epoch}, loss={total_loss/max(1,n_batches):.4f}, "
            f"train_cindex={train_c:.4f}, val_loss={val_eval['loss']:.4f}, "
            f"val_cindex={val_eval['cindex']:.4f}, skipped={skipped}"
        )
        early(val_eval["cindex"], val_eval["loss"], model, epoch)
        if early.early_stop:
            break

    model = early.restore(model)
    model.eval()
    return model, early.state_dict()


def load_model_checkpoint(path: Path, cfg, device):
    model = make_model(cfg, device)
    obj = torch.load(path, map_location="cpu")
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state, strict=True)
    return model.eval()


def compute_final_fgt_bwt(metric_matrix: np.ndarray) -> Dict[str, float]:
    mat = np.asarray(metric_matrix, dtype=float)
    num_tasks = mat.shape[1]
    final = float(np.nanmean(mat[-1, :]))
    fgt_vals, bwt_vals = [], []
    for task_id in range(max(0, num_tasks - 1)):
        final_perf = mat[-1, task_id]
        learned_perf = mat[task_id, task_id]
        history = mat[task_id:, task_id]
        history = history[np.isfinite(history)]
        if np.isfinite(final_perf) and history.size > 0:
            fgt_vals.append(float(np.nanmax(history) - final_perf))
        if np.isfinite(final_perf) and np.isfinite(learned_perf):
            bwt_vals.append(float(final_perf - learned_perf))
    return {
        "final": final,
        "fgt": float(np.mean(fgt_vals)) if fgt_vals else float("nan"),
        "bwt": float(np.mean(bwt_vals)) if bwt_vals else float("nan"),
    }


def compute_headline_metrics(cindex_mat, ipcw_mat):
    """Compute final performance, forgetting and BWT for both C-index and IPCW.

    Keys:
      cindex       : final-step mean Harrell C-index over all tasks.
      cindex_ipcw  : final-step mean IPCW C-index over all tasks.
      fgt          : forgetting computed from Harrell C-index matrix.
      bwt          : backward transfer computed from Harrell C-index matrix.
      fgt_ipcw     : forgetting computed from IPCW C-index matrix.
      bwt_ipcw     : backward transfer computed from IPCW C-index matrix.
    """
    c = compute_final_fgt_bwt(cindex_mat)
    i = compute_final_fgt_bwt(ipcw_mat)
    return {
        "cindex": c["final"],
        "cindex_ipcw": i["final"],
        "fgt": c["fgt"],
        "bwt": c["bwt"],
        "fgt_ipcw": i["fgt"],
        "bwt_ipcw": i["bwt"],
    }


def summarize_metric_dicts(fold_metrics: List[Dict[str, float]]):
    keys = fold_metrics[0].keys()
    out = {}
    for key in keys:
        vals = np.asarray([m[key] for m in fold_metrics], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[key] = {
            "mean": float(np.mean(vals)) if vals.size else float("nan"),
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else (0.0 if vals.size == 1 else float("nan")),
        }
    return out


def fmt_mean_std(stat):
    mean, std = stat.get("mean", float("nan")), stat.get("std", float("nan"))
    if not np.isfinite(mean):
        return "N/A"
    if not np.isfinite(std):
        return f"{mean:.4f} (+-N/A)"
    return f"{mean:.4f} (+-{std:.4f})"


def print_final_metrics(metrics, prefix=""):
    label = f"{prefix} " if prefix else ""
    print(f"{label}C-Index:             {fmt_mean_std(metrics['cindex'])}")
    print(f"{label}C-index IPCW:        {fmt_mean_std(metrics['cindex_ipcw'])}")
    print(f"{label}Forget C-index (?):   {fmt_mean_std(metrics['fgt'])}")
    print(f"{label}BWT C-index (?):      {fmt_mean_std(metrics['bwt'])}")
    if "fgt_ipcw" in metrics:
        print(f"{label}Forget IPCW (?):      {fmt_mean_std(metrics['fgt_ipcw'])}")
    if "bwt_ipcw" in metrics:
        print(f"{label}BWT IPCW (?):         {fmt_mean_std(metrics['bwt_ipcw'])}")
