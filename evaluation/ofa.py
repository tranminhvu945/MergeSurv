from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from core.config import add_config_arg, ensure_dir, get_task_order, load_config, seed_everything
from data_utils.datasets import Sequential_Generic_MIL_Survival_Dataset
from core.survival import concordance_index, concordance_index_ipcw

EMBED_DIM = 768
TITAN_PS_ARG = 1024


class TaskSpecificCoxHead(nn.Module):
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

    def forward(self, z):
        z = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return self.risk(self.net(z)).squeeze(-1).clamp(min=-20.0, max=20.0)


def stable_int_hash(value, base_seed=0):
    key = f"{base_seed}_{str(value)}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def sample_patches(features, coords, k, mode="fixed_random", seed=None):
    k = min(int(k), int(features.shape[0])); n = int(features.shape[0])
    if mode == "fixed_random":
        g = torch.Generator(device="cpu"); g.manual_seed(int(seed or 0) % (2**31 - 1))
        idx = torch.randperm(n, generator=g, device="cpu")[:k].to(features.device)
    elif mode == "first":
        idx = torch.arange(k, device=features.device)
    elif mode == "all":
        idx = torch.arange(n, device=features.device)
    else:
        raise ValueError(mode)
    return features[idx], coords[idx]


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
            raise RuntimeError("Use updated collate returning list of bags, or set batch_size=1.")
        features, coords = [features], [coords]
    return features, coords, time_bin, censorship, survival_time, slide_id


def load_backbone(base_model, path: Path, device):
    state = torch.load(path, map_location="cpu")
    base_model.vision_encoder.load_state_dict(state, strict=False)
    return base_model.vision_encoder.to(device).eval()


def load_head(path: Path, cfg, device):
    head = TaskSpecificCoxHead(
        hidden_dim=int(cfg.training.get("survival_hidden_dim", 512)),
        dropout=float(cfg.training.get("survival_dropout", 0.25)),
    ).to(device)
    state = torch.load(path, map_location="cpu")
    head.load_state_dict(state, strict=True)
    return head.eval()


def dataset_times_events(dataset):
    if hasattr(dataset, "slide_data") and hasattr(dataset, "time_col"):
        df = dataset.slide_data
        return df[dataset.time_col].values.astype(float).tolist(), (1 - df["censorship"].values.astype(int)).astype(int).tolist()
    return None, None


def collect(loader, backbone, head, device, num_patches, mode, base_seed):
    ps = torch.tensor(TITAN_PS_ARG).int().to(device)
    times, events, risks = [], [], []
    with torch.no_grad():
        for batch in loader:
            features_list, coords_list, _tb, censorship, survival_time, slide_ids = unpack_batch(batch)
            for i, (features, coords) in enumerate(zip(features_list, coords_list)):
                sid = slide_ids[i] if isinstance(slide_ids, (list, tuple)) else slide_ids
                seed = stable_int_hash(sid if sid is not None else len(risks), base_seed=base_seed)
                features = torch.nan_to_num(features.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
                coords = coords.to(device).long()
                features, coords = sample_patches(features, coords, num_patches, mode=mode, seed=seed)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    z = backbone(features, coords, ps)
                    risk = head(z).view(-1)[0]
                risks.append(float(risk.detach().cpu().item()))
            times.extend(np.asarray(survival_time, dtype=float).tolist())
            events.extend((1 - np.asarray(censorship, dtype=int)).astype(int).tolist())
    return times, events, risks



def compute_final_fgt_bwt(metric_matrix: np.ndarray) -> Dict[str, float]:
    """Continual-learning summary for a higher-is-better metric matrix.

    metric_matrix[step, task] is performance on `task` after learning/merging up to `step`.

    Returned metrics:
      - final: mean performance on all tasks after the last step.
      - fgt: forgetting, mean(max historical performance on each old task - final performance).
        Positive FGT means the model forgot performance after learning later tasks.
      - bwt: backward transfer, mean(final performance - performance right after the task was learned).
        Negative BWT means forgetting; positive BWT means later tasks helped old tasks.
    """
    mat = np.asarray(metric_matrix, dtype=float)
    num_tasks = mat.shape[1]
    final_values = mat[-1, :]
    final = float(np.nanmean(final_values))

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

    fgt = float(np.mean(fgt_vals)) if fgt_vals else float("nan")
    bwt = float(np.mean(bwt_vals)) if bwt_vals else float("nan")
    return {"final": final, "fgt": fgt, "bwt": bwt}


def compute_headline_metrics(cindex_mat: np.ndarray, ipcw_mat: np.ndarray) -> Dict[str, float]:
    """Compute the four requested metrics from one fold/matrix.

    C-Index and C-index IPCW are final-step mean performance over all tasks.
    FGT and BWT are computed from C-index, following the continual-learning convention.
    IPCW FGT/BWT are also stored for analysis, but are not printed as headline metrics.
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


def summarize_metric_dicts(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Return mean/std over folds for every metric key."""
    keys = fold_metrics[0].keys()
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = np.asarray([m[key] for m in fold_metrics], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
        else:
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            }
    return summary


def _fmt_mean_std(stat: Dict[str, float]) -> str:
    mean = stat.get("mean", float("nan"))
    std = stat.get("std", float("nan"))
    if not np.isfinite(mean):
        return "N/A"
    if not np.isfinite(std):
        return f"{mean:.4f} (+-N/A)"
    return f"{mean:.4f} (+-{std:.4f})"


def print_final_metrics(metrics_summary: Dict[str, Dict[str, float]], prefix: str = "") -> None:
    """Print exactly the four requested final metrics as mean (+-std) over folds."""
    label = f"{prefix} " if prefix else ""
    print(f"{label}C-Index:        {_fmt_mean_std(metrics_summary['cindex'])}")
    print(f"{label}C-index IPCW:   {_fmt_mean_std(metrics_summary['cindex_ipcw'])}")
    print(f"{label}FGT:            {_fmt_mean_std(metrics_summary['fgt'])}")
    print(f"{label}BWT:            {_fmt_mean_std(metrics_summary['bwt'])}")

def main():
    parser = argparse.ArgumentParser(description="Evaluation: Strategy 1 OPCM backbone + directly averaged Cox head")
    add_config_arg(parser)
    parser.add_argument("--merged_root", type=str, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_num_patches", type=int, default=None)
    parser.add_argument("--sampling_mode", type=str, default="fixed_random", choices=["fixed_random", "first", "all"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(cfg.training.seed), device)
    task_order = get_task_order(cfg)[: int(args.num_tasks or cfg.training.num_tasks)]
    num_tasks = len(task_order)
    eval_num_patches = int(args.eval_num_patches or cfg.training.get("eval_num_patches", cfg.training.get("train_num_patches", 400)))
    merged_root = Path(args.merged_root or (Path(cfg.paths.merged_checkpoints) / "strategy1_merge_heads"))
    out_dir = ensure_dir(args.output_dir or merged_root / "eval_merged_head")

    seq_dataset = Sequential_Generic_MIL_Survival_Dataset(cfg=cfg)
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    cmat = np.full((num_tasks, num_tasks), np.nan); imat = np.full((num_tasks, num_tasks), np.nan)
    details = []
    fold_metrics = []
    for fold_idx in range(int(cfg.training.num_folds)):
        fold = f"fold_{fold_idx}"; fold_dir = merged_root / fold
        fc = np.full((num_tasks, num_tasks), np.nan); fi = np.full((num_tasks, num_tasks), np.nan)
        for step in range(num_tasks):
            package = json.loads((fold_dir / f"merged_package_task_{step}.json").read_text())
            backbone = load_backbone(base_model, Path(package["backbone_path"]), device)
            head = load_head(Path(package["head_path"]), cfg, device)
            for task_id in range(step + 1):
                train_loader, val_loader, _ = seq_dataset.get_data_loaders(fold_idx, task_id)
                times, events, risks = collect(val_loader, backbone, head, device, eval_num_patches, args.sampling_mode, int(cfg.training.seed))
                tr_times, tr_events = dataset_times_events(train_loader.dataset)
                if tr_times is None: tr_times, tr_events = times, events
                cidx = concordance_index(times, risks, events)
                ipcw = concordance_index_ipcw(tr_times, tr_events, times, risks, events)
                fc[step, task_id] = cidx; fi[step, task_id] = ipcw
                details.append({"fold": fold_idx, "step": step, "task_id": task_id, "task": task_order[task_id], "cindex": cidx, "ipcw": ipcw})
                print(f"[{fold}] step={step} task={task_order[task_id]} C={cidx:.4f} IPCW={ipcw:.4f}")
        cmat = np.nan_to_num(cmat, nan=0.0) + np.nan_to_num(fc, nan=0.0)
        imat = np.nan_to_num(imat, nan=0.0) + np.nan_to_num(fi, nan=0.0)
        np.save(out_dir / f"cindex_matrix_{fold}.npy", fc)
        np.save(out_dir / f"ipcw_matrix_{fold}.npy", fi)
        fold_metrics.append(compute_headline_metrics(fc, fi))
    cmat /= int(cfg.training.num_folds); imat /= int(cfg.training.num_folds)
    matrix_mean_metrics = compute_headline_metrics(cmat, imat)
    metrics = summarize_metric_dicts(fold_metrics)
    print_final_metrics(metrics)
    summary = {
        "strategy": "strategy1_direct_average_survival_head",
        "metrics": metrics,
        "matrix_mean_metrics": matrix_mean_metrics,
        "fold_metrics": fold_metrics,
        "details": details,
    }
    np.save(out_dir / "cindex_matrix_mean.npy", cmat)
    np.save(out_dir / "ipcw_matrix_mean.npy", imat)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
