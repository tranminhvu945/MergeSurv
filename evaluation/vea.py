from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from core.config import add_config_arg, ensure_dir, get_task_order, load_config, seed_everything
from data_utils.datasets import Sequential_Generic_MIL_Survival_Dataset
from core.survival import concordance_index, concordance_index_ipcw

try:
    from data_utils.prompts import task_level_prompts
except Exception:
    task_level_prompts = None

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
        hidden = self.net(z)
        return self.risk(hidden).squeeze(-1).clamp(min=-20.0, max=20.0)


def stable_int_hash(value, base_seed: int = 0) -> int:
    if isinstance(value, (list, tuple)):
        value = "_".join(str(v) for v in value)
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


def default_task_prompts(task: str):
    names = {
        "blca": "bladder urothelial carcinoma",
        "brca": "breast invasive carcinoma",
        "luad": "lung adenocarcinoma",
        "ucec": "uterine corpus endometrial carcinoma",
    }
    name = names.get(task, task)
    prompts = [[name, f"{name} histopathology", f"TCGA-{task.upper()}"]]
    templates = [
        "CLASSNAME.",
        "a histopathological whole slide image of CLASSNAME.",
        "an H&E stained whole slide image of CLASSNAME.",
    ]
    return prompts, templates


def build_task_embeddings(task_order, device, cache_path: Path | None = None):
    if cache_path is not None and cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu")
        if "task_prompt_embeddings" in obj:
            return obj["task_prompt_embeddings"].float()
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)
    embs = []
    for task in task_order:
        if task_level_prompts is not None:
            prompts, templates = task_level_prompts(task)
        else:
            prompts, templates = default_task_prompts(task)
        with torch.autocast("cuda", torch.float16, enabled=torch.cuda.is_available()), torch.inference_mode():
            cls = titan.zero_shot_classifier(prompts, templates, device=device)
        embs.append(cls.float().squeeze(-1).detach().cpu())
    del titan
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return torch.stack(embs, dim=0)


def load_backbone(base_model, path: Path, device):
    state = torch.load(path, map_location="cpu")
    base_model.vision_encoder.load_state_dict(state, strict=False)
    return base_model.vision_encoder.to(device).eval()


def load_head(state, cfg, device):
    head = TaskSpecificCoxHead(
        hidden_dim=int(cfg.training.get("survival_hidden_dim", 512)),
        dropout=float(cfg.training.get("survival_dropout", 0.25)),
    ).to(device)
    head.load_state_dict(state, strict=True)
    return head.eval()


def dataset_times_events(dataset):
    if hasattr(dataset, "slide_data") and hasattr(dataset, "time_col"):
        df = dataset.slide_data
        return df[dataset.time_col].values.astype(float).tolist(), (1 - df["censorship"].values.astype(int)).astype(int).tolist()
    return None, None


def risk_for_bag(backbone, heads, task_embs, seen_ids, features, coords, device, ps, num_patches, sampling_mode, seed, temperature, normalize_stats=None):
    features = torch.nan_to_num(features.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    coords = coords.to(device).long()
    features, coords = sample_patches(features, coords, num_patches, mode=sampling_mode, seed=seed)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()), torch.no_grad():
        z = backbone(features, coords, ps).float()  # [1, D]
    z_norm = F.normalize(z, dim=1)
    task_e = F.normalize(task_embs[seen_ids].to(device).float(), dim=1)
    scores = (z_norm @ task_e.T).squeeze(0)
    probs = torch.softmax(scores / max(float(temperature), 1e-6), dim=0)
    risks = []
    with torch.no_grad():
        for local_idx, task_id in enumerate(seen_ids):
            r = heads[task_id](z).view(-1)[0].float()
            if normalize_stats is not None and task_id in normalize_stats:
                mean, std = normalize_stats[task_id]
                r = (r - mean) / max(std, 1e-6)
            risks.append(r)
    risks_t = torch.stack(risks)
    soft_risk = torch.sum(probs * risks_t)
    pred_task = seen_ids[int(torch.argmax(probs).detach().cpu().item())]
    return float(soft_risk.detach().cpu().item()), pred_task



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
    parser = argparse.ArgumentParser(description="CLASS-IL evaluation: Strategy 2 task-specific heads + soft routing")
    add_config_arg(parser)
    parser.add_argument("--merged_root", type=str, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_num_patches", type=int, default=None)
    parser.add_argument("--sampling_mode", type=str, default="fixed_random", choices=["fixed_random", "first", "all"])
    parser.add_argument("--routing_temperature", type=float, default=0.1)
    parser.add_argument("--task_embedding_cache", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(cfg.training.seed), device)
    task_order = get_task_order(cfg)[: int(args.num_tasks or cfg.training.num_tasks)]
    num_tasks = len(task_order)
    eval_num_patches = int(args.eval_num_patches or cfg.training.get("eval_num_patches", cfg.training.get("train_num_patches", 400)))
    merged_root = Path(args.merged_root or (Path(cfg.paths.merged_checkpoints) / "strategy2_keep_task_heads"))
    out_dir = ensure_dir(args.output_dir or merged_root / "eval_classIL_soft_routing")

    task_embs = build_task_embeddings(task_order, device, Path(args.task_embedding_cache) if args.task_embedding_cache else None)
    seq_dataset = Sequential_Generic_MIL_Survival_Dataset(cfg=cfg)
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    cindex_mat = np.full((num_tasks, num_tasks), np.nan, dtype=float)
    ipcw_mat = np.full((num_tasks, num_tasks), np.nan, dtype=float)
    route_acc_mat = np.full((num_tasks, num_tasks), np.nan, dtype=float)
    details = []
    fold_metrics = []

    for fold_idx in range(int(cfg.training.num_folds)):
        fold = f"fold_{fold_idx}"; fold_dir = merged_root / fold
        fold_c = np.full((num_tasks, num_tasks), np.nan); fold_i = np.full((num_tasks, num_tasks), np.nan); fold_r = np.full((num_tasks, num_tasks), np.nan)
        for step in range(num_tasks):
            package = json.loads((fold_dir / f"merged_package_task_{step}.json").read_text())
            # Additional
            expected_task_names = task_order[: step + 1]
            actual_task_names = [str(task).lower() for task in package["seen_task_names"]]

            if actual_task_names != expected_task_names:
              raise RuntimeError(
                "Merged package does not match the evaluation order.\n"
                f"Expected: {expected_task_names}\n"
                f"Found:    {actual_task_names}\n"
                f"Package:  {fold_dir / f'merged_package_task_{step}.json'}"
              )
            # End
                      
            backbone = load_backbone(base_model, Path(package["backbone_path"]), device)
            heads_obj = torch.load(package["task_heads_path"], map_location="cpu")
            task_heads_state = heads_obj["task_heads"]
            seen_ids = list(range(step + 1))
            heads = {tid: load_head(task_heads_state[tid], cfg, device) for tid in seen_ids}
            for task_id in seen_ids:
                train_loader, val_loader, _ = seq_dataset.get_data_loaders(fold_idx, task_id)
                times, events, risks, pred_tasks = [], [], [], []
                ps = torch.tensor(TITAN_PS_ARG).int().to(device)
                with torch.no_grad():
                    for batch in val_loader:
                        features_list, coords_list, _tb, censorship, survival_time, slide_ids = unpack_batch(batch)
                        for j, (features, coords) in enumerate(zip(features_list, coords_list)):
                            sid = slide_ids[j] if isinstance(slide_ids, (list, tuple)) else slide_ids
                            seed = stable_int_hash(sid if sid is not None else len(risks), base_seed=int(cfg.training.seed))
                            risk, pred_task = risk_for_bag(
                                backbone, heads, task_embs, seen_ids, features, coords, device, ps,
                                eval_num_patches, args.sampling_mode, seed, args.routing_temperature,
                            )
                            risks.append(risk); pred_tasks.append(pred_task)
                        times.extend(np.asarray(survival_time, dtype=float).tolist())
                        c_np = np.asarray(censorship, dtype=int)
                        events.extend((1 - c_np).astype(int).tolist())
                tr_times, tr_events = dataset_times_events(train_loader.dataset)
                if tr_times is None: tr_times, tr_events = times, events
                cidx = concordance_index(times, risks, events)
                ipcw = concordance_index_ipcw(tr_times, tr_events, times, risks, events)
                racc = float(np.mean(np.asarray(pred_tasks) == task_id)) if pred_tasks else float("nan")
                fold_c[step, task_id] = cidx; fold_i[step, task_id] = ipcw; fold_r[step, task_id] = racc
                details.append({"fold": fold_idx, "step": step, "task_id": task_id, "task": task_order[task_id], "cindex": cidx, "ipcw": ipcw, "routing_acc": racc})
                print(f"[{fold}] step={step} task={task_order[task_id]} C={cidx:.4f} IPCW={ipcw:.4f} route_acc={racc:.4f}")
            del heads
        cindex_mat = np.nan_to_num(cindex_mat, nan=0.0) + np.nan_to_num(fold_c, nan=0.0)
        ipcw_mat = np.nan_to_num(ipcw_mat, nan=0.0) + np.nan_to_num(fold_i, nan=0.0)
        route_acc_mat = np.nan_to_num(route_acc_mat, nan=0.0) + np.nan_to_num(fold_r, nan=0.0)
        np.save(out_dir / f"cindex_matrix_{fold}.npy", fold_c)
        np.save(out_dir / f"ipcw_matrix_{fold}.npy", fold_i)
        np.save(out_dir / f"routing_acc_matrix_{fold}.npy", fold_r)
        fold_metrics.append(compute_headline_metrics(fold_c, fold_i))

    nfolds = int(cfg.training.num_folds)
    cindex_mat /= nfolds; ipcw_mat /= nfolds; route_acc_mat /= nfolds
    matrix_mean_metrics = compute_headline_metrics(cindex_mat, ipcw_mat)
    metrics = summarize_metric_dicts(fold_metrics)
    print_final_metrics(metrics)
    final_r = float(np.nanmean(route_acc_mat[-1]))
    print(f"Final routing accuracy: {final_r:.4f}")
    summary = {
        "strategy": "strategy2_classIL_soft_routing",
        "metrics": metrics,
        "matrix_mean_metrics": matrix_mean_metrics,
        "fold_metrics": fold_metrics,
        "final_routing_acc": final_r,
        "details": details,
    }
    np.save(out_dir / "cindex_matrix_mean.npy", cindex_mat)
    np.save(out_dir / "ipcw_matrix_mean.npy", ipcw_mat)
    np.save(out_dir / "routing_acc_matrix_mean.npy", route_acc_mat)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
