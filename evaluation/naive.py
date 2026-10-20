from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baselines.common import (
    compute_headline_metrics,
    dataset_times_events,
    evaluate_loader,
    load_model_checkpoint,
    print_final_metrics,
    summarize_metric_dicts,
)
from core.config import add_config_arg, ensure_dir, get_task_order, load_config, seed_everything
from data_utils.datasets import Sequential_Generic_MIL_Survival_Dataset
from core.survival import concordance_index_ipcw


def main():
    parser = argparse.ArgumentParser(description="Evaluate SeqFT-Cox baseline with continual matrix, FGT, BWT")
    add_config_arg(parser)
    parser.add_argument("--checkpoint_root", type=str, default=None, help="Default: paths.merged_checkpoints/baselines/seq_ft")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_num_patches", type=int, default=None)
    parser.add_argument("--sampling_mode", type=str, default="fixed_random", choices=["fixed_random", "first", "all"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(cfg.training.seed), device)
    task_order = get_task_order(cfg)[: int(cfg.training.num_tasks)]
    num_tasks = len(task_order)
    eval_num_patches = int(args.eval_num_patches or cfg.training.get("eval_num_patches", cfg.training.get("train_num_patches", 400)))
    root = Path(args.checkpoint_root or (Path(cfg.paths.merged_checkpoints) / "baselines" / "seq_ft"))
    out_dir = ensure_dir(args.output_dir or root / "eval_seq_ft")
    seq_dataset = Sequential_Generic_MIL_Survival_Dataset(cfg=cfg)

    fold_metrics, details = [], []
    cindex_mean = np.full((num_tasks, num_tasks), np.nan)
    ipcw_mean = np.full((num_tasks, num_tasks), np.nan)

    for fold_idx in range(int(cfg.training.num_folds)):
        fold_c = np.full((num_tasks, num_tasks), np.nan)
        fold_i = np.full((num_tasks, num_tasks), np.nan)
        for step in range(num_tasks):
            ckpt = root / f"fold_{fold_idx}" / f"seq_task_{step}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
            model = load_model_checkpoint(ckpt, cfg, device)
            for task_id in range(step + 1):
                train_loader, val_loader, _ = seq_dataset.get_data_loaders(fold_idx, task_id)
                val = evaluate_loader(val_loader, model, device, eval_num_patches, cfg, sampling_mode=args.sampling_mode, sampling_seed=int(cfg.training.seed), return_loss=False)
                tr_times, tr_events = dataset_times_events(train_loader.dataset)
                if tr_times is None:
                    tr_times, tr_events = val["times"], val["events"]
                ipcw = concordance_index_ipcw(tr_times, tr_events, val["times"], val["risks"], val["events"])
                fold_c[step, task_id] = val["cindex"]
                fold_i[step, task_id] = ipcw
                details.append({"fold": fold_idx, "step": step, "task_id": task_id, "task": task_order[task_id], "cindex": val["cindex"], "ipcw": ipcw})
                print(f"[fold_{fold_idx}] step={step} task={task_order[task_id]} C={val['cindex']:.4f} IPCW={ipcw:.4f}")
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        np.save(out_dir / f"cindex_matrix_fold_{fold_idx}.npy", fold_c)
        np.save(out_dir / f"ipcw_matrix_fold_{fold_idx}.npy", fold_i)
        fold_metrics.append(compute_headline_metrics(fold_c, fold_i))
        cindex_mean = np.nan_to_num(cindex_mean, nan=0.0) + np.nan_to_num(fold_c, nan=0.0)
        ipcw_mean = np.nan_to_num(ipcw_mean, nan=0.0) + np.nan_to_num(fold_i, nan=0.0)

    cindex_mean /= int(cfg.training.num_folds)
    ipcw_mean /= int(cfg.training.num_folds)
    metrics = summarize_metric_dicts(fold_metrics)
    print_final_metrics(metrics, prefix="SeqFT-Cox")
    summary = {
        "baseline": "seq_ft_cox",
        "metrics": metrics,
        "matrix_mean_metrics": compute_headline_metrics(cindex_mean, ipcw_mean),
        "fold_metrics": fold_metrics,
        "details": details,
    }
    np.save(out_dir / "cindex_matrix_mean.npy", cindex_mean)
    np.save(out_dir / "ipcw_matrix_mean.npy", ipcw_mean)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
