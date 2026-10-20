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
    parser = argparse.ArgumentParser(description="Evaluate Joint-Cox baseline on each task validation split")
    add_config_arg(parser)
    parser.add_argument("--checkpoint_root", type=str, default=None, help="Default: paths.merged_checkpoints/baselines/joint")
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
    root = Path(args.checkpoint_root or (Path(cfg.paths.merged_checkpoints) / "baselines" / "joint"))
    out_dir = ensure_dir(args.output_dir or root / "eval_joint")
    seq_dataset = Sequential_Generic_MIL_Survival_Dataset(cfg=cfg)

    fold_metrics, details = [], []
    final_c_all = np.full((int(cfg.training.num_folds), num_tasks), np.nan)
    final_i_all = np.full((int(cfg.training.num_folds), num_tasks), np.nan)

    for fold_idx in range(int(cfg.training.num_folds)):
        ckpt = root / f"fold_{fold_idx}" / "joint_model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model = load_model_checkpoint(ckpt, cfg, device)
        c_row, i_row = [], []
        for task_id, task in enumerate(task_order):
            train_loader, val_loader, _ = seq_dataset.get_data_loaders(fold_idx, task_id)
            val = evaluate_loader(val_loader, model, device, eval_num_patches, cfg, sampling_mode=args.sampling_mode, sampling_seed=int(cfg.training.seed), return_loss=False)
            tr_times, tr_events = dataset_times_events(train_loader.dataset)
            if tr_times is None:
                tr_times, tr_events = val["times"], val["events"]
            ipcw = concordance_index_ipcw(tr_times, tr_events, val["times"], val["risks"], val["events"])
            c_row.append(val["cindex"]); i_row.append(ipcw)
            details.append({"fold": fold_idx, "task_id": task_id, "task": task, "cindex": val["cindex"], "ipcw": ipcw})
            print(f"[fold_{fold_idx}] task={task} C={val['cindex']:.4f} IPCW={ipcw:.4f}")
        final_c_all[fold_idx] = np.asarray(c_row); final_i_all[fold_idx] = np.asarray(i_row)
        # Joint is not an incremental CL model, so FGT/BWT are not defined. Store them as 0 for table compatibility.
        fold_metrics.append({"cindex": float(np.mean(c_row)), "cindex_ipcw": float(np.mean(i_row)), "fgt": 0.0, "bwt": 0.0})
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    metrics = summarize_metric_dicts(fold_metrics)
    print_final_metrics(metrics, prefix="Joint-Cox")
    summary = {
        "baseline": "joint_cox",
        "note": "Joint training is an offline upper bound; FGT/BWT are set to 0 for table compatibility, not true continual-learning quantities.",
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "details": details,
    }
    np.save(out_dir / "cindex_per_fold_task.npy", final_c_all)
    np.save(out_dir / "ipcw_per_fold_task.npy", final_i_all)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
