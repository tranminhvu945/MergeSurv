from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from baselines.common import make_model, train_one_phase
from core.config import add_config_arg, ensure_dir, get_task_order, load_config, seed_everything
from data_utils.datasets import Sequential_Generic_MIL_Survival_Dataset


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser(description="SeqFT-Cox baseline: sequentially fine-tune one TITAN+Cox model task by task")
    add_config_arg(parser)
    parser.add_argument("--save_dir", type=str, default=None, help="Default: paths.merged_checkpoints/baselines/seq_ft")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(cfg.training.seed), device)
    task_order = get_task_order(cfg)[: int(cfg.training.num_tasks)]
    save_root = ensure_dir(args.save_dir or (Path(cfg.paths.merged_checkpoints) / "baselines" / "seq_ft"))
    seq_dataset = Sequential_Generic_MIL_Survival_Dataset(cfg=cfg)

    for fold_idx in range(int(cfg.training.num_folds)):
        fold_dir = ensure_dir(save_root / f"fold_{fold_idx}")
        print(f"\n========== SeqFT baseline | fold {fold_idx} ==========")
        model = make_model(cfg, device)
        all_meta = []
        for task_id, task in enumerate(task_order):
            print(f"\n----- Sequential fine-tune fold={fold_idx}, task={task_id}:{task} -----")
            train_loader, val_loader, _ = seq_dataset.get_data_loaders(fold_idx, task_id)
            start = time.time()
            model, best_info = train_one_phase(train_loader, val_loader, model, cfg, device, phase_name=f"seq/fold_{fold_idx}/task_{task_id}_{task}")
            elapsed = time.time() - start
            ckpt = fold_dir / f"seq_task_{task_id}.pt"
            torch.save({"model": model.state_dict(), "best_info": best_info, "task_id": task_id, "task_name": task}, ckpt)
            meta = {"baseline": "seq_ft_cox", "fold": fold_idx, "task_id": task_id, "task_name": task, "checkpoint": str(ckpt), "early_stopping": best_info, "elapsed_sec": elapsed}
            all_meta.append(meta)
            (fold_dir / f"seq_task_{task_id}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"Saved {ckpt} in {elapsed:.2f}s")
        (fold_dir / "seq_metadata.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
