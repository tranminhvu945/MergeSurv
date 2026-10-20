from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from baselines.common import make_model, train_one_phase
from core.config import add_config_arg, ensure_dir, load_config, seed_everything
from data_utils.datasets import Sequential_Generic_MIL_Survival_Dataset


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser(description="Joint-Cox baseline: train one TITAN+Cox model on all tasks jointly per fold")
    add_config_arg(parser)
    parser.add_argument("--save_dir", type=str, default=None, help="Default: paths.merged_checkpoints/baselines/joint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(cfg.training.seed), device)
    save_root = ensure_dir(args.save_dir or (Path(cfg.paths.merged_checkpoints) / "baselines" / "joint"))

    seq_dataset = Sequential_Generic_MIL_Survival_Dataset(cfg=cfg)
    for fold_idx in range(int(cfg.training.num_folds)):
        fold_dir = ensure_dir(save_root / f"fold_{fold_idx}")
        print(f"\n========== Joint baseline | fold {fold_idx} ==========")
        train_loader, val_loader, _ = seq_dataset.get_joint_data_loaders(fold_idx)
        model = make_model(cfg, device)
        start = time.time()
        model, best_info = train_one_phase(train_loader, val_loader, model, cfg, device, phase_name=f"joint/fold_{fold_idx}")
        elapsed = time.time() - start
        ckpt = fold_dir / "joint_model.pt"
        torch.save({"model": model.state_dict(), "best_info": best_info}, ckpt)
        meta = {"baseline": "joint_cox", "fold": fold_idx, "checkpoint": str(ckpt), "early_stopping": best_info, "elapsed_sec": elapsed}
        (fold_dir / "joint_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved {ckpt} in {elapsed:.2f}s")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
