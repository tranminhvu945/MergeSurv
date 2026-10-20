from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from baselines.common import make_model
from baselines.continual_common import EWCHelper, WSIReplayBuffer, freeze_model, train_one_task_cl
from core.config import add_config_arg, ensure_dir, get_task_order, load_config, seed_everything
from data_utils.datasets import Sequential_Generic_MIL_Survival_Dataset


METHODS = ["ewc", "lwf", "er", "der", "derpp"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="WSI-only Cox continual-learning baselines: EWC, LwF, ER, DER, DER++"
    )
    add_config_arg(parser)
    parser.add_argument("--method", type=str, required=True, choices=METHODS)
    parser.add_argument("--save_dir", type=str, default=None, help="Default: paths.merged_checkpoints/baselines/<method>")

    # Replay / dark replay
    parser.add_argument("--buffer_size", type=int, default=32)
    parser.add_argument("--replay_batch_size", type=int, default=None, help="Default: dataloader.batch_size")
    parser.add_argument("--er_beta", type=float, default=1.0, help="ER supervised replay Cox loss weight")
    parser.add_argument("--der_alpha", type=float, default=0.5, help="DER distillation weight")
    parser.add_argument("--der_beta", type=float, default=1.0, help="DER++ supervised replay Cox loss weight")
    parser.add_argument("--distill_risk_weight", type=float, default=1.0)
    parser.add_argument("--distill_z_weight", type=float, default=1.0)

    # LwF
    parser.add_argument("--lwf_alpha", type=float, default=0.5)
    parser.add_argument("--lwf_risk_weight", type=float, default=1.0)
    parser.add_argument("--lwf_z_weight", type=float, default=1.0)

    # EWC
    parser.add_argument("--ewc_lambda", type=float, default=1.0e-4)
    parser.add_argument("--ewc_gamma", type=float, default=0.9)
    parser.add_argument(
        "--ewc_include_regex",
        type=str,
        default=".*",
        help="Regex over parameter names. For a lighter first run try: 'mlp'",
    )
    parser.add_argument("--ewc_exclude_regex", type=str, default="")
    parser.add_argument("--ewc_max_fisher_batches", type=int, default=0, help="0 = all train batches")
    return parser

def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = build_parser().parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(cfg.training.seed), device)
    task_order = get_task_order(cfg)[: int(cfg.training.num_tasks)]
    save_root = ensure_dir(args.save_dir or (Path(cfg.paths.merged_checkpoints) / "baselines" / args.method))
    seq_dataset = Sequential_Generic_MIL_Survival_Dataset(cfg=cfg)
    replay_batch_size = int(args.replay_batch_size or cfg.dataloader.batch_size)

    run_config = vars(args).copy()
    run_config["task_order"] = task_order
    (save_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    for fold_idx in range(int(cfg.training.num_folds)):
        fold_dir = ensure_dir(save_root / f"fold_{fold_idx}")
        print(f"\n========== {args.method.upper()} baseline | fold {fold_idx} ==========")
        model = make_model(cfg, device)

        buffer = None
        if args.method in {"er", "der", "derpp"}:
            buffer = WSIReplayBuffer(args.buffer_size, seed=int(cfg.training.seed) + 1009 * fold_idx)

        ewc = None
        if args.method == "ewc":
            ewc = EWCHelper(
                gamma=args.ewc_gamma,
                include_regex=args.ewc_include_regex,
                exclude_regex=args.ewc_exclude_regex,
                max_batches=args.ewc_max_fisher_batches,
            )

        old_model = None
        all_meta = []
        for task_id, task in enumerate(task_order):
            print(f"\n----- {args.method.upper()} fold={fold_idx}, task={task_id}:{task} -----")
            train_loader, val_loader, _ = seq_dataset.get_data_loaders(fold_idx, task_id)

            if args.method == "lwf" and task_id > 0:
                old_model = copy.deepcopy(model).to(device)
                freeze_model(old_model)
            else:
                old_model = None

            start = time.time()
            model, best_info = train_one_task_cl(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                cfg=cfg,
                device=device,
                method=args.method,
                task_id=task_id,
                buffer=buffer,
                ewc=ewc,
                old_model=old_model,
                phase_name=f"{args.method}/fold_{fold_idx}/task_{task_id}_{task}",
                buffer_batch_size=replay_batch_size,
                er_beta=args.er_beta,
                der_alpha=args.der_alpha,
                der_beta=args.der_beta,
                distill_risk_weight=args.distill_risk_weight,
                distill_z_weight=args.distill_z_weight,
                lwf_alpha=args.lwf_alpha,
                lwf_risk_weight=args.lwf_risk_weight,
                lwf_z_weight=args.lwf_z_weight,
                ewc_lambda=args.ewc_lambda,
            )

            ewc_info = None
            if args.method == "ewc" and ewc is not None:
                print(f"[EWC] estimating Fisher after task {task_id}:{task} ...")
                ewc_info = ewc.update_after_task(
                    model, train_loader, cfg, device, task_id,
                    num_patches=int(cfg.training.train_num_patches),
                )
                print(f"[EWC] Fisher info: {ewc_info}")

            elapsed = time.time() - start
            ckpt = fold_dir / f"{args.method}_task_{task_id}.pt"
            payload = {
                "model": model.state_dict(),
                "best_info": best_info,
                "task_id": task_id,
                "task_name": task,
                "method": args.method,
                "args": run_config,
            }
            if args.method == "ewc" and ewc is not None:
                payload["ewc"] = ewc.state_dict()
            if args.method in {"er", "der", "derpp"} and buffer is not None:
                # Stored for debugging/resuming, not needed for evaluation.
                payload["buffer_meta"] = {"buffer_size": buffer.buffer_size, "num_seen": buffer.num_seen, "num_items": len(buffer)}
            torch.save(payload, ckpt)

            meta = {
                "baseline": f"{args.method}_cox",
                "fold": fold_idx,
                "task_id": task_id,
                "task_name": task,
                "checkpoint": str(ckpt),
                "early_stopping": best_info,
                "ewc_info": ewc_info,
                "buffer_items": len(buffer) if buffer is not None else 0,
                "elapsed_sec": elapsed,
            }
            all_meta.append(meta)
            (fold_dir / f"{args.method}_task_{task_id}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"Saved {ckpt} in {elapsed:.2f}s")

            if old_model is not None:
                del old_model
                old_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        (fold_dir / f"{args.method}_metadata.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
