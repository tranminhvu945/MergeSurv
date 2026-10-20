from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import AutoModel

from core.config import add_config_arg, ensure_dir, get_task_order, load_config, resolve_finetuned_checkpoints_by_name
from core.utils import get_task_vector_norm, is_leaf_module, svd


# -----------------------------------------------------------------------------
# Strategy 2: OPCM backbone + keep task-specific Cox heads
# -----------------------------------------------------------------------------
# Use this strategy as the upper-bound / safest survival-head baseline:
#   - Merge TITAN slide aggregator/backbone with OPCM.
#   - Keep one small task-specific Cox head per seen task.
# This grows linearly with tasks, but Cox heads are small. It tells you how much
# performance is lost when forcing the heads into one merged/shared head.
# -----------------------------------------------------------------------------


def merge_linear_weights(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    param_name: str,
    alpha: float,
    previous_lambda_t: float,
    lambda_t: float,
    accelerator: str = "cpu",
) -> Tensor:
    original_device = merged_W.device
    merged_W = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W = task_W.to(accelerator)
    previous_merged_tv = merged_W - pretrained_W
    task_tv = task_W - pretrained_W
    u, _s, v = svd(previous_merged_tv)
    projected_task_tv = u.T @ task_tv @ v
    projected_task_tv.diag().fill_(0)
    cleaned_task_tv = u @ projected_task_tv @ v.T
    new_merged_W = pretrained_W + (previous_lambda_t * previous_merged_tv + cleaned_task_tv) / lambda_t
    return new_merged_W.to(original_device)


def merge_other_parameters(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    param_name: str,
    previous_lambda_t: float,
    lambda_t: float,
    accelerator: str = "cpu",
) -> Tensor:
    original_device = merged_W.device
    merged_W = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W = task_W.to(accelerator)
    previous_merged_tv = merged_W - pretrained_W
    task_tv = task_W - pretrained_W
    new_merged_W = pretrained_W + (previous_lambda_t * previous_merged_tv + task_tv) / lambda_t
    return new_merged_W.to(original_device)


def strip_backbone_only(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    backbone_state = {k.replace("backbone.", "", 1): v.detach().cpu() for k, v in state_dict.items() if k.startswith("backbone.")}
    if not backbone_state:
        raise KeyError("Checkpoint has no backbone.* parameters.")
    return backbone_state


def strip_head_only(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    head_state = {k.replace("mlp.", "", 1): v.detach().cpu() for k, v in state_dict.items() if k.startswith("mlp.")}
    if not head_state:
        raise KeyError("Checkpoint has no mlp.* Cox head parameters.")
    return head_state


def opcm_merge_backbone_incremental(task_paths: List[Path], base_model, alpha: float, dst_fold_dir: Path) -> None:
    base_weight = {k: v.detach().cpu() for k, v in base_model.vision_encoder.state_dict().items()}
    all_task_vector_norm: List[float] = []

    task_weight = strip_backbone_only(torch.load(task_paths[0], map_location="cpu"))
    merged_weight = dict(task_weight)
    all_task_vector_norm.append(float(get_task_vector_norm(task_weight, base_weight)))
    previous_lambda_t = 1.0
    torch.save({k: v.detach().cpu() for k, v in merged_weight.items()}, dst_fold_dir / "merged_backbone_task_0.pth")

    for model_idx, task_path in enumerate(task_paths[1:], start=1):
        task_weight = strip_backbone_only(torch.load(task_path, map_location="cpu"))
        all_task_vector_norm.append(float(get_task_vector_norm(task_weight, base_weight)))
        avg_task_vector_norm = float(np.mean(all_task_vector_norm))
        lambda_t = 1.0

        for module_name, module in tqdm(list(base_model.vision_encoder.named_modules()), desc=f"OPCM backbone merge task {model_idx}", leave=False):
            if not is_leaf_module(module):
                continue
            if isinstance(module, nn.Linear):
                weight_key = f"{module_name}.weight"
                merged_weight[weight_key] = merge_linear_weights(
                    merged_weight[weight_key].detach(),
                    base_model.vision_encoder.get_submodule(module_name).weight.detach().cpu(),
                    task_weight[weight_key].detach(),
                    weight_key,
                    alpha,
                    previous_lambda_t,
                    lambda_t,
                )
                if module.bias is not None:
                    bias_key = f"{module_name}.bias"
                    merged_weight[bias_key] = merge_other_parameters(
                        merged_weight[bias_key].detach(),
                        base_model.vision_encoder.get_submodule(module_name).bias.detach().cpu(),
                        task_weight[bias_key].detach(),
                        bias_key,
                        previous_lambda_t,
                        lambda_t,
                    )
            else:
                for param_name, _ in module.named_parameters(recurse=False):
                    key = f"{module_name}.{param_name}"
                    merged_weight[key] = merge_other_parameters(
                        merged_weight[key].detach(),
                        base_model.vision_encoder.get_submodule(module_name).get_parameter(param_name).detach().cpu(),
                        task_weight[key].detach(),
                        key,
                        previous_lambda_t,
                        lambda_t,
                    )

        task_vector_norm = float(get_task_vector_norm(merged_weight, base_weight))
        lambda_t *= task_vector_norm / max(avg_task_vector_norm, 1e-12)
        previous_lambda_t = lambda_t

        for param_name, _ in base_model.vision_encoder.named_parameters():
            base_param = base_model.vision_encoder.get_parameter(param_name).detach().cpu()
            merged_weight[param_name] = base_param + (merged_weight[param_name] - base_param) * (
                avg_task_vector_norm / max(float(task_vector_norm), 1e-12)
            )
        torch.save({k: v.detach().cpu() for k, v in merged_weight.items()}, dst_fold_dir / f"merged_backbone_task_{model_idx}.pth")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy 2: OPCM backbone + keep task-specific Cox heads")
    add_config_arg(parser)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--src_finedtuned_checkpoints", type=str, default=None)
    parser.add_argument("--des_merged_checkpoints", type=str, default=None)
    parser.add_argument("--copy_prompt_aux", action="store_true", help="Also copy prompt auxiliary checkpoints for interpretability/debugging.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_order = get_task_order(cfg)[: int(args.num_tasks or cfg.training.num_tasks)]
    num_tasks = len(task_order)
    src_root = Path(args.src_finedtuned_checkpoints or cfg.paths.finetuned_checkpoints)
    dst_root = ensure_dir(Path(args.des_merged_checkpoints or cfg.paths.merged_checkpoints) / "strategy2_keep_task_heads")

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)

    for fold_idx in range(int(cfg.training.num_folds)):
        fold = f"fold_{fold_idx}"
        dst_fold_dir = ensure_dir(dst_root / fold)
#        task_paths = [src_root / fold / f"finetune_task_{task_id}.pt" for task_id in range(num_tasks)]
        task_paths = resolve_finetuned_checkpoints_by_name(src_root=src_root, fold=fold, task_order=task_order,)
        missing = [str(p) for p in task_paths if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing fine-tuned checkpoints:\n" + "\n".join(missing))

        print(f"[{fold}] Merging backbone with OPCM ...")
        opcm_merge_backbone_incremental(task_paths, base_model, float(getattr(cfg.training, "opcm_alpha", 0.5)), dst_fold_dir)

        seen_heads: Dict[int, Dict[str, Tensor]] = {}
        for task_id, task_path in enumerate(task_paths):
            seen_heads[task_id] = strip_head_only(torch.load(task_path, map_location="cpu"))
            out_heads = dst_fold_dir / f"task_heads_until_task_{task_id}.pth"
            torch.save({"task_heads": seen_heads, "task_names": task_order[: task_id + 1]}, out_heads)

            package = {
                "strategy": "strategy2_keep_task_heads",
                "fold": fold,
                "seen_task_ids": list(range(task_id + 1)),
                "seen_task_names": task_order[: task_id + 1],
                "backbone_path": str(dst_fold_dir / f"merged_backbone_task_{task_id}.pth"),
                "task_heads_path": str(out_heads),
            }
            if args.copy_prompt_aux:
                aux_dir = ensure_dir(dst_fold_dir / "prompt_aux")
                copied = []
                for seen_id in range(task_id + 1):
                    src_aux = src_root / fold / f"prompt_aux_task_{seen_id}.pt"
                    if src_aux.exists():
                        dst_aux = aux_dir / f"prompt_aux_task_{seen_id}.pt"
                        shutil.copy2(src_aux, dst_aux)
                        copied.append(str(dst_aux))
                package["prompt_aux_paths"] = copied

            with open(dst_fold_dir / f"merged_package_task_{task_id}.json", "w", encoding="utf-8") as f:
                json.dump(package, f, indent=2)
            print(f"Saved {out_heads}")


if __name__ == "__main__":
    main()
