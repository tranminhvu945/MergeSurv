from __future__ import annotations

import argparse
import copy
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
import json

CANONICAL_TASK_ORDER = ["blca", "ucec", "luad", "brca"]
DEFAULT_NUM_PROMPT_BINS = 5


class DotDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _to_dotdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(v) for v in obj]
    return obj


def _lookup(root: Mapping[str, Any], dotted_key: str) -> Any:
    cur: Any = root
    for part in dotted_key.split("."):
        cur = cur[part]
    return cur


def _resolve_interpolation(obj: Any, root: Mapping[str, Any]) -> Any:
    pattern = re.compile(r"\$\{([^}]+)\}")
    if isinstance(obj, dict):
        return {k: _resolve_interpolation(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_interpolation(v, root) for v in obj]
    if isinstance(obj, str):
        def repl(match: re.Match[str]) -> str:
            return str(_lookup(root, match.group(1)))
        return pattern.sub(repl, obj)
    return obj


def load_config(config_path: str | os.PathLike[str] = "configs/forward.yaml") -> DotDict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    resolved = copy.deepcopy(raw)
    for _ in range(5):
        new_resolved = _resolve_interpolation(resolved, resolved)
        if new_resolved == resolved:
            break
        resolved = new_resolved
    return _to_dotdict(resolved)


def add_config_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default="configs/forward.yaml", help="Path to YAML config file.")
    return parser


def get_task_order(cfg: Mapping[str, Any]) -> list[str]:
    order_name = cfg["dataset"].get("order", "forward")
    try:
        order = list(cfg["dataset"]["tasks"][order_name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset.order={order_name!r}; expected one of {list(cfg['dataset']['tasks'].keys())}"
        ) from exc
    return order


def get_num_prompt_bins(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        return DEFAULT_NUM_PROMPT_BINS
    return int(cfg.get("survival", {}).get("num_prompt_bins", DEFAULT_NUM_PROMPT_BINS))


def get_prompt_bin_counts(task_order: Iterable[str], cfg: Mapping[str, Any] | None = None) -> list[int]:
    n = get_num_prompt_bins(cfg)
    return [n for _ in task_order]


# Backward-compatible names for scripts originally adapted from classification.
def get_num_bins(task_order: Iterable[str], cfg: Mapping[str, Any] | None = None) -> list[int]:
    return get_prompt_bin_counts(task_order, cfg)


def get_num_classes(task_order: Iterable[str], cfg: Mapping[str, Any] | None = None) -> list[int]:
    return get_prompt_bin_counts(task_order, cfg)


def get_bin_slices(task_order: Iterable[str], cfg: Mapping[str, Any] | None = None) -> dict[int, tuple[int, int]]:
    slices: dict[int, tuple[int, int]] = {}
    start = 0
    for task_id, _task_name in enumerate(task_order):
        n = get_num_prompt_bins(cfg)
        slices[task_id] = (start, start + n)
        start += n
    return slices


def get_class_slices(task_order: Iterable[str], cfg: Mapping[str, Any] | None = None) -> dict[int, tuple[int, int]]:
    return get_bin_slices(task_order, cfg)


def seed_everything(seed: int, device: torch.device | str | None = None) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def resolve_finetuned_checkpoints_by_name(
    src_root: str | os.PathLike[str],
    fold: str,
    task_order: Iterable[str],
) -> list[Path]:
    fold_dir = Path(src_root) / fold

    if not fold_dir.exists():
        raise FileNotFoundError(
            f"Stage-1 fold directory does not exist: {fold_dir}"
        )

    checkpoint_by_task: dict[str, Path] = {}

    metadata_paths = sorted(
        fold_dir.glob("finetune_task_*_metadata.json")
    )

    if not metadata_paths:
        raise FileNotFoundError(
            f"No Stage-1 metadata files found in: {fold_dir}"
        )

    for metadata_path in metadata_paths:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        if "task_name" not in metadata:
            raise KeyError(
                f"Missing task_name in {metadata_path}"
            )

        task_name = str(metadata["task_name"]).lower()
        original_task_id = int(metadata["task_id"])

        # Prefer the checkpoint located beside its metadata.
        checkpoint_path = (
            fold_dir / f"finetune_task_{original_task_id}.pt"
        )

        # Fall back to the path stored in metadata.
        if not checkpoint_path.exists():
            stored_path = metadata.get("checkpoint")
            if stored_path:
                stored_path = Path(stored_path)
                if stored_path.exists():
                    checkpoint_path = stored_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for task={task_name} not found. "
                f"Expected: {checkpoint_path}"
            )

        if task_name in checkpoint_by_task:
            raise RuntimeError(
                f"Duplicate Stage-1 metadata for task={task_name}"
            )

        checkpoint_by_task[task_name] = checkpoint_path

    normalized_order = [
        str(task).lower() for task in task_order
    ]

    missing_tasks = [
        task
        for task in normalized_order
        if task not in checkpoint_by_task
    ]

    if missing_tasks:
        raise RuntimeError(
            f"Missing Stage-1 checkpoints for tasks: {missing_tasks}. "
            f"Available tasks: {sorted(checkpoint_by_task)}"
        )

    resolved_paths = [
        checkpoint_by_task[task]
        for task in normalized_order
    ]

    print(f"[{fold}] Resolved merging order:")
    for merge_position, (task, checkpoint) in enumerate(
        zip(normalized_order, resolved_paths)
    ):
        print(
            f"  position={merge_position}: "
            f"{task.upper()} <- {checkpoint.name}"
        )

    return resolved_paths