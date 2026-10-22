import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union, Tuple
import torch.nn as nn
from torch import Tensor
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             log_loss, roc_auc_score)
from tqdm import tqdm
import copy

StateDictType = Dict[str, Tensor]

# zeroshot prompt templates
TEMPLATES = [
    "CLASSNAME.",
    "an image of CLASSNAME.",
    "the image shows CLASSNAME.",
    "the image displays CLASSNAME.",
    "the image exhibits CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "I observe CLASSNAME.",
    "the pathology image shows CLASSNAME.",
    "a pathology image shows CLASSNAME.",
    "the pathology slide shows CLASSNAME.",
    "shows CLASSNAME.",
    "contains CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "CLASSNAME is observed.",
    "the pathology image reveals CLASSNAME.",
    "a microscopic image of showing CLASSNAME.",
    "histology shows CLASSNAME.",
    "CLASSNAME can be seen.",
    "the tissue shows CLASSNAME.",
    "CLASSNAME is identified.",
]

def seed_torch(device, seed=0):
    # ------------------------------------------------------------------------------------------
    # References:
    # HIPT: https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/main.py
    # ------------------------------------------------------------------------------------------
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def state_dict_sub(
    a: StateDictType, b: StateDictType, strict: bool = True, device=None
):
    """
    Returns the difference between two state dicts `a-b`.

    Args:
        a (StateDictType): The first state dict.
        b (StateDictType): The second state dict.
        strict (bool): Whether to check if the keys of the two state dicts are the same.

    Returns:
        StateDictType: The difference between the two state dicts.
    """
    if strict:
        assert set(a.keys()) == set(b.keys())

    diff = OrderedDict()
    for k in a:
        if k in b:
            diff[k] = a[k] - b[k]
            if device is not None:
                diff[k] = diff[k].to(device, non_blocking=True)
    return diff
  
def state_dict_to_vector(
    state_dict: StateDictType,
    remove_keys: Optional[List[str]] = None,
):
    """
    Convert a state dictionary to a vector.

    Args:
        state_dict (dict): The state dictionary to convert.
        remove_keys (list, optional): List of keys to remove from the state dictionary. Defaults to [].

    Returns:
        torch.Tensor: The converted vector.
    """
    remove_keys = remove_keys if remove_keys is not None else []
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )

def _svd(w: Tensor, full_matrices=True) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform Singular Value Decomposition (SVD) on a tensor.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    u, s, vh = torch.linalg.svd(
        w, full_matrices=full_matrices, driver="gesvd" if w.is_cuda else None
    )
    v = vh.T
    return u, s, v


def svd(
    w: Tensor, full_matrices=True, accelerator=None
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform SVD on a tensor, optionally using a specified accelerator.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.
        accelerator (str): The device to perform the computation on.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    if accelerator is None:
        return _svd(w, full_matrices=full_matrices)
    original_device = w.device
    w = w.to(accelerator)
    u, s, v = _svd(w)
    return u.to(original_device), s.to(original_device), v.to(original_device)


def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def get_task_vector_norm(model: nn.Module, pretrained_model: nn.Module) -> Tensor:
    """
    Get the vector norm of the task model.

    Args:
        model (nn.Module): The task model.
        pretrained_model (nn.Module): The pretrained model.

    Returns:
        Tensor: The vector norm of the task model.
    """
    return torch.linalg.norm(
        state_dict_to_vector(
            state_dict_sub(model, pretrained_model)
        )
    )