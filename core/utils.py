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

def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    unique_classes: Optional[List[int]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics and return the evaluation metrics.

    Args:
        targets_all (array-like): True target values.
        preds_all (array-like): Predicted target values.
        probs_all (array-like, optional): Predicted probabilities for each class. Defaults to None.
        get_report (bool, optional): Whether to include the classification report in the results. Defaults to True.
        prefix (str, optional): Prefix to add to the result keys. Defaults to "".
        roc_kwargs (dict, optional): Additional keyword arguments for calculating ROC AUC. Defaults to {}.

    Returns:
        dict: Dictionary containing the evaluation metrics.

    """
    unique_classes = unique_classes if unique_classes is not None else np.unique(targets_all)
    bacc = balanced_accuracy_score(targets_all, preds_all) if len(targets_all) > 1 else 0
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    nw_kappa = cohen_kappa_score(targets_all, preds_all, weights="linear")
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0, labels=unique_classes)

    eval_metrics = {
        f"{prefix}/acc": acc,
        f"{prefix}/bacc": bacc,
        f"{prefix}/kappa": kappa,
        f"{prefix}/nw_kappa": nw_kappa,
        f"{prefix}/weighted_f1": cls_rep["weighted avg"]["f1-score"],
    }

    if probs_all is not None:
        if len(np.unique(targets_all)) > 1:
            try: 
                loss = log_loss(targets_all, probs_all, labels=unique_classes)
                roc_auc = roc_auc_score(targets_all, probs_all, labels=unique_classes, **roc_kwargs)
            except ValueError:
                roc_auc = -1
                loss = -1
            eval_metrics[f"{prefix}/loss"] = loss
            eval_metrics[f"{prefix}/auroc"] = roc_auc

    return eval_metrics


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
    
    
def merge_dict(dict1, dict2):
    for k in dict2.keys():
        if k in dict1.keys():
            dict1[k].append(dict2[k])
        else:
            dict1[k] = [dict2[k]]
    return dict1

    
def bootstrap(results_dict=None, preds_all=None, targets_all=None, probs_all=None, n=1000, alpha=0.95, format_as_str=False):
    if results_dict is not None:
        targets_all = results_dict['targets']
        probs_key = 'logits' if 'logits' in results_dict.keys() else 'probs'
        if probs_key == 'logits':
            results_dict[probs_key] = softmax(results_dict[probs_key], axis=1)
        probs_all = results_dict[probs_key] if probs_key in results_dict.keys() else None
        preds_all = results_dict['preds'] if 'preds' in results_dict.keys() else None
        if probs_all is None:
            assert 'preds' in results_dict.keys()
            preds_all = results_dict['preds']
        if preds_all is None:
            preds_all = np.argmax(probs_all, axis=1)
    
    num_classes = len(np.unique(targets_all))
    if probs_all is not None and len(probs_all.shape) == 2:
        probs_all = probs_all[:, 1] if num_classes == 2 else probs_all
    roc_kwargs = {'average': 'macro', 'multi_class': 'ovo'} if num_classes > 2 else {}
    overall_scores = get_eval_metrics(probs_all=probs_all, preds_all=preds_all, targets_all=targets_all, roc_kwargs=roc_kwargs)

    all_scores = {}
    for seed in tqdm(range(n)):
        np.random.seed(seed)
        bootstrap_ind = list(pd.Series(targets_all).sample(n=len(targets_all), replace=True, random_state=seed).index)
        collision = 0
        while len(np.unique(targets_all[bootstrap_ind])) != num_classes:
            bootstrap_ind = list(pd.Series(targets_all).sample(n=len(targets_all), replace=True, random_state=seed+collision+n).index)
            collision += 1
            if collision % 100 == 0:
                print(collision)
        sample_targets_all = targets_all[bootstrap_ind]
        sample_preds_all = preds_all[bootstrap_ind] if preds_all is not None else None
        sample_probs_all = probs_all[bootstrap_ind] if probs_all is not None else None
        results = get_eval_metrics(probs_all=sample_probs_all, preds_all=sample_preds_all, targets_all=sample_targets_all, roc_kwargs=roc_kwargs)
        merge_dict(all_scores, results)
    
    ci_dict = {}
    ci_as_str_dict = {}
    mean_dict = {}
    std_dict = {}
    for k in all_scores.keys():
        scores = np.array(all_scores[k])
        mean_dict[k] = scores.mean()
        std_dict[k] = scores.std()
        
    return mean_dict, std_dict

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


def frobenius_inner_product(w1: Tensor, w2: Tensor) -> Tensor:
    return torch.trace(w1.T @ w2)


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

def get_task_vector_state_dict(model: nn.Module, pretrained_model: nn.Module) -> Tensor:
    """
    Get the vector norm of the task model.

    Args:
        model (nn.Module): The task model.
        pretrained_model (nn.Module): The pretrained model.

    Returns:
        Tensor: The vector norm of the task model.
    """
    return state_dict_to_vector(
            state_dict_sub(model, pretrained_model)
        )
