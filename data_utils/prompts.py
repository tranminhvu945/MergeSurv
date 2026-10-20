from __future__ import annotations

from typing import Dict, List, Tuple

# Prompt template style follows MergeSlide's class-aware prompt ensemble idea,
# but the class name is replaced by an ordinal prognosis concept.
SURVIVAL_TEMPLATES = [
    "CLASSNAME.",
    "a histopathological image showing CLASSNAME.",
    "a histopathological image of CLASSNAME.",
    "an H&E stained image showing CLASSNAME.",
    "an H&E image of CLASSNAME.",
    "a whole slide image associated with CLASSNAME.",
    "a pathology image indicating CLASSNAME.",
    "tumor tissue associated with CLASSNAME.",
    "a histopathological whole slide image associated with CLASSNAME.",
    "an H&E whole slide image associated with CLASSNAME.",
]

TASK_TEMPLATES = [
    "CLASSNAME.",
    "a histopathological whole slide image of CLASSNAME.",
    "an H&E stained whole slide image of CLASSNAME.",
    "a pathology image from CLASSNAME.",
]

PROGNOSIS_LEVELS = [
    "very poor prognosis",
    "poor prognosis",
    "intermediate prognosis",
    "favorable prognosis",
    "very favorable prognosis",
]

RISK_LEVELS = [
    "very high risk of early death",
    "high risk of death",
    "intermediate survival risk",
    "low survival risk",
    "very low survival risk and long overall survival",
]

SHORT_OS_LEVELS = [
    "very short overall survival",
    "short overall survival",
    "moderate overall survival",
    "long overall survival",
    "very long overall survival",
]


def _ordinal_survival_prompts(cancer_name: str, aliases: list[str] | None = None, num_levels: int = 5) -> Tuple[List[List[str]], List[str]]:
    if num_levels != 5:
        raise ValueError(
            "This prompt file currently defines 5 ordered prognosis levels. "
            "Set survival.num_prompt_bins=5 or extend PROGNOSIS_LEVELS/RISK_LEVELS."
        )
    names = [cancer_name] + (aliases or [])
    prompts: list[list[str]] = []
    for k in range(num_levels):
        concepts = []
        for name in names:
            concepts.extend(
                [
                    f"{name} with {PROGNOSIS_LEVELS[k]}",
                    f"{name} associated with {RISK_LEVELS[k]}",
                    f"{name} associated with {SHORT_OS_LEVELS[k]}",
                ]
            )
        prompts.append(concepts)
    return prompts, SURVIVAL_TEMPLATES


def blca_survival_prompts(num_levels: int = 5):
    return _ordinal_survival_prompts("bladder urothelial carcinoma", ["bladder cancer", "TCGA-BLCA"], num_levels)


def brca_survival_prompts(num_levels: int = 5):
    return _ordinal_survival_prompts("breast invasive carcinoma", ["breast cancer", "TCGA-BRCA"], num_levels)


def luad_survival_prompts(num_levels: int = 5):
    return _ordinal_survival_prompts("lung adenocarcinoma", ["lung cancer adenocarcinoma", "TCGA-LUAD"], num_levels)


def ucec_survival_prompts(num_levels: int = 5):
    return _ordinal_survival_prompts("uterine corpus endometrial carcinoma", ["endometrial cancer", "TCGA-UCEC"], num_levels)


TASK_NAMES: Dict[str, List[str]] = {
    "blca": ["bladder urothelial carcinoma", "bladder cancer", "TCGA-BLCA"],
    "brca": ["breast invasive carcinoma", "breast cancer", "TCGA-BRCA"],
    "luad": ["lung adenocarcinoma", "lung cancer adenocarcinoma", "TCGA-LUAD"],
    "ucec": ["uterine corpus endometrial carcinoma", "endometrial cancer", "TCGA-UCEC"],
}


def task_level_prompts(task: str) -> Tuple[List[List[str]], List[str]]:
    """Task-level prompts for CLASS-IL routing after merging."""
    if task not in TASK_NAMES:
        raise KeyError(f"Unknown task {task!r}. Available: {sorted(TASK_NAMES)}")
    return [[name for name in TASK_NAMES[task]]], TASK_TEMPLATES
