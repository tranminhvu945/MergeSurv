from __future__ import annotations

import bisect
import os
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from core.config import get_task_order, load_config


def collate_MIL_survival(batch):
    """Collate WSI bags without concatenating different slides.

    Stage-1 uses Cox/DeepSurv scalar risk. Cox loss needs multiple patients in
    a mini-batch, but each WSI bag must still be forwarded separately through
    the TITAN slide aggregator. Therefore, this collate function returns lists
    of feature/coord tensors plus batched labels.
    """
    features = [item[0] for item in batch]
    coords = [item[1] for item in batch]
    time_bin = torch.LongTensor([item[2] for item in batch])
    censorship = torch.FloatTensor([item[3] for item in batch])
    survival_time = torch.FloatTensor([item[4] for item in batch])
    slide_id = [item[5] for item in batch]
    return features, coords, time_bin, censorship, survival_time, slide_id


class Generic_Survival_MIL_Dataset:
    """MIL survival dataset for pre-extracted WSI features.

    Returned censorship convention follows MambaMIL/CLAM survival code:
        censorship=0 observed death/event, censorship=1 censored.

    Prompt bins are task-specific quantile bins used only for VLSA-style
    auxiliary losses. They are NOT the output bins of the main survival head.
    """

    def __init__(self, csv_path, data_dir, time_col, event_col, event_observed_value=1, quantiles=(0.2, 0.4, 0.6, 0.8)):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.time_col = time_col
        self.event_col = event_col
        self.event_observed_value = event_observed_value
        self.quantiles = tuple(float(q) for q in quantiles)

        slide_data = pd.read_csv(csv_path)
        required = {"slide_id", time_col, event_col}
        missing = required - set(slide_data.columns)
        if missing:
            raise KeyError(f"{csv_path} is missing required columns: {sorted(missing)}")

        slide_data = slide_data.dropna(subset=["slide_id", time_col, event_col]).reset_index(drop=True)
        slide_data[time_col] = pd.to_numeric(slide_data[time_col], errors="coerce")
        slide_data = slide_data.dropna(subset=[time_col]).reset_index(drop=True)

        raw_status = pd.to_numeric(slide_data[event_col], errors="coerce")
        if raw_status.isna().any():
            raise ValueError(f"Column {event_col!r} must be numeric or convertible to numeric in {csv_path}.")

        status_name = str(event_col).lower()
        if status_name in {"censorship", "censor", "censored", "is_censored"}:
            slide_data["censorship"] = raw_status.astype(int)
        else:
            event = (raw_status == int(event_observed_value)).astype(int)
            slide_data["censorship"] = (1 - event).astype(int)

        unique_censorship = set(slide_data["censorship"].unique().tolist())
        if not unique_censorship.issubset({0, 1}):
            raise ValueError(f"Converted censorship values must be binary 0/1. Got {sorted(unique_censorship)}.")

        n_event = int((slide_data["censorship"] == 0).sum())
        n_censored = int((slide_data["censorship"] == 1).sum())
        print(
            f"[Survival CSV] {os.path.basename(str(csv_path))}: rows={len(slide_data)}, "
            f"events={n_event}, censored={n_censored}, event_col={event_col!r}"
        )
        self.slide_data = slide_data

    @staticmethod
    def _normalize_id(x) -> str:
        x = str(x).strip()
        x = os.path.basename(x)
        for suffix in [".svs", ".h5", ".pt"]:
            if x.endswith(suffix):
                x = x[: -len(suffix)]
        return x

    @classmethod
    def _match_split(cls, slide_data: pd.DataFrame, ids: Iterable[str]) -> pd.DataFrame:
        ids_norm = {cls._normalize_id(x) for x in ids if pd.notna(x) and str(x).strip() != ""}
        if not ids_norm:
            return slide_data.iloc[0:0].reset_index(drop=True)

        df = slide_data.copy()
        masks = []

        if "case_id" in df.columns:
            masks.append(df["case_id"].apply(cls._normalize_id).isin(ids_norm))

        slide_norm = df["slide_id"].apply(cls._normalize_id)
        masks.append(slide_norm.isin(ids_norm))

        derived_case = slide_norm.apply(lambda x: "-".join(x.split("-")[:3]) if len(x.split("-")) >= 3 else x)
        masks.append(derived_case.isin(ids_norm))

        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m
        return df[mask].reset_index(drop=True)

    def _compute_boundaries(self, train_df: pd.DataFrame) -> np.ndarray:
        observed = train_df[train_df["censorship"] == 0][self.time_col].values.astype(float)
        if len(observed) < len(self.quantiles) + 1:
            observed = train_df[self.time_col].values.astype(float)
        if len(observed) == 0:
            raise ValueError("Cannot compute prompt bin boundaries because training split is empty.")
        boundaries = np.quantile(observed, self.quantiles).astype(float)
        # Prevent duplicated boundaries from breaking searchsorted semantics.
        boundaries = np.maximum.accumulate(boundaries)
        return boundaries

    @staticmethod
    def _assign_bins(df: pd.DataFrame, time_col: str, boundaries: np.ndarray) -> pd.DataFrame:
        out = df.copy()
        out["time_bin"] = np.searchsorted(boundaries, out[time_col].values.astype(float), side="right").astype(int)
        out["time_bin"] = out["time_bin"].clip(0, len(boundaries))
        return out.reset_index(drop=True)

    @staticmethod
    def _event_summary(df: pd.DataFrame) -> str:
        if len(df) == 0 or "censorship" not in df.columns:
            return "events=0, censored=0"
        events = int((df["censorship"] == 0).sum())
        censored = int((df["censorship"] == 1).sum())
        return f"events={events}, censored={censored}"

    def return_splits(self, from_id=False, csv_path=None):
        if from_id:
            raise NotImplementedError("This configured version reads split CSV files.")
        if csv_path is None:
            raise ValueError("csv_path must be provided")

        all_splits = pd.read_csv(csv_path)
        if "train" not in all_splits.columns:
            raise KeyError(f"Missing 'train' column in {csv_path}. Available: {list(all_splits.columns)}")
        if "val" not in all_splits.columns:
            raise KeyError(f"Missing 'val' column in {csv_path}. Available: {list(all_splits.columns)}")

        train_ids = all_splits["train"].dropna().tolist()
        val_ids = all_splits["val"].dropna().tolist()
        test_ids = val_ids

        train_df = self._match_split(self.slide_data, train_ids)
        val_df = self._match_split(self.slide_data, val_ids)
        test_df = self._match_split(self.slide_data, test_ids)

        print(
            f"[Split] {os.path.basename(str(csv_path))} | "
            f"train ids={len(train_ids)}, matched slides={len(train_df)} ({self._event_summary(train_df)}) | "
            f"val ids={len(val_ids)}, matched slides={len(val_df)} ({self._event_summary(val_df)}) | "
            f"test=val, matched slides={len(test_df)}"
        )

        boundaries = self._compute_boundaries(train_df)
        print(f"[Prompt bins] task-specific quantile boundaries={boundaries.tolist()}")

        train_df = self._assign_bins(train_df, self.time_col, boundaries)
        val_df = self._assign_bins(val_df, self.time_col, boundaries)
        test_df = self._assign_bins(test_df, self.time_col, boundaries)

        return (
            Generic_Survival_Split(train_df, self.data_dir, self.time_col, boundaries),
            Generic_Survival_Split(val_df, self.data_dir, self.time_col, boundaries),
            Generic_Survival_Split(test_df, self.data_dir, self.time_col, boundaries),
        )


class Generic_Survival_Split(Dataset):
    def __init__(self, slide_data, data_dir, time_col, boundaries):
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.time_col = time_col
        self.boundaries = np.asarray(boundaries, dtype=float)
        self.num_prompt_bins = len(self.boundaries) + 1

    def __len__(self):
        return len(self.slide_data)

    @staticmethod
    def _stem_candidates(slide_id: str, case_id: str | None = None):
        stem = os.path.basename(str(slide_id)).replace(".svs", "").replace(".h5", "").replace(".pt", "")
        candidates = [stem, stem.split(".")[0]]
        parts = stem.split("-")
        if len(parts) >= 3:
            candidates.append("-".join(parts[:3]))
        if case_id is not None and str(case_id) != "nan":
            candidates.append(str(case_id))
        seen, out = set(), []
        for c in candidates:
            if c and c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def _find_feature_path(self, row):
        slide_id = str(row["slide_id"])
        case_id = str(row["case_id"]) if "case_id" in row.index else None
        stems = self._stem_candidates(slide_id, case_id)

        search_dirs = [self.data_dir, os.path.join(self.data_dir, "h5_files")]
        searched = []
        for stem in stems:
            for d in search_dirs:
                path = os.path.join(d, f"{stem}.h5")
                searched.append(path)
                if os.path.exists(path):
                    return path

        raise FileNotFoundError(
            "Cannot find H5 feature file. PT fallback is disabled.\n"
            f"slide_id={slide_id}\ncase_id={case_id}\nsearched paths=\n  " + "\n  ".join(searched[:20])
        )

    def __getitem__(self, idx):
        row = self.slide_data.iloc[idx]
        h5_path = self._find_feature_path(row)

        with h5py.File(h5_path, "r") as hdf5_file:
            if "features" not in hdf5_file:
                raise KeyError(f"H5 file {h5_path} does not contain dataset 'features'.")
            if "coords" not in hdf5_file:
                raise KeyError(f"H5 file {h5_path} does not contain dataset 'coords'.")
            features = hdf5_file["features"][:]
            coords = hdf5_file["coords"][:]

        features = torch.from_numpy(features) if not torch.is_tensor(features) else features
        coords = torch.from_numpy(coords) if not torch.is_tensor(coords) else coords
        return features, coords, int(row["time_bin"]), int(row["censorship"]), float(row[self.time_col]), str(row["slide_id"])


class ConcatDataset(Dataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super().__init__()
        assert len(datasets) > 0, "datasets should not be empty"
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class Sequential_Generic_MIL_Survival_Dataset:
    NAME = "seq-wsi-survival"
    SETTING = "class-il"

    def __init__(self, cfg=None, config_path="configs/forward.yaml"):
        self.cfg = cfg or load_config(config_path)
        self.task_order = get_task_order(self.cfg)[: int(self.cfg.training.num_tasks)]
        self.N_TASKS = int(self.cfg.training.num_tasks)
        self.batch_size = int(self.cfg.dataloader.batch_size)
        self.num_workers = int(self.cfg.dataloader.num_workers)
        self.datasets, self.split_dirs = self._build_datasets()
        self.test_loaders = []
        self.train_loader = None
        self.val_loader = None

    def _build_datasets(self):
        datasets, split_dirs = [], []
        surv_cfg = self.cfg.survival
        quantiles = surv_cfg.get("boundary_quantiles", [0.2, 0.4, 0.6, 0.8])
        for task in self.task_order:
            task_cfg = self.cfg.dataset[task]
            ds = Generic_Survival_MIL_Dataset(
                csv_path=task_cfg.csv,
                data_dir=task_cfg.features,
                time_col=surv_cfg.time_col,
                event_col=surv_cfg.event_col,
                event_observed_value=surv_cfg.get("event_observed_value", 1),
                quantiles=quantiles,
            )
            datasets.append(ds)
            split_dirs.append(task_cfg.splits)
        return datasets, split_dirs

    def _split_csv(self, fold: int, task_id: int) -> str:
        return os.path.join(self.split_dirs[task_id], f"splits_{fold}.csv")

    def get_data_loaders(self, FOLD, task_id):
        dataset = self.datasets[task_id]
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=self._split_csv(FOLD, task_id))
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            collate_fn=collate_MIL_survival, pin_memory=torch.cuda.is_available(), drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            collate_fn=collate_MIL_survival, pin_memory=torch.cuda.is_available(), drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            collate_fn=collate_MIL_survival, pin_memory=torch.cuda.is_available(), drop_last=False,
        )
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        self.val_loader = val_loader
        return train_loader, val_loader, test_loader

    def get_joint_data_loaders(self, FOLD):
        train_datasets, val_datasets = [], []
        test_loader = None
        for n in range(self.N_TASKS):
            dataset = self.datasets[n]
            train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=self._split_csv(FOLD, n))
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_MIL_survival)
            self.test_loaders.append(test_loader)
        train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_MIL_survival)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_MIL_survival)
        self.train_loader = train_loader
        self.val_loader = val_loader
        return train_loader, val_loader, test_loader
