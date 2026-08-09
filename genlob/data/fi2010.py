"""FI-2010 benchmark loader utilities.

FI-2010 files are normalized benchmark matrices, not replayable LOB data. The
processed adapter keeps the benchmark protocol separate from GenLOB
``snapshot_l2`` datasets: arrays are feature tables plus official labels, and
windowing is performed dynamically by the PyTorch dataset below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class FI2010WindowDataset(Dataset):
    """Create DeepLOB-style windows from processed FI-2010 features.

    Args:
        features: ``[N, F]`` feature table. For DeepLOB-style work this is
            usually the first 40 FI-2010 LOB feature rows.
        labels: ``[N, 5]`` official FI-2010 labels. Values are expected to be
            1/2/3; returned labels are zero-based 0/1/2.
        window_len: Number of historical events per sample.
        label_index: Which of the five official label rows to use.
        stride: Window stride.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window_len: int = 100,
        label_index: int = 4,
        stride: int = 1,
    ) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must be [N, F], got {features.shape}")
        if labels.ndim != 2 or labels.shape[1] < 1:
            raise ValueError(f"labels must be [N, L], got {labels.shape}")
        if len(features) != len(labels):
            raise ValueError(
                f"features/labels length mismatch: {len(features)} vs {len(labels)}"
            )
        if not 0 <= label_index < labels.shape[1]:
            raise ValueError(f"label_index {label_index} out of range for {labels.shape[1]} labels")
        if window_len <= 0:
            raise ValueError("window_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels[:, label_index], dtype=np.int64) - 1)
        self.window_len = int(window_len)
        self.label_index = int(label_index)
        self.stride = int(stride)
        self.starts = np.arange(0, max(0, len(features) - window_len + 1), stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        end = start + self.window_len
        x = self.features[start:end]
        y = self.labels[end - 1]
        return x, y


def _load_split(processed_dir: Path, split: str, use_all_features: bool) -> tuple[np.ndarray, np.ndarray]:
    feature_name = "all_features" if use_all_features else "lob_features"
    features = np.load(processed_dir / f"{split}_{feature_name}.npy")
    labels = np.load(processed_dir / f"{split}_labels.npy")
    return features, labels


def load_fi2010_arrays(
    processed_dir: str | Path = "data/processed/fi2010/noauction_zscore_cf7",
    use_all_features: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load processed FI-2010 arrays keyed by split name."""

    root = Path(processed_dir)
    if not root.exists():
        raise FileNotFoundError(f"FI-2010 processed dir not found: {root}")

    splits = ["train", "test_cf7", "test_cf8", "test_cf9"]
    return {split: _load_split(root, split, use_all_features) for split in splits}


def load_fi2010_fold(
    processed_dir: str | Path,
    fold: int,
    use_all_features: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load one official anchored train/test fold with CF7 compatibility."""
    if fold < 1 or fold > 9:
        raise ValueError("FI-2010 fold must be in [1, 9]")
    root = Path(processed_dir)
    train_name = f"train_cf{fold}"
    test_name = f"test_cf{fold}"
    feature_name = "all_features" if use_all_features else "lob_features"
    if not (root / f"{train_name}_{feature_name}.npy").exists() and fold == 7:
        train_name = "train"
    return {
        "train": _load_split(root, train_name, use_all_features),
        "test": _load_split(root, test_name, use_all_features),
    }


def get_fi2010_loaders(
    processed_dir: str | Path = "data/processed/fi2010/noauction_zscore_cf7",
    window_len: int = 100,
    label_index: int = 4,
    batch_size: int = 128,
    stride: int = 1,
    num_workers: int = 0,
    use_all_features: bool = False,
) -> Mapping[str, DataLoader]:
    """Create DataLoaders for the processed FI-2010 benchmark arrays."""

    arrays = load_fi2010_arrays(processed_dir, use_all_features=use_all_features)
    loaders: dict[str, DataLoader] = {}
    for split, (features, labels) in arrays.items():
        ds = FI2010WindowDataset(
            features,
            labels,
            window_len=window_len,
            label_index=label_index,
            stride=stride,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
