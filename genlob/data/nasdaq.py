"""
NASDAQ LOB Dataset Loader for GenLOB

Loads LOB snapshots parsed from ITCH 5.0 binary files.
Applies mid-price relative normalization (preserves LOB structure).

Data format: .npy files with shape [N, 2, 20]
  Channel 0: prices (ask_p1..10, bid_p1..10) in dollars
  Channel 1: volumes (ask_v1..10, bid_v1..10) in shares
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional

from .normalize import normalize_lob, compute_norm_stats, NormStats


def normalize_anchor_relative_sequence(
    raw_sequence: np.ndarray | torch.Tensor,
    hist_len: int,
    normalization: dict,
) -> torch.Tensor:
    """Normalize one raw history+future sequence around the last history mid.

    Unlike the retained per-frame-centered representation, this preserves the
    future mid-price path while remaining translation invariant to the absolute
    stock price level.
    """
    # Preserve exchange tick differences at high nominal prices during the
    # subtraction; converting GOOG-like dollar prices to float32 first can turn
    # a $0.01 tick into roughly $0.009887.
    raw = torch.as_tensor(raw_sequence, dtype=torch.float64)
    if raw.ndim != 3 or raw.shape[1:] != (2, 20):
        raise ValueError(f"Expected [T, 2, 20] raw sequence, got {tuple(raw.shape)}")
    if not 0 < hist_len <= len(raw):
        raise ValueError(f"hist_len={hist_len} is invalid for sequence length {len(raw)}")

    price_scale = float(normalization.get("price_scale", 100.0))
    price_std = float(normalization.get("price_global_std", 0.0))
    volume_clip = float(normalization.get("volume_clip", 0.0))
    volume_scale = float(normalization.get("volume_scale", 15.0))
    tick_size = float(normalization.get("tick_size_raw", 0.0))
    ticks_per_unit = float(normalization.get("ticks_per_price_unit", 0.0))
    tick_scaled = tick_size > 0 and ticks_per_unit > 0
    if (not tick_scaled and price_std <= 0) or volume_clip <= 0 or volume_scale <= 0:
        raise ValueError("Anchor-relative normalization requires positive train-only normalization scales")

    anchor_frame = raw[hist_len - 1, 0]
    anchor_mid = (anchor_frame[0] + anchor_frame[10]) / 2.0
    if tick_scaled:
        prices = (raw[:, 0] - anchor_mid) / (tick_size * ticks_per_unit)
    else:
        prices = (raw[:, 0] - anchor_mid) * price_scale / price_std
    volumes = torch.sqrt(torch.clamp(raw[:, 1], min=0.0, max=volume_clip)) / volume_scale
    return torch.stack([prices, volumes], dim=1).float()


def normalized_price_unit(normalization: dict | None) -> float | None:
    """Raw price units represented by one normalized price unit."""
    if not normalization:
        return None
    tick_size = float(normalization.get("tick_size_raw", 0.0))
    ticks_per_unit = float(normalization.get("ticks_per_price_unit", 0.0))
    if tick_size > 0 and ticks_per_unit > 0:
        return tick_size * ticks_per_unit
    price_scale = float(normalization.get("price_scale", 0.0))
    price_std = float(normalization.get("price_global_std", 0.0))
    if price_scale <= 0 or price_std <= 0:
        return None
    return price_std / price_scale


def resolve_price_reference(requested: str, has_raw_states: bool) -> str:
    """Resolve the sequence price reference used by train/eval protocols."""
    if requested not in {"auto", "dataset", "history_last_mid"}:
        raise ValueError(f"Unknown price_reference: {requested}")
    if requested == "auto":
        return "history_last_mid" if has_raw_states else "dataset"
    if requested == "history_last_mid" and not has_raw_states:
        raise ValueError("history_last_mid requires states_raw.npy")
    return requested


def resolve_lob_constraints(
    normalization: dict | None,
    min_spread: float | None = None,
    min_gap: float | None = None,
) -> tuple[float, float]:
    """Resolve normalized hard constraints from explicit values or metadata.

    Older processed datasets did not store a separate minimum adjacent-level
    gap. For those datasets, the normalized minimum spread is the conservative
    fallback because both quantities are one exchange tick in a clean L2 book.
    """
    metadata_spread = None
    metadata_gap = None
    if normalization:
        metadata_spread = normalization.get("min_spread_normalized")
        metadata_gap = normalization.get("min_level_gap_normalized", metadata_spread)

    resolved_spread = float(min_spread if min_spread is not None else metadata_spread or 1e-6)
    resolved_gap = float(min_gap if min_gap is not None else metadata_gap or 1e-8)
    if resolved_spread <= 0 or resolved_gap < 0:
        raise ValueError(
            f"LOB constraints must satisfy min_spread>0 and min_gap>=0; "
            f"got {resolved_spread}, {resolved_gap}"
        )
    return resolved_spread, resolved_gap


def select_evaluation_indices(
    n_windows: int,
    n_batches: int,
    batch_size: int,
    sampling_mode: str,
) -> np.ndarray:
    """Select chronological evaluation windows without a head-only bias."""
    if n_windows <= 0:
        return np.empty(0, dtype=np.int64)
    if sampling_mode == "full":
        return np.arange(n_windows, dtype=np.int64)
    if sampling_mode not in {"uniform", "head"}:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
    target = min(n_windows, n_batches * batch_size)
    if target <= 0:
        raise ValueError("n_batches * batch_size must be positive unless sampling_mode=full")
    if sampling_mode == "head":
        return np.arange(target, dtype=np.int64)
    return np.linspace(0, n_windows - 1, target, dtype=np.int64)


def parse_snapshot_dataset_id(dataset: str) -> tuple[str, str]:
    """Parse canonical and backward-compatible snapshot dataset ids.

    Accepted:
        INTC_9day
        nasdaq_INTC_9day
    """
    parts = dataset.split("_")
    if parts[0].lower() == "nasdaq":
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(
            f"Expected dataset id like INTC_9day, GOOG_public_01302019, "
            f"or nasdaq_INTC_9day, got {dataset}"
        )
    return parts[0].upper(), "_".join(parts[1:])


class NASDAQGenerationDataset(Dataset):
    """
    NASDAQ LOB dataset for generation tasks.

    Creates (history, future) pairs with mid-price relative normalization.
    Output: [2, 20, T] tensors.
    """

    def __init__(
        self,
        data_norm: np.ndarray,
        hist_len: int = 50,
        pred_len: int = 10,
        stride: int = 1,
    ):
        self.data = torch.from_numpy(data_norm).float()  # [N, 2, 20]
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.stride = stride
        total_len = hist_len + pred_len
        self.n_samples = max(0, (len(data_norm) - total_len) // stride + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        history = self.data[start:start + self.hist_len]                         # [H, 2, 20]
        future = self.data[start + self.hist_len:start + self.hist_len + self.pred_len]  # [P, 2, 20]
        return history.permute(1, 2, 0), future.permute(1, 2, 0)  # [2, 20, T]


def load_nasdaq_stock(
    data_dir: str,
    stock: str = "TSLA",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict:
    """
    Load and normalize NASDAQ stock LOB data.

    Args:
        data_dir: directory containing {STOCK}_lob.npy files
        stock: stock ticker
        train_ratio, val_ratio: split ratios (test = 1 - train - val)

    Returns:
        dict with train/val/test normalized data and norm_stats
    """
    path = Path(data_dir) / f"{stock}_lob.npy"
    raw = np.load(path)  # [N, 2, 20]

    # Compute stats from training portion
    N = len(raw)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    stats = compute_norm_stats(raw[:n_train])

    # Normalize entire dataset with training stats
    data_norm, stats = normalize_lob(raw, stats)

    return {
        "train": data_norm[:n_train],
        "val": data_norm[n_train:n_train + n_val],
        "test": data_norm[n_train + n_val:],
        "stats": stats,
        "stock": stock,
    }


def _find_snapshot_l2_dataset_dir(data_dir: Path, dataset_id: str) -> Path | None:
    candidates = [
        data_dir / dataset_id,
        data_dir / "processed" / "snapshot_l2" / dataset_id,
        data_dir / "snapshot_l2" / dataset_id,
        Path("data") / "processed" / "snapshot_l2" / dataset_id,
    ]
    for candidate in candidates:
        if (candidate / "states.npy").exists() and (candidate / "actions.npy").exists():
            return candidate
    return None


def _split_by_ratio(states: np.ndarray, actions: np.ndarray,
                    train_ratio: float, val_ratio: float) -> dict:
    N = len(states)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    return {
        "train": states[:n_train], "train_actions": actions[:n_train],
        "val": states[n_train:n_train + n_val], "val_actions": actions[n_train:n_train + n_val],
        "test": states[n_train + n_val:], "test_actions": actions[n_train + n_val:],
    }


def _align_state_action_arrays(
    states: np.ndarray,
    actions: np.ndarray,
    split: np.ndarray | None,
    day_id: np.ndarray | None,
    dataset_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if len(states) == len(actions) + 1:
        states = states[:-1]
        if split is not None:
            split = split[:-1]
        if day_id is not None:
            day_id = day_id[:-1]
    elif len(states) != len(actions):
        raise ValueError(
            f"{dataset_dir}: states length must equal actions or actions+1; "
            f"got states={len(states)} actions={len(actions)}"
        )
    if split is not None and len(split) != len(actions):
        raise ValueError(
            f"{dataset_dir}: split length must align with actions; "
            f"got split={len(split)} actions={len(actions)}"
        )
    if day_id is not None and len(day_id) != len(actions):
        raise ValueError(
            f"{dataset_dir}: day_id length must align with actions; "
            f"got day_id={len(day_id)} actions={len(actions)}"
        )
    return states, actions, split, day_id


def _split_by_manifest(states: np.ndarray, actions: np.ndarray, split: np.ndarray,
                       dataset_dir: Path, day_id: np.ndarray | None = None) -> dict:
    states, actions, split, day_id = _align_state_action_arrays(
        states, actions, split, day_id, dataset_dir,
    )

    result = {}
    for split_value, split_name in [(0, "train"), (1, "val"), (2, "test")]:
        mask = split == split_value
        result[split_name] = states[mask]
        result[f"{split_name}_actions"] = actions[mask]
        if day_id is not None:
            result[f"{split_name}_day_id"] = day_id[mask]
    return result


def load_nasdaq_multiday(data_dir: str, stock: str = "TSLA", nday: str = "9day",
                         train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict:
    """Load multi-day normalized NASDAQ data with actions.

    Preferred layout:
        data/processed/snapshot_l2/{STOCK}_{Nday}/states.npy
        data/processed/snapshot_l2/{STOCK}_{Nday}/actions.npy
        data/processed/snapshot_l2/{STOCK}_{Nday}/split.npy

    The legacy layout ``{STOCK}_{Nday}_norm.npy`` is still accepted when
    present, but new experiments should use the processed snapshot schema.
    """
    data_dir = Path(data_dir)
    dataset_id = f"{stock}_{nday}"
    dataset_dir = _find_snapshot_l2_dataset_dir(data_dir, dataset_id)

    if dataset_dir is not None:
        states = np.load(dataset_dir / "states.npy")      # [N, 2, 20]
        actions = np.load(dataset_dir / "actions.npy")    # [N-1, A]
        split_path = dataset_dir / "split.npy"
        day_id_path = dataset_dir / "day_id.npy"
        raw_state_path = dataset_dir / "states_raw.npy"
        segment_id_path = dataset_dir / "segment_id.npy"
        normalization_path = dataset_dir / "normalization.json"
        day_id = np.load(day_id_path) if day_id_path.exists() else None
        raw_states = np.load(raw_state_path) if raw_state_path.exists() else None
        segment_id = np.load(segment_id_path) if segment_id_path.exists() else None
        normalization = None
        if normalization_path.exists():
            with open(normalization_path) as f:
                normalization = json.load(f)
        if split_path.exists():
            split = np.load(split_path)
            states, actions, split, day_id = _align_state_action_arrays(
                states, actions, split, day_id, dataset_dir,
            )
            aligned_len = len(states)
            if raw_states is not None:
                raw_states = raw_states[:aligned_len]
            if segment_id is not None:
                segment_id = segment_id[:aligned_len]

            result = {}
            for split_value, split_name in [(0, "train"), (1, "val"), (2, "test")]:
                mask = split == split_value
                result[split_name] = states[mask]
                result[f"{split_name}_actions"] = actions[mask]
                if day_id is not None:
                    result[f"{split_name}_day_id"] = day_id[mask]
                if raw_states is not None:
                    result[f"{split_name}_raw"] = raw_states[mask]
                if segment_id is not None:
                    result[f"{split_name}_segment_id"] = segment_id[mask]
        else:
            aligned_states = states[:-1] if len(states) == len(actions) + 1 else states
            result = _split_by_ratio(aligned_states, actions, train_ratio, val_ratio)
            if raw_states is not None:
                aligned_raw = raw_states[:len(aligned_states)]
                n_train = len(result["train"])
                n_val = len(result["val"])
                result["train_raw"] = aligned_raw[:n_train]
                result["val_raw"] = aligned_raw[n_train:n_train + n_val]
                result["test_raw"] = aligned_raw[n_train + n_val:]
        result["dataset_dir"] = str(dataset_dir)
        result["dataset_id"] = dataset_id
        result["normalization"] = normalization
        result["has_raw_states"] = raw_states is not None
        return result

    legacy_norm = data_dir / f"{stock}_{nday}_norm.npy"
    legacy_actions = data_dir / f"{stock}_{nday}_actions.npy"
    if legacy_norm.exists() and legacy_actions.exists():
        states = np.load(legacy_norm)        # [N, 2, 20]
        actions = np.load(legacy_actions)    # [N-1, A]
        aligned_states = states[:-1] if len(states) == len(actions) + 1 else states
        result = _split_by_ratio(aligned_states, actions, train_ratio, val_ratio)
        result["dataset_dir"] = str(data_dir)
        result["dataset_id"] = dataset_id
        return result

    raise FileNotFoundError(
        f"Could not find {dataset_id}. Looked for processed snapshot_l2 layout "
        f"under {data_dir} and legacy files {legacy_norm.name}/{legacy_actions.name}."
    )

# Keep backward compat
load_nasdaq_6day = lambda *a, **kw: load_nasdaq_multiday(*a, nday="6day", **kw)


class NASDAQMultiDayDataset(Dataset):
    """NASDAQ multi-day dataset with actions for generation."""

    def __init__(self, data, actions, hist_len=50, pred_len=10, stride=1,
                 day_id: np.ndarray | None = None,
                 segment_id: np.ndarray | None = None,
                 raw_data: np.ndarray | None = None,
                 normalization: dict | None = None,
                 price_reference: str = "dataset"):
        self.data = torch.from_numpy(data).float()
        self.actions = torch.from_numpy(actions).float()
        self.day_id = np.asarray(day_id) if day_id is not None else None
        self.segment_id = np.asarray(segment_id) if segment_id is not None else None
        self.raw_data = np.asarray(raw_data) if raw_data is not None else None
        self.normalization = normalization
        self.price_reference = price_reference
        if price_reference not in {"dataset", "history_last_mid"}:
            raise ValueError(f"Unknown price_reference: {price_reference}")
        if price_reference == "history_last_mid" and (self.raw_data is None or not normalization):
            raise ValueError("history_last_mid requires states_raw.npy and normalization.json")
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.stride = stride
        total = hist_len + pred_len
        raw_starts = np.arange(0, max(0, len(data) - total + 1), stride, dtype=np.int64)
        self.starts = self._filter_group_aware_starts(raw_starts, total)
        self.n_samples = len(self.starts)

    def _filter_group_aware_starts(self, starts: np.ndarray, total: int) -> np.ndarray:
        if starts.size == 0:
            return starts
        groups = []
        if self.day_id is not None:
            unique = np.unique(self.day_id)
            if not (unique.size == 1 and int(unique[0]) < 0):
                groups.append(self.day_id)
        if self.segment_id is not None:
            groups.append(self.segment_id)
        if not groups:
            return starts

        valid = []
        for start in starts:
            keep = True
            for group in groups:
                window_group = group[start:start + total]
                if window_group.size != total or np.any(window_group < 0):
                    keep = False
                    break
                if not np.all(window_group == window_group[0]):
                    keep = False
                    break
            if keep:
                valid.append(int(start))
        return np.asarray(valid, dtype=np.int64)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        mid = start + self.hist_len
        if self.price_reference == "history_last_mid":
            raw_window = self.raw_data[start:mid + self.pred_len]
            window = normalize_anchor_relative_sequence(
                raw_window,
                hist_len=self.hist_len,
                normalization=self.normalization,
            )
            history = window[:self.hist_len].permute(1, 2, 0)
            future = window[self.hist_len:].permute(1, 2, 0)
        else:
            history = self.data[start:mid].permute(1, 2, 0)
            future = self.data[mid:mid+self.pred_len].permute(1, 2, 0)
        action = self.actions[mid - 1]
        return history, future, action

# Backward compat alias
NASDAQ6DayDataset = NASDAQMultiDayDataset


def get_nasdaq_generation_loaders(
    data_dir: str = "data",
    stock: str = "TSLA",
    hist_len: int = 50,
    pred_len: int = 10,
    batch_size: int = 128,
    stride: int = 1,
    num_workers: int = 4,
    use_6day: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for NASDAQ LOB generation."""

    if use_6day:
        raw = load_nasdaq_6day(data_dir, stock)
        loaders = []
        for split in ["train", "val", "test"]:
            ds = NASDAQ6DayDataset(
                raw[split], raw[f"{split}_actions"],
                hist_len, pred_len, stride,
            )
            loaders.append(DataLoader(
                ds, batch_size=batch_size, shuffle=(split == "train"),
                num_workers=num_workers, pin_memory=True,
            ))
        return tuple(loaders)

    data = load_nasdaq_stock(data_dir, stock)

    loaders = []
    for split in ["train", "val", "test"]:
        ds = NASDAQGenerationDataset(data[split], hist_len, pred_len, stride)
        loader = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True,
        )
        loaders.append(loader)

    return tuple(loaders)


if __name__ == "__main__":
    from .normalize import verify_normalization

    for stock in ["TSLA", "AAPL"]:
        print(f"\n=== {stock} ===")
        data = load_nasdaq_stock("data", stock)
        print(f"Train: {data['train'].shape}")
        print(f"Val:   {data['val'].shape}")
        print(f"Test:  {data['test'].shape}")
        verify_normalization(data['train'], stock)

    print("\n=== DataLoader test ===")
    train_loader, val_loader, test_loader = get_nasdaq_generation_loaders(
        "data", stock="TSLA", hist_len=50, pred_len=10, batch_size=32, num_workers=0,
    )
    hist, fut = next(iter(train_loader))
    print(f"History: {hist.shape}")
    print(f"Future:  {fut.shape}")
