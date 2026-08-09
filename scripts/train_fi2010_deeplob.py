#!/usr/bin/env python3
"""Train a DeepLOB-style classifier on processed FI-2010 arrays.

This is a prediction benchmark runner, not a GenLOB generation experiment. It
keeps FI-2010 isolated from snapshot_l2 replay/projection claims.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.data.fi2010 import FI2010WindowDataset, load_fi2010_fold
from genlob.models.deeplob import DeepLOBStyleClassifier
from genlob.utils.device import device_summary, select_device


@dataclass
class ClassificationMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    support: list[int]
    confusion_matrix: list[list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLOB-style FI-2010 benchmark baseline")
    parser.add_argument("--processed-dir", default="data/processed/fi2010/noauction_zscore_cf7")
    parser.add_argument("--fold", type=int, default=7, choices=range(1, 10))
    parser.add_argument("--save-dir", default="checkpoints_fi2010/deeplob_style_smoke")
    parser.add_argument("--window-len", type=int, default=100)
    parser.add_argument("--label-index", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--use-all-features", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=None,
        help="Future label horizon used for purging; defaults to FI-2010 horizon for label-index.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--inception-channels", type=int, default=32)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument("--shuffle-train", action="store_true", default=True)
    parser.add_argument("--no-shuffle-train", dest="shuffle_train", action="store_false")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_train_val_subsets(
    dataset: FI2010WindowDataset,
    val_fraction: float,
    label_horizon: int,
) -> tuple[Subset, Subset]:
    """Chronological split with window-overlap and future-label purging."""
    if not 0 < val_fraction < 0.5:
        raise ValueError("--val-fraction must be in (0, 0.5)")
    n = len(dataset)
    val_n = max(1, int(round(n * val_fraction)))
    train_n = n - val_n
    if train_n <= 0:
        raise ValueError("not enough FI-2010 windows after validation split")
    val_idx = np.arange(train_n, n, dtype=np.int64)
    val_first_raw_start = int(dataset.starts[val_idx[0]])
    train_candidates = np.arange(0, train_n, dtype=np.int64)
    train_label_end = (
        dataset.starts[train_candidates]
        + dataset.window_len
        - 1
        + int(label_horizon)
    )
    train_idx = train_candidates[train_label_end < val_first_raw_start]
    if train_idx.size == 0:
        raise ValueError("purged FI-2010 split leaves no training windows")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def class_weights_from_subset(subset: Subset, num_classes: int = 3) -> torch.Tensor:
    labels = []
    ds = subset.dataset
    if not isinstance(ds, FI2010WindowDataset):
        raise TypeError("expected FI2010WindowDataset")
    for idx in subset.indices:
        start = int(ds.starts[int(idx)])
        labels.append(int(ds.labels[start + ds.window_len - 1]))
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    inv = counts.sum() / np.maximum(counts, 1.0)
    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_loaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], int]:
    arrays = load_fi2010_fold(
        args.processed_dir,
        fold=args.fold,
        use_all_features=args.use_all_features,
    )
    train_features, train_labels = arrays["train"]
    input_dim = train_features.shape[1]
    train_dataset = FI2010WindowDataset(
        train_features,
        train_labels,
        window_len=args.window_len,
        label_index=args.label_index,
        stride=args.stride,
    )
    fi_horizons = [10, 20, 30, 50, 100]
    label_horizon = args.label_horizon
    if label_horizon is None:
        label_horizon = fi_horizons[args.label_index]
    train_subset, val_subset = make_train_val_subsets(
        train_dataset,
        args.val_fraction,
        label_horizon=label_horizon,
    )

    pin = torch.cuda.is_available()
    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=args.shuffle_train,
            num_workers=args.num_workers,
            pin_memory=pin,
        ),
        "val": DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
        ),
    }

    features, labels = arrays["test"]
    ds = FI2010WindowDataset(
        features,
        labels,
        window_len=args.window_len,
        label_index=args.label_index,
        stride=args.stride,
    )
    loaders["test"] = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    args.label_horizon = label_horizon
    return loaders, input_dim


def update_confusion(confusion: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    for target, pred in zip(y_true.view(-1), y_pred.view(-1)):
        confusion[int(target), int(pred)] += 1


def metrics_from_confusion(confusion: torch.Tensor, loss: float) -> ClassificationMetrics:
    cm = confusion.cpu().numpy().astype(np.int64)
    support = cm.sum(axis=1)
    true_positive = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = support.astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(recall), where=(precision + recall) > 0)
    accuracy = float(true_positive.sum() / max(cm.sum(), 1))
    balanced_accuracy = float(recall.mean())
    return ClassificationMetrics(
        loss=float(loss),
        accuracy=accuracy,
        macro_f1=float(f1.mean()),
        balanced_accuracy=balanced_accuracy,
        support=support.tolist(),
        confusion_matrix=cm.tolist(),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    clip_grad: float = 1.0,
    max_batches: int | None = None,
) -> ClassificationMetrics:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_examples = 0
    confusion = torch.zeros(3, 3, dtype=torch.long)

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        batch_n = y.numel()
        total_loss += float(loss.item()) * batch_n
        total_examples += batch_n
        pred = torch.argmax(logits.detach(), dim=1)
        update_confusion(confusion, y.detach().cpu(), pred.cpu())

    avg_loss = total_loss / max(total_examples, 1)
    return metrics_from_confusion(confusion, avg_loss)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loaders, input_dim = build_loaders(args)
    model = DeepLOBStyleClassifier(
        input_dim=input_dim,
        conv_channels=args.conv_channels,
        inception_channels=args.inception_channels,
        lstm_hidden=args.lstm_hidden,
        dropout=args.dropout,
    ).to(device)

    class_weights = None
    if args.class_balance:
        class_weights = class_weights_from_subset(loaders["train"].dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = {
        "args": vars(args),
        "device": device_summary(args.device),
        "input_dim": input_dim,
        "model_params": model.get_num_params(),
        "train_windows": len(loaders["train"].dataset),
        "val_windows": len(loaders["val"].dataset),
        "test_windows": len(loaders["test"].dataset),
        "validation_protocol": {
            "type": "chronological_purged_holdout_from_train_cf7",
            "window_len": args.window_len,
            "label_horizon": args.label_horizon,
            "condition": "train_window_end_plus_label_horizon_before_first_validation_window_start",
            "official_anchored_fold": args.fold,
        },
    }
    write_json(save_dir / "config.json", config)
    print(f"Device: {device}")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Train windows: {config['train_windows']} | Val windows: {config['val_windows']}")

    best_val = -1.0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            clip_grad=args.clip_grad,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )
        row = {
            "epoch": epoch,
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
            "elapsed_seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} "
            f"val_macro_f1={val_metrics.macro_f1:.4f} "
            f"elapsed={row['elapsed_seconds']:.1f}s"
        )
        if val_metrics.macro_f1 > best_val:
            best_val = val_metrics.macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "input_dim": input_dim,
                    "val_metrics": asdict(val_metrics),
                    "config": config,
                },
                save_dir / "best.pt",
            )

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "input_dim": input_dim,
            "config": config,
        },
        save_dir / "final.pt",
    )
    write_json(save_dir / "history.json", {"epochs": history, "total_seconds": time.time() - start})

    best = torch.load(save_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    test_results = {}
    metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        optimizer=None,
        max_batches=args.max_test_batches,
    )
    test_results[f"test_cf{args.fold}"] = asdict(metrics)
    print(
        f"test_cf{args.fold}: loss={metrics.loss:.4f} acc={metrics.accuracy:.4f} "
        f"macro_f1={metrics.macro_f1:.4f} bal_acc={metrics.balanced_accuracy:.4f}"
    )

    result = {
        "best_epoch": int(best["epoch"]),
        "best_val_metrics": best["val_metrics"],
        "test": test_results,
        "scope": {
            "dataset": f"FI-2010 no-auction z-score anchored fold CF{args.fold}",
            "task": "DeepLOB-style mid-price movement classification",
            "not_supported": [
                "LOB replay",
                "queue/fill calibration",
                "LegalLOB projection ablation",
                "trading profitability claims",
            ],
        },
    }
    write_json(save_dir / "metrics.json", result)


if __name__ == "__main__":
    main()
