#!/usr/bin/env python3
"""Generate paper figures from frozen LegalLOB evaluation artifacts.

The script is offline: it does not train or sample a model. It reads saved
arrays and paper-facing CSV summaries, then writes vector PDF figures plus a
machine-readable provenance manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_financial_sensitivity import functional_vectors


COLORS = {
    "real": "#222222",
    "raw": "#C44E52",
    "raw_post": "#D55E00",
    "matched": "#7E57C2",
    "matched_post": "#CC8A00",
    "inloop": "#0072B2",
    "legal": "#2A8452",
    "ask": "#C44E52",
    "bid": "#3B6FB6",
    "grid": "#D8DADF",
}

DATASET_ORDER = (
    "AAPL_PUBLIC_ITCH_1S",
    "AMZN_PUBLIC_ITCH_1S",
    "GOOG_DATABENTO_RTH_V2",
    "GOOG_PUBLIC_ITCH_1S",
)

DATASET_LABELS = {
    "AAPL_PUBLIC_ITCH_1S": "AAPL-1Hz",
    "AMZN_PUBLIC_ITCH_1S": "AMZN-1Hz",
    "GOOG_DATABENTO_RTH_V2": "GOOG-Event",
    "GOOG_PUBLIC_ITCH_1S": "GOOG-1Hz",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=240)
    plt.close(fig)
    return [str(pdf.relative_to(REPO_ROOT)), str(png.relative_to(REPO_ROOT))]


def underline_text_artists(fig: plt.Figure, artists: list[mpl.text.Text]) -> None:
    """Draw stable, print-visible underlines beneath selected annotations."""
    if not artists:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    for artist in artists:
        bbox = artist.get_window_extent(renderer=renderer)
        (x0, y0), (x1, _) = inverse.transform(
            ((bbox.x0, bbox.y0 - 0.8), (bbox.x1, bbox.y0 - 0.8))
        )
        fig.add_artist(
            Line2D(
                (x0, x1),
                (y0, y0),
                transform=fig.transFigure,
                color=artist.get_color(),
                linewidth=0.65,
                solid_capstyle="butt",
                clip_on=False,
            )
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def aggregate_field(
    frame: pd.DataFrame,
    method_id: str,
    field: str,
) -> dict[str, tuple[float, float]]:
    selected = frame.loc[frame["method_id"] == method_id].copy()
    selected[field] = pd.to_numeric(selected[field], errors="coerce")
    selected = selected.dropna(subset=[field])
    within_checkpoint = (
        selected.groupby(["dataset_id", "train_seed"], as_index=False)[field].mean()
    )
    result: dict[str, tuple[float, float]] = {}
    for dataset_id, group in within_checkpoint.groupby("dataset_id"):
        values = group[field].to_numpy(dtype=float)
        result[str(dataset_id)] = (
            float(values.mean()),
            float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        )
    return result


def snapshot_flags(snapshot: np.ndarray, minimum: float = 0.1) -> dict[str, bool]:
    price = np.asarray(snapshot[0], dtype=float)
    depth = np.asarray(snapshot[1], dtype=float)
    ask = price[:10]
    bid = price[10:]
    tol = 2e-6
    return {
        "crossed": bool(ask[0] - bid[0] < minimum - tol),
        "ask_ladder": bool(np.any(np.diff(ask) < minimum - tol)),
        "bid_ladder": bool(np.any(-np.diff(bid) < minimum - tol)),
        "negative_depth": bool(np.any(depth < -tol)),
    }


def select_representative_snapshot(raw: np.ndarray, projected: np.ndarray) -> tuple[int, int]:
    prices = np.asarray(raw[:, 0], dtype=float)
    ask = prices[:, :10]
    bid = prices[:, 10:]
    crossed = ask[:, 0] < bid[:, 0]
    ask_inversion = np.any(np.diff(ask, axis=1) < 0.0, axis=1)
    bid_inversion = np.any(np.diff(bid, axis=1) > 0.0, axis=1)

    mask = crossed & (ask_inversion | bid_inversion)
    if not np.any(mask):
        mask = crossed | ask_inversion | bid_inversion
    candidates = np.argwhere(mask)
    if candidates.size == 0:
        raise ValueError("No visually explicit raw price violation found")

    repair = np.sqrt(np.mean((np.asarray(projected) - np.asarray(raw)) ** 2, axis=(1, 2)))
    candidate_repairs = repair[mask]
    target = float(np.median(candidate_repairs))
    chosen = candidates[int(np.argmin(np.abs(candidate_repairs - target)))]
    return int(chosen[0]), int(chosen[1])


def plot_ladder(
    axis: plt.Axes,
    snapshot: np.ndarray,
    title: str,
    *,
    valid: bool,
    raw_overlay: np.ndarray | None = None,
) -> None:
    prices = np.asarray(snapshot[0], dtype=float) * 10.0
    depth = np.square(np.maximum(np.asarray(snapshot[1], dtype=float), 0.0) * 15.0)
    ask_price = prices[:10]
    bid_price = prices[10:]
    ask_depth = depth[:10]
    bid_depth = depth[10:]
    bid_x = np.arange(-10, 0)
    ask_x = np.arange(1, 11)

    if raw_overlay is not None:
        raw_price = np.asarray(raw_overlay[0], dtype=float) * 10.0
        raw_bid = raw_price[10:][::-1]
        raw_ask = raw_price[:10]
        axis.plot(
            bid_x,
            raw_bid,
            color="#9AA0A6",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        axis.plot(
            ask_x,
            raw_ask,
            color="#9AA0A6",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        axis.scatter(
            bid_x,
            raw_bid,
            s=20,
            facecolors="white",
            edgecolors="#858B91",
            linewidths=0.8,
            zorder=2,
        )
        axis.scatter(
            ask_x,
            raw_ask,
            s=20,
            facecolors="white",
            edgecolors="#858B91",
            linewidths=0.8,
            zorder=2,
        )

    max_depth = max(float(depth.max()), 1.0)
    bid_size = 15.0 + 35.0 * np.sqrt(bid_depth[::-1] / max_depth)
    ask_size = 15.0 + 35.0 * np.sqrt(ask_depth / max_depth)
    axis.plot(bid_x, bid_price[::-1], color=COLORS["bid"], zorder=2)
    axis.plot(ask_x, ask_price, color=COLORS["ask"], zorder=2)
    axis.scatter(bid_x, bid_price[::-1], s=bid_size, color=COLORS["bid"], zorder=3)
    axis.scatter(ask_x, ask_price, s=ask_size, color=COLORS["ask"], zorder=3)
    if raw_overlay is not None:
        displacement = np.concatenate(
            [np.abs(bid_price[::-1] - raw_bid), np.abs(ask_price - raw_ask)]
        )
        arrow_data = list(
            zip(
                np.concatenate([bid_x, ask_x]),
                np.concatenate([raw_bid, raw_ask]),
                np.concatenate([bid_price[::-1], ask_price]),
                displacement,
            )
        )
        for x_value, start, end, _ in sorted(
            arrow_data, key=lambda item: item[3], reverse=True
        )[:6]:
            if abs(end - start) < 0.2:
                continue
            axis.annotate(
                "",
                xy=(x_value, end),
                xytext=(x_value, start),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#777D83",
                    "lw": 0.65,
                    "alpha": 0.85,
                    "shrinkA": 2.5,
                    "shrinkB": 2.5,
                },
                zorder=2,
            )
    axis.axvline(0, color="#8A8A8A", linewidth=0.7)
    axis.axhline(0, color="#B7BAC0", linewidth=0.55, linestyle=":")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.65)
    axis.set_title(title, pad=3.0, fontweight="semibold")
    axis.set_xticks([-10, -5, -1, 1, 5, 10])
    axis.set_xticklabels(["B10", "B5", "B1", "A1", "A5", "A10"])
    axis.tick_params(axis="x", labelsize=8.0, labelrotation=30)
    for index, label in enumerate(axis.get_xticklabels()):
        if index == 2:
            label.set_rotation(90)
            label.set_horizontalalignment("center")
            label.set_verticalalignment("top")
        elif index == 3:
            label.set_rotation(90)
            label.set_horizontalalignment("center")
            label.set_verticalalignment("top")
        else:
            label.set_horizontalalignment("right")
    axis.set_xlabel("Book level")
    axis.set_ylabel("Price offset (ticks)")
    axis.tick_params(length=2.5)

    badge = "VALID" if valid else "INFEASIBLE"
    badge_color = COLORS["legal"] if valid else COLORS["raw"]
    axis.text(
        0.98,
        0.96,
        badge,
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="white",
        fontsize=8.0,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.23", "facecolor": badge_color, "edgecolor": "none"},
    )

    flags = snapshot_flags(snapshot)
    spread = float(ask_price[0] - bid_price[0])
    if flags["crossed"]:
        axis.annotate(
            f"crossed {spread:.1f} ticks",
            xy=(0, 0.5 * (ask_price[0] + bid_price[0])),
            xytext=(0.04, 0.11),
            textcoords="axes fraction",
            color=COLORS["raw"],
            fontsize=8.0,
            arrowprops={"arrowstyle": "->", "color": COLORS["raw"], "lw": 0.8},
        )
    elif valid:
        axis.annotate(
            "",
            xy=(0, ask_price[0]),
            xytext=(0, bid_price[0]),
            arrowprops={
                "arrowstyle": "<->",
                "color": COLORS["legal"],
                "lw": 0.75,
            },
        )
        axis.text(
            0.20,
            0.08,
            f"spread {spread:.1f} ticks",
            transform=axis.transAxes,
            color=COLORS["legal"],
            fontsize=8.0,
        )

    raw_price = np.asarray(snapshot[0], dtype=float)
    bad_ask = np.flatnonzero(np.diff(raw_price[:10]) < 0.1 - 2e-6)
    bad_bid = np.flatnonzero(-np.diff(raw_price[10:]) < 0.1 - 2e-6)
    if (bad_ask.size or bad_bid.size) and not valid:
        axis.text(
            0.04,
            0.02,
            "ladder gap violation",
            transform=axis.transAxes,
            color=COLORS["raw"],
            fontsize=8.0,
        )


def add_invalidity_panel(axis: plt.Axes, frame: pd.DataFrame) -> None:
    methods = (
        ("raw_no_projection", "Raw", COLORS["raw"]),
        ("matched_no_projection", "Matched", COLORS["matched"]),
        ("legal_no_projection", "LegalLOB", COLORS["legal"]),
    )
    x = np.arange(len(DATASET_ORDER), dtype=float)
    width = 0.18
    center_offsets = (-0.22, 0.0, 0.22)
    for offset, (method_id, label, color) in enumerate(methods):
        values = aggregate_field(frame, method_id, "invalid_raw")
        mean = np.array([values[item][0] * 100.0 for item in DATASET_ORDER])
        std = np.array([values[item][1] * 100.0 for item in DATASET_ORDER])
        axis.bar(
            x + center_offsets[offset],
            mean,
            width=width,
            yerr=std,
            capsize=2,
            color=color,
            label=label,
            linewidth=0,
        )
        if method_id == "legal_no_projection":
            legal_x = x + center_offsets[offset]
            axis.scatter(
                legal_x,
                np.zeros_like(legal_x),
                marker="D",
                s=24,
                color=COLORS["legal"],
                edgecolor="white",
                linewidth=0.5,
                clip_on=False,
                zorder=5,
            )
            for x_value in legal_x:
                axis.annotate(
                    "0",
                    (x_value, 0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.0,
                    color=COLORS["legal"],
                    fontweight="semibold",
                )
    axis.set_title("(e) Pre-projection invalid snapshots", loc="left", fontweight="semibold")
    axis.set_ylabel("Invalid snapshots (%)")
    axis.set_xticks(x)
    axis.set_xticklabels([DATASET_LABELS[item] for item in DATASET_ORDER])
    axis.tick_params(axis="x", labelrotation=15)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.set_ylim(0, 108)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    axis.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.60, 1.005),
        fontsize=7.0,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.2,
    )


def add_repair_panel(axis: plt.Axes, frame: pd.DataFrame) -> None:
    methods = (
        ("raw_posthoc_pava", "Raw + PAVA", COLORS["raw_post"], "o"),
        ("matched_posthoc_pava", "Matched + PAVA", COLORS["matched_post"], "s"),
    )
    # Data-coordinate positions remain stable when the figure is scaled into
    # the two-column paper layout and keep labels away from threshold text.
    label_positions = {
        "AAPL_PUBLIC_ITCH_1S": (84.5, 0.018),
        "AMZN_PUBLIC_ITCH_1S": (43.0, 0.43),
        "GOOG_DATABENTO_RTH_V2": (86.0, 0.27),
        "GOOG_PUBLIC_ITCH_1S": (66.5, 0.68),
    }
    points: dict[str, dict[str, tuple[float, float]]] = {
        dataset_id: {} for dataset_id in DATASET_ORDER
    }
    for method_id, label, color, marker in methods:
        x_values = aggregate_field(frame, method_id, "repair_snapshot_modified_rate")
        y_values = aggregate_field(frame, method_id, "repair_ticks_p95")
        for dataset_id in DATASET_ORDER:
            x_mean, x_std = x_values[dataset_id]
            y_mean, y_std = y_values[dataset_id]
            axis.errorbar(
                x_mean * 100.0,
                y_mean,
                xerr=x_std * 100.0,
                yerr=y_std,
                marker=marker,
                markersize=5.0,
                color=color,
                capsize=2,
                linestyle="none",
                label=label if dataset_id == DATASET_ORDER[0] else None,
            )
            points[dataset_id][method_id] = (x_mean * 100.0, y_mean)
    for dataset_id, method_points in points.items():
        raw_point = method_points["raw_posthoc_pava"]
        matched_point = method_points["matched_posthoc_pava"]
        axis.plot(
            [raw_point[0], matched_point[0]],
            [raw_point[1], matched_point[1]],
            color="#A9ADB2",
            linewidth=0.7,
            zorder=0,
        )
        midpoint = (
            0.5 * (raw_point[0] + matched_point[0]),
            float(np.sqrt(raw_point[1] * matched_point[1])),
        )
        axis.annotate(
            DATASET_LABELS[dataset_id],
            midpoint,
            xytext=label_positions[dataset_id],
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8.0,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.3},
        )
    axis.axhline(0.1, color="#9AA0A6", linestyle=":", linewidth=0.8)
    axis.axhline(1.0, color="#6F7378", linestyle="--", linewidth=0.8)
    axis.text(42, 0.11, "0.1 tick", fontsize=8.0, color="#73777B")
    axis.text(42, 1.08, "1 tick", fontsize=8.0, color="#55595D")
    axis.set_yscale("log")
    axis.set_xlim(41, 108)
    axis.set_ylim(0.003, 3.0)
    axis.set_xlabel("Snapshots modified by exact PAVA (%)")
    axis.set_ylabel("p95 modified-level repair (ticks)")
    axis.set_title("(f) Repair incidence and severity", loc="left", fontweight="semibold")
    axis.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.6, which="both")
    axis.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower left")


def generate_mechanism_evidence(
    output_dir: Path,
    feasibility_csv: Path,
    result_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    raw_dir = result_root / "raw_posthoc_pava"
    real = np.load(raw_dir / "real_future.npy", mmap_mode="r")
    raw = np.load(raw_dir / "generated_raw.npy", mmap_mode="r")
    projected = np.load(raw_dir / "generated_projected.npy", mmap_mode="r")
    legal = np.load(result_root / "legal_no_projection" / "generated_raw.npy", mmap_mode="r")
    history_index, future_index = select_representative_snapshot(raw, projected)

    real_snapshot = np.asarray(real[history_index, :, :, future_index])
    raw_snapshot = np.asarray(raw[history_index, :, :, future_index])
    projected_snapshot = np.asarray(projected[history_index, :, :, future_index])
    legal_snapshot = np.asarray(legal[history_index, :, :, future_index])
    if any(snapshot_flags(projected_snapshot).values()):
        raise AssertionError("Selected projected snapshot is not feasible")
    if any(snapshot_flags(legal_snapshot).values()):
        raise AssertionError("Selected LegalLOB snapshot is not feasible")

    frame = pd.read_csv(feasibility_csv)
    fig = plt.figure(figsize=(7.15, 3.95), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=(1.1, 1.0),
        left=0.065,
        right=0.995,
        bottom=0.105,
        top=0.94,
        wspace=0.42,
        hspace=0.58,
    )
    snapshots = (
        (real_snapshot, "(a) Real future", True, None),
        (raw_snapshot, "(b) Raw output", False, None),
        (projected_snapshot, "(c) Raw + exact PAVA", True, raw_snapshot),
        (legal_snapshot, "(d) LegalLOB output", True, None),
    )
    ladder_axes = []
    for column, (snapshot, title, valid, overlay) in enumerate(snapshots):
        axis = fig.add_subplot(grid[0, column])
        plot_ladder(axis, snapshot, title, valid=valid, raw_overlay=overlay)
        ladder_axes.append(axis)
        if column:
            axis.set_ylabel("")
    all_prices = np.concatenate(
        [np.asarray(item[0][0], dtype=float).reshape(-1) * 10.0 for item in snapshots]
    )
    margin = max(8.0, 0.05 * float(np.ptp(all_prices)))
    limits = (float(all_prices.min() - margin), float(all_prices.max() + margin))
    for axis in ladder_axes:
        axis.set_ylim(*limits)
    add_invalidity_panel(fig.add_subplot(grid[1, :2]), frame)
    add_repair_panel(fig.add_subplot(grid[1, 2:]), frame)
    files = save_figure(fig, output_dir, "figure_mechanism_evidence")

    provenance = {
        "figure": "figure_mechanism_evidence",
        "selection_rule": (
            "Among train-seed-0/eval-seed-0 Raw snapshots with an actual crossed book "
            "and an actual ask or bid inversion, select the case nearest median paired "
            "normalized repair L2."
        ),
        "legal_draw_rule": (
            "Use the same history/future array index from the fixed train-seed-0/"
            "eval-seed-0 LegalLOB output; no LegalLOB-to-real distance or visual "
            "criterion is used."
        ),
        "legal_selection_uses_fidelity": False,
        "result_root": str(result_root.relative_to(REPO_ROOT)),
        "history_index": history_index,
        "future_index": future_index,
        "snapshot_flags": {
            "real": snapshot_flags(real_snapshot),
            "raw": snapshot_flags(raw_snapshot),
            "projected": snapshot_flags(projected_snapshot),
            "legal": snapshot_flags(legal_snapshot),
        },
        "raw_spread_ticks": float((raw_snapshot[0, 0] - raw_snapshot[0, 10]) * 10.0),
        "projected_spread_ticks": float(
            (projected_snapshot[0, 0] - projected_snapshot[0, 10]) * 10.0
        ),
        "aggregation": (
            "evaluation seeds averaged within each checkpoint; mean and sample SD "
            "computed across three training seeds"
        ),
        "feasibility_csv": str(feasibility_csv.relative_to(REPO_ROOT)),
        "outputs": files,
    }
    return files, provenance


def generate_structural_distributions(
    output_dir: Path,
    result_root: Path,
    normalization_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    normalization = load_json(normalization_path)
    real = np.load(result_root / "raw_posthoc_pava" / "real_future.npy", mmap_mode="r")
    real_vectors, target_size = functional_vectors(real, normalization)
    sources = (
        ("Real", real_vectors, COLORS["real"]),
        (
            "Raw + PAVA",
            functional_vectors(
                np.load(
                    result_root / "raw_posthoc_pava" / "generated_projected.npy",
                    mmap_mode="r",
                ),
                normalization,
                target_size,
            )[0],
            COLORS["raw_post"],
        ),
        (
            "Matched + PAVA",
            functional_vectors(
                np.load(
                    result_root / "matched_posthoc_pava" / "generated_projected.npy",
                    mmap_mode="r",
                ),
                normalization,
                target_size,
            )[0],
            COLORS["matched"],
        ),
        (
            "LegalLOB",
            functional_vectors(
                np.load(
                    result_root / "legal_no_projection" / "generated_raw.npy",
                    mmap_mode="r",
                ),
                normalization,
                target_size,
            )[0],
            COLORS["legal"],
        ),
    )
    panels = (
        ("spread", "Spread", "Ticks"),
        ("imbalance_l10", "Ten-level depth imbalance", "Imbalance"),
        ("liquidity_5tick", "Displayed liquidity within 5 ticks", "Liquidity / Q"),
        ("market_order_slippage", "Fixed-size market-order slippage", "Ticks"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.2), constrained_layout=False)
    for panel_index, (axis, (field, title, xlabel)) in enumerate(zip(axes.flat, panels)):
        pooled = []
        for _, vectors, _ in sources:
            values = np.asarray(vectors[field], dtype=float)
            pooled.append(values[np.isfinite(values)])
        combined = np.concatenate(pooled)
        lower = float(np.min(combined))
        upper = float(np.max(combined))
        bins = np.linspace(lower, upper, 70)
        for label, vectors, color in sources:
            values = np.asarray(vectors[field], dtype=float)
            values = values[np.isfinite(values)]
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=color,
                label=label,
            )
        axis.set_title(f"({chr(97 + panel_index)}) {title}", loc="left", fontweight="semibold")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Density")
        axis.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.105, top=0.86, wspace=0.25, hspace=0.43)
    files = save_figure(fig, output_dir, "figure_structural_distributions")
    provenance = {
        "figure": "figure_structural_distributions",
        "scope": "representative fixed train-seed-0/eval-seed-0 GOOG-Event outputs",
        "result_root": str(result_root.relative_to(REPO_ROOT)),
        "normalization": str(normalization_path.relative_to(REPO_ROOT)),
        "target_size_clipped_shares": target_size,
        "outputs": files,
    }
    return files, provenance


def _ecdf_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a compressed empirical CDF with an explicit initial jump."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot draw an ECDF from an empty finite sample")
    unique, counts = np.unique(finite, return_counts=True)
    cumulative = np.cumsum(counts, dtype=np.float64) / float(finite.size)
    return np.concatenate(([unique[0]], unique)), np.concatenate(([0.0], cumulative))


def generate_structural_distributions_ecdf(
    output_dir: Path,
    result_root: Path,
    normalization_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Draw bin-free ECDFs for the representative GOOG-Event functionals."""
    normalization = load_json(normalization_path)
    real = np.load(result_root / "raw_posthoc_pava" / "real_future.npy", mmap_mode="r")
    real_vectors, target_size = functional_vectors(real, normalization)
    sources = (
        ("Real", real_vectors, "#222222", "-", 1.45, 2, "o"),
        (
            "Raw + PAVA",
            functional_vectors(
                np.load(
                    result_root / "raw_posthoc_pava" / "generated_projected.npy",
                    mmap_mode="r",
                ),
                normalization,
                target_size,
            )[0],
            "#D55E00",
            (0, (5, 2)),
            1.7,
            3,
            "s",
        ),
        (
            "Matched + PAVA",
            functional_vectors(
                np.load(
                    result_root / "matched_posthoc_pava" / "generated_projected.npy",
                    mmap_mode="r",
                ),
                normalization,
                target_size,
            )[0],
            "#0072B2",
            (0, (3, 1.4, 1, 1.4)),
            1.7,
            4,
            "^",
        ),
        (
            "LegalLOB",
            functional_vectors(
                np.load(
                    result_root / "legal_no_projection" / "generated_raw.npy",
                    mmap_mode="r",
                ),
                normalization,
                target_size,
            )[0],
            "#009E73",
            (0, (1.2, 1.2)),
            1.85,
            5,
            "D",
        ),
    )
    panels = (
        ("spread", "Spread", "Ticks"),
        ("imbalance_l10", "Ten-level depth imbalance", "Imbalance"),
        ("liquidity_5tick", "Displayed liquidity within 5 ticks", "Liquidity / $Q$"),
        ("market_order_slippage", "Fixed-size market-order slippage", "Ticks"),
    )
    zero_tolerance = 1e-8
    cleaned: dict[str, dict[str, np.ndarray]] = {}
    sample_counts: dict[str, dict[str, int]] = {}
    zero_mass: dict[str, float] = {}
    x_limits: dict[str, list[float]] = {}

    for field, _, _ in panels:
        cleaned[field] = {}
        sample_counts[field] = {}
        for label, vectors, *_ in sources:
            values = np.asarray(vectors[field], dtype=np.float64)
            values = values[np.isfinite(values)]
            if field == "market_order_slippage":
                values = values.copy()
                values[np.abs(values) <= zero_tolerance] = 0.0
                zero_mass[label] = float(np.mean(values == 0.0))
            cleaned[field][label] = values
            sample_counts[field][label] = int(values.size)

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.2), constrained_layout=False)
    legend_handles = {
        label: Line2D(
            [],
            [],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=4.1,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.85,
            label=label,
        )
        for label, _, color, linestyle, linewidth, _, marker in sources
    }
    marker_probabilities: dict[str, dict[str, list[float]]] = {}
    for panel_index, (axis, (field, title, xlabel)) in enumerate(zip(axes.flat, panels)):
        pooled = np.concatenate(list(cleaned[field].values()))
        data_min = float(np.min(pooled))
        data_max = float(np.max(pooled))
        span = max(data_max - data_min, 1e-9)
        if field in {"spread", "liquidity_5tick"}:
            lower = min(0.0, data_min - 0.015 * span)
            upper = data_max + 0.025 * span
        elif field == "market_order_slippage":
            lower = -max(0.5, 0.018 * data_max)
            upper = data_max + 0.025 * max(data_max, 1.0)
        else:
            lower = data_min - 0.035 * span
            upper = data_max + 0.035 * span
        x_limits[field] = [lower, upper]

        # Dashed method curves sit above the solid reference where ECDFs overlap.
        draw_order = sources
        if field == "market_order_slippage":
            probabilities_by_label = {
                "Real": np.asarray((0.56, 0.66, 0.76, 0.86, 0.96)),
                "Raw + PAVA": np.asarray((0.59, 0.69, 0.79, 0.89, 0.97)),
                "Matched + PAVA": np.asarray((0.62, 0.72, 0.82, 0.92, 0.98)),
                "LegalLOB": np.asarray((0.65, 0.75, 0.85, 0.94, 0.99)),
            }
        else:
            probabilities_by_label = {
                "Real": np.asarray((0.15, 0.35, 0.55, 0.75, 0.92)),
                "Raw + PAVA": np.asarray((0.20, 0.40, 0.60, 0.80, 0.94)),
                "Matched + PAVA": np.asarray((0.25, 0.45, 0.65, 0.85, 0.96)),
                "LegalLOB": np.asarray((0.30, 0.50, 0.70, 0.90, 0.98)),
            }
        marker_probabilities[field] = {
            label: values.tolist()
            for label, values in probabilities_by_label.items()
        }
        for label, _, color, linestyle, linewidth, zorder, marker in draw_order:
            x, y = _ecdf_points(cleaned[field][label])
            axis.step(
                x,
                y,
                where="post",
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
                zorder=zorder,
            )
            probabilities = probabilities_by_label[label]
            quantiles = np.quantile(cleaned[field][label], probabilities, method="higher")
            axis.scatter(
                quantiles,
                probabilities,
                marker=marker,
                s=16,
                facecolors="white",
                edgecolors=color,
                linewidths=0.9,
                zorder=zorder + 0.25,
                clip_on=True,
            )

        axis.set_title(
            f"({chr(97 + panel_index)}) {title}",
            loc="left",
            fontweight="semibold",
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Cumulative probability")
        axis.set_xlim(lower, upper)
        axis.set_ylim(0.0, 1.01)
        axis.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
        axis.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.65)
        axis.set_axisbelow(True)

        if field == "market_order_slippage":
            annotation = (
                "$P(\\mathrm{slip}=0)$\n"
                f"Real {100.0 * zero_mass['Real']:.1f}%  ·  "
                f"Raw {100.0 * zero_mass['Raw + PAVA']:.1f}%\n"
                f"Matched {100.0 * zero_mass['Matched + PAVA']:.1f}%  ·  "
                f"Legal {100.0 * zero_mass['LegalLOB']:.1f}%"
            )
            axis.text(
                0.975,
                0.055,
                annotation,
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.4,
                color="#4A4D52",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#C8CBD1",
                    "linewidth": 0.55,
                    "alpha": 0.92,
                },
            )

    legend_order = ("Real", "Raw + PAVA", "Matched + PAVA", "LegalLOB")
    fig.legend(
        [legend_handles[label] for label in legend_order],
        legend_order,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.7,
    )
    fig.subplots_adjust(
        left=0.082,
        right=0.995,
        bottom=0.105,
        top=0.86,
        wspace=0.27,
        hspace=0.43,
    )
    files = save_figure(fig, output_dir, "figure_structural_distributions_ecdf")
    provenance = {
        "figure": "figure_structural_distributions_ecdf",
        "plot_type": "empirical cumulative distribution function",
        "scope": "representative fixed train-seed-0/eval-seed-0 GOOG-Event outputs",
        "result_root": str(result_root.relative_to(REPO_ROOT)),
        "normalization": str(normalization_path.relative_to(REPO_ROOT)),
        "target_size_clipped_shares": target_size,
        "zero_tolerance_ticks": zero_tolerance,
        "slippage_zero_mass": zero_mass,
        "sample_counts": sample_counts,
        "curve_encoding": {
            "Real": "black solid line with circle markers",
            "Raw + PAVA": "vermillion dashed line with square markers",
            "Matched + PAVA": "blue dash-dot line with triangle markers",
            "LegalLOB": "bluish-green dotted line with diamond markers",
        },
        "marker_cumulative_probabilities": marker_probabilities,
        "x_limits": x_limits,
        "outputs": files,
    }
    return files, provenance


def rollout_aggregate(
    frame: pd.DataFrame,
    method_id: str,
    field: str,
) -> pd.DataFrame:
    selected = frame.loc[frame["method_id"] == method_id].copy()
    selected[field] = pd.to_numeric(selected[field], errors="coerce")
    selected = selected.dropna(subset=[field])
    checkpoint = (
        selected.groupby(["train_seed", "horizon"], as_index=False)[field].mean()
    )
    rows = []
    for horizon, group in checkpoint.groupby("horizon"):
        values = group[field].to_numpy(dtype=float)
        rows.append(
            {
                "horizon": int(horizon),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("horizon")


def generate_rollout_burden(output_dir: Path, rollout_csv: Path) -> tuple[list[str], dict[str, Any]]:
    frame = pd.read_csv(rollout_csv)
    methods = (
        ("raw_posthoc_pava", "Raw+Post", COLORS["raw_post"], 10),
        ("matched_posthoc_pava", "Matched+Post", COLORS["matched_post"], 10),
        ("raw_inloop_pava", "Raw+In-loop", COLORS["inloop"], 200),
        ("legal_lob", "LegalLOB", COLORS["legal"], 0),
    )
    h100 = frame.loc[frame["horizon"] == 100].copy()

    calls = []
    per_call_mean = []
    per_call_std = []
    cumulative_mean = []
    cumulative_std = []
    for method_id, _, _, call_count in methods:
        calls.append(float(call_count))
        if call_count == 0:
            per_call_mean.append(0.0)
            per_call_std.append(0.0)
            cumulative_mean.append(0.0)
            cumulative_std.append(0.0)
            continue
        selected = h100.loc[h100["method_id"] == method_id].copy()
        selected["path_cumulative_projection_l2"] = pd.to_numeric(
            selected["path_cumulative_projection_l2"], errors="coerce"
        )
        checkpoint = selected.groupby("train_seed")["path_cumulative_projection_l2"].mean()
        values = checkpoint.to_numpy(dtype=float)
        cumulative_mean.append(float(values.mean()))
        cumulative_std.append(float(values.std(ddof=1)))
        per_call = values / float(call_count)
        per_call_mean.append(float(per_call.mean()))
        per_call_std.append(float(per_call.std(ddof=1)))

    labels = [item[1] for item in methods]
    colors = [item[2] for item in methods]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), constrained_layout=False)
    panels = (
        (calls, np.zeros(len(calls)), "(a) Calls per path $\\downarrow$", "Projection calls", None),
        (
            per_call_mean,
            per_call_std,
            "(b) Correction per call $\\downarrow$",
            "Normalized projection $L_2$",
            0.004,
        ),
        (
            cumulative_mean,
            cumulative_std,
            "(c) Cumulative repair burden $\\downarrow$",
            "Normalized projection $L_2$",
            0.03,
        ),
    )
    # Match the manuscript's table convention: best is bold and second-best is
    # underlined.  Tied second-best methods share the underline.  LegalLOB's
    # correction per call is undefined because it invokes no projection and is
    # therefore excluded from the panel-(b) ranking.
    annotation_styles = (
        {0: "second", 1: "second", 3: "best"},
        {0: "second", 1: "best", 3: "na"},
        {1: "second", 3: "best"},
    )
    value_formats = ("{:.0f}", "{:.4g}", "{:.4g}")
    second_best_artists: list[mpl.text.Text] = []
    for panel_index, (values, errors, title, ylabel, linthresh) in enumerate(panels):
        axis = axes[panel_index]
        bars = axis.bar(x, values, yerr=errors, color=colors, capsize=2, width=0.68)
        if linthresh is not None:
            axis.set_yscale("symlog", linthresh=linthresh)
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.65, which="both")
        upper_value = max(
            float(value) + float(error)
            for value, error in zip(values, errors)
        )
        if linthresh is None:
            axis.set_ylim(0, max(upper_value * 1.28, upper_value + 25.0))
        else:
            axis.set_ylim(0, max(upper_value * 4.0, linthresh * 20.0))
        for method_index, (bar, value) in enumerate(zip(bars, values)):
            style = annotation_styles[panel_index].get(method_index, "plain")
            display_value = (
                "N/A"
                if style == "na"
                else value_formats[panel_index].format(value)
            )
            text_color = (
                "#55595D"
                if style == "na"
                else COLORS["legal"] if method_index == 3 else "#111111"
            )
            artist = axis.annotate(
                display_value,
                (bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=text_color,
                fontsize=7.0,
                fontweight="bold" if style == "best" else "normal",
            )
            if style == "second":
                second_best_artists.append(artist)
    fig.subplots_adjust(left=0.070, right=0.995, bottom=0.28, top=0.86, wspace=0.35)
    underline_text_artists(fig, second_best_artists)
    files = save_figure(fig, output_dir, "figure_rollout_projection_burden")
    provenance = {
        "figure": "figure_rollout_projection_burden",
        "rollout_csv": str(rollout_csv.relative_to(REPO_ROOT)),
        "horizon": 100,
        "projection_calls_per_path": dict(zip(labels, calls)),
        "projection_call_definition": (
            "state-changing method operators only; offline diagnostic PAVA calls "
            "used to measure distance are excluded"
        ),
        "correction_per_call_mean": dict(zip(labels, per_call_mean)),
        "cumulative_correction_mean": dict(zip(labels, cumulative_mean)),
        "metric_direction": {
            "calls_per_path": "lower_is_better",
            "correction_per_call": "lower_is_better_among_applicable_methods",
            "cumulative_repair_burden": "lower_is_better",
        },
        "ranking_style": "best bold; second-best underlined; ties share rank",
        "legal_lob_correction_per_call": "not_applicable_no_projection_calls",
        "aggregation": (
            "evaluation seeds averaged within each checkpoint; mean and sample SD "
            "computed across three training seeds"
        ),
        "outputs": files,
    }
    return files, provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("paper_figures"))
    parser.add_argument(
        "--feasibility-csv",
        type=Path,
        default=Path("paper_data/cross_asset_feasibility_per_run.csv"),
    )
    parser.add_argument(
        "--rollout-csv",
        type=Path,
        default=Path("paper_data/all_method_rollout_goog_event_640_3eval_per_run.csv"),
    )
    parser.add_argument(
        "--representative-result-root",
        type=Path,
        default=Path("results/goog_databento_rth_v2_full/train_seed0"),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path(
            "data/processed/snapshot_l2/"
            "GOOG_databento_20190130_mbp10_rth_v2/normalization.json"
        ),
    )
    args = parser.parse_args()
    configure_style()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    feasibility_csv = (REPO_ROOT / args.feasibility_csv).resolve()
    rollout_csv = (REPO_ROOT / args.rollout_csv).resolve()
    result_root = (REPO_ROOT / args.representative_result_root).resolve()
    normalization = (REPO_ROOT / args.normalization).resolve()

    manifest: dict[str, Any] = {
        "scope": "offline visualization from frozen arrays and summaries; no training or sampling",
        "figures": [],
    }
    _, evidence = generate_mechanism_evidence(output_dir, feasibility_csv, result_root)
    manifest["figures"].append(evidence)
    _, evidence = generate_structural_distributions(output_dir, result_root, normalization)
    manifest["figures"].append(evidence)
    _, evidence = generate_structural_distributions_ecdf(
        output_dir, result_root, normalization
    )
    manifest["figures"].append(evidence)
    _, evidence = generate_rollout_burden(output_dir, rollout_csv)
    manifest["figures"].append(evidence)

    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest['figures'])} figures and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
