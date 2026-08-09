#!/usr/bin/env python3
"""Summarize fixed-checkpoint sampling ensembles on common test windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _seed_dirs(root: Path, method: str, seeds: list[int]) -> list[tuple[int, Path]]:
    found = []
    for seed in seeds:
        path = root / method if seed == 0 else root / "seeds" / f"seed{seed}" / method
        if (path / "generated_raw.npy").exists() and (path / "metrics.json").exists():
            found.append((seed, path))
    return found


def _evaluated_output(path: Path) -> np.ndarray:
    metrics = json.loads((path / "metrics.json").read_text())
    projection_mode = metrics.get("projection_mode", "none")
    name = "generated_projected.npy" if projection_mode in {"exact", "fast", "auto"} else "generated_raw.npy"
    return np.load(path / name, mmap_mode="r")


def _ensemble_metrics(samples: np.ndarray, real: np.ndarray, persistence: np.ndarray) -> tuple[dict, dict]:
    # samples [S,N,2,20,T]; all scores normalize vector L2 by sqrt(D).
    sample64 = samples.astype(np.float64, copy=False)
    real64 = real.astype(np.float64, copy=False)
    persistence64 = persistence.astype(np.float64, copy=False)
    ensemble_mean = sample64.mean(axis=0)
    dims = int(np.prod(real64.shape[1:]))
    term1 = np.sqrt(np.mean((sample64 - real64[None]) ** 2, axis=(2, 3, 4))).mean(axis=0)
    pair_values = []
    pair_abs_values = []
    for left in range(len(sample64)):
        for right in range(len(sample64)):
            diff = sample64[left] - sample64[right]
            pair_values.append(np.sqrt(np.mean(diff ** 2, axis=(1, 2, 3))))
            pair_abs_values.append(np.mean(np.abs(diff), axis=(1, 2, 3)))
    pair_rmse = np.mean(np.stack(pair_values), axis=0)
    pair_abs = np.mean(np.stack(pair_abs_values), axis=0)
    energy = term1 - 0.5 * pair_rmse
    first_abs = np.mean(
        np.abs(sample64 - real64[None]), axis=(0, 2, 3, 4)
    )
    crps = first_abs - 0.5 * pair_abs
    lower = sample64.min(axis=0)
    upper = sample64.max(axis=0)
    coverage = ((real64 >= lower) & (real64 <= upper)).mean(axis=(1, 2, 3))

    def mid_return(x: np.ndarray) -> np.ndarray:
        mid = 0.5 * (x[..., 0, 0, :] + x[..., 0, 10, :])
        return np.diff(mid, axis=-1)

    sample_returns = mid_return(sample64)
    real_returns = mid_return(real64)
    return_coverage = (
        (real_returns >= sample_returns.min(axis=0))
        & (real_returns <= sample_returns.max(axis=0))
    ).mean(axis=1)
    ensemble_mse_per_sample = np.mean((ensemble_mean - real64) ** 2, axis=(1, 2, 3))
    persistence_mse_per_sample = np.mean((persistence64 - real64) ** 2, axis=(1, 2, 3))
    summary = {
        "n_sampling_seeds": int(len(sample64)),
        "n_test_windows": int(len(real64)),
        "state_dimensions": dims,
        "ensemble_mean_mse": float(ensemble_mse_per_sample.mean()),
        "persistence_mse": float(persistence_mse_per_sample.mean()),
        "ensemble_mean_persistence_skill": float(
            1.0 - ensemble_mse_per_sample.mean() / max(persistence_mse_per_sample.mean(), 1e-12)
        ),
        "energy_score": float(energy.mean()),
        "scalar_crps": float(crps.mean()),
        "pairwise_sample_rmse": float(pair_rmse.mean()),
        "ensemble_variance": float(np.var(sample64, axis=0).mean()),
        "minmax_element_coverage": float(coverage.mean()),
        "minmax_mid_return_coverage": float(return_coverage.mean()),
        "scope": "fixed-checkpoint sampling ensemble; not training-seed uncertainty",
    }
    per_sample = {
        "ensemble_mean_mse": ensemble_mse_per_sample,
        "persistence_mse": persistence_mse_per_sample,
        "energy_score": energy,
        "scalar_crps": crps,
        "minmax_element_coverage": coverage,
        "minmax_mid_return_coverage": return_coverage,
        "pairwise_sample_rmse": pair_rmse,
    }
    return summary, per_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.results_dir)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    methods = (
        [item.strip() for item in args.methods.split(",") if item.strip()]
        if args.methods
        else sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "metrics.json").exists())
    )
    rows = []
    for method in methods:
        paths = _seed_dirs(root, method, seeds)
        if not paths:
            continue
        first_path = paths[0][1]
        real = np.load(first_path / "real_future.npy", mmap_mode="r")
        persistence = np.load(first_path / "persistence.npy", mmap_mode="r")
        first_indices = np.load(first_path / "per_sample_metrics.npz")["dataset_index"]
        arrays = []
        used_seeds = []
        for seed, path in paths:
            candidate_real = np.load(path / "real_future.npy", mmap_mode="r")
            candidate_indices = np.load(path / "per_sample_metrics.npz")["dataset_index"]
            if candidate_real.shape != real.shape or not np.array_equal(candidate_indices, first_indices):
                raise ValueError(f"{method} seed {seed} does not use the common test windows")
            if not np.array_equal(candidate_real, real):
                raise ValueError(f"{method} seed {seed} real futures differ")
            arrays.append(np.asarray(_evaluated_output(path)))
            used_seeds.append(seed)
        samples = np.stack(arrays)
        summary, per_sample = _ensemble_metrics(samples, np.asarray(real), np.asarray(persistence))
        summary.update({"method": method, "sampling_seeds": used_seeds})
        rows.append(summary)
        np.savez_compressed(root / f"ensemble_per_sample_{method}.npz", **per_sample)

    output = Path(args.output) if args.output else root / "ensemble_summary.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    md = output.with_suffix(".md")
    columns = [
        "method", "n_sampling_seeds", "ensemble_mean_mse", "persistence_mse",
        "ensemble_mean_persistence_skill", "energy_score", "scalar_crps",
        "pairwise_sample_rmse", "ensemble_variance", "minmax_element_coverage",
        "minmax_mid_return_coverage",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(
            f"{row[col]:.6g}" if isinstance(row.get(col), float) else str(row.get(col, ""))
            for col in columns
        ) + " |")
    md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output}")
    print(f"Wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
