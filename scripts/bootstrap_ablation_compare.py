#!/usr/bin/env python3
"""Paired bootstrap comparisons for seeded projection ablation metrics.

The script compares two methods over matched seeds from a results directory.
It is intended for fixed-checkpoint sampling seeds and for future full
training-seed result directories that use the same layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_seed_ablation import _collect_rows


DEFAULT_METRICS = [
    "mse_raw",
    "raw_crossed_book_rate",
    "raw_ask_ladder_violation",
    "raw_bid_ladder_violation",
    "projection_distance",
    "sampler_cumulative_projection_l2_mean",
]


def parse_metric_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def paired_rows(rows: list[dict[str, Any]], method_a: str, method_b: str) -> list[tuple[int, dict, dict]]:
    by_method_seed = {(row["method"], int(row["seed"])): row for row in rows}
    seeds_a = {seed for method, seed in by_method_seed if method == method_a}
    seeds_b = {seed for method, seed in by_method_seed if method == method_b}
    seeds = sorted(seeds_a & seeds_b)
    return [(seed, by_method_seed[(method_a, seed)], by_method_seed[(method_b, seed)]) for seed in seeds]


def bootstrap_diff(values_a: np.ndarray, values_b: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    diff = values_b - values_a
    n = len(diff)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = diff[idx].mean()
    mean = float(diff.mean())
    low, high = np.percentile(boots, [2.5, 97.5])
    # Two-sided sign probability around zero from the bootstrap distribution.
    p_low = float((boots <= 0).mean())
    p_high = float((boots >= 0).mean())
    p_two_sided = min(1.0, 2.0 * min(p_low, p_high))
    return {
        "mean_a": float(values_a.mean()),
        "mean_b": float(values_b.mean()),
        "mean_diff_b_minus_a": mean,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_two_sided_p": p_two_sided,
    }


def write_markdown(path: Path, payload: dict) -> None:
    rows = payload["comparisons"]
    lines = [
        "# Paired Bootstrap Ablation Comparison",
        "",
        f"Method A: `{payload['method_a']}`",
        f"Method B: `{payload['method_b']}`",
        f"Seeds: `{payload['paired_seeds']}`",
        "",
        "| Metric | Mean A | Mean B | Diff B-A | 95% CI | Bootstrap p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if "mean_a" not in row:
            continue
        lines.append(
            "| {metric} | {a:.6g} | {b:.6g} | {d:.6g} | [{lo:.6g}, {hi:.6g}] | {p:.6g} |".format(
                metric=row["metric"],
                a=row["mean_a"],
                b=row["mean_b"],
                d=row["mean_diff_b_minus_a"],
                lo=row["ci95_low"],
                hi=row["ci95_high"],
                p=row["bootstrap_two_sided_p"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison over ablation seeds")
    parser.add_argument("--results-dir", default="results/ablation_projection")
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--allow-exploratory-small-n", action="store_true",
                        help="Allow fewer than five paired seeds; output is explicitly exploratory.")
    args = parser.parse_args()

    root = Path(args.results_dir)
    rows = _collect_rows(root, include_smoke=False)
    pairs = paired_rows(rows, args.method_a, args.method_b)
    if not pairs:
        raise SystemExit(f"No paired seeds found for {args.method_a} vs {args.method_b} under {root}")
    if len(pairs) < 5 and not args.allow_exploratory_small_n:
        raise SystemExit(
            f"Only {len(pairs)} paired seeds found. Seed-level bootstrap p-values are unstable at this n. "
            "Use training-seed mean/std plus block_bootstrap_ablation.py, or pass "
            "--allow-exploratory-small-n without treating the result as confirmatory."
        )

    comparisons = []
    for metric in parse_metric_list(args.metrics):
        values_a = []
        values_b = []
        seeds = []
        for seed, row_a, row_b in pairs:
            a = row_a.get(metric)
            b = row_b.get(metric)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                values_a.append(float(a))
                values_b.append(float(b))
                seeds.append(seed)
        if not values_a:
            comparisons.append({"metric": metric, "error": "metric not available for paired rows"})
            continue
        result = bootstrap_diff(
            np.asarray(values_a, dtype=np.float64),
            np.asarray(values_b, dtype=np.float64),
            n_boot=args.n_boot,
            seed=args.seed,
        )
        result["metric"] = metric
        result["paired_seeds"] = seeds
        comparisons.append(result)

    payload = {
        "results_dir": str(root),
        "method_a": args.method_a,
        "method_b": args.method_b,
        "paired_seeds": ",".join(str(seed) for seed, _, _ in pairs),
        "n_boot": args.n_boot,
        "comparisons": comparisons,
        "interpretation_note": (
            "Positive diff means method_b has a larger metric than method_a. "
            "With fewer than five paired seeds this output is exploratory, not a formal significance test."
        ),
    }

    output_json = Path(args.output_json) if args.output_json else root / f"bootstrap_{args.method_a}_vs_{args.method_b}.json"
    output_md = Path(args.output_md) if args.output_md else root / f"bootstrap_{args.method_a}_vs_{args.method_b}.md"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(output_md, payload)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
