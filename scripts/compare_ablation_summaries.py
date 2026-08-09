#!/usr/bin/env python3
"""Compare canonical and reproduced ablation summaries within tolerance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


IGNORED_KEYS = {"source"}


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _key_for_row(row: dict[str, Any]) -> tuple[Any, ...]:
    if "kind" in row:
        return (row.get("method"), row.get("kind"))
    return (row.get("method"),)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare_value(
    ref: Any,
    cand: Any,
    path: str,
    abs_tol: float,
    rel_tol: float,
    errors: list[str],
) -> None:
    if _is_number(ref) or _is_number(cand):
        if not (_is_number(ref) and _is_number(cand)):
            errors.append(f"{path}: numeric/type mismatch ref={ref!r} cand={cand!r}")
            return
        if math.isnan(float(ref)) and math.isnan(float(cand)):
            return
        if not math.isclose(float(ref), float(cand), abs_tol=abs_tol, rel_tol=rel_tol):
            errors.append(f"{path}: ref={ref!r} cand={cand!r}")
        return
    if ref != cand:
        errors.append(f"{path}: ref={ref!r} cand={cand!r}")


def _compare_rows(
    ref_rows: list[dict[str, Any]],
    cand_rows: list[dict[str, Any]],
    label: str,
    abs_tol: float,
    rel_tol: float,
) -> list[str]:
    errors: list[str] = []
    ref_map = {_key_for_row(row): row for row in ref_rows}
    cand_map = {_key_for_row(row): row for row in cand_rows}
    if set(ref_map) != set(cand_map):
        errors.append(
            f"{label}: row key mismatch ref_only={sorted(set(ref_map) - set(cand_map))} "
            f"cand_only={sorted(set(cand_map) - set(ref_map))}"
        )
        return errors

    for key in sorted(ref_map):
        ref = ref_map[key]
        cand = cand_map[key]
        ref_keys = set(ref) - IGNORED_KEYS
        cand_keys = set(cand) - IGNORED_KEYS
        if ref_keys != cand_keys:
            errors.append(
                f"{label}/{key}: key mismatch ref_only={sorted(ref_keys - cand_keys)} "
                f"cand_only={sorted(cand_keys - ref_keys)}"
            )
            continue
        for col in sorted(ref_keys):
            _compare_value(
                ref[col],
                cand[col],
                f"{label}/{key}/{col}",
                abs_tol,
                rel_tol,
                errors,
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--abs-tol", type=float, default=1e-5)
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    args = parser.parse_args()

    reference = Path(args.reference)
    candidate = Path(args.candidate)
    checks = [
        ("summary.json", "single_seed_summary"),
        ("seed_summary.json", "seed_aggregate_summary"),
    ]
    all_errors: list[str] = []
    for filename, label in checks:
        ref_rows = _load_json(reference / filename)
        cand_rows = _load_json(candidate / filename)
        if not isinstance(ref_rows, list) or not isinstance(cand_rows, list):
            raise TypeError(f"{filename} must contain a list of rows")
        all_errors.extend(
            _compare_rows(ref_rows, cand_rows, label, args.abs_tol, args.rel_tol)
        )

    if all_errors:
        print("Ablation summary comparison FAILED")
        for error in all_errors[:100]:
            print(f"- {error}")
        if len(all_errors) > 100:
            print(f"... {len(all_errors) - 100} more")
        return 1

    print("Ablation summary comparison passed")
    print(f"reference: {reference}")
    print(f"candidate: {candidate}")
    print(f"abs_tol: {args.abs_tol}")
    print(f"rel_tol: {args.rel_tol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
