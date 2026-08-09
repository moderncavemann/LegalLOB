#!/usr/bin/env python3
"""Generate SHA256 hashes for frozen experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CORE_ARTIFACTS = [
    "checkpoints_raw/intc_9day_full_50ep/best.pt",
    "checkpoints_legal/intc_9day_full_50ep/best.pt",
    "results/ablation_projection/summary.md",
    "results/ablation_projection/summary.json",
    "results/ablation_projection/summary.csv",
    "results/ablation_projection/seed_summary.md",
    "results/ablation_projection/seed_summary.json",
    "results/ablation_projection/seed_summary.csv",
    "results/ablation_projection/intc_raw_no_projection/metrics.json",
    "results/ablation_projection/intc_raw_posthoc_pava/metrics.json",
    "results/ablation_projection/intc_raw_inloop_dmpp/metrics.json",
    "results/ablation_projection/intc_legal_no_projection/metrics.json",
    "results/ablation_projection/intc_legal_snowball_100_replay/snowball_stats.json",
    "results/goog_public_short/summary.md",
    "results/goog_public_short/summary.json",
    "results/goog_public_short/summary.csv",
    "README.md",
    "EXPERIMENT_LOG.md",
    "MILESTONE.md",
    "PAPER_DRAFT.md",
    "data/processed/snapshot_l2/INTC_9day/DATASET_CARD.md",
    "DATA_REQUIREMENTS.md",
    "DATA_GAP_STATUS.md",
    "DATASET_EXTENSION.md",
    "ARTIFACT_INDEX.md",
    "data/manifests/datasets.yaml",
    "data/manifests/baselines.yaml",
    "data/manifests/experiments.yaml",
    "scripts/prepare_lobster_snapshot_l2.py",
    "scripts/prepare_goog_9day.sh",
    "scripts/list_nasdaq_itch_public.py",
    "scripts/fetch_nasdaq_itch_public.py",
    "scripts/prepare_public_itch_snapshot_l2.py",
    "scripts/apply_snapshot_purge_gap.py",
]


def _paths_from_markdown(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
    paths = []
    for match in re.findall(r"`([^`]+)`", text):
        if any(ch.isspace() for ch in match):
            continue
        if "/" not in match and not Path(match).exists():
            continue
        if match.endswith(
            (".md", ".json", ".csv", ".yaml", ".yml", ".txt", ".pt", ".sh", ".py", ".png")
        ):
            paths.append(match)
    return paths


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "pt":
        return "checkpoint"
    if suffix in {"json", "csv", "md", "yaml", "yml", "txt", "png"}:
        return suffix
    if suffix in {"sh", "py"}:
        return "script"
    return "file"


def _record(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return {
            "path": path_str,
            "missing": True,
            "artifact_type": _artifact_type(path),
        }
    stat = path.stat()
    return {
        "path": path_str,
        "sha256": _sha256(path),
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "artifact_type": _artifact_type(path),
        "missing": False,
    }


def _format_md(records: list[dict[str, Any]]) -> str:
    columns = ["path", "artifact_type", "size_bytes", "sha256", "modified_time", "missing"]
    lines = [
        "# Artifact Hashes",
        "",
        "Generated for the frozen cross-asset continuous-feasibility paper milestone.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in records:
        values = [str(record.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    missing = [r["path"] for r in records if r.get("missing")]
    lines.append(f"Missing files: {len(missing)}")
    if missing:
        for path in missing:
            lines.append(f"- `{path}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-index", default="ARTIFACT_INDEX.md")
    parser.add_argument("--output-json", default="artifact_hashes.json")
    parser.add_argument("--output-md", default="ARTIFACT_HASHES.md")
    args = parser.parse_args()

    artifact_index = Path(args.artifact_index)
    output_paths = {args.output_json, args.output_md}
    paths = [
        path
        for path in list(CORE_ARTIFACTS) + _paths_from_markdown(artifact_index)
        if path not in output_paths
    ]
    paths = sorted(dict.fromkeys(paths))
    records = [_record(path) for path in paths]
    missing = [record["path"] for record in records if record.get("missing")]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_index": str(artifact_index),
        "missing_count": len(missing),
        "duplicate_path_count": len(paths) - len(set(paths)),
        "artifacts": records,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")
    Path(args.output_md).write_text(_format_md(records))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Artifacts: {len(records)}")
    print(f"Missing: {len(missing)}")
    if missing:
        for path in missing:
            print(f"Missing artifact: {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
