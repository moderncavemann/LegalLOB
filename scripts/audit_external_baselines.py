#!/usr/bin/env python3
"""Audit external LOB baseline checkouts without claiming they are runnable."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.utils.device import device_summary


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    name: str
    path: str
    pinned_commit: str
    task: str
    required_paths: tuple[str, ...]
    data_gate: str


SPECS = (
    BaselineSpec(
        key="difflob",
        name="DiffLOB",
        path="DiffLOB",
        pinned_commit="5f15bf882ab6894b92043c24bb47b025c426cf1a",
        task="snapshot_counterfactual_generation",
        required_paths=(
            "README.md",
            "pyproject.toml",
            "src/difflob/training/diffusion.py",
            "src/difflob/sampling/diffusion.py",
            "nets/diff_wavenet_motion_control.py",
        ),
        data_gate="one_hz_top10_and_regime_labels",
    ),
    BaselineSpec(
        key="difflob_legacy_source",
        name="DiffLOB official pre-reorganization source",
        path="DiffLOB-legacy",
        pinned_commit="9ee0e0a25bff83cc8df5351f20475bbad3f794a9",
        task="snapshot_counterfactual_generation",
        required_paths=(
            "diffusion_trainer.py",
            "diffusion_sampler.py",
            "utils/preprocessing.py",
            "nets/diff_wavenet_motion_control.py",
            "nets/diff_csdi.py",
            "nets/diff_s4.py",
        ),
        data_gate="one_hz_top10_and_regime_labels",
    ),
    BaselineSpec(
        key="trades",
        name="TRADES / DeepMarket",
        path="DeepMarket",
        pinned_commit="8f1f89b7285c79a73f528b88b60f74ce58faadc4",
        task="message_generation_and_simulation",
        required_paths=("README.md", "main.py", "models", "preprocessing", "evaluation"),
        data_gate="event_l3_and_simulator",
    ),
    BaselineSpec(
        key="lob_bench",
        name="LOB-Bench",
        path="lob_bench",
        pinned_commit="276baf3039bf7630b555891b9d04ad49a92b3356",
        task="message_distribution_and_impact_benchmark",
        required_paths=("README.md", "data_loading.py", "scoring.py", "impact.py", "metrics.py"),
        data_gate="real_and_generated_lobster_format_streams",
    ),
    BaselineSpec(
        key="lobs5",
        name="LOBS5",
        path="LOBS5",
        pinned_commit="aa36029e9d348a017d00efc37c6b6dda89fab150",
        task="autoregressive_message_generation_and_replay",
        required_paths=("README.md", "run_train.py", "lob/preproc.py", "lob/lobster_dataloader.py"),
        data_gate="lobster_message_orderbook_and_jax_lob",
    ),
    BaselineSpec(
        key="lobdif",
        name="LOBDIF",
        path="LOBDIF",
        pinned_commit="5f6e235a90118802f8f69a9aab09e87a4d5be662",
        task="event_time_and_type_prediction",
        required_paths=("README.md", "run_model.py", "data_parser.py", "model/DiffusionModel.py"),
        data_gate="event_times_types_and_market_state",
    ),
)


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def required_path_checks(root: Path, required: tuple[str, ...]) -> dict[str, bool]:
    return {item: (root / item).exists() for item in required}


def scan_device_markers(root: Path) -> dict[str, int]:
    markers = {
        "mps_mentions": 0,
        "cuda_mentions": 0,
        "hardcoded_cuda_device": 0,
        "cuda_only_selection": 0,
        "amp_mentions": 0,
    }
    if not root.exists():
        return markers
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        lower = text.lower()
        markers["mps_mentions"] += lower.count("mps")
        markers["cuda_mentions"] += lower.count("cuda")
        markers["hardcoded_cuda_device"] += lower.count("torch.device('cuda')")
        markers["hardcoded_cuda_device"] += lower.count('torch.device("cuda")')
        markers["cuda_only_selection"] += lower.count('"cuda" if torch.cuda.is_available() else "cpu"')
        markers["cuda_only_selection"] += lower.count("'cuda' if torch.cuda.is_available() else 'cpu'")
        markers["amp_mentions"] += lower.count("autocast(") + lower.count("gradscaler(")
    return markers


def _compile_python(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"status": "not_run", "reason": "checkout_missing"}
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
    }


def _known_blockers(key: str, root: Path) -> list[str]:
    blockers: list[str] = []
    if key == "difflob":
        if not (root / "src/difflob/data/preprocessing.py").exists():
            blockers.append("official_cli_imports_missing_difflob.data.preprocessing")
        if not (root / "src/difflob/models/diffusion").exists():
            blockers.append("official_trainers_import_missing_difflob.models.diffusion")
        blockers.append("official_device_selection_has_no_mps_branch")
        blockers.append("mps_requires_amp_and_grad_scaler_compatibility_patch")
    elif key == "difflob_legacy_source":
        blockers.append("historical_official_commit_required_because_current_packaging_is_incomplete")
        blockers.append("device_selection_has_no_mps_branch")
        blockers.append("mps_requires_amp_and_grad_scaler_compatibility_patch")
        blockers.append("published_configs_contain_machine_specific_absolute_data_paths")
    elif key == "trades":
        blockers.append("fair_retraining_requires_event_l3_source_and_abides_alignment")
    elif key == "lob_bench":
        blockers.append("full_impact_protocol_requires_message_level_lobster_format")
    elif key == "lobs5":
        alpha_trade = root / "AlphaTrade"
        if not alpha_trade.exists() or not any(alpha_trade.iterdir()):
            blockers.append("alphatrade_jax_lob_submodule_not_initialized")
        blockers.append("requires_lobster_message_orderbook_pairs")
    elif key == "lobdif":
        blockers.append("run_model_hardcodes_cuda")
        blockers.append("expected_lobster_wrds_csv_files_are_not_in_checkout")
    return blockers


def inspect_baseline(spec: BaselineSpec, references_root: Path) -> dict[str, Any]:
    root = references_root / spec.path
    paths = required_path_checks(root, spec.required_paths)
    commit = _git_head(root)
    markers = scan_device_markers(root)
    blockers = _known_blockers(spec.key, root) if root.exists() else ["checkout_missing"]
    source_ready = root.exists() and all(paths.values()) and commit is not None
    commit_matches = commit == spec.pinned_commit
    if source_ready and not commit_matches:
        blockers.append("checkout_commit_differs_from_pinned_commit")
    mps_status = "unsupported_or_patch_required"
    if markers["mps_mentions"] > 0 and markers["hardcoded_cuda_device"] == 0 and markers["cuda_only_selection"] == 0:
        mps_status = "mentions_mps_requires_runtime_smoke"
    return {
        "name": spec.name,
        "task": spec.task,
        "checkout": str(root),
        "source_ready": source_ready,
        "pinned_commit": spec.pinned_commit,
        "observed_commit": commit,
        "commit_matches": commit_matches,
        "required_paths": paths,
        "syntax_compile": _compile_python(root),
        "device_markers": markers,
        "mps_status": mps_status,
        "cuda_status": "requires_runtime_smoke" if source_ready else "not_testable",
        "data_gate": spec.data_gate,
        "blockers": blockers,
        "ready_for_full_experiment": source_ready and commit_matches and not blockers,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# External Baseline Readiness",
        "",
        f"Generated: `{report['generated_at']}`.",
        "",
        "Source availability is not equivalent to experiment readiness.",
        "",
        "| Baseline | Source | Commit | MPS | Data gate | Full run |",
        "|---|---:|---:|---|---|---:|",
    ]
    for item in report["baselines"].values():
        lines.append(
            f"| {item['name']} | {'yes' if item['source_ready'] else 'no'} | "
            f"{'match' if item['commit_matches'] else 'mismatch'} | {item['mps_status']} | "
            f"{item['data_gate']} | {'yes' if item['ready_for_full_experiment'] else 'no'} |"
        )
    lines.extend(["", "## Blockers", ""])
    for key, item in report["baselines"].items():
        lines.append(f"### {item['name']} (`{key}`)")
        lines.append("")
        if item["blockers"]:
            lines.extend(f"- `{blocker}`" for blocker in item["blockers"])
        else:
            lines.append("- None detected by the static audit; runtime smoke is still required.")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references-root", default="references")
    parser.add_argument("--output-json", default="EXTERNAL_BASELINE_READINESS.json")
    parser.add_argument("--output-md", default="EXTERNAL_BASELINE_READINESS.md")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any baseline is not full-run ready.")
    args = parser.parse_args()

    references_root = Path(args.references_root)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "device": device_summary("auto"),
        },
        "baselines": {spec.key: inspect_baseline(spec, references_root) for spec in SPECS},
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(output_md, report)
    print(json.dumps(report, indent=2))
    if args.strict and not all(item["ready_for_full_experiment"] for item in report["baselines"].values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
