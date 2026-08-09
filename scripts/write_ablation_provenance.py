#!/usr/bin/env python3
"""Write config and provenance files for projection ablation result directories."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RAW_CKPT = "checkpoints_raw/intc_9day_full_50ep/best.pt"
LEGAL_CKPT = "checkpoints_legal/intc_9day_full_50ep/best.pt"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def _git_status_short() -> str:
    return _run(["git", "status", "--short"])


def _hardware_text() -> str:
    lines = [
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        f"processor: {platform.processor() or 'unavailable'}",
    ]
    cpu_brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if cpu_brand != "unavailable":
        lines.append(f"cpu_brand: {cpu_brand}")
    try:
        import torch

        lines.extend(
            [
                f"torch: {torch.__version__}",
                f"mps_available: {torch.backends.mps.is_available()}",
                f"cuda_available: {torch.cuda.is_available()}",
            ]
        )
        if torch.cuda.is_available():
            lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
    except Exception as exc:
        lines.append(f"torch: unavailable ({exc})")
    return "\n".join(lines) + "\n"


def _seed_from_path(root: Path, result_dir: Path) -> int:
    rel = result_dir.relative_to(root)
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "seeds" and parts[1].startswith("seed"):
        return int(parts[1].replace("seed", ""))
    return 0


def _method_from_path(result_dir: Path) -> str:
    return result_dir.name


def _checkpoint_for_method(method: str, raw_ckpt: str, legal_ckpt: str) -> str:
    return legal_ckpt if "legal" in method else raw_ckpt


def _command_for_result(
    root: Path,
    result_dir: Path,
    metrics: dict[str, Any] | None,
    dataset: str,
    raw_ckpt: str,
    legal_ckpt: str,
) -> str:
    method = _method_from_path(result_dir)
    seed = _seed_from_path(root, result_dir)
    output_dir = str(result_dir)

    if (result_dir / "snowball_stats.json").exists():
        stats = json.loads((result_dir / "snowball_stats.json").read_text())
        cfg = stats.get("config", {})
        ckpt = stats.get("checkpoint", {}).get("path") or legal_ckpt
        return "\n".join(
            [
                "python scripts/snowball_test.py \\",
                f"  --genlob-ckpt {ckpt} \\",
                "  --weights ema \\",
                f"  --dataset {cfg.get('dataset', dataset)} \\",
                f"  --data-dir {cfg.get('data_dir', 'data')} \\",
                f"  --hist-len {cfg.get('hist_len', 50)} \\",
                f"  --pred-len {cfg.get('pred_len', 10)} \\",
                f"  --horizon {cfg.get('horizon', 100)} \\",
                f"  --ddim-steps {cfg.get('ddim_steps', 20)} \\",
                "  --representation auto \\",
                f"  --projection-mode {cfg.get('projection_mode', 'none')} \\",
                "  --compute-projection-distance \\",
                f"  --action-mode {cfg.get('action_mode', 'replay')} \\",
                f"  --batch-size {cfg.get('batch_size', 128)} \\",
                f"  --n-batches {cfg.get('n_histories', 6400) // cfg.get('batch_size', 128)} \\",
                f"  --output-dir {output_dir} \\",
                f"  --seed {seed}",
            ]
        ) + "\n"

    checkpoint = _checkpoint_for_method(method, raw_ckpt=raw_ckpt, legal_ckpt=legal_ckpt)
    projection_mode = metrics.get("projection_mode", "none") if metrics else "none"
    sampler_projection_mode = metrics.get("sampler_projection_mode", "none") if metrics else "none"
    return "\n".join(
        [
            "python scripts/evaluate_v2.py \\",
            f"  --checkpoint {checkpoint} \\",
            "  --weights ema \\",
            f"  --dataset {dataset} \\",
            "  --data-dir data \\",
            "  --split test \\",
            f"  --output-dir {output_dir} \\",
            "  --n-batches 50 \\",
            "  --batch-size 128 \\",
            "  --ddim-steps 20 \\",
            f"  --projection-mode {projection_mode} \\",
            f"  --sampler-projection-mode {sampler_projection_mode} \\",
            "  --representation auto \\",
            f"  --seed {seed}",
        ]
    ) + "\n"


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace('"', '\\"')
    return f'"{text}"'


def _config_for_result(
    root: Path,
    result_dir: Path,
    metrics: dict[str, Any] | None,
    experiment: str,
    dataset: str,
    raw_ckpt: str,
    legal_ckpt: str,
) -> dict[str, Any]:
    method = _method_from_path(result_dir)
    seed = _seed_from_path(root, result_dir)
    is_snowball = (result_dir / "snowball_stats.json").exists()
    config = {
        "experiment": experiment,
        "method": method,
        "dataset": dataset,
        "data_dir": "data",
        "result_dir": str(result_dir),
        "seed": seed,
        "weights": "ema",
        "device_observed": "mps",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if is_snowball:
        stats = json.loads((result_dir / "snowball_stats.json").read_text())
        cfg = stats.get("config", {})
        ckpt = stats.get("checkpoint", {}).get("path") or legal_ckpt
        config.update(
            {
                "script": "scripts/snowball_test.py",
                "checkpoint": ckpt,
                "hist_len": cfg.get("hist_len", 50),
                "pred_len": cfg.get("pred_len", 10),
                "horizon": cfg.get("horizon", 100),
                "ddim_steps": cfg.get("ddim_steps", 20),
                "projection_mode": cfg.get("projection_mode", "none"),
                "compute_projection_distance": cfg.get("compute_projection_distance", True),
                "action_mode": cfg.get("action_mode", "replay"),
                "batch_size": cfg.get("batch_size", 128),
                "n_batches": cfg.get("n_histories", 6400) // cfg.get("batch_size", 128),
            }
        )
    else:
        config.update(
            {
                "script": "scripts/evaluate_v2.py",
                "checkpoint": _checkpoint_for_method(method, raw_ckpt=raw_ckpt, legal_ckpt=legal_ckpt),
                "split": "test",
                "n_batches": 50,
                "batch_size": 128,
                "ddim_steps": 20,
                "representation": "auto",
                "projection_mode": metrics.get("projection_mode", "none") if metrics else "none",
                "sampler_projection_mode": metrics.get("sampler_projection_mode", "none") if metrics else "none",
            }
        )
    if "smoke" in method:
        config["note"] = "smoke output retained for pipeline diagnostics, not paper result"
    return config


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    lines = []
    for key, value in data.items():
        lines.append(f"{key}: {_yaml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n")


def _result_dirs(root: Path) -> list[Path]:
    dirs = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        if (path / "metrics.json").exists() or (path / "snowball_stats.json").exists():
            dirs.append(path)
    return dirs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/ablation_projection")
    parser.add_argument("--experiment", default="intc_9day_projection_ablation")
    parser.add_argument("--dataset", default="INTC_9day")
    parser.add_argument("--raw-checkpoint", default=RAW_CKPT)
    parser.add_argument("--legal-checkpoint", default=LEGAL_CKPT)
    args = parser.parse_args()

    root = Path(args.results_dir)
    git_hash = _run(["git", "rev-parse", "HEAD"])
    git_status = _git_status_short()
    dirty = bool(git_status.strip())
    hardware = _hardware_text()

    dirs = _result_dirs(root)
    if not dirs:
        raise SystemExit(f"No result directories found under {root}")

    for result_dir in dirs:
        metrics_path = result_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        _write_yaml(
            result_dir / "config.yaml",
            _config_for_result(
                root,
                result_dir,
                metrics,
                experiment=args.experiment,
                dataset=args.dataset,
                raw_ckpt=args.raw_checkpoint,
                legal_ckpt=args.legal_checkpoint,
            ),
        )
        (result_dir / "command.txt").write_text(
            _command_for_result(
                root,
                result_dir,
                metrics,
                dataset=args.dataset,
                raw_ckpt=args.raw_checkpoint,
                legal_ckpt=args.legal_checkpoint,
            )
        )
        (result_dir / "git_commit.txt").write_text(
            f"commit: {git_hash}\n"
            f"dirty_worktree: {'true' if dirty else 'false'}\n"
            "status_short:\n"
            + "\n".join(f"  {line}" for line in git_status.splitlines())
            + ("\n" if git_status else "")
        )
        (result_dir / "hardware.txt").write_text(hardware)
        print(f"Wrote provenance for {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
