#!/usr/bin/env python3
"""Check CUDA, Apple MPS, and unified-memory readiness for GenLOB."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return None


def _bytes_to_gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024**3)


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _mps_memory_report() -> dict[str, Any]:
    mps = getattr(torch, "mps", None)
    if mps is None:
        return {}

    report: dict[str, Any] = {}
    for name in ["current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"]:
        fn = getattr(mps, name, None)
        if callable(fn):
            try:
                value = int(fn())
            except Exception as exc:
                report[name] = {"error": str(exc)}
            else:
                report[name] = value
                report[f"{name}_gb"] = _bytes_to_gb(value)
    return report


def _cuda_report() -> dict[str, Any]:
    report = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        report["current_device"] = torch.cuda.current_device()
        report["devices"] = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            report["devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "total_memory_gb": _bytes_to_gb(int(props.total_memory)),
                    "major": int(props.major),
                    "minor": int(props.minor),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
    return report


def _run_cuda_smoke_test(matrix_size: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"ok": False, "skipped": True, "reason": "CUDA backend is not available"}

    device = torch.device("cuda")
    try:
        torch.manual_seed(0)
        x = torch.randn(matrix_size, matrix_size, device=device)
        y = torch.randn(matrix_size, matrix_size, device=device)
        z = (x @ y).mean()
        value = float(z.detach().cpu())
        torch.cuda.synchronize()
        return {
            "ok": True,
            "skipped": False,
            "device": str(device),
            "matrix_size": matrix_size,
            "result_mean": value,
            "memory": {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "allocated_gb": _bytes_to_gb(int(torch.cuda.memory_allocated(device))),
                "reserved_gb": _bytes_to_gb(int(torch.cuda.memory_reserved(device))),
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "skipped": False,
            "device": str(device),
            "matrix_size": matrix_size,
            "error": repr(exc),
        }


def _run_mps_smoke_test(matrix_size: int) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"ok": False, "skipped": True, "reason": "MPS backend is not available"}

    device = torch.device("mps")
    try:
        torch.manual_seed(0)
        x = torch.randn(matrix_size, matrix_size, device=device)
        y = torch.randn(matrix_size, matrix_size, device=device)
        z = (x @ y).mean()
        z.backward() if z.requires_grad else None
        value = float(z.detach().cpu())
        allocated = _mps_memory_report()
        return {
            "ok": True,
            "skipped": False,
            "device": str(device),
            "matrix_size": matrix_size,
            "result_mean": value,
            "memory": allocated,
        }
    except Exception as exc:
        return {
            "ok": False,
            "skipped": False,
            "device": str(device),
            "matrix_size": matrix_size,
            "error": repr(exc),
        }


def build_report(matrix_size: int) -> dict[str, Any]:
    mem_bytes = _safe_int(_sysctl("hw.memsize"))
    arm64_flag = _sysctl("hw.optional.arm64")
    cpu_brand = _sysctl("machdep.cpu.brand_string")
    machine = platform.machine()
    apple_silicon = machine == "arm64" and bool(cpu_brand and cpu_brand.startswith("Apple"))

    report = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": machine,
        },
        "hardware": {
            "cpu_brand": cpu_brand,
            "arm64_optional": arm64_flag,
            "apple_silicon": apple_silicon,
            "physical_memory_bytes": mem_bytes,
            "physical_memory_gb": _bytes_to_gb(mem_bytes),
            "unified_memory_expected": apple_silicon,
        },
        "torch": {
            "version": torch.__version__,
            "cuda": _cuda_report(),
            "cuda_available": torch.cuda.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_memory": _mps_memory_report(),
        },
        "environment": {
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
        },
    }
    report["cuda_smoke_test"] = _run_cuda_smoke_test(matrix_size)
    report["mps_smoke_test"] = _run_mps_smoke_test(matrix_size)
    report["summary"] = {
        "cuda_ready": bool(torch.cuda.is_available() and report["cuda_smoke_test"]["ok"]),
        "mps_ready": bool(torch.backends.mps.is_built() and torch.backends.mps.is_available() and report["mps_smoke_test"]["ok"]),
        "unified_memory_ready": bool(apple_silicon and mem_bytes and mem_bytes > 0),
        "preferred_device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PyTorch CUDA/MPS and Apple unified memory readiness")
    parser.add_argument("--matrix-size", type=int, default=1024, help="MPS smoke-test matrix size")
    parser.add_argument("--json-out", default="results/env_check/mps_environment.json")
    parser.add_argument("--require-cuda", action="store_true", help="Exit non-zero if CUDA is not ready")
    parser.add_argument("--require-mps", action="store_true", help="Exit non-zero if MPS is not ready")
    parser.add_argument(
        "--require-unified-memory",
        action="store_true",
        help="Exit non-zero if Apple Silicon unified memory is not detected",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.matrix_size)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")

    if args.require_cuda and not report["summary"]["cuda_ready"]:
        return 1
    if args.require_mps and not report["summary"]["mps_ready"]:
        return 1
    if args.require_unified_memory and not report["summary"]["unified_memory_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
