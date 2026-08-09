#!/usr/bin/env python3
"""Forward-smoke the complete official pre-reorganization DiffLOB network."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.utils.device import device_summary, select_device


PINNED_COMMIT = "9ee0e0a25bff83cc8df5351f20475bbad3f794a9"


def _head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", default="references/DiffLOB-legacy")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda", "auto"))
    parser.add_argument("--output", default="DIFFLOB_COMPAT_READINESS.json")
    args = parser.parse_args()

    checkout = Path(args.checkout).resolve()
    observed_commit = _head(checkout)
    if observed_commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected DiffLOB {PINNED_COMMIT}, found {observed_commit}")
    sys.path.insert(0, str(checkout))
    from nets.diff_wavenet_motion_control import WaveNetJoint

    device = select_device(args.device)
    batch, levels, horizon = 2, 20, 8
    model = WaveNetJoint(
        input_dim=2,
        output_dim=2,
        base_channels=16,
        # The official MotionModule hard-codes GlobalFiLM emb_dim=128.
        emb_dim=128,
        num_layers=2,
        n_features=levels,
        num_heads=4,
        num_steps=100,
    ).to(device)
    x = torch.randn(batch, 2, levels, horizon, device=device)
    t = torch.randint(0, 100, (batch,), device=device)
    cond = (
        torch.randn(batch, 1, levels, horizon, device=device),
        torch.rand(batch, 1, levels, horizon, device=device),
        torch.randn(batch, device=device),
        torch.rand(batch, device=device),
        torch.rand(batch, horizon, device=device),
        torch.randn(batch, horizon, device=device),
        torch.linspace(0, 1, horizon, device=device).expand(batch, -1),
    )
    with torch.no_grad():
        output = model(x, t, cond, enable_motion=True, enable_control=True)
    passed = output.shape == x.shape and bool(torch.isfinite(output).all().item())
    report = {
        "baseline": "DiffLOB official pre-reorganization WaveNet motion-control",
        "checkout": str(checkout),
        "commit": observed_commit,
        "device": device_summary(device),
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all().item()),
        "forward_smoke": "passed" if passed else "failed",
        "scope": "network forward only; preprocessing, training, sampling, and metric adapters remain gated",
        "compatibility_note": "official MotionModule requires the default global embedding width of 128",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
