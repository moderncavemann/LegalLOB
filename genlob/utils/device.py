"""Device selection helpers for CUDA, Apple MPS, and CPU."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceStatus:
    """Runtime accelerator availability."""

    cuda_available: bool
    mps_built: bool
    mps_available: bool

    @property
    def available(self) -> tuple[str, ...]:
        devices = []
        if self.cuda_available:
            devices.append("cuda")
        if self.mps_available:
            devices.append("mps")
        devices.append("cpu")
        return tuple(devices)


def get_device_status() -> DeviceStatus:
    """Return CUDA/MPS availability for the current PyTorch runtime."""
    return DeviceStatus(
        cuda_available=torch.cuda.is_available(),
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
    )


def select_device(
    requested: str | torch.device | None = None,
    prefer: tuple[str, ...] = ("cuda", "mps", "cpu"),
) -> torch.device:
    """
    Select a torch device with explicit validation.

    Args:
        requested: `None`/`"auto"` or an explicit device string such as
            `"cuda"`, `"cuda:0"`, `"mps"`, or `"cpu"`.
        prefer: auto-selection order. Default is CUDA first for cluster/GPU
            machines, then MPS for Apple Silicon, then CPU.
    """
    if isinstance(requested, torch.device):
        requested = str(requested)
    requested = (requested or "auto").lower()
    status = get_device_status()

    if requested == "auto":
        for name in prefer:
            if name == "cuda" and status.cuda_available:
                return torch.device("cuda")
            if name == "mps" and status.mps_available:
                return torch.device("mps")
            if name == "cpu":
                return torch.device("cpu")
        return torch.device("cpu")

    base = requested.split(":", 1)[0]
    if base == "cuda":
        if not status.cuda_available:
            raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False")
        return torch.device(requested)
    if base == "mps":
        if not status.mps_built:
            raise RuntimeError("Requested MPS device, but this PyTorch build has no MPS support")
        if not status.mps_available:
            raise RuntimeError("Requested MPS device, but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    if base == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device '{requested}'. Expected auto, cuda, mps, or cpu.")


def device_summary(device: torch.device | str | None = None) -> dict[str, str | bool]:
    """Small serializable device summary for logs and reports."""
    selected = select_device(device)
    status = get_device_status()
    return {
        "selected": str(selected),
        "cuda_available": status.cuda_available,
        "mps_built": status.mps_built,
        "mps_available": status.mps_available,
        "available": ",".join(status.available),
    }
