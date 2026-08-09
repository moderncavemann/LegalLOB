import pytest
import torch

from genlob.utils.device import get_device_status, select_device


def test_select_device_cpu_always_available():
    assert select_device("cpu").type == "cpu"


def test_select_device_auto_returns_available_backend():
    device = select_device()
    assert device.type in get_device_status().available


def test_select_device_rejects_unavailable_explicit_backend():
    status = get_device_status()
    if not status.cuda_available:
        with pytest.raises(RuntimeError):
            select_device("cuda")
    if not status.mps_available:
        with pytest.raises(RuntimeError):
            select_device("mps")
