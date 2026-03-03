"""MPS/CPU device selection with fallback."""

import torch


def get_device(force_cpu: bool = False) -> torch.device:
    """Select the best available device.

    Priority: MPS (Apple Silicon GPU) > CPU.
    Use force_cpu=True for ONNX export (required workaround).
    """
    if force_cpu:
        print("Device: cpu (forced)")
        return torch.device("cpu")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Device: mps (Apple Silicon GPU)")
        return torch.device("mps")

    print("Device: cpu (MPS not available)")
    return torch.device("cpu")
