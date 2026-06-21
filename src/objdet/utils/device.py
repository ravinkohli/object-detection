"""Device selection for Apple Silicon (MPS) with a CPU fallback.

This machine has no CUDA GPU; PyTorch's Metal (MPS) backend is the accelerator.
A few ops are still unimplemented on MPS — set PYTORCH_ENABLE_MPS_FALLBACK=1 so
those silently run on CPU instead of crashing the whole training run.
"""

from __future__ import annotations

import os

import torch


def get_device(prefer: str = "mps") -> torch.device:
    """Pick the best available device.

    Args:
        prefer: "mps", "cuda", or "cpu". Falls back gracefully if unavailable.

    Returns:
        A ``torch.device``.

    TODO:
        1. If prefer == "mps" and ``torch.backends.mps.is_available()`` -> mps.
        2. Else if prefer == "cuda" and ``torch.cuda.is_available()`` -> cuda.
        3. Else -> cpu.
        4. Consider calling ``enable_mps_fallback()`` here when returning mps.
    """
    raise NotImplementedError("Implement get_device")


def enable_mps_fallback() -> None:
    """Set PYTORCH_ENABLE_MPS_FALLBACK=1 so missing MPS ops fall back to CPU.

    Must be set *before* the offending op runs (ideally at process start). Safe
    to call unconditionally.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and torch for reproducibility.

    TODO:
        1. ``random.seed`` / ``numpy.random.seed`` / ``torch.manual_seed``.
        2. (Optional) ``torch.use_deterministic_algorithms(True)`` — note some
           ops have no deterministic MPS/CPU impl, so this may raise.
    """
    raise NotImplementedError("Implement set_seed")
