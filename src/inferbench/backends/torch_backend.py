"""PyTorch, on whatever accelerator this machine has.

This is the baseline every other backend is judged against, and the reference
the accuracy check compares to.
"""
from __future__ import annotations

import numpy as np

from .base import Availability


class TorchBackend:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.name = f"pytorch-{device}"
        self._model = None
        self._torch = None

    @classmethod
    def for_device(cls, device: str) -> "TorchBackend":
        return cls(device)

    def available(self) -> Availability:  # type: ignore[override]
        try:
            import torch
        except ImportError as exc:
            return Availability(False, f"torch not installed ({exc})")
        if self.device == "cuda" and not torch.cuda.is_available():
            return Availability(False, "no CUDA device")
        if self.device == "mps" and not torch.backends.mps.is_available():
            return Availability(False, "no MPS device")
        return Availability(True)

    def load(self, spec, batch: int) -> None:
        import torch

        from ..model import build_torch_model
        self._torch = torch
        self._model = build_torch_model(spec.name).to(self.device).eval()

    def _sync(self) -> None:
        t = self._torch
        if self.device == "cuda":
            t.cuda.synchronize()
        elif self.device == "mps":
            t.mps.synchronize()

    def infer(self, x: np.ndarray) -> np.ndarray:
        t = self._torch
        with t.no_grad():
            out = self._model(t.from_numpy(x).to(self.device))
            self._sync()          # the queue is not the computation
            return out.detach().to("cpu").numpy()

    def close(self) -> None:
        self._model = None
