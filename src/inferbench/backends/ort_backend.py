"""ONNX Runtime, over any execution provider this machine offers.

One ONNX file, several providers: CPU everywhere, CoreML on Apple silicon,
CUDA and the TensorRT execution provider on an NVIDIA box. Comparing providers
against each other isolates the runtime from the graph, because the graph is
byte-identical.
"""
from __future__ import annotations

import numpy as np

from .base import Availability


class OrtBackend:
    def __init__(self, provider: str = "CPUExecutionProvider"):
        self.provider = provider
        short = provider.replace("ExecutionProvider", "").lower()
        self.name = f"onnxruntime-{short}"
        self._sess = None

    def available(self) -> Availability:  # type: ignore[override]
        try:
            import onnxruntime as ort
        except ImportError as exc:
            return Availability(False, f"onnxruntime not installed ({exc})")
        if self.provider not in ort.get_available_providers():
            return Availability(False, f"{self.provider} not built into this wheel")
        return Availability(True)

    def load(self, spec, batch: int) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(spec.onnx_path), opts,
                                          providers=[self.provider])
        self._input = self._sess.get_inputs()[0].name

    def infer(self, x: np.ndarray) -> np.ndarray:
        # InferenceSession.run is synchronous; the result is already on the host.
        return self._sess.run(None, {self._input: x})[0]

    def close(self) -> None:
        self._sess = None
