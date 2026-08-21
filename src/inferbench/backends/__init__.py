"""Backend discovery.

`all_backends()` returns every backend this repository knows how to drive.
Which of them can actually run is a separate question, answered by each one's
`available()`. Nothing is filtered out silently - an unavailable backend is
reported with its reason so the comparison table shows the gaps instead of
hiding them.
"""
from __future__ import annotations

from .base import Availability, Backend
from .ort_backend import OrtBackend
from .torch_backend import TorchBackend
from .trt_backend import TensorRTBackend
from .triton_backend import TritonBackend

__all__ = ["Availability", "Backend", "OrtBackend", "TorchBackend",
           "TensorRTBackend", "TritonBackend", "all_backends", "REFERENCE"]

REFERENCE = "pytorch-cpu"          # what accuracy is measured against


def all_backends(triton_url: str = "localhost:8000") -> list:
    return [
        TorchBackend("cpu"),
        TorchBackend("mps"),          # Apple silicon
        TorchBackend("cuda"),
        OrtBackend("CPUExecutionProvider"),
        OrtBackend("CoreMLExecutionProvider"),
        OrtBackend("CUDAExecutionProvider"),
        OrtBackend("TensorrtExecutionProvider"),
        TensorRTBackend("fp32"),
        TensorRTBackend("fp16"),
        TritonBackend(triton_url),
    ]
