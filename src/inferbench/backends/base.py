"""The backend contract.

Every runtime - PyTorch, ONNX Runtime, TensorRT, Triton - is reduced to the
same three operations, so the benchmark measures runtimes rather than measuring
four different pieces of harness code.

`available()` is the important one. A backend that cannot run says so and gives
a reason; it never returns a number. The failure mode this whole repository
exists to correct is a comparison table full of figures nobody measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Availability:
    ok: bool
    reason: str = ""


@runtime_checkable
class Backend(Protocol):
    name: str

    def available(self) -> Availability:
        """Whether this backend can run here, and if not, why not."""
        ...

    def load(self, spec, batch: int) -> None: ...

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Run one batch and return the output, fully materialised.

        Implementations MUST synchronise the device before returning. An async
        runtime that returns as soon as the work is queued reports the time to
        enqueue, not the time to compute - which is how a backend ends up
        looking twenty times faster than it is.
        """
        ...

    def close(self) -> None: ...
