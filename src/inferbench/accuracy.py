"""Cross-backend numerical agreement.

A faster wrong answer is not an optimisation. FP16 and INT8 trade precision for
speed, and graph-level rewrites in ONNX Runtime or TensorRT can reorder
arithmetic even at FP32 - so every speed number in this repository is paired
with how far that backend drifted from the PyTorch CPU reference.

Two measures, because they fail differently:

* **max absolute difference** on the raw outputs catches numerical drift.
* **top-1 agreement** catches whether the drift changed the answer, which is
  the part a deployment actually cares about.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AccuracyStats:
    backend: str
    reference: str
    samples: int
    max_abs_diff: float
    mean_abs_diff: float
    top1_agreement: float          # fraction of rows whose argmax matches

    def as_dict(self) -> dict:
        return asdict(self)


def compare(name: str, out: np.ndarray, ref: np.ndarray,
            reference_name: str) -> AccuracyStats:
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if out.shape != ref.shape:
        raise ValueError(f"{name}: shape {out.shape} != reference {ref.shape}")
    diff = np.abs(out - ref)
    return AccuracyStats(
        backend=name, reference=reference_name, samples=int(out.shape[0]),
        max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
        top1_agreement=float((out.argmax(1) == ref.argmax(1)).mean()),
    )
