"""The model under test, and its ONNX export.

ResNet-18 with random weights. Random is deliberate: nothing here measures
accuracy against ground truth, only whether two runtimes computing the same
graph agree with each other, and how fast each one is. Random weights give the
real architecture and the real FLOPs without a 45 MB download, so the benchmark
runs offline on a fresh checkout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_shape: tuple[int, int, int]     # CHW, batch is dynamic
    output_size: int
    onnx_path: Path
    opset: int = 18

    def sample(self, batch: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((batch, *self.input_shape), dtype=np.float32)


WEIGHT_SEED = 20260821


def build_torch_model(name: str = "resnet18", seed: int = WEIGHT_SEED):
    """Build the model with deterministic weights.

    The seed is not a detail. Without it every call initialises a different
    random network, so the ONNX file, the PyTorch reference and each backend
    all run *different models* - and the accuracy comparison silently measures
    the difference between two random initialisations rather than between two
    runtimes. The first run of this harness reported max |delta| = 9.65 and 0%
    top-1 agreement across the board, which is what that looks like.
    """
    import torch

    torch.manual_seed(seed)
    if name == "resnet18":
        from torchvision.models import resnet18
        model = resnet18(weights=None)
    elif name == "tinycnn":
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(64, 1000),
        )
    else:
        raise ValueError(f"unknown model {name!r}")
    return model.eval()


def export_onnx(name: str, out_dir: Path, opset: int = 18,
                input_shape: tuple[int, int, int] = (3, 224, 224)) -> ModelSpec:
    """Export with a dynamic batch axis.

    A fixed batch would force a separate engine per batch size and quietly make
    the throughput sweep meaningless, so the batch dimension stays symbolic all
    the way through to the TensorRT optimisation profile.
    """
    import inspect
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.onnx"

    model = build_torch_model(name)
    dummy = torch.randn(1, *input_shape)
    # Torch 2.4's exporter uses dynamic_axes; newer dynamo exporters use a
    # real torch.export.Dim in dynamic_shapes. Keep the pinned CPU dependency
    # working while retaining the symbolic batch axis on both APIs.
    if "dynamic_shapes" in inspect.signature(torch.onnx.export).parameters:
        batch = torch.export.Dim("batch")
        torch.onnx.export(
            model, (dummy,), str(path),
            input_names=["input"], output_names=["output"],
            dynamic_shapes={"x": {0: batch}} if name == "resnet18" else ({0: batch},),
            opset_version=opset, dynamo=True,
        )
    else:
        torch.onnx.export(
            model, (dummy,), str(path),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset,
        )
    with torch.no_grad():
        out = model(dummy)
    return ModelSpec(name=name, input_shape=input_shape,
                     output_size=int(out.shape[1]), onnx_path=path, opset=opset)
