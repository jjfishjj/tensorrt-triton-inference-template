"""Generate a Triton model repository from the exported model.

Triton's layout is rigid and its `config.pbtxt` is easy to get subtly wrong -
the two mistakes that cost the most time are worth spelling out, because both
produce a server that starts happily and then behaves oddly:

* **`max_batch_size > 0` means the batch dimension is implicit.** `dims` must
  then describe one sample without its batch axis. Writing the batch axis into
  `dims` as well gives a model that loads and rejects every request.
* **`dynamic_batching` is what makes a server fast**, and it is off by default.
  Triton will queue up to `max_queue_delay_microseconds` to assemble a larger
  batch; that delay is a latency/throughput dial, not a formality.
"""
from __future__ import annotations

from pathlib import Path
from shutil import copyfile

CONFIG = """\
name: "{name}"
platform: "{platform}"
max_batch_size: {max_batch}

input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [ {input_dims} ]
  }}
]

output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ {output_dims} ]
  }}
]

instance_group [
  {{
    count: {instances}
    kind: {kind}
  }}
]

# Off by default. Without it Triton runs one request at a time and the GPU
# spends most of its life idle between small requests.
dynamic_batching {{
  preferred_batch_size: [ {preferred} ]
  max_queue_delay_microseconds: {queue_delay}
}}
"""


def render_config(spec, max_batch_size: int = 32, instances: int = 1,
                  platform: str = "onnxruntime_onnx", kind: str = "KIND_GPU",
                  queue_delay: int = 100) -> str:
    preferred = [b for b in (4, 8, 16, 32) if b <= max_batch_size] or [max_batch_size]
    return CONFIG.format(
        name=spec.name, platform=platform, max_batch=max_batch_size,
        input_name="input", output_name="output",
        # max_batch_size > 0 => the batch axis is implicit and must not appear here
        input_dims=", ".join(str(d) for d in spec.input_shape),
        output_dims=str(spec.output_size),
        instances=instances, kind=kind,
        preferred=", ".join(str(b) for b in preferred),
        queue_delay=queue_delay,
    )


def write_repository(spec, root: Path, max_batch_size: int = 32,
                     instances: int = 1, platform: str = "onnxruntime_onnx",
                     version: int = 1) -> Path:
    root = Path(root)
    model_dir = root / spec.name
    (model_dir / str(version)).mkdir(parents=True, exist_ok=True)

    filename = "model.onnx" if platform == "onnxruntime_onnx" else "model.plan"
    if platform == "onnxruntime_onnx":
        copyfile(spec.onnx_path, model_dir / str(version) / filename)

    (model_dir / "config.pbtxt").write_text(
        render_config(spec, max_batch_size, instances, platform),
        encoding="utf-8")
    return root
