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

import hashlib
import os
import stat
from pathlib import Path
import tempfile

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
response_cache {{
  enable: {response_cache}
}}
"""


def copy_plan(source: Path, destination: Path) -> Path:
    """Atomically copy a readable engine plan and verify its bytes.

    An existing destination directory receives ``source.name``.  Same-file
    copies are a verified no-op; all other failures include both paths.
    """
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"TensorRT plan source is not a file: {source}")
    if destination.exists() and destination.is_dir():
        destination = destination / source.name
    destination = destination.absolute()
    try:
        if destination.exists() and source.samefile(destination):
            with source.open("rb") as src:
                src.read(1)  # A same-file no-op still requires a readable source.
            return destination
    except OSError as exc:
        raise OSError(f"cannot compare TensorRT plan {source} to {destination}: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        digest = hashlib.sha256()
        with source.open("rb") as src, temporary.open("wb") as dst:
            while chunk := src.read(1024 * 1024):
                digest.update(chunk)
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        os.chmod(temporary, stat.S_IMODE(source.stat().st_mode) & 0o777)
        copied_digest = hashlib.sha256()
        with temporary.open("rb") as copied:
            while chunk := copied.read(1024 * 1024):
                copied_digest.update(chunk)
        if copied_digest.digest() != digest.digest():
            raise IOError(f"byte verification failed copying {source} to {destination}")
        os.replace(temporary, destination)
        temporary = None
    except OSError as exc:
        raise OSError(f"cannot copy TensorRT plan {source} to {destination}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return destination


def render_config(spec, max_batch_size: int = 32, instances: int = 1,
                  platform: str = "onnxruntime_onnx", kind: str = "KIND_GPU",
                  queue_delay: int = 100, response_cache: bool = False) -> str:
    if queue_delay < 0:
        raise ValueError("queue_delay must be non-negative")
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
        response_cache=str(response_cache).lower(),
    )


def write_repository(spec, root: Path, max_batch_size: int = 32,
                     instances: int = 1, platform: str = "onnxruntime_onnx",
                     version: int = 1, plan_path: Path | None = None,
                     queue_delay: int = 100, response_cache: bool = False) -> Path:
    root = Path(root)
    model_dir = root / spec.name
    (model_dir / str(version)).mkdir(parents=True, exist_ok=True)

    filename = "model.onnx" if platform == "onnxruntime_onnx" else "model.plan"
    if platform == "onnxruntime_onnx":
        copy_plan(spec.onnx_path, model_dir / str(version) / filename)
    elif platform == "tensorrt_plan":
        if plan_path is None:
            raise ValueError("plan_path is required when platform='tensorrt_plan'")
        copy_plan(plan_path, model_dir / str(version) / filename)
    else:
        raise ValueError(f"unsupported Triton platform: {platform}")

    (model_dir / "config.pbtxt").write_text(
        render_config(spec, max_batch_size, instances, platform,
                      queue_delay=queue_delay, response_cache=response_cache),
        encoding="utf-8")
    return root
