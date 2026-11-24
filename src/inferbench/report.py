"""Rendering results as a table and as JSON.

The table is the deliverable. Unavailable backends keep their row and carry
their reason, because a comparison that silently drops what it could not run is
how four invented numbers end up in a slide.
"""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version as package_version, PackageNotFoundError
from pathlib import Path
from uuid import uuid4


def energy_record(result, context: dict | None) -> dict | None:
    if not context or not context.get("requested"):
        return None
    backend = result.backend
    local_gpu = (backend in {"pytorch-cuda", "onnxruntime-cuda",
                             "onnxruntime-tensorrt"} or
                 backend.startswith("tensorrt-"))
    if backend == "triton-http" and not context.get("assume_triton_local_gpu", False):
        return {"status": "not_applicable", "scope": "gpu_board",
                "reason": "Triton server location is unknown; client-side GPU energy is not server energy"}
    if backend == "triton-http" and context.get("assume_triton_local_gpu", False):
        local_gpu = True
    if not local_gpu:
        return {"status": "not_applicable", "scope": "gpu_board",
                "reason": "this backend does not represent a local NVIDIA GPU run; host/CPU energy is not measured"}
    if not result.available:
        return {"status": "skipped", "scope": "gpu_board", "reason": result.reason}
    if result.error:
        return {"status": "failed", "scope": "gpu_board", "reason": result.error}

    capture = result.energy
    if capture is None or capture.status != "measured":
        reason = capture.reason if capture is not None else context.get(
            "unavailable_reason", "energy capture did not run")
        return {"status": "unavailable", "scope": "gpu_board", "reason": reason}

    idle = context.get("idle_baseline") or {}
    idle_joules = idle.get("joules") if idle.get("status") == "measured" else None
    idle_duration = idle.get("duration_seconds") if idle.get("status") == "measured" else None
    idle_watts = (idle_joules / idle_duration
                  if idle_joules is not None and idle_duration else None)
    successful = result.latency.successful_inferences if result.latency else 0
    duration = capture.duration_seconds or 0.0
    adjusted_joules = (max(0.0, capture.joules - idle_watts * duration)
                       if idle_watts is not None else None)
    return {
        "status": "measured", "scope": "gpu_board",
        "method": capture.method,
        "gpu": {"index": capture.gpu_index, "name": capture.gpu_name,
                "uuid": capture.gpu_uuid},
        "board_energy_joules": capture.joules,
        "joules_per_successful_inference": (capture.joules / successful
                                              if successful else None),
        "idle_baseline_joules": idle_joules,
        "idle_baseline_duration_seconds": idle_duration,
        "idle_power_watts": idle_watts,
        "idle_adjusted_joules_per_successful_inference": (
            adjusted_joules / successful if adjusted_joules is not None and successful else None),
        "successful_inferences": successful,
        "duration_seconds": duration,
        "counter_reads": capture.counter_reads,
        "notes": ("NVML measures the whole physical GPU board; unrelated processes or "
                  "co-tenants contribute. Idle-adjusted energy subtracts the measured "
                  "idle baseline and is shown separately from raw board energy."),
    }


def table(results, accuracies: dict, host: dict, reference: str,
          energy_context: dict | None = None) -> str:
    acc = {a.backend: a for a in accuracies.values()}
    w = 22
    lines = [
        f"host: {host['processor']} · {host['platform']}",
        "",
        f"{'backend':<{w}} {'p95 ms':>8} {'p99 ms':>8} {'inf/s':>9} {'J/infer':>9} "
        f"{'max |Δ|':>10} {'top-1':>7}",
        "-" * (w + 66),
    ]
    for r in results:
        if not r.available:
            lines.append(f"{r.backend:<{w}} {'—':>8} {'—':>8} {'—':>9} {'—':>9}   "
                         f"unavailable: {r.reason}")
            continue
        if r.error:
            lines.append(f"{r.backend:<{w}} {'—':>8} {'—':>8} {'—':>9} {'—':>9}   "
                         f"error: {r.error}")
            continue
        s = r.latency
        a = acc.get(r.backend)
        energy = energy_record(r, energy_context)
        joules = (energy.get("joules_per_successful_inference")
                  if energy and energy.get("status") == "measured" else None)
        energy_text = f"{joules:.3g}" if joules is not None else "—"
        adiff = f"{a.max_abs_diff:.2e}" if a else ("reference" if r.backend == reference else "—")
        top1 = f"{a.top1_agreement * 100:.1f}%" if a else ("—" if r.backend != reference else "—")
        lines.append(f"{r.backend:<{w}} {s.p95_ms:8.2f} {s.p99_ms:8.2f} "
                     f"{s.throughput_ips:9.1f} {energy_text:>9} {adiff:>10} {top1:>7}")
    return "\n".join(lines)


def write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _git_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, False
    return {"commit": commit, "dirty": dirty}


def _installed(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _gpu_info() -> tuple[str | None, str | None, list[dict]]:
    """Best-effort metadata probe; absent CUDA is represented explicitly."""
    runtime = driver = None
    gpus: list[dict] = []
    try:
        import torch
        runtime = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                capability = f"{props.major}.{props.minor}"
                gpus.append({"index": index, "name": props.name,
                             "compute_capability": capability,
                             "memory_bytes": int(props.total_memory)})
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        query = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
        rows = [line.strip().split(", ") for line in query.splitlines() if line.strip()]
        driver = rows[0][2] if rows and len(rows[0]) >= 3 else None
        for index, row in enumerate(rows):
            if index >= len(gpus) and len(row) >= 2:
                memory = int(float(row[1]) * 1024 * 1024)
                gpus.append({"index": index, "name": row[0],
                             "compute_capability": None, "memory_bytes": memory})
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return runtime, driver, gpus


def standard_payload(model: str, batch: int, iterations: int, warmup: int,
                     host: dict, results, reference: str, accuracies: dict,
                     energy_context: dict | None = None,
                     run_parameters: dict | None = None) -> dict:
    """Build the shared benchmark-results.schema.json envelope.

    Timing samples are retained in seconds. Unavailable backends are skipped,
    never represented as zeroes or estimated GPU measurements.
    """
    measurements = []
    for result in results:
        if result.latency is not None and not result.error:
            stats = result.latency
            samples = list(stats.samples_seconds)
            if not samples:
                raise ValueError(f"completed result {result.backend} has no samples")
            measurements.append({
                "case_id": f"{model}-b{batch}", "backend": result.backend,
                "status": "completed", "unit": "seconds", "input_rows": batch,
                "output_rows": stats.successful_inferences // max(1, iterations),
                "successful_inferences": stats.successful_inferences,
                "warmup_iterations": warmup,
                "samples": samples,
                "statistics": {"median": statistics.median(samples),
                                "mean": stats.mean_ms / 1000,
                                "stdev": statistics.pstdev(samples),
                                "min": stats.min_ms / 1000,
                                "max": stats.max_ms / 1000,
                                "p95": stats.p95_ms / 1000,
                                "p99": stats.p99_ms / 1000,
                                "throughput_inferences_per_second": stats.throughput_ips},
                "notes": ("raw wall-clock samples in seconds; stdev is population "
                          "stdev (statistics.pstdev); percentiles use nearest rank. "
                          "PyTorch synchronizes CUDA/MPS "
                          "before return; TensorRT synchronizes its CUDA stream; "
                          "ONNX Runtime and Triton calls are blocking."),
            })
        else:
            status = "skipped" if not result.available else "failed"
            reason = result.reason or result.error or "backend unavailable"
            measurements.append({
                "case_id": f"{model}-b{batch}", "backend": result.backend,
                "status": status, "unit": "seconds", "input_rows": batch,
                "output_rows": None, "warmup_iterations": warmup, "samples": [],
                "statistics": {"median": None, "mean": None, "stdev": None,
                                "min": None, "max": None}, "notes": reason,
            })
        measured_energy = energy_record(result, energy_context)
        if measured_energy is not None:
            measurements[-1]["energy"] = measured_energy
    cuda_runtime, cuda_driver, gpus = _gpu_info()
    return {
        "schema_version": "1.0.0", "benchmark_id": "tensorrt-triton-inference",
        "run_id": uuid4().hex, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(),
        "environment": {
            "os": platform.system(), "os_release": platform.release(),
            "architecture": host.get("machine", platform.machine()),
            "python": host.get("python", platform.python_version()),
            "cuda_runtime": cuda_runtime, "cuda_driver": cuda_driver,
            "software": {name: _installed(name) for name in
                          ("torch", "torchvision", "onnx", "onnxruntime", "numpy",
                           "tensorrt", "cuda-python", "tritonclient", "nvidia-ml-py")},
            "gpus": gpus,
        },
        "parameters": {"model": model, "batch": batch, "iterations": iterations,
                        "warmup": warmup, "reference": reference,
                        "accuracy": {k: v.as_dict() for k, v in accuracies.items()},
                        "energy": energy_context,
                        **(run_parameters or {})},
        "measurements": measurements,
    }
