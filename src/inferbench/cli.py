"""CLI.

    inferbench export
    inferbench bench --batch 1 --iterations 200
    inferbench triton-repo --out model_repository
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import accuracy as acc_mod
from . import report
from .backends import REFERENCE, all_backends
from .bench import host_info, run_backend
from .energy import EnergyUnavailableError, NvidiaEnergyCounter
from .model import export_onnx


def _cmd_export(args) -> int:
    spec = export_onnx(args.model, args.artifacts, opset=args.opset)
    print(f"exported {spec.name} -> {spec.onnx_path} "
          f"(input {spec.input_shape}, output {spec.output_size}, opset {spec.opset})")
    return 0


def _cmd_bench(args) -> int:
    if args.energy and args.energy_idle_seconds <= 0:
        raise ValueError("--energy-idle-seconds must be positive")
    if args.triton_max_queue_delay_us is not None and args.triton_max_queue_delay_us < 0:
        raise ValueError("--triton-max-queue-delay-us must be non-negative")
    if args.triton_response_cache == "enabled" and (
            args.triton_cache_size_bytes is None or args.triton_cache_size_bytes < 1024):
        raise ValueError("enabled Triton response cache requires --triton-cache-size-bytes >= 1024")
    if args.triton_response_cache == "disabled" and args.triton_cache_size_bytes is not None:
        raise ValueError("--triton-cache-size-bytes is inconsistent with disabled response cache")
    spec = export_onnx(args.model, args.artifacts, opset=args.opset)
    backends = all_backends(args.triton_url)
    if args.only:
        wanted = set(args.only.split(","))
        backends = [b for b in backends if b.name in wanted]

    energy_context = None
    energy_counter = None
    if args.energy:
        energy_context = {
            "requested": True,
            "scope": "gpu_board",
            "gpu_index": args.energy_gpu,
            "idle_baseline_seconds_requested": args.energy_idle_seconds,
            "assume_triton_local_gpu": args.assume_triton_local_gpu,
        }
        try:
            energy_counter = NvidiaEnergyCounter(args.energy_gpu)
            _, idle = energy_counter.capture(lambda: time.sleep(args.energy_idle_seconds))
            energy_context["idle_baseline"] = idle.as_dict()
            if idle.status != "measured":
                energy_context["unavailable_reason"] = idle.reason
        except EnergyUnavailableError as exc:
            energy_context["unavailable_reason"] = str(exc)
            print(f"GPU energy unavailable: {exc}")

    def is_local_gpu(name: str) -> bool:
        return (name in {"pytorch-cuda", "onnxruntime-cuda", "onnxruntime-tensorrt"}
                or name.startswith("tensorrt-")
                or (name == "triton-http" and args.assume_triton_local_gpu))

    results = [run_backend(
        b, spec, args.batch, args.iterations, args.warmup,
        energy_meter=(energy_counter if energy_counter and is_local_gpu(b.name) else None),
        request_pattern=args.request_pattern)
        for b in backends]

    # accuracy: every runnable backend against the PyTorch CPU reference
    ref_out = None
    for b in all_backends(args.triton_url):
        if b.name != REFERENCE:
            continue
        if b.available().ok:
            b.load(spec, args.accuracy_batch)
            ref_out = b.infer(spec.sample(args.accuracy_batch, seed=1234))
            b.close()

    accuracies: dict = {}
    if ref_out is not None:
        x = spec.sample(args.accuracy_batch, seed=1234)
        for b in all_backends(args.triton_url):
            if b.name == REFERENCE or not b.available().ok:
                continue
            if args.only and b.name not in set(args.only.split(",")):
                continue
            try:
                b.load(spec, args.accuracy_batch)
                accuracies[b.name] = acc_mod.compare(b.name, b.infer(x), ref_out, REFERENCE)
            except Exception as exc:
                print(f"  accuracy for {b.name} failed: {type(exc).__name__}: {exc}")
            finally:
                try:
                    b.close()
                except Exception:
                    pass

    host = host_info()
    print()
    print(report.table(results, accuracies, host, REFERENCE, energy_context))

    payload = report.standard_payload(
        spec.name, args.batch, args.iterations, args.warmup,
        host, results, REFERENCE, accuracies, energy_context,
        {"request_pattern": args.request_pattern,
         "triton_server": {"url": args.triton_url,
                           "response_cache": args.triton_response_cache,
                           "cache_size_bytes": args.triton_cache_size_bytes,
                           "max_queue_delay_microseconds": args.triton_max_queue_delay_us}})
    out = report.write_json(Path(args.out) / f"bench-b{args.batch}.json", payload)
    print(f"\nwrote {out}")
    if energy_counter:
        energy_counter.close()
    return 0


def _cmd_triton_repo(args) -> int:
    from .triton_repo import write_repository

    if args.response_cache and args.cache_size_bytes < 1024:
        raise ValueError("response cache requires --cache-size-bytes of at least 1024")
    if not args.response_cache and args.cache_size_bytes:
        raise ValueError("--cache-size-bytes requires --response-cache")
    spec = export_onnx(args.model, args.artifacts, opset=args.opset)
    root = write_repository(spec, Path(args.out), max_batch_size=args.max_batch,
                            instances=args.instances, platform=args.platform,
                            plan_path=args.plan, queue_delay=args.max_queue_delay_us,
                            response_cache=args.response_cache)
    cache_arg = (f" --cache-config local,size={args.cache_size_bytes}"
                 if args.response_cache else "")
    print(f"Triton model repository: {root}")
    print(f"  serve with:  docker run --gpus=all --rm -p8000-8002:8000-8002 \\\n"
          f"                 -v {root.resolve()}:/models "
          f"nvcr.io/nvidia/tritonserver:24.08-py3 \\\n"
          f"                 tritonserver --model-repository=/models{cache_arg}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="inferbench")
    p.add_argument("--model", default="resnet18", choices=["resnet18", "tinycnn"])
    p.add_argument("--artifacts", default=Path("artifacts"), type=Path)
    p.add_argument("--opset", default=18, type=int,
                   help="the dynamo exporter emits 18 natively; forcing a lower "
                        "value makes onnx attempt a down-conversion that fails")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="export the model to ONNX")
    e.set_defaults(func=_cmd_export)

    b = sub.add_parser("bench", help="measure every available backend")
    b.add_argument("--batch", default=1, type=int)
    b.add_argument("--iterations", default=100, type=int)
    b.add_argument("--warmup", default=20, type=int)
    b.add_argument("--accuracy-batch", default=8, type=int)
    b.add_argument("--triton-url", default="localhost:8000")
    b.add_argument("--request-pattern", choices=["unique", "repeat"], default="unique",
                   help="use unique deterministic inputs or repeat one fixed batch each call")
    b.add_argument("--triton-response-cache", choices=["unknown", "disabled", "enabled"],
                   default="unknown", help="declared response-cache state of the running Triton server")
    b.add_argument("--triton-cache-size-bytes", type=int,
                   help="declared local Triton cache size for this benchmark run")
    b.add_argument("--triton-max-queue-delay-us", type=int,
                   help="declared running-server dynamic batching queue delay in microseconds")
    b.add_argument("--only", default="", help="comma-separated backend names")
    b.add_argument("--out", default=Path("results"), type=Path)
    b.add_argument("--energy", action="store_true",
                   help="measure NVIDIA GPU-board energy with the NVML cumulative counter")
    b.add_argument("--energy-gpu", default=0, type=int,
                   help="NVIDIA GPU index for the energy counter (default: 0)")
    b.add_argument("--energy-idle-seconds", default=2.0, type=float,
                   help="idle baseline duration before measured runs (default: 2 seconds)")
    b.add_argument("--assume-triton-local-gpu", action="store_true",
                   help="attribute local GPU-board energy to Triton; only use for a local server")
    b.set_defaults(func=_cmd_bench)

    t = sub.add_parser("triton-repo", help="generate a Triton model repository")
    t.add_argument("--out", default=Path("model_repository"), type=Path)
    t.add_argument("--max-batch", default=32, type=int)
    t.add_argument("--instances", default=1, type=int)
    t.add_argument("--max-queue-delay-us", default=100, type=int,
                   help="Triton dynamic batching queue delay in microseconds (default: 100)")
    t.add_argument("--response-cache", action="store_true",
                   help="enable Triton response cache for the model and local server cache")
    t.add_argument("--cache-size-bytes", default=0, type=int,
                   help="Triton local response-cache size in bytes (requires --response-cache)")
    t.add_argument("--platform", default="onnxruntime_onnx",
                   choices=["onnxruntime_onnx", "tensorrt_plan"])
    t.add_argument("--plan", type=Path,
                   help="source TensorRT engine plan; required with --platform tensorrt_plan")
    t.set_defaults(func=_cmd_triton_repo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
