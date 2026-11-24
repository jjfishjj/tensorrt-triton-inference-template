"""CLI.

    inferbench export
    inferbench bench --batch 1 --iterations 200
    inferbench triton-repo --out model_repository
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import accuracy as acc_mod
from . import report
from .backends import REFERENCE, all_backends
from .bench import host_info, run_backend
from .model import export_onnx


def _cmd_export(args) -> int:
    spec = export_onnx(args.model, args.artifacts, opset=args.opset)
    print(f"exported {spec.name} -> {spec.onnx_path} "
          f"(input {spec.input_shape}, output {spec.output_size}, opset {spec.opset})")
    return 0


def _cmd_bench(args) -> int:
    spec = export_onnx(args.model, args.artifacts, opset=args.opset)
    backends = all_backends(args.triton_url)
    if args.only:
        wanted = set(args.only.split(","))
        backends = [b for b in backends if b.name in wanted]

    results = [run_backend(b, spec, args.batch, args.iterations, args.warmup)
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
    print(report.table(results, accuracies, host, REFERENCE))

    payload = report.standard_payload(
        spec.name, args.batch, args.iterations, args.warmup,
        host, results, REFERENCE, accuracies)
    out = report.write_json(Path(args.out) / f"bench-b{args.batch}.json", payload)
    print(f"\nwrote {out}")
    return 0


def _cmd_triton_repo(args) -> int:
    from .triton_repo import write_repository

    spec = export_onnx(args.model, args.artifacts, opset=args.opset)
    root = write_repository(spec, Path(args.out), max_batch_size=args.max_batch,
                            instances=args.instances, platform=args.platform,
                            plan_path=args.plan)
    print(f"Triton model repository: {root}")
    print(f"  serve with:  docker run --gpus=all --rm -p8000-8002:8000-8002 \\\n"
          f"                 -v {root.resolve()}:/models "
          f"nvcr.io/nvidia/tritonserver:24.08-py3 \\\n"
          f"                 tritonserver --model-repository=/models")
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
    b.add_argument("--only", default="", help="comma-separated backend names")
    b.add_argument("--out", default=Path("results"), type=Path)
    b.set_defaults(func=_cmd_bench)

    t = sub.add_parser("triton-repo", help="generate a Triton model repository")
    t.add_argument("--out", default=Path("model_repository"), type=Path)
    t.add_argument("--max-batch", default=32, type=int)
    t.add_argument("--instances", default=1, type=int)
    t.add_argument("--platform", default="onnxruntime_onnx",
                   choices=["onnxruntime_onnx", "tensorrt_plan"])
    t.add_argument("--plan", type=Path,
                   help="source TensorRT engine plan; required with --platform tensorrt_plan")
    t.set_defaults(func=_cmd_triton_repo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
