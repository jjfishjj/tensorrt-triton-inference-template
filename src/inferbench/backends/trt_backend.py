"""Native TensorRT: build an engine from the ONNX graph, then execute it.

Kept separate from ONNX Runtime's TensorRT execution provider on purpose. The
EP is convenient and falls back to CUDA for any layer TensorRT will not take,
which quietly mixes two runtimes in one measurement. Building the engine
directly means what runs is TensorRT or nothing.

Requires a CUDA device; reports itself unavailable everywhere else.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Availability


class TensorRTBackend:
    def __init__(self, precision: str = "fp16", workspace_gb: float = 4.0,
                 cache_dir: Path | None = None):
        self.precision = precision
        self.workspace_gb = workspace_gb
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/engines")
        self.name = f"tensorrt-{precision}"
        self._ctx = self._engine = self._stream = None
        self._bindings: dict = {}

    def available(self) -> Availability:  # type: ignore[override]
        try:
            import tensorrt  # noqa: F401
        except ImportError as exc:
            return Availability(False, f"tensorrt not installed ({exc})")
        try:
            from cuda import cudart
        except ImportError as exc:
            return Availability(False, f"cuda-python not installed ({exc})")
        err, n = cudart.cudaGetDeviceCount()
        if int(err) != 0 or n == 0:
            return Availability(False, "no CUDA device")
        return Availability(True)

    def load(self, spec, batch: int) -> None:
        import tensorrt as trt
        from cuda import cudart

        self._trt, self._cudart = trt, cudart
        logger = trt.Logger(trt.Logger.WARNING)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        plan = self.cache_dir / f"{spec.name}-{self.precision}-b{batch}.plan"

        if plan.exists():
            data = plan.read_bytes()
        else:
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(spec.onnx_path.read_bytes()):
                msgs = "; ".join(str(parser.get_error(i))
                                 for i in range(parser.num_errors))
                raise RuntimeError(f"ONNX parse failed: {msgs}")

            cfg = builder.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolFlag.WORKSPACE,
                                      int(self.workspace_gb * (1 << 30)))
            if self.precision == "fp16":
                cfg.set_flag(trt.BuilderFlag.FP16)
            elif self.precision == "int8":
                # Without a calibrator INT8 falls back per-layer, which is a
                # measurement of nothing in particular. Wire a calibrator to the
                # real validation set before trusting an INT8 number.
                cfg.set_flag(trt.BuilderFlag.INT8)

            profile = builder.create_optimization_profile()
            shape = (1, *spec.input_shape)
            profile.set_shape("input", min=shape,
                              opt=(batch, *spec.input_shape),
                              max=(batch, *spec.input_shape))
            cfg.add_optimization_profile(profile)

            data = builder.build_serialized_network(network, cfg)
            if data is None:
                raise RuntimeError("TensorRT engine build returned nothing")
            plan.write_bytes(data)

        runtime = trt.Runtime(logger)
        self._engine = runtime.deserialize_cuda_engine(data)
        self._ctx = self._engine.create_execution_context()
        self._ctx.set_input_shape("input", (batch, *spec.input_shape))
        _, self._stream = cudart.cudaStreamCreate()

        self._out = np.empty((batch, spec.output_size), dtype=np.float32)
        nbytes_in = batch * int(np.prod(spec.input_shape)) * 4
        _, self._d_in = cudart.cudaMalloc(nbytes_in)
        _, self._d_out = cudart.cudaMalloc(self._out.nbytes)
        self._nbytes_in = nbytes_in
        self._ctx.set_tensor_address("input", int(self._d_in))
        self._ctx.set_tensor_address("output", int(self._d_out))

    def infer(self, x: np.ndarray) -> np.ndarray:
        cudart = self._cudart
        kind_h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        kind_d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        cudart.cudaMemcpyAsync(self._d_in, x.ctypes.data, self._nbytes_in,
                               kind_h2d, self._stream)
        self._ctx.execute_async_v3(self._stream)
        cudart.cudaMemcpyAsync(self._out.ctypes.data, self._d_out, self._out.nbytes,
                               kind_d2h, self._stream)
        cudart.cudaStreamSynchronize(self._stream)   # measure compute, not enqueue
        return self._out.copy()

    def close(self) -> None:
        if self._stream is not None:
            self._cudart.cudaFree(self._d_in)
            self._cudart.cudaFree(self._d_out)
            self._cudart.cudaStreamDestroy(self._stream)
        self._ctx = self._engine = self._stream = None
