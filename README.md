# tensorrt-triton-inference-template

A benchmark harness for inference runtimes that **refuses to make numbers up**.

Point it at a model and it measures every runtime the machine can actually run —
PyTorch, ONNX Runtime, TensorRT, Triton — reports latency percentiles and
throughput for each, and checks that the fast ones still compute the same answer
as the slow one. A runtime it cannot run gets a row saying *why*, never a figure.

That last rule is the reason this exists. Comparison tables of inference
backends are easy to write and almost never measured; this one starts empty and
fills in only what it observed.

## Quick start

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
PYTHONPATH=src ./.venv/bin/python -m inferbench.cli bench --batch 1
```

No GPU required — the harness measures whatever is present and says what is not.
On an NVIDIA box, add `-r requirements-gpu.txt` and the TensorRT and Triton rows
fill themselves in.

## Measured

Apple M1, macOS, ResNet-18, batch 1, 100 iterations after 20 warm-up:

```
backend                  p50 ms   p99 ms     inf/s    max |Δ|   top-1
--------------------------------------------------------------------
pytorch-cpu               23.82   111.18      36.4  reference       —
pytorch-mps                7.47    20.33     124.4   6.68e-06  100.0%
onnxruntime-cpu           36.00    90.27      24.3   6.20e-06  100.0%
onnxruntime-coreml         1.62    11.68     511.6   1.20e-02  100.0%
pytorch-cuda                  —        —         —   unavailable: no CUDA device
onnxruntime-cuda              —        —         —   unavailable: CUDAExecutionProvider not built into this wheel
tensorrt-fp16                 —        —         —   unavailable: tensorrt not installed
triton-http                   —        —         —   unavailable: tritonclient not installed
```

### The ranking changes with batch size

| backend | b=1 p50 | b=8 inf/s | b=32 inf/s |
| --- | ---: | ---: | ---: |
| pytorch-cpu | 23.82 ms | 47.4 | 20.5 |
| pytorch-mps | 7.47 ms | 253.9 | 222.2 |
| onnxruntime-cpu | 36.00 ms | 25.9 | 24.9 |
| onnxruntime-coreml | **1.62 ms** | **1256.8** | **1583.0** |

ONNX Runtime on CPU is the **slowest** backend at batch 1 and **beats PyTorch
CPU** at batch 32. PyTorch CPU's throughput peaks at batch 8 and then falls.
A single headline number cannot express either of those, which is why the
harness separates latency (batch 1 — what one caller waits) from throughput
(large batch — what the hardware delivers) and sweeps between them.

### Speed has a precision cost, and it is measurable

CoreML is 15× faster than the PyTorch CPU reference. It is also the only backend
whose numerical drift is visible:

| backend | max abs difference | top-1 agreement |
| --- | ---: | ---: |
| pytorch-mps | 6.68e-06 | 100% |
| onnxruntime-cpu | 6.20e-06 | 100% |
| onnxruntime-coreml | **1.20e-02** | 100% |

Three to four orders of magnitude more drift than the FP32 backends, because the
Apple Neural Engine computes in reduced precision. Top-1 still agrees here, but
that is a property of this model and this input — not a guarantee. Every speed
number in this repository is paired with the drift that bought it, and the same
column is what will hold TensorRT's FP16 and INT8 engines to account.

## Two bugs the harness caught in itself

**Every backend ran a different model.** The first run reported `max |Δ| = 9.65`
and **0% top-1 agreement** for all three backends. Not drift — the model factory
initialised fresh random weights on every call, so the ONNX file, the reference
and each backend held different networks. The accuracy column existed for
exactly one run before it earned its keep. `build_torch_model` now seeds, and a
test asserts two builds are bit-identical.

**Async runtimes lie about latency.** A GPU backend that returns as soon as work
is queued measures the enqueue, not the compute — and looks spectacular. Every
backend's `infer()` is required to synchronise before returning: `mps.synchronize`,
`cuda.synchronize`, `cudaStreamSynchronize` for TensorRT. Triton and ONNX Runtime
are blocking already.

## The GPU path

Both GPU backends are written and neither has been run — there is no NVIDIA
device here, and the harness says so rather than estimating. On a CUDA box:

```bash
./.venv/bin/pip install -r requirements-gpu.txt
PYTHONPATH=src python -m inferbench.cli bench --batch 32     # trt rows fill in
PYTHONPATH=src python -m inferbench.cli triton-repo --out model_repository
docker run --gpus=all --rm -p8000-8002:8000-8002 \
  -v $PWD/model_repository:/models nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models
PYTHONPATH=src python -m inferbench.cli bench --batch 32 --only triton-http
```

`tensorrt-fp16` builds the engine from the ONNX graph directly rather than going
through ONNX Runtime's TensorRT execution provider. The EP is more convenient,
but it silently falls back to CUDA for any layer TensorRT will not take, which
mixes two runtimes into one measurement. Building the engine means what runs is
TensorRT or nothing.

## `config.pbtxt`, and the two ways to get it wrong

`inferbench triton-repo` generates the model repository. Both of these produce a
server that starts cleanly and then misbehaves:

- **`max_batch_size > 0` makes the batch axis implicit.** `dims` must then
  describe one sample *without* it. Include it and the model loads and rejects
  every request.
- **`dynamic_batching` is off by default**, and it is most of what makes a
  server fast. `max_queue_delay_microseconds` is a real latency/throughput dial,
  not boilerplate.

Both are covered by tests.

## Layout

```
src/inferbench/
  model.py            ResNet-18 + ONNX export, deterministic weights, dynamic batch
  backends/
    base.py           the contract: available() / load() / infer() / close()
    torch_backend.py  cpu · mps · cuda
    ort_backend.py    CPU · CoreML · CUDA · TensorRT execution providers
    trt_backend.py    native engine build + execution        (CUDA only)
    triton_backend.py HTTP client                            (server required)
  bench.py            warm-up, percentiles, latency vs throughput
  accuracy.py         cross-backend agreement against the reference
  triton_repo.py      model repository + config.pbtxt
  report.py           the table, and the JSON behind it
```

Results land in `results/bench-b{N}.json` with the host recorded alongside them,
because a latency number without the machine it was measured on is not a result.

## Status

- [x] Backend abstraction with honest availability reporting
- [x] PyTorch (CPU/MPS/CUDA) and ONNX Runtime (4 providers) backends
- [x] Latency percentiles, throughput, batch sweep
- [x] Cross-backend accuracy comparison
- [x] Triton model repository generation
- [x] Unit tests (9)
- [ ] TensorRT and Triton rows measured on real hardware
- [ ] INT8 calibration against a real validation set
- [ ] Nsight Systems trace capture alongside the timings
