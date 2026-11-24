# Shared GPU benchmark standard (schema 1.0.0)

The cuDF repository is the pilot for a common result contract used by the cuDF, CUDA Data Residency Lab, and TensorRT benchmark projects. Each repository stores a copy of `benchmark-results.schema.json` so that every result can be validated locally and consumed by the same downstream tooling.

## Environment lock

- Python: CPython 3.12.x.
- GPU dataframe stack: `cudf-cu12==26.10.2` (CUDA 12 wheels); its direct CPU dependencies and development tools are pinned in the cuDF `pyproject.toml`, and the cuDF `uv.lock` pins that project’s complete resolved dependency graph, including wheel hashes. Each other project documents its own environment separately.
- CUDA/cuDF results must record the actual CUDA runtime and driver, GPU model, compute capability, memory, and installed package versions. A version pin does not replace NVIDIA's host-driver/GPU prerequisites.
- Use `uv sync --group dev` for the CPU baseline and `uv sync --group gpu --group dev` for the GPU run. Do not silently change a lockfile to match a particular machine; if the compatible runtime changes, update the explicit environment metadata and lock together.

## Run and comparison rules

- Make inputs deterministic and record seed, input dimensions, and every material configuration in `parameters`.
- Run CPU and GPU backends in separate fresh processes with the same seed, rows, and operation definitions. Do not compare different sizes or mix setup/transfer time into an operation timer without explicitly reporting it.
- Report all measured samples in seconds, warm-up count, and median/mean/standard deviation/min/max plus p95 and p99 latency. Record the successful output rows used in throughput and per-inference metrics. Percentiles use the nearest-rank definition.
- Synchronize the GPU before starting and after each timed operation; otherwise CUDA's asynchronous execution can make timings misleading.
- Store the result as JSON conforming to `benchmark-results.schema.json`. Record the source commit and dirty working-tree state. Never fill missing measurements with estimates or publish a result as GPU-measured unless it was actually captured on a GPU.
- For performance claims, preserve the raw result files and compare medians on the same hardware/software environment. Treat one-off values without raw samples and environment metadata as historical, not reproducible measurements.

## Energy measurement

- When available, use NVML's cumulative GPU energy counter, whose unit is millijoules, and record the measured counter delta, GPU identity, elapsed interval, and counter method. Unsupported devices/counters must be marked unavailable; do not substitute a short power sample or a model-based estimate.
- Measure an idle baseline separately. Report raw GPU-board joules per successful inference and the idle-baseline-adjusted value as distinct metrics; never silently replace raw energy with the adjusted value.
- NVML energy describes the physical GPU board and can include activity from other processes or co-tenants. It is not process-level energy or whole-system energy. Do not attribute a remote Triton server's GPU energy to a client-side GPU.
- Keep CPU/host energy out of GPU-only measurements. Any broader system-energy comparison needs a separately identified, validated measurement source and power boundary.

## Result contract

Each run contains the schema version, benchmark/run IDs, UTC timestamp, Git revision, host/runtime/software/GPU metadata, workload parameters, and one measurement record per operation/backend. Failed or skipped measurements should say so rather than reporting a fabricated timing. GPU-only unavailable hardware is an environment limitation, not a zero-second result.
