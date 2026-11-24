"""Latency and throughput measurement.

Three things separate this from timing a loop:

* **Warm-up is discarded.** The first calls pay for lazy kernel compilation,
  memory pool growth and, on some providers, graph capture. Including them
  makes a fast backend look slow and a cached one look magic.
* **Percentiles, not means.** A serving decision is made on p99. A mean hides
  exactly the tail that decides whether a latency budget holds.
* **Latency and throughput are measured separately.** Latency is batch 1 -
  what one caller waits. Throughput is the largest batch that fits - what the
  hardware delivers. Quoting one number for both is the most common way a
  benchmark misleads.
"""
from __future__ import annotations

import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field

from .energy import EnergyCapture


@dataclass(frozen=True)
class LatencyStats:
    backend: str
    batch: int
    iterations: int
    warmup: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    throughput_ips: float          # inferences per second
    successful_inferences: int = 0
    samples_seconds: list[float] = field(default_factory=list)
    energy_capture: EnergyCapture | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchResult:
    backend: str
    available: bool
    reason: str = ""
    latency: LatencyStats | None = None
    error: str = ""
    energy: EnergyCapture | None = None


def host_info() -> dict:
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _output_rows(output) -> int:
    shape = getattr(output, "shape", None)
    if shape is not None and len(shape):
        return int(shape[0])
    try:
        return len(output)
    except TypeError as exc:
        raise ValueError("backend output has no batch dimension") from exc


def measure(backend, spec, batch: int = 1, iterations: int = 100,
            warmup: int = 20, seed: int = 0,
            energy_meter=None, request_pattern: str = "unique") -> LatencyStats:
    if batch <= 0:
        raise ValueError("batch must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if request_pattern not in {"repeat", "unique"}:
        raise ValueError("request_pattern must be 'repeat' or 'unique'")
    repeated_input = spec.sample(batch, seed=seed)

    def request_at(index: int):
        if request_pattern == "repeat":
            return repeated_input
        return spec.sample(batch, seed=seed + index + 1)

    for warmup_index in range(warmup):
        backend.infer(request_at(warmup_index))

    samples: list[float] = []
    successful_inferences = 0

    def run_iterations() -> None:
        nonlocal successful_inferences
        for iteration in range(iterations):
            x = request_at(warmup + iteration)
            t0 = time.perf_counter_ns()
            output = backend.infer(x)
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            output_rows = _output_rows(output)
            if output_rows != batch:
                raise ValueError(
                    f"backend returned {output_rows} rows for a batch of {batch}"
                )
            successful_inferences += output_rows
            samples.append(elapsed)

    energy = None
    if energy_meter is None:
        run_iterations()
    else:
        _, energy = energy_meter.capture(run_iterations)

    samples.sort()

    def pct(p: float) -> float:
        # Nearest-rank percentile: rank = ceil(p*n), converted to a zero-based index.
        return samples[max(0, min(len(samples) - 1, math.ceil(p * len(samples)) - 1))]

    mean = statistics.fmean(samples)
    stats = LatencyStats(
        backend=backend.name, batch=batch, iterations=iterations, warmup=warmup,
        mean_ms=mean, p50_ms=pct(0.50), p90_ms=pct(0.90), p95_ms=pct(0.95),
        p99_ms=pct(0.99),
        min_ms=samples[0], max_ms=samples[-1],
        # Population stdev: these are the complete samples collected for this
        # benchmark run, and report.py uses the same definition.
        stdev_ms=statistics.pstdev(samples),
        throughput_ips=successful_inferences / (mean / 1000.0 * iterations),
        successful_inferences=successful_inferences,
        samples_seconds=[sample / 1000.0 for sample in samples],
        energy_capture=energy,
    )
    return stats


def run_backend(backend, spec, batch: int, iterations: int, warmup: int,
                energy_meter=None, request_pattern: str = "unique") -> BenchResult:
    avail = backend.available()
    if not avail.ok:
        return BenchResult(backend=backend.name, available=False, reason=avail.reason)
    try:
        backend.load(spec, batch)
        stats = measure(backend, spec, batch, iterations, warmup,
                        energy_meter=energy_meter,
                        request_pattern=request_pattern)
        return BenchResult(backend=backend.name, available=True,
                           latency=stats, energy=stats.energy_capture)
    except Exception as exc:               # a backend that breaks says so
        return BenchResult(backend=backend.name, available=True,
                           error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            backend.close()
        except Exception:
            pass
