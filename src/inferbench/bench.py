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

import platform
import statistics
import time
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class LatencyStats:
    backend: str
    batch: int
    iterations: int
    warmup: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    throughput_ips: float          # inferences per second
    samples_seconds: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchResult:
    backend: str
    available: bool
    reason: str = ""
    latency: LatencyStats | None = None
    error: str = ""


def host_info() -> dict:
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def measure(backend, spec, batch: int = 1, iterations: int = 100,
            warmup: int = 20, seed: int = 0) -> LatencyStats:
    x = spec.sample(batch, seed=seed)

    for _ in range(warmup):
        backend.infer(x)

    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        backend.infer(x)
        samples.append((time.perf_counter_ns() - t0) / 1e6)

    samples.sort()

    def pct(p: float) -> float:
        return samples[min(len(samples) - 1, int(p * len(samples)))]

    mean = statistics.fmean(samples)
    return LatencyStats(
        backend=backend.name, batch=batch, iterations=iterations, warmup=warmup,
        mean_ms=mean, p50_ms=pct(0.50), p90_ms=pct(0.90), p99_ms=pct(0.99),
        min_ms=samples[0], max_ms=samples[-1],
        # Population stdev: these are the complete samples collected for this
        # benchmark run, and report.py uses the same definition.
        stdev_ms=statistics.pstdev(samples),
        throughput_ips=batch / (mean / 1000.0),
        samples_seconds=[sample / 1000.0 for sample in samples],
    )


def run_backend(backend, spec, batch: int, iterations: int, warmup: int) -> BenchResult:
    avail = backend.available()
    if not avail.ok:
        return BenchResult(backend=backend.name, available=False, reason=avail.reason)
    try:
        backend.load(spec, batch)
        stats = measure(backend, spec, batch, iterations, warmup)
        return BenchResult(backend=backend.name, available=True, latency=stats)
    except Exception as exc:               # a backend that breaks says so
        return BenchResult(backend=backend.name, available=True,
                           error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            backend.close()
        except Exception:
            pass
