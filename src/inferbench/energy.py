"""GPU-board energy measurement using NVML's cumulative energy counter.

The counter is in millijoules and measures the whole physical GPU, not just a
process. Unsupported devices and NVML errors are reported rather than replaced
with estimates. See the NVIDIA NVML Device Queries reference for support and
units: https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, TypeVar

T = TypeVar("T")


class EnergyUnavailableError(RuntimeError):
    """Raised when this host cannot expose a trustworthy GPU energy counter."""


@dataclass(frozen=True)
class EnergyCapture:
    status: str
    method: str = "nvml_total_energy_millijoules"
    gpu_index: int | None = None
    gpu_name: str | None = None
    gpu_uuid: str | None = None
    joules: float | None = None
    duration_seconds: float | None = None
    counter_reads: int = 0
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class NvidiaEnergyCounter:
    """Capture GPU-board energy deltas with NVML's cumulative mJ counter."""

    method = "nvml_total_energy_millijoules"

    def __init__(self, gpu_index: int = 0, nvml_module=None):
        if gpu_index < 0:
            raise ValueError("gpu_index must be non-negative")
        if nvml_module is None:
            try:
                import pynvml as nvml_module  # provided by optional nvidia-ml-py
            except ImportError as exc:
                raise EnergyUnavailableError(
                    "optional nvidia-ml-py package is not installed"
                ) from exc
        self._nvml = nvml_module
        self.gpu_index = gpu_index
        self._initialized = False
        try:
            self._nvml.nvmlInit()
            self._initialized = True
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(gpu_index)
            getter = getattr(self._nvml, "nvmlDeviceGetTotalEnergyConsumption", None)
            if getter is None:
                raise EnergyUnavailableError(
                    "NVML binding does not expose the total-energy counter"
                )
            self._energy_getter = getter
            self._read_millijoules()  # probe support before starting a benchmark
            self.gpu_name = self._decode(self._nvml.nvmlDeviceGetName(self._handle))
            try:
                self.gpu_uuid = self._decode(self._nvml.nvmlDeviceGetUUID(self._handle))
            except Exception:
                self.gpu_uuid = None
        except EnergyUnavailableError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise EnergyUnavailableError(
                f"NVML energy counter unavailable for GPU {gpu_index}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _decode(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _read_millijoules(self) -> int:
        value = self._energy_getter(self._handle)
        value = int(value)
        if value < 0:
            raise EnergyUnavailableError("NVML returned a negative energy counter")
        return value

    def capture(self, operation: Callable[[], T]) -> tuple[T, EnergyCapture]:
        """Run an operation once and return its result plus a measured energy delta.

        If the counter fails during capture, the operation still runs exactly
        once and receives an explicit unavailable energy record.
        """
        try:
            before_mj = self._read_millijoules()
        except Exception as exc:
            result = operation()
            return result, EnergyCapture(
                status="unavailable", gpu_index=self.gpu_index,
                gpu_name=self.gpu_name, gpu_uuid=self.gpu_uuid,
                reason=f"energy counter read failed before measurement: {type(exc).__name__}: {exc}",
            )

        started = perf_counter()
        result = operation()
        elapsed = perf_counter() - started
        try:
            after_mj = self._read_millijoules()
            if after_mj < before_mj:
                raise EnergyUnavailableError("NVML energy counter moved backwards")
            if after_mj == before_mj:
                raise EnergyUnavailableError(
                    "NVML energy counter did not advance during measurement; "
                    "the workload may be below the counter resolution"
                )
        except Exception as exc:
            return result, EnergyCapture(
                status="unavailable", gpu_index=self.gpu_index,
                gpu_name=self.gpu_name, gpu_uuid=self.gpu_uuid,
                duration_seconds=elapsed, counter_reads=1,
                reason=f"energy result unavailable after workload: {type(exc).__name__}: {exc}",
            )
        return result, EnergyCapture(
            status="measured", gpu_index=self.gpu_index,
            gpu_name=self.gpu_name, gpu_uuid=self.gpu_uuid,
            joules=(after_mj - before_mj) / 1000.0,
            duration_seconds=elapsed, counter_reads=2,
        )

    def close(self) -> None:
        if self._initialized:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False

    def __enter__(self) -> "NvidiaEnergyCounter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
