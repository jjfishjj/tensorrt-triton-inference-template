"""Triton Inference Server, over HTTP.

Measures something the other backends cannot: what a caller across the network
actually waits for. Triton's own compute time is TensorRT's, so anything above
that number is serialisation, HTTP and scheduling - which is exactly what a
deployment decision turns on.
"""
from __future__ import annotations

import numpy as np

from .base import Availability


class TritonBackend:
    def __init__(self, url: str = "localhost:8000", model: str | None = None,
                 model_version: str = "1"):
        self.url, self.model, self.model_version = url, model, model_version
        self.name = "triton-http"
        self._client = None

    def available(self) -> Availability:  # type: ignore[override]
        try:
            import tritonclient.http as httpclient
        except ImportError as exc:
            return Availability(False, f"tritonclient not installed ({exc})")
        try:
            c = httpclient.InferenceServerClient(url=self.url, verbose=False)
            if not c.is_server_live():
                return Availability(False, f"server at {self.url} not live")
        except Exception as exc:
            return Availability(False, f"cannot reach {self.url}: {exc}")
        return Availability(True)

    def load(self, spec, batch: int) -> None:
        import tritonclient.http as httpclient

        self._http = httpclient
        self._client = httpclient.InferenceServerClient(url=self.url)
        self.model = self.model or spec.name
        if not self._client.is_model_ready(self.model, self.model_version):
            self._client.load_model(self.model)
        meta = self._client.get_model_metadata(self.model, self.model_version)
        self._in_name = meta["inputs"][0]["name"]
        self._out_name = meta["outputs"][0]["name"]

    def infer(self, x: np.ndarray) -> np.ndarray:
        inp = self._http.InferInput(self._in_name, list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        out = self._http.InferRequestedOutput(self._out_name)
        # A blocking call: the response is the completed inference.
        res = self._client.infer(self.model, [inp], outputs=[out],
                                 model_version=self.model_version)
        return res.as_numpy(self._out_name)

    def close(self) -> None:
        self._client = None
