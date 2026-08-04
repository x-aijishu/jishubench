"""edge-eval-agent monitor client (/v1/eval/*)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from host.config import Settings


@dataclass
class TraceRecord:
    trace_id: str
    benchmark: str
    sample_id: str
    model: str
    phase: str = "measured"
    extra: dict[str, Any] = field(default_factory=dict)


class EvalMonitorClient:
    """Monitor control and metrics readout for edge-eval-agent."""

    def __init__(self, settings: Settings) -> None:
        self._base = settings.target.monitor_base_url.rstrip("/")
        self._timeout = settings.target.monitor_timeout_s
        self._device_info: dict[str, Any] | None = None

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    @property
    def device_info(self) -> dict[str, Any] | None:
        """Return the last fetched DeviceInfo, or None if health() was never called."""
        return self._device_info

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(self._url("/v1/eval/health"))
            resp.raise_for_status()
            data = resp.json()
            self._device_info = data.get("device_info")
            return data

    def create_trace(self, record: TraceRecord) -> dict[str, Any]:
        payload = {
            "trace_id": record.trace_id,
            "benchmark": record.benchmark,
            "sample_id": record.sample_id,
            "model": record.model,
            "phase": record.phase,
            **record.extra,
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(self._url("/v1/eval/traces"), json=payload)
            resp.raise_for_status()
            return resp.json()

    def start_trace(self, trace_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(self._url(f"/v1/eval/traces/{trace_id}/start"))
            resp.raise_for_status()
            return resp.json()

    def mark(self, trace_id: str, stage: str, **extra: Any) -> dict[str, Any]:
        payload = {"stage": stage, **extra}
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                self._url(f"/v1/eval/traces/{trace_id}/mark"),
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    def stop_trace(self, trace_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(self._url(f"/v1/eval/traces/{trace_id}/stop"))
            resp.raise_for_status()
            return resp.json()

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(self._url(f"/v1/eval/traces/{trace_id}"))
            resp.raise_for_status()
            return resp.json()
