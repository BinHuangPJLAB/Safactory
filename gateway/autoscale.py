from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatewayScaleSnapshot:
    inflight_requests: int
    active_streams: int
    queued_telemetry: int
    requests_per_second: float = 0.0
    p95_latency_ms: float = 0.0
    error_rate: float = 0.0


def collect_scale_snapshot(
    *,
    inflight_requests: int = 0,
    active_streams: int = 0,
    queued_telemetry: int = 0,
    requests_per_second: float = 0.0,
    p95_latency_ms: float = 0.0,
    error_rate: float = 0.0,
) -> GatewayScaleSnapshot:
    return GatewayScaleSnapshot(
        inflight_requests=inflight_requests,
        active_streams=active_streams,
        queued_telemetry=queued_telemetry,
        requests_per_second=requests_per_second,
        p95_latency_ms=p95_latency_ms,
        error_rate=error_rate,
    )


def recommend_replica_delta(snapshot: GatewayScaleSnapshot) -> int:
    if snapshot.error_rate >= 0.2 or snapshot.queued_telemetry >= 50000:
        return 1
    if snapshot.inflight_requests >= 1500 or snapshot.active_streams >= 800:
        return 1
    if snapshot.inflight_requests <= 128 and snapshot.active_streams <= 32 and snapshot.queued_telemetry <= 100:
        return -1
    return 0
