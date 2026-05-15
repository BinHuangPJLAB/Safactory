from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from gateway.admission_control import AdmissionController, AdmissionRejected
from gateway.config import GatewayConfig, load_gateway_config
from gateway.inference_forwarder import (
    InferenceForwarder,
    SessionClosedError,
    StreamForwardContext,
)
from gateway.llm_router import LLMRouteTarget, LLMRouter
from gateway.models import GatewayRequestContext, GatewaySessionBinding
from gateway.session_resolver import SessionResolver
from gateway.storage import GatewayStorage
from gateway.telemetry import StreamTelemetryStats, TelemetryRecorder

log = logging.getLogger("gateway.app")


def create_app(cfg: GatewayConfig | None = None, storage: GatewayStorage | None = None) -> FastAPI:
    cfg = cfg or load_gateway_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.gateway_ready = False
        app.state.gateway_draining = False
        app.state.gateway_config = cfg
        app.state.gateway_storage = storage or await GatewayStorage.from_config(cfg)
        app.state.gateway_router = LLMRouter(cfg)
        app.state.gateway_admission = AdmissionController(cfg)
        app.state.gateway_forwarder = InferenceForwarder(cfg)
        app.state.gateway_resolver = SessionResolver(cfg)
        app.state.gateway_telemetry = TelemetryRecorder(cfg, app.state.gateway_storage)
        await app.state.gateway_telemetry.start()
        app.state.gateway_ready = True
        try:
            yield
        finally:
            app.state.gateway_ready = False
            app.state.gateway_draining = True
            app.state.gateway_admission.draining = True
            await _wait_for_drain(app, cfg.drain_timeout_s)
            await app.state.gateway_telemetry.stop()
            await app.state.gateway_forwarder.close()
            await app.state.gateway_storage.close()

    app = FastAPI(
        title="AIEvo API Gateway",
        version="0.2.0",
        lifespan=lifespan,
    )
    session_root = cfg.base_session_path.rstrip("/")

    async def handle_inference_request(
        request: Request,
        *,
        endpoint: str,
        path_session_id: str,
    ) -> Response:
        started = time.perf_counter()
        payload: dict[str, Any] = {}
        ctx: GatewayRequestContext | None = None
        binding: GatewaySessionBinding | None = None
        target: LLMRouteTarget | None = None
        route_reserved = False
        release_in_finally = True

        resolver: SessionResolver = request.app.state.gateway_resolver
        admission: AdmissionController = request.app.state.gateway_admission
        router: LLMRouter = request.app.state.gateway_router
        forwarder: InferenceForwarder = request.app.state.gateway_forwarder
        telemetry: TelemetryRecorder = request.app.state.gateway_telemetry

        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")

            ctx = await resolver.resolve(
                payload,
                endpoint=endpoint,
                path_session_id=path_session_id,
            )
            binding = await resolver.get_or_create_binding(ctx)
            if binding.status == "closed":
                raise SessionClosedError(f"session {ctx.session_id} is closed")
            if telemetry.should_reject_new_requests():
                raise AdmissionRejected("telemetry queue is full", 503)

            await admission.acquire_request(ctx, binding)
            target = await router.select_target(ctx, binding)
            await admission.acquire_llm_route(ctx, target)
            route_reserved = True
            await router.on_acquire(target.route_model, is_stream=ctx.is_stream)

            headers = forwarder.build_upstream_headers(target)
            if ctx.is_stream:
                opened = await _open_stream(forwarder, target, endpoint, payload, headers)
                release_in_finally = False
                return StreamingResponse(
                    _stream_and_finalize(
                        opened=opened,
                        started=started,
                        ctx=ctx,
                        binding=binding,
                        target=target,
                        payload=payload,
                        router=router,
                        telemetry=telemetry,
                        admission=admission,
                    ),
                    status_code=opened.status_code,
                    media_type=opened.media_type,
                )

            result = await _forward_json(forwarder, target, endpoint, payload, headers)
            latency_ms = (time.perf_counter() - started) * 1000
            await router.mark_route_result(
                target.route_model,
                True,
                result.upstream_latency_ms,
                result.status_code,
            )
            await telemetry.enqueue_success(
                ctx,
                binding,
                target,
                payload,
                result.body,
                latency_ms,
                upstream_latency_ms=result.upstream_latency_ms,
            )
            return JSONResponse(result.body, status_code=result.status_code)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            status_code, error_body = forwarder.normalize_error(exc)
            if target is not None:
                await router.mark_route_result(target.route_model, False, latency_ms, status_code)
            if ctx is not None and binding is not None:
                await telemetry.enqueue_failure(
                    ctx,
                    binding,
                    target,
                    payload,
                    str(exc),
                    status_code,
                    latency_ms,
                )
            return JSONResponse(error_body, status_code=status_code)
        finally:
            if release_in_finally:
                if route_reserved and target is not None and ctx is not None:
                    await router.on_release(target.route_model, is_stream=ctx.is_stream)
                await admission.release(ctx, binding, target)

    async def handle_session_chat_completions(session_id: str, request: Request) -> Response:
        return await handle_inference_request(
            request,
            endpoint="chat/completions",
            path_session_id=session_id,
        )

    async def handle_session_responses(session_id: str, request: Request) -> Response:
        return await handle_inference_request(
            request,
            endpoint="responses",
            path_session_id=session_id,
        )

    async def get_session_status(session_id: str) -> Response:
        resolver: SessionResolver = app.state.gateway_resolver
        status = await resolver.get_status(session_id)
        if status is None:
            return JSONResponse({"error": {"type": "not_found", "message": "session not found"}}, status_code=404)
        return JSONResponse(status)

    async def close_session(session_id: str, request: Request) -> dict[str, Any]:
        resolver: SessionResolver = app.state.gateway_resolver
        telemetry: TelemetryRecorder = app.state.gateway_telemetry
        reason = "gateway_close"
        try:
            body = await request.json()
            if isinstance(body, dict) and body.get("reason"):
                reason = str(body["reason"])
        except Exception:
            pass
        binding = await resolver.close_session(session_id, reason=reason)
        await telemetry.enqueue_session_close(binding)
        return {"session_id": session_id, "status": binding.status}

    app.add_api_route(
        f"{session_root}/{{session_id}}/chat/completions",
        handle_session_chat_completions,
        methods=["POST"],
    )
    app.add_api_route(
        f"{session_root}/{{session_id}}/responses",
        handle_session_responses,
        methods=["POST"],
    )
    app.add_api_route(
        f"{session_root}/{{session_id}}",
        get_session_status,
        methods=["GET"],
    )
    app.add_api_route(
        f"{session_root}/{{session_id}}/close",
        close_session,
        methods=["POST"],
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> Response:
        if not app.state.gateway_ready:
            return JSONResponse({"status": "draining" if app.state.gateway_draining else "starting"}, status_code=503)
        return JSONResponse({"status": "ready"})

    @app.get("/metrics")
    async def metrics() -> Response:
        admission: AdmissionController = app.state.gateway_admission
        router: LLMRouter = app.state.gateway_router
        telemetry: TelemetryRecorder = app.state.gateway_telemetry
        admission_snapshot = await admission.snapshot()
        route_snapshot = await router.snapshot()
        telemetry_snapshot = await telemetry.metrics_snapshot()
        lines = [
            "# TYPE gateway_inflight_requests gauge",
            f"gateway_inflight_requests {admission_snapshot['inflight_requests']}",
            "# TYPE gateway_active_streams gauge",
            f"gateway_active_streams {admission_snapshot['active_streams']}",
            "# TYPE gateway_telemetry_queue_depth gauge",
            f"gateway_telemetry_queue_depth {telemetry.queue_depth()}",
            "# TYPE gateway_telemetry_dropped_total counter",
        ]
        if telemetry_snapshot["dropped_by_reason"]:
            for reason, count in telemetry_snapshot["dropped_by_reason"].items():
                lines.append(f'gateway_telemetry_dropped_total{{reason="{_metric_label(reason)}"}} {count}')
        else:
            lines.append('gateway_telemetry_dropped_total{reason="none"} 0')

        lines.extend(
            [
                "# TYPE gateway_requests_accepted_total counter",
                f"gateway_requests_accepted_total {admission_snapshot['accepted_total']}",
                "# TYPE gateway_requests_rejected_total counter",
                f"gateway_requests_rejected_total {admission_snapshot['rejected_total']}",
                "# TYPE gateway_inference_requests_total counter",
            ]
        )
        for (endpoint, status_code), count in telemetry_snapshot["request_totals"].items():
            lines.append(
                f'gateway_inference_requests_total{{endpoint="{_metric_label(endpoint)}",status_code="{status_code}"}} {count}'
            )

        lines.append("# TYPE gateway_request_duration_seconds summary")
        for (endpoint, model), count in telemetry_snapshot["duration_count"].items():
            total_ms = telemetry_snapshot["duration_sum_ms"][(endpoint, model)]
            label = f'endpoint="{_metric_label(endpoint)}",model="{_metric_label(model)}"'
            lines.append(f"gateway_request_duration_seconds_sum{{{label}}} {total_ms / 1000}")
            lines.append(f"gateway_request_duration_seconds_count{{{label}}} {count}")

        lines.append("# TYPE gateway_ttft_seconds summary")
        for model, count in telemetry_snapshot["ttft_count"].items():
            label = f'model="{_metric_label(model)}"'
            lines.append(f"gateway_ttft_seconds_sum{{{label}}} {telemetry_snapshot['ttft_sum_ms'][model] / 1000}")
            lines.append(f"gateway_ttft_seconds_count{{{label}}} {count}")

        lines.append("# TYPE gateway_llm_route_inflight gauge")
        for route_model, state in route_snapshot.items():
            label = f'model="{_metric_label(route_model)}"'
            lines.append(f"gateway_llm_route_inflight{{{label}}} {state['inflight_requests']}")
            lines.append(f"gateway_llm_route_error_rate{{{label}}} {state['recent_error_rate']}")
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return app


async def _forward_json(
    forwarder: InferenceForwarder,
    target: LLMRouteTarget,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
):
    if endpoint == "chat/completions":
        return await forwarder.forward_chat(target, payload, headers)
    if endpoint == "responses":
        return await forwarder.forward_responses(target, payload, headers)
    raise ValueError(f"unsupported endpoint {endpoint}")


async def _open_stream(
    forwarder: InferenceForwarder,
    target: LLMRouteTarget,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> StreamForwardContext:
    if endpoint == "chat/completions":
        return await forwarder.open_chat_stream(target, payload, headers)
    if endpoint == "responses":
        return await forwarder.open_responses_stream(target, payload, headers)
    raise ValueError(f"unsupported endpoint {endpoint}")


async def _stream_and_finalize(
    *,
    opened: StreamForwardContext,
    started: float,
    ctx: GatewayRequestContext,
    binding: GatewaySessionBinding,
    target: LLMRouteTarget,
    payload: dict[str, Any],
    router: LLMRouter,
    telemetry: TelemetryRecorder,
    admission: AdmissionController,
) -> AsyncIterator[bytes]:
    first_chunk_at: float | None = None
    chunk_count = 0
    output_bytes = 0
    status_code = opened.status_code
    error_text: str | None = None
    client_cancelled = False
    upstream_cancelled = False
    stream_response_body: dict[str, Any] = {}
    stream_metadata_buffer = ""

    try:
        async for chunk in opened.response.content.iter_any():
            if chunk:
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                chunk_count += 1
                output_bytes += len(chunk)
                stream_metadata_buffer = _collect_stream_metadata(
                    chunk,
                    stream_metadata_buffer,
                    stream_response_body,
                )
            yield chunk
    except asyncio.CancelledError:
        client_cancelled = True
        status_code = 499
        error_text = "client cancelled streaming response"
        raise
    except Exception as exc:
        upstream_cancelled = True
        status_code = 502
        error_text = str(exc)
        raise
    finally:
        opened.response.close()
        latency_ms = (time.perf_counter() - started) * 1000
        ttft_ms = (first_chunk_at - started) * 1000 if first_chunk_at is not None else None
        stats = StreamTelemetryStats(
            ttft_ms=ttft_ms,
            output_chunk_count=chunk_count,
            output_bytes=output_bytes,
            client_cancelled=client_cancelled,
            upstream_cancelled=upstream_cancelled,
        )
        ok = status_code < 400
        await router.mark_route_result(target.route_model, ok, latency_ms, status_code)
        try:
            if ok:
                await telemetry.enqueue_success(
                    ctx,
                    binding,
                    target,
                    payload,
                    stream_response_body or None,
                    latency_ms,
                    upstream_latency_ms=opened.upstream_latency_ms,
                    stream_stats=stats,
                )
            else:
                await telemetry.enqueue_failure(
                    ctx,
                    binding,
                    target,
                    payload,
                    error_text or "stream failed",
                    status_code,
                    latency_ms,
                    upstream_latency_ms=opened.upstream_latency_ms,
                    stream_stats=stats,
                )
        finally:
            await router.on_release(target.route_model, is_stream=ctx.is_stream)
            await admission.release(ctx, binding, target)


async def _wait_for_drain(app: FastAPI, drain_timeout_s: int) -> None:
    admission: AdmissionController = app.state.gateway_admission
    deadline = time.monotonic() + max(0, drain_timeout_s)
    while time.monotonic() < deadline:
        snapshot = await admission.snapshot()
        if snapshot["inflight_requests"] <= 0 and snapshot["active_streams"] <= 0:
            return
        await asyncio.sleep(0.05)


def _metric_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _collect_stream_metadata(chunk: bytes, buffer: str, summary: dict[str, Any]) -> str:
    buffer += chunk.decode("utf-8", errors="ignore")
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            summary.setdefault("status", "completed")
            continue
        try:
            event = json.loads(data)
        except ValueError:
            continue
        if isinstance(event, dict):
            _merge_stream_event(summary, event)
    if len(buffer) > 65536:
        return buffer[-4096:]
    return buffer


def _merge_stream_event(summary: dict[str, Any], event: dict[str, Any]) -> None:
    for key in ("id", "object"):
        value = event.get(key)
        if value is not None:
            summary.setdefault(key, value)

    usage = event.get("usage")
    if isinstance(usage, dict):
        summary["usage"] = usage

    choices = event.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict) and choice.get("finish_reason"):
                summary["choices"] = [{"finish_reason": choice["finish_reason"]}]
                break

    response = event.get("response")
    if isinstance(response, dict):
        for key in ("id", "object", "status"):
            value = response.get(key)
            if value is not None:
                summary[key] = value
        response_usage = response.get("usage")
        if isinstance(response_usage, dict):
            summary["usage"] = response_usage

    event_type = event.get("type")
    if event_type == "response.completed":
        summary["status"] = "completed"
    elif event_type == "response.failed":
        summary["status"] = "failed"
    elif event_type == "response.cancelled":
        summary["status"] = "cancelled"


app = create_app()
