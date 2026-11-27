# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cloud-side direct-DP HTTP coordination.

One bridge owns all per-DP command queues and the shared event queue. Each DP
has a thin FastAPI listener: ``/v1/lwd/probe`` performs a short read-only peek,
while ``/v1/lwd/activate`` registers usage before returning its SSE response.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import queue
import socket
import threading
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.logger import init_logger
from vllm.lwd_logger import lwd_logger
from vllm.v1.lwd_control.lwd_comm.lwd_id_adapter import wrap_req_id
from vllm.v1.lwd_control.lwd_comm.lwd_mm_identity import (
    compute_processor_fingerprint,
    mm_abi_header_value,
    negotiate_lwd_protocol,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    HEADER_CLOUD_ID,
    HEADER_EDGE_ID,
    HEADER_ENGINE_INDEX,
    HEADER_MM_ABI,
    HEADER_PROBE_NONCE,
    HEADER_PROTOCOL,
    HEADER_REQUEST_ID,
    PROBE_HEADROOM_S,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)

logger = init_logger(__name__)

_USAGE_HEARTBEAT_INTERVAL = 10.0
_USAGE_ADMISSION_TIMEOUT_S = 600.0
_CLOUD_PROBE_POLL_BUDGET = 32
_SERVER_START_TIMEOUT = 30.0
_SERVER_STOP_TIMEOUT = 5.0


@dataclass
class _LwdUsageRegistration:
    engine_index: int
    prefix_nonce: int
    future: asyncio.Future[LwdUsage | None]
    started: asyncio.Event = field(default_factory=asyncio.Event)


class LwdCloudControlBridge:
    """Dispatch direct-DP probe and winner usage events to asyncio futures."""

    def __init__(self, command_queue: Any, event_queue: Any) -> None:
        if isinstance(command_queue, (list, tuple)):
            self._command_queues = list(command_queue)
        else:
            self._command_queues = [command_queue]
        if not self._command_queues:
            raise ValueError("cloud control requires at least one command queue")
        self._event_queue = event_queue
        self._probe_futures: dict[str, tuple[int, asyncio.Future[LwdProbeResult]]] = {}
        self._usage_futures: dict[str, _LwdUsageRegistration] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_events, name="lwd-ctrl-dispatch", daemon=True
        )

    def start(self) -> None:
        self._dispatch_thread.start()

    def close(self) -> None:
        """Stop dispatching and fail all pending futures."""
        self._closed = True
        with self._lock:
            pending = [future for _, future in self._probe_futures.values()] + [
                registration.future for registration in self._usage_futures.values()
            ]
            self._probe_futures.clear()
            self._usage_futures.clear()
            loop = self._loop
        alive = [future for future in pending if not future.done()]
        if loop is None or not alive:
            return

        def _fail_pending() -> None:
            for future in alive:
                if future.done():
                    continue
                future.set_exception(RuntimeError("cloud control plane stopped"))
                future.exception()

        try:
            loop.call_soon_threadsafe(_fail_pending)
        except RuntimeError:
            logger.info(
                "[LWD][ctrl] close: loop already stopped, drop %d waiters",
                len(alive),
            )

    async def probe(
        self,
        dp_idx: int,
        manifest: LwdPrefixManifest,
        *,
        reply_key: str,
        timeout: float = 20.0,
    ) -> LwdProbeResult:
        """Peek one DP without creating a reservation or usage registration."""
        if not 0 <= dp_idx < len(self._command_queues):
            raise IndexError(f"invalid cloud DP index {dp_idx}")
        if not reply_key:
            raise ValueError("reply_key must not be empty")
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._closed:
                raise RuntimeError("cloud control plane stopped")
            self._loop = loop
            stale = self._probe_futures.pop(reply_key, None)
            probe_future: asyncio.Future[LwdProbeResult] = loop.create_future()
            self._probe_futures[reply_key] = (dp_idx, probe_future)
        if stale is not None and not stale[1].done():
            stale[1].set_exception(
                RuntimeError(f"probe reply key {reply_key} was superseded")
            )
        now = time.monotonic()
        self._command_queues[dp_idx].put(
            {
                "type": "probe",
                "manifest": manifest,
                "engine_index": dp_idx,
                "reply_key": reply_key,
                "t_cmd": now,
                "deadline": now + timeout,
            }
        )
        try:
            return await asyncio.wait_for(probe_future, timeout)
        except BaseException:
            with self._lock:
                registered = self._probe_futures.get(reply_key)
                if registered is not None and registered[1] is probe_future:
                    self._probe_futures.pop(reply_key, None)
            raise

    def activate(
        self, request_id: str, dp_idx: int, prefix_nonce: int
    ) -> _LwdUsageRegistration:
        """Register one DP/nonce usage generation before response headers."""
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._closed:
                raise RuntimeError("cloud control plane stopped")
            self._loop = loop
            stale = self._usage_futures.pop(request_id, None)
            registration = _LwdUsageRegistration(
                engine_index=dp_idx,
                prefix_nonce=prefix_nonce,
                future=loop.create_future(),
            )
            self._usage_futures[request_id] = registration
        if stale is not None and not stale.future.done():
            logger.warning(
                "[LWD][ctrl] duplicate activate req=%s: superseding usage",
                request_id,
            )
            stale.future.set_result(None)
        return registration

    def registration(self, request_id: str) -> _LwdUsageRegistration | None:
        with self._lock:
            return self._usage_futures.get(request_id)

    async def wait_usage(
        self,
        request_id: str,
        timeout: float = _USAGE_ADMISSION_TIMEOUT_S,
        *,
        registration: _LwdUsageRegistration | None = None,
    ) -> LwdUsage | None:
        """Bound only admission; started requests wait for real terminal usage."""
        with self._lock:
            current = self._usage_futures.get(request_id)
        if current is None or (
            registration is not None and current is not registration
        ):
            if current is None:
                logger.warning(
                    "[LWD][ctrl] usage wait without activation req=%s", request_id
                )
            return None
        registration = current
        started_wait = asyncio.create_task(registration.started.wait())
        try:
            done, _ = await asyncio.wait(
                {registration.future, started_wait},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if registration.future in done:
                return registration.future.result()
            if started_wait in done:
                return await registration.future
            logger.warning("[LWD][ctrl] usage admission timeout req=%s", request_id)
            return None
        finally:
            started_wait.cancel()
            with self._lock:
                if self._usage_futures.get(request_id) is registration:
                    self._usage_futures.pop(request_id, None)

    def release(
        self,
        request_id: str,
        registration: _LwdUsageRegistration | None = None,
    ) -> None:
        """Release an activation generation, preserving a newer activation."""
        with self._lock:
            current = self._usage_futures.get(request_id)
            if registration is None or current is registration:
                self._usage_futures.pop(request_id, None)

    def _dispatch_events(self) -> None:
        """Forward the single shared event stream to the shared asyncio loop."""
        while not self._closed:
            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            kind = event.get("type")
            request_id = event.get("request_id")
            with self._lock:
                if kind == "probe":
                    registered = request_id in self._probe_futures
                else:
                    registered = request_id in self._usage_futures
                loop = self._loop
            logger.debug(
                "[LWD][ctrl] dispatch got event type=%s req=%s ok=%s engine=%s",
                kind,
                request_id,
                event.get("ok"),
                event.get("engine_index"),
            )
            if not registered or loop is None:
                logger.warning(
                    "[LWD][ctrl] orphan %s event req=%s (no waiter/loop)",
                    kind,
                    request_id,
                )
                continue
            try:
                loop.call_soon_threadsafe(self._deliver_event, event)
            except RuntimeError:
                logger.info("[LWD][ctrl] dispatch stopping: loop closed")
                return

    def _deliver_event(self, event: dict[str, Any]) -> None:
        """Settle futures only on the event-loop thread."""
        kind = event.get("type")
        request_id = event.get("request_id")
        if kind == "probe":
            with self._lock:
                registered = self._probe_futures.get(request_id)
                if registered is None or registered[0] != event.get("engine_index"):
                    return
                self._probe_futures.pop(request_id, None)
                future = registered[1]
            if future.done():
                return
            if event.get("ok"):
                future.set_result(event.get("result"))
            else:
                future.set_exception(RuntimeError(event.get("error") or "probe failed"))
            return
        if kind in ("usage_start", "usage"):
            with self._lock:
                registration = self._usage_futures.get(request_id)
            if (
                registration is None
                or registration.engine_index != event.get("engine_index")
                or registration.prefix_nonce != event.get("prefix_nonce")
            ):
                return
            if kind == "usage_start":
                registration.started.set()
                return
            future = registration.future
            if future.done():
                logger.warning(
                    "[LWD][ctrl] usage event arrived too late req=%s", request_id
                )
                return
            future.set_result(event.get("usage"))


class LwdCloudControlProcessor:
    """Child-side non-blocking command poll using read-only prefix peeks."""

    def __init__(
        self,
        pending_queue: Any,
        event_queue: Any,
        poll_budget: int = _CLOUD_PROBE_POLL_BUDGET,
    ) -> None:
        self._pending_queue = pending_queue
        self._event_queue = event_queue
        self._poll_budget = poll_budget

    def poll(self, coordinator: Any) -> int:
        consumed = 0
        while consumed < self._poll_budget:
            try:
                command = self._pending_queue.get_nowait()
            except queue.Empty:
                break
            consumed += 1
            if command.get("type") != "probe":
                continue
            manifest: LwdPrefixManifest = command["manifest"]
            reply_key = command["reply_key"]
            engine_index = command["engine_index"]
            t_take = time.monotonic()
            t_cmd = command.get("t_cmd")
            cmd_wait_ms = (t_take - t_cmd) * 1000.0 if t_cmd is not None else -1.0
            deadline = command.get("deadline")
            if deadline is not None and t_take >= deadline:
                event = {
                    "type": "probe",
                    "request_id": reply_key,
                    "engine_index": engine_index,
                    "ok": False,
                    "error": "probe command expired",
                }
            else:
                try:
                    result = coordinator.peek(manifest)
                    logger.info(
                        "[LWD][perf][cloud] probe_consumed req=%s "
                        "cmd_wait_ms=%.1f probe_ms=%.1f",
                        manifest.request_id,
                        cmd_wait_ms,
                        (time.monotonic() - t_take) * 1000.0,
                    )
                    event = {
                        "type": "probe",
                        "request_id": reply_key,
                        "engine_index": engine_index,
                        "ok": True,
                        "result": result,
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.exception("[LWD][ctrl] probe failed")
                    event = {
                        "type": "probe",
                        "request_id": reply_key,
                        "engine_index": engine_index,
                        "ok": False,
                        "error": str(exc),
                    }
            self._event_queue.put(event)
        return consumed


def _negotiate_lwd_protocol_headers(
    headers: Mapping[str, str],
    expected_mm_abi: str | None,
    *,
    enforce_mm_abi_match: bool,
) -> str | None:
    normalized = {k.lower(): v for k, v in headers.items()}
    return negotiate_lwd_protocol(
        normalized.get(HEADER_PROTOCOL.lower()),
        normalized.get(HEADER_MM_ABI.lower()),
        expected_mm_abi=expected_mm_abi,
        enforce_mm_abi_match=enforce_mm_abi_match,
    )


def _sse_data(obj: Any) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def _build_chunk(request_id: str, model: str, usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": usage,
    }


def _required_decimal_header(headers: Mapping[str, str], name: str) -> int:
    value = headers.get(name.lower())
    if value is None:
        raise ValueError(f"missing required header {name}")
    if not (value.isascii() and value.isdecimal()):
        raise ValueError(f"{name} must be a non-negative ASCII decimal integer")
    return int(value)


def create_lwd_cloud_control_app(
    bridge: LwdCloudControlBridge,
    cloud_id: int,
    dp_idx: int,
    processor_fingerprint: bytes | None = None,
    *,
    enforce_mm_abi_match: bool = True,
    probe_timeout: float = 25.0,
) -> Any:
    """Build one fixed-DP app with short probe and activate/usage routes."""
    if cloud_id < 0 or dp_idx < 0:
        raise ValueError("cloud_id and dp_idx must be non-negative")
    app = FastAPI()
    expected_mm_abi = (
        mm_abi_header_value(processor_fingerprint)
        if processor_fingerprint is not None
        else None
    )

    async def parse_request(
        request: Request,
    ) -> tuple[LwdPrefixManifest, Mapping[str, Any], str, str | None, int]:
        headers = dict(request.headers)
        body = await request.json()
        if not isinstance(body, Mapping):
            raise ValueError("request body must be a JSON object")
        mm_abi_header = _negotiate_lwd_protocol_headers(
            headers,
            expected_mm_abi,
            enforce_mm_abi_match=enforce_mm_abi_match,
        )
        parse_headers: Mapping[str, str] = headers
        if mm_abi_header is not None:
            parse_headers = {
                k: v for k, v in headers.items() if k.lower() != HEADER_PROTOCOL.lower()
            }
            parse_headers[HEADER_PROTOCOL] = PROTOCOL_VERSION
        manifest = LwdPrefixManifest.from_openai_request(parse_headers, body)
        headers_lc = {k.lower(): v for k, v in headers.items()}
        edge_id = _required_decimal_header(headers_lc, HEADER_EDGE_ID)
        probe_nonce = _required_decimal_header(headers_lc, HEADER_PROBE_NONCE)
        for name, expected in (
            (HEADER_CLOUD_ID, cloud_id),
            (HEADER_ENGINE_INDEX, dp_idx),
        ):
            actual = _required_decimal_header(headers_lc, name)
            if actual != expected:
                raise ValueError(f"{name} does not match this DP listener")
        display_request_id = manifest.request_id
        manifest = dataclass_replace(
            manifest, request_id=wrap_req_id(edge_id, display_request_id)
        )
        return manifest, body, display_request_id, mm_abi_header, probe_nonce

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/lwd/probe")
    async def probe(request: Request) -> Any:
        try:
            manifest, _, display_id, mm_abi, nonce = await parse_request(request)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)
        reply_key = f"{manifest.request_id}#probe:{nonce}:{cloud_id}:{dp_idx}"
        try:
            result = await bridge.probe(
                dp_idx, manifest, reply_key=reply_key, timeout=probe_timeout
            )
        except TimeoutError as exc:
            logger.warning(
                "[LWD][ctrl] probe timeout req=%s dp=%d: %s",
                manifest.request_id,
                dp_idx,
                exc,
            )
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[LWD][ctrl] handler probe raised")
            return JSONResponse({"error": str(exc)}, status_code=503)
        result = dataclass_replace(
            result,
            request_id=display_id,
            cloud_id=cloud_id,
            engine_index=dp_idx,
        )
        protocol = PROTOCOL_VERSION_MM if mm_abi is not None else PROTOCOL_VERSION
        control_message = result.to_control_message(protocol, mm_abi)
        headers_out = result.to_headers()
        headers_out[HEADER_PROTOCOL] = protocol
        if mm_abi is not None:
            headers_out[HEADER_MM_ABI] = mm_abi
        return JSONResponse(control_message, headers=headers_out)

    @app.post("/v1/lwd/activate")
    async def activate(request: Request) -> Any:
        try:
            manifest, body, display_id, mm_abi, nonce = await parse_request(request)
            registration = bridge.activate(manifest.request_id, dp_idx, nonce)
        except ValueError as exc:
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=503)
        model = body.get("model", "edge-cloud-prefix")

        async def events_inner() -> AsyncIterator[str]:
            usage_task = asyncio.create_task(
                bridge.wait_usage(manifest.request_id, registration=registration)
            )
            try:
                while not usage_task.done():
                    done, _ = await asyncio.wait(
                        {usage_task}, timeout=_USAGE_HEARTBEAT_INTERVAL
                    )
                    if not done:
                        yield ": edge-cloud-usage-pending\n\n"
                usage = usage_task.result()
            except asyncio.CancelledError:
                usage_task.cancel()
                await asyncio.gather(usage_task, return_exceptions=True)
                raise
            if usage is None:
                logger.error(
                    "[LWD][ctrl] usage unavailable req=%s: ending without usage",
                    manifest.request_id,
                )
                yield ": edge-cloud-usage-unavailable\n\n"
                yield "data: [DONE]\n\n"
                return
            yield _sse_data(_build_chunk(display_id, model, usage.to_openai_dict()))
            yield "data: [DONE]\n\n"

        async def events() -> AsyncIterator[str]:
            try:
                async for frame in events_inner():
                    yield frame
            finally:
                bridge.release(manifest.request_id, registration)

        protocol = PROTOCOL_VERSION_MM if mm_abi is not None else PROTOCOL_VERSION
        headers_out = {
            HEADER_PROTOCOL: protocol,
            HEADER_REQUEST_ID: display_id,
            HEADER_CLOUD_ID: str(cloud_id),
            HEADER_ENGINE_INDEX: str(dp_idx),
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        if mm_abi is not None:
            headers_out[HEADER_MM_ABI] = mm_abi
        return StreamingResponse(
            events(), media_type="text/event-stream", headers=headers_out
        )

    return app


def _bind_control_sockets(host: str, cloud_dps: Any) -> list[socket.socket]:
    sockets: list[socket.socket] = []
    try:
        for dp in cloud_dps:
            if dp.coord_port is None:
                raise ValueError(f"cloud DP {dp.dp_idx} has no coord_port")
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            listener = socket.socket(family, socket.SOCK_STREAM)
            try:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((host, dp.coord_port))
                listener.listen(socket.SOMAXCONN)
            except BaseException:
                listener.close()
                raise
            sockets.append(listener)
    except BaseException:
        for listener in sockets:
            listener.close()
        raise
    return sockets


def start_lwd_cloud_control_server(
    vllm_config: Any,
) -> tuple[list[Any] | None, Any | None]:
    """Start all DP listeners on one loop, or fail the cloud startup."""
    lwd_config = getattr(vllm_config, "lwd_config", None)
    coord_cfg = getattr(lwd_config, "coordination", None)
    if coord_cfg is None or not coord_cfg.enabled or lwd_config.role != "cloud":
        return None, None

    cloud_dps = tuple(lwd_config.instance.dp)
    if not cloud_dps:
        raise ValueError("cloud coordination requires at least one DP")
    for queue_idx, dp in enumerate(cloud_dps):
        if dp.dp_idx != queue_idx:
            raise ValueError("cloud DP indices must be contiguous and zero-based")
    sockets = _bind_control_sockets(coord_cfg.listen_host, cloud_dps)

    import uvicorn

    from vllm.utils.system_utils import get_mp_context

    mp_ctx = get_mp_context()
    command_queues = [mp_ctx.Queue() for _ in cloud_dps]
    event_queue = mp_ctx.Queue()
    bridge = LwdCloudControlBridge(command_queues, event_queue)
    bridge.start()
    cloud_probe_timeout = max(coord_cfg.probe_timeout - PROBE_HEADROOM_S, 1.0)
    fingerprint = compute_processor_fingerprint(vllm_config.model_config)
    lwd_logger.info(
        "[mm-abi] cloud_id=%s fingerprint=%s",
        lwd_config.instance.id,
        mm_abi_header_value(fingerprint),
    )
    servers = []
    for dp in cloud_dps:
        app = create_lwd_cloud_control_app(
            bridge,
            lwd_config.instance.id,
            dp.dp_idx,
            fingerprint,
            enforce_mm_abi_match=coord_cfg.enforce_mm_abi_match,
            probe_timeout=cloud_probe_timeout,
        )
        servers.append(uvicorn.Server(uvicorn.Config(app, log_level="warning")))

    loop = asyncio.new_event_loop()
    ready = threading.Event()
    startup_errors: list[BaseException] = []

    async def serve_all() -> None:
        tasks = [
            asyncio.create_task(server.serve(sockets=[listener]))
            for server, listener in zip(servers, sockets, strict=True)
        ]
        try:
            while not all(server.started for server in servers):
                if any(task.done() for task in tasks):
                    raise RuntimeError("a cloud control listener exited during startup")
                await asyncio.sleep(0.01)
            ready.set()
            await asyncio.gather(*tasks)
        except BaseException as exc:
            if not ready.is_set():
                startup_errors.append(exc)
                ready.set()
            else:
                logger.error("[LWD] cloud control server stopped: %s", exc)
        finally:
            for server in servers:
                server.should_exit = True
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def run_servers() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(serve_all())
        except BaseException as exc:  # noqa: BLE001
            if not ready.is_set():
                startup_errors.append(exc)
                ready.set()
        finally:
            loop.close()

    server_thread = threading.Thread(
        target=run_servers, name="lwd-cloud-ctrl", daemon=True
    )
    server_thread.start()
    closed = threading.Event()

    def close() -> None:
        if closed.is_set():
            return
        closed.set()
        bridge.close()
        if loop.is_running():

            def request_shutdown() -> None:
                for server in servers:
                    server.should_exit = True

            loop.call_soon_threadsafe(request_shutdown)
        server_thread.join(timeout=_SERVER_STOP_TIMEOUT)
        for listener in sockets:
            with contextlib.suppress(OSError):
                listener.close()

    if not ready.wait(timeout=_SERVER_START_TIMEOUT):
        close()
        raise TimeoutError("cloud control listeners did not become ready")
    if startup_errors:
        close()
        raise RuntimeError("failed to start all cloud control listeners") from (
            startup_errors[0]
        )
    atexit.register(close)
    logger.info(
        "[LWD] cloud control HTTP ready on %s ports=%s cloud_id=%s dp_size=%d",
        coord_cfg.listen_host,
        [dp.coord_port for dp in cloud_dps],
        lwd_config.instance.id,
        len(cloud_dps),
    )
    return command_queues, event_queue
