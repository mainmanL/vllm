# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cloud-side prefix negotiation HTTP control plane (SSE gateway form).

Derived from the reference branch's ``edge_cloud/cloud_control.py``: a parent
FastAPI application and bridge (LwdCloudControlBridge) communicate with a
child-process processor (LwdCloudControlProcessor, with non-blocking poll)
through two multiprocessing queues. The host is ``LwdCloudEngineCore``.

The control plane carries probe and usage events over **OpenAI-compatible
SSE**:
  1. Put the probe result in a JSON control message in ``delta.content`` so
     gateway DTO conversion preserves it;
  2. Immediately send an empty delta frame to flush gateway buffering of the
     final chunk (NewAPI and similar gateways);
  3. Send an SSE comment heartbeat every 10 seconds while waiting, preventing
     gateway upstream idle timeouts;
  4. After inference finishes, send a standard OpenAI usage chunk and
     ``[DONE]`` for gateway accounting. If real usage is unavailable, send only
     a comment and ``[DONE]``; never fabricate a zero-usage frame.

Protocol gating: text uses ``edge-cloud-prefix-v1``; images use
``edge-cloud-prefix-v2`` and must carry the ``X-Edge-Cloud-MM-ABI``
fingerprint (the chain construction identity for media-covered blocks). The
two stacks must not downgrade into each other. ``X-Edge-Cloud-*`` response
headers remain available for direct debugging and gateway observability, but
do not participate in negotiation.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import queue
import threading
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace as dataclass_replace
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.logger import init_logger
from vllm.v1.lwd_control.lwd_comm.lwd_id_adapter import wrap_req_id
from vllm.v1.lwd_control.lwd_comm.lwd_mm_identity import (
    compute_processor_fingerprint,
    mm_abi_header_value,
    validate_mm_abi_header,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    HEADER_EDGE_ID,
    HEADER_MM_ABI,
    HEADER_PROTOCOL,
    HEADER_REQUEST_ID,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)

logger = init_logger(__name__)

# SSE heartbeat interval in seconds. Keep it below the gateway's
# stream_idle_timeout, with sufficient headroom.
_USAGE_HEARTBEAT_INTERVAL = 10.0
# Total usage wait timeout in seconds. It covers the full inference duration;
# on timeout, end the stream without fabricating usage.
_USAGE_WAIT_TIMEOUT = 600.0
# The cloud probe limit is the edge lwd_coordination.probe_timeout minus this
# headroom. The cloud must answer first (200 or 503); otherwise the edge may
# abandon the request while the cloud still waits for its child, leaving both
# the reservation and accounting stream hanging. On 503 the edge fails open to
# full prefill, preserving existing behavior.
_CLOUD_PROBE_HEADROOM_S = 5.0


class LwdCloudControlBridge:
    """Connect parent-process HTTP coroutines to child probe/usage events.

    Waiting uses **per-request asyncio.Future objects**, consuming no threads
    (the reference ``edge_cloud/cloud_control.py`` shape validated with
    NewAPI/Higress). The dispatch thread only forwards events through
    ``loop.call_soon_threadsafe``; handler coroutines await directly. ``_loop``
    is captured in ``probe()`` inside the handler coroutine, so it is uvicorn's
    actual running loop rather than a loop captured on the startup thread.

    ``run_in_executor(None, waiter.get)`` is unsuitable for usage waits, which
    remain pending until inference completes (observed hold around 8 s). One
    inflight request would occupy one executor thread; the default pool limit
    of ``min(32, cpu+4)`` would eventually prevent new probes from even waiting
    for results (observed in 8cfb9f499: busy_usage=32, exec_delay_ms=-1, probe
    total_ms=60000). Worse, a started thread cannot be cancelled when
    ``wait_for`` expires, and removing its waiter leaves late events unable to
    wake it: every timeout would permanently leak a thread until the control
    plane stops responding. With futures, timeout only cancels a coroutine,
    decoupling inflight capacity from thread count. An earlier docstring here
    claimed cross-thread futures could hang mysteriously and switched to
    queues, but no such incident appears in the history (this shape came from
    the initial port in 2e0979767).
    """

    def __init__(self, command_queue: Any, event_queue: Any) -> None:
        # A mapping preserves global ranks for services managing a subset of
        # DPs. Lists/single queues retain the ordinary probe API compatibility.
        if isinstance(command_queue, dict):
            self._command_queue_map = dict(command_queue)
            self._command_queues = list(command_queue.values())
        elif isinstance(command_queue, (list, tuple)):
            self._command_queues: list[Any] = list(command_queue)
            self._command_queue_map = dict(enumerate(self._command_queues))
        else:
            self._command_queues = [command_queue]
            self._command_queue_map = {0: command_queue}
        self._engine_ranks = list(self._command_queue_map)
        from vllm.v1.lwd_control.pd.cloud_http import LwdPdCloudHttpBridge

        self.pd = LwdPdCloudHttpBridge(self._command_queue_map)
        self._command_queue = self._command_queues[0]
        self._event_queue = event_queue
        self._probe_futures: dict[str, asyncio.Future] = {}
        self._usage_futures: dict[str, asyncio.Future] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        # Fanout reply collectors, keyed by reply_key ("<rid>#probe"), are
        # isolated from the single-engine future namespace. Late replies with
        # no waiter are dropped.
        self._fanout_waiters: dict[str, queue.Queue] = {}
        # Rotate among engines tied for the longest hit to avoid hot spots.
        self._rr_counter = 0
        self._lock = threading.Lock()
        self._closed = False
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_events, name="lwd-ctrl-dispatch", daemon=True
        )

    def start(self) -> None:
        self._dispatch_thread.start()

    def close(self) -> None:
        """Stop dispatching and fail all pending futures.

        Called by the atexit hook registered in
        ``start_lwd_cloud_control_server``. This branch runs uvicorn on a
        daemon thread without a shutdown channel; without the hook, process
        exit could only hard-kill the daemon dispatch thread."""
        self._closed = True
        self.pd.close()
        with self._lock:
            pending = [*self._probe_futures.values(), *self._usage_futures.values()]
            self._probe_futures.clear()
            self._usage_futures.clear()
            loop = self._loop
        alive = [future for future in pending if not future.done()]
        if loop is None or not alive:
            return
        error = RuntimeError("cloud control plane stopped")

        def _fail_pending() -> None:
            for future in alive:
                if future.done():
                    continue
                future.set_exception(error)
                # A usage future is often unclaimed because its handler exited
                # after an edge disconnect. Retrieve its exception once to avoid
                # "exception was never retrieved" when the loop exits.
                future.exception()

        try:
            loop.call_soon_threadsafe(_fail_pending)
        except RuntimeError:
            # The loop has stopped, so no coroutine can await these futures.
            logger.info(
                "[LWD][ctrl] close: loop already stopped, drop %d waiters",
                len(alive),
            )

    async def probe(
        self, manifest: LwdPrefixManifest, timeout: float = 20.0
    ) -> LwdProbeResult:
        """Send a probe and await its result; timeout raises TimeoutError.

        The single-engine path registers probe and usage futures **together**:
        cloud accounting can push usage at any time, potentially before the
        handler reaches ``wait_usage``, so late registration could lose it.
        Registration precedes command submission, so no event can arrive first.

        The timeout is aligned with the edge: its probe watchdog abandons the
        request at ``lwd_coordination.probe_timeout`` + 5, while this limit is
        derived with headroom so the **cloud answers first** (200 or 503). Under
        load, a probe command may remain queued until timeout while the engine
        loop executes data-plane steps; its result is then observational only.

        Resending a request ID is normal (client retry or same-ID resubmission
        after abort). The engine ID comes from ``X-Request-Id`` when present,
        otherwise a random UUID (entrypoints/openai/engine/serving.py:387-396),
        so replaying that header reuses the ID. Edge nonces and coordinator
        same-key probe replacement exist for this case. Duplicates therefore
        **supersede** rather than being rejected: the old usage future resolves
        to None (usage belongs to the new stream; the old stream closes without
        usage), and the old probe future gets an exception (its handler returns
        503 before any 200 or accounting frame). The reference ``edge_client``
        rejected duplicates because its strict ``probe.block_size`` equality
        check had no second same-ID probe; copying that behavior would break
        retries here.

        For cloud-local dp>1, ``_probe_fanout`` performs read-only per-engine
        queries, placement, and winner reservation. Its replies are collected
        through ``_fanout_waiters``, not the ordinary probe future namespace.
        """
        request_id = manifest.request_id
        logger.debug("[LWD][ctrl] bridge probe req=%s", request_id)
        if len(self._command_queues) > 1:
            # Cloud-local DP: query every engine read-only, place the request,
            # then reserve on the winning engine.
            return await self._probe_fanout(manifest, timeout)
        loop = asyncio.get_running_loop()
        self._loop = loop
        with self._lock:
            stale_probe = self._probe_futures.pop(request_id, None)
            stale_usage = self._usage_futures.pop(request_id, None)
            probe_future: asyncio.Future = loop.create_future()
            usage_future: asyncio.Future = loop.create_future()
            self._probe_futures[request_id] = probe_future
            self._usage_futures[request_id] = usage_future
        if stale_probe is not None or stale_usage is not None:
            logger.warning(
                "[LWD][ctrl] duplicate control request req=%s: superseding "
                "the previous registration",
                request_id,
            )
            if stale_usage is not None and not stale_usage.done():
                stale_usage.set_result(None)
            if stale_probe is not None and not stale_probe.done():
                stale_probe.set_exception(
                    RuntimeError(
                        f"control request {request_id} superseded by a newer probe"
                    )
                )
        # [LWD][perf] I8a: t_cmd is command submission time in the parent
        # (CLOCK_MONOTONIC). Parent and child share a host clock; never compare
        # edge and cloud timestamps across nodes.
        self._command_queue.put(
            {
                "type": "probe",
                "manifest": manifest,
                "t_cmd": time.monotonic(),
            }
        )
        try:
            return await asyncio.wait_for(probe_future, timeout)
        except BaseException:
            # On probe failure (timeout, cancellation, or cloud error), this
            # control request is over and its handler will not await usage.
            # Remove both registrations; success is cleaned by _deliver_event
            # and wait_usage.
            with self._lock:
                if self._probe_futures.get(request_id) is probe_future:
                    self._probe_futures.pop(request_id, None)
                if self._usage_futures.get(request_id) is usage_future:
                    self._usage_futures.pop(request_id, None)
            raise

    def release(self, request_id: str) -> None:
        """Remove a control request's registrations (idempotently).

        The SSE generator must call this when it ends. If the client disconnects
        after the initial messages, ``wait_usage`` is never called and only
        process ``close()`` could otherwise collect the entry. Late accounting
        events are then dropped as orphans."""
        with self._lock:
            self._probe_futures.pop(request_id, None)
            self._usage_futures.pop(request_id, None)

    async def _probe_fanout(
        self, manifest: LwdPrefixManifest, timeout: float
    ) -> LwdProbeResult:
        """Fan out a cloud-local DP probe and select its placement (DP > 1).

        First broadcast read-only hit queries (reserve=False) to every engine
        processor. Then select the winner by descending hit length, rotating
        ties, and send the formal probe (reserve=True) only to that engine.
        Cross-engine reservations would be false hits: the reservation and the
        subsequent data plane must use the same engine. Return the winner's hit
        values and engine_index.

        Replies use the shared event queue. Commands carry engine_index and
        reply_key, which Processor.poll echoes so the bridge can route them to
        this method's collection queue. Fanout replies never settle the
        single-engine probe futures.
        """
        request_id = manifest.request_id
        n = len(self._command_queues)
        probe_key = f"{request_id}#probe"
        waiter: queue.Queue = queue.Queue(maxsize=n)
        loop = asyncio.get_running_loop()
        self._loop = loop
        with self._lock:
            stale_probe = self._probe_futures.pop(request_id, None)
            stale_usage = self._usage_futures.pop(request_id, None)
            stale_waiter = self._fanout_waiters.get(probe_key)
            usage_future: asyncio.Future = loop.create_future()
            self._usage_futures[request_id] = usage_future
            self._fanout_waiters[probe_key] = waiter
        if (
            stale_probe is not None
            or stale_usage is not None
            or (stale_waiter is not None)
        ):
            logger.warning(
                "[LWD][ctrl] duplicate control request req=%s: superseding "
                "the previous registration",
                request_id,
            )
            if stale_usage is not None and not stale_usage.done():
                stale_usage.set_result(None)
            if stale_probe is not None and not stale_probe.done():
                stale_probe.set_exception(
                    RuntimeError(
                        f"control request {request_id} superseded by a newer probe"
                    )
                )
            if stale_waiter is not None:
                with contextlib.suppress(queue.Full):
                    stale_waiter.put_nowait((False, -1, "superseded"))
        results: dict[int, LwdProbeResult] = {}
        try:
            for idx, cmd_queue in self._command_queue_map.items():
                cmd_queue.put(
                    {
                        "type": "probe",
                        "manifest": manifest,
                        "reserve": False,
                        "engine_index": idx,
                        "reply_key": probe_key,
                    }
                )
            deadline = loop.time() + timeout
            try:
                while len(results) < n:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        ok, eng_idx, payload = await asyncio.wait_for(
                            loop.run_in_executor(None, waiter.get), remaining
                        )
                    except asyncio.TimeoutError:
                        break
                    if eng_idx == -1 and payload == "superseded":
                        raise RuntimeError(
                            f"control request {request_id} superseded by a newer probe"
                        )
                    if ok:
                        results[eng_idx] = payload
                    else:
                        logger.warning(
                            "[LWD][ctrl] fanout probe failed req=%s engine=%s: %s",
                            request_id,
                            eng_idx,
                            payload,
                        )
            finally:
                with self._lock:
                    if self._fanout_waiters.get(probe_key) is waiter:
                        self._fanout_waiters.pop(probe_key, None)
                # A timed-out executor thread cannot be cancelled while blocked
                # in waiter.get. Poison it so every attempt releases its thread.
                with contextlib.suppress(Exception):
                    waiter.put_nowait((False, -1, "poison"))
            missing = [i for i in self._engine_ranks if i not in results]
            if missing:
                # Slow or failed engines count as zero-hit for observability but
                # are never placement candidates. Fail only if none answered.
                logger.warning(
                    "[LWD][ctrl] fanout probe partial req=%s missing=%s "
                    "(treated as zero hit)",
                    request_id,
                    missing,
                )
                if not results:
                    raise TimeoutError(
                        f"prefix probe fanout timed out after {timeout}s"
                    )
            # Select argmax(hit_tokens), rotating among tied live engines.
            hits = {
                i: results[i].hit_tokens if i in results else 0
                for i in self._engine_ranks
            }
            top = max(hits[i] for i in results) if results else 0
            candidates = [i for i in results if hits[i] == top]
            if len(candidates) > 1:
                winner = candidates[self._rr_counter % len(candidates)]
                self._rr_counter += 1
            else:
                winner = candidates[0]
            logger.info(
                "[LWD][ctrl] fanout placement req=%s hits=%s winner=%d",
                request_id,
                hits,
                winner,
            )
            with self._lock:
                if self._usage_futures.get(request_id) is not usage_future:
                    raise RuntimeError(
                        f"control request {request_id} superseded by a newer probe"
                    )
            return dataclass_replace(results[winner], engine_index=winner)
        except BaseException:
            # No handler will wait for usage after a failed probe. Remove only
            # this generation so a concurrent retry's registration survives.
            with self._lock:
                if self._usage_futures.get(request_id) is usage_future:
                    self._usage_futures.pop(request_id, None)
            raise

    async def wait_usage(
        self, request_id: str, timeout: float = _USAGE_WAIT_TIMEOUT
    ) -> LwdUsage | None:
        """Await the terminal usage event; timeout or no request returns None.

        The future is registered during ``probe()``, so early accounting is not
        lost. None lets the handler close the stream without a usage frame.
        """
        with self._lock:
            future = self._usage_futures.get(request_id)
        if future is None:
            logger.warning(
                "[LWD][ctrl] usage wait without control request req=%s",
                request_id,
            )
            return None
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            logger.warning("[LWD][ctrl] usage wait timeout req=%s", request_id)
            return None
        finally:
            with self._lock:
                # Remove only this generation: a same-ID probe may already
                # have installed its replacement.
                if self._usage_futures.get(request_id) is future:
                    self._usage_futures.pop(request_id, None)

    def _dispatch_events(self) -> None:
        """One resident thread forwards events without touching futures.

        Future settlement remains affine to the event-loop thread.
        """
        while not self._closed:
            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            kind = event.get("type")
            if kind in ("pd_register", "pd_export", "pd_result"):
                self.pd.dispatch(event)
                continue
            request_id = event.get("request_id")
            with self._lock:
                table = self._probe_futures if kind == "probe" else self._usage_futures
                registered = request_id in table
                loop = self._loop
            # One debug log per event, consistent with this branch's hot-path
            # logging policy.
            logger.debug(
                "[LWD][ctrl] dispatch got event type=%s req=%s ok=%s engine=%s",
                kind,
                request_id,
                event.get("ok"),
                event.get("engine_index"),
            )
            # Cloud-local DP fanout/reservation replies carry engine_index and
            # use reply_key to find their collection queue. Late replies after
            # the waiter exits are harmlessly dropped; they never settle the
            # ordinary futures.
            if kind == "probe" and "engine_index" in event:
                with self._lock:
                    waiter = self._fanout_waiters.get(request_id)
                if waiter is not None:
                    with contextlib.suppress(queue.Full):
                        waiter.put_nowait(
                            (
                                bool(event.get("ok")),
                                int(event.get("engine_index", 0)),
                                event.get("result")
                                if event.get("ok")
                                else event.get("error"),
                            )
                        )
                continue
            if not registered or loop is None:
                # The waiter timed out or disconnected, or this accounting
                # event preceded any probe (a data-plane request can finish
                # without one). Drop it without raising.
                logger.warning(
                    "[LWD][ctrl] orphan %s event req=%s (no waiter/loop)",
                    kind,
                    request_id,
                )
                continue
            try:
                loop.call_soon_threadsafe(self._deliver_event, event)
            except RuntimeError:
                # The loop closed during uvicorn shutdown versus atexit. No
                # coroutine can now be settled, and continuing would emit a
                # traceback per event. Stop dispatching, symmetrically with
                # close().
                logger.info(
                    "[LWD][ctrl] dispatch stopping: loop closed (%d control "
                    "request(s) abandoned)",
                    len(self._probe_futures) + len(self._usage_futures),
                )
                return

    def _deliver_event(self, event: dict[str, Any]) -> None:
        """Settle futures on the event-loop thread (the only settlement site)."""
        kind = event.get("type")
        request_id = event.get("request_id")
        if kind == "probe":
            with self._lock:
                future = self._probe_futures.pop(request_id, None)
            if future is None or future.done():
                return  # Timed out/cancelled; the edge already failed open.
            if event.get("ok"):
                result = event.get("result")
                if len(self._engine_ranks) == 1:
                    result = dataclass_replace(
                        result, engine_index=self._engine_ranks[0]
                    )
                future.set_result(result)
            else:
                future.set_exception(RuntimeError(event.get("error") or "probe failed"))
            return
        if kind == "usage":
            # Accounting carries only request_id, not a probe generation, so it
            # must settle the currently registered future. This does not assume
            # the cloud gently deduplicates requests: duplicate RangeNotify
            # suppression in lwd_cloud_engine covers only the gate-pool window
            # before admission. After that window, same-ID submission reaches
            # native Scheduler.add_request (scheduler.py:1823-1836), where a
            # streaming input session may fold into an update but otherwise the
            # "duplicate request id" assertion fires. One inflight attempt per
            # request ID is therefore a whole-stack invariant maintained by the
            # edge nonce/abort tombstones, coordinator reservation replacement,
            # and the gate-pool window, with that assertion as a backstop.
            with self._lock:
                future = self._usage_futures.get(request_id)
            if future is None or future.done():
                logger.warning(
                    "[LWD][ctrl] usage event arrived too late req=%s", request_id
                )
                return
            future.set_result(event.get("usage"))


class LwdCloudControlProcessor:
    """Child-side non-blocking poll at step-loop entry or after WAKEUP.

    Commands come from a local thread-safe queue; the wakeup thread already
    moved them from the multiprocessing queue and sent WAKEUP. This avoids the
    mp.Queue.get_nowait feeder race.
    """

    def __init__(self, pending_queue: Any, event_queue: Any) -> None:
        self._pending_queue = pending_queue
        self._event_queue = event_queue

    def poll(self, coordinator: Any, *, pd_handler: Any = None) -> int:
        """Consume pending commands and return the count ([LWD][perf] I4).

        A count above one directly shows probes waiting for the cloud engine
        step boundary and being consumed as a batch; probe service capacity is
        constrained by cloud step frequency.
        """
        consumed = 0
        while True:
            try:
                command = self._pending_queue.get_nowait()
            except queue.Empty:
                return consumed
            consumed += 1
            logger.debug(
                "[LWD][ctrl] poll consumed command type=%s", command.get("type")
            )
            if command.get("type") == "pd_register":
                if pd_handler is None:
                    self._event_queue.put(
                        {
                            "type": "pd_register",
                            "request_id": wrap_req_id(
                                command["envelope"]["edge_id"],
                                command["envelope"]["rid"],
                            ),
                            "attempt_id": command["envelope"]["attempt_id"],
                            "ok": False,
                            "error": "PD is not enabled on this engine",
                        }
                    )
                else:
                    pd_handler(command)
                continue
            if command.get("type") == "probe":
                manifest: LwdPrefixManifest = command["manifest"]
                # Cloud-local DP protocol fields (defaults preserve legacy
                # single-queue semantics):
                #   reserve=False: read-only fanout query, no reservation;
                #   reply_key: response routing key for fanout/reservation;
                #   engine_index: echoed so the bridge can identify the engine.
                reserve = command.get("reserve", True)
                reply_key = command.get("reply_key", manifest.request_id)
                # [LWD][perf] I8a: cmd_wait_ms is time from parent submission
                # to engine-loop collection (mp queue + wakeup thread + waiting
                # for a step boundary). If near the edge's http_ms, the engine
                # loop was too busy to collect; if near zero, the remaining time
                # was spent returning and settling the result.
                t_take = time.monotonic()
                t_cmd = command.get("t_cmd")
                cmd_wait_ms = (t_take - t_cmd) * 1000.0 if t_cmd is not None else -1.0
                try:
                    if reserve:
                        result = coordinator.probe(manifest)
                    else:
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
                        "ok": True,
                        "result": result,
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.exception("[LWD][ctrl] probe failed")
                    event = {
                        "type": "probe",
                        "request_id": reply_key,
                        "ok": False,
                        "error": str(exc),
                    }
                if "engine_index" in command:
                    event["engine_index"] = command["engine_index"]
                self._event_queue.put(event)


def _negotiate_lwd_protocol_headers(
    headers: Mapping[str, str],
    expected_mm_abi: str | None,
    *,
    enforce_mm_abi_match: bool,
) -> str | None:
    """Return the validated MM-ABI header for v2, or None for v1.

    Preserve v1 semantics. Reject MM-ABI on v1 and missing or malformed MM-ABI
    on v2: the protocol stacks must not downgrade into each other, which would
    account for image chains as text and silently misalign hits.
    ``enforce_mm_abi_match`` defaults to enabled: edge and cloud compare their
    locally computed processor fingerprints, and the cloud response advertises
    only its own identity rather than echoing the edge value.
    """
    normalized = {k.lower(): v for k, v in headers.items()}
    protocol = normalized.get(HEADER_PROTOCOL.lower())
    mm_abi = normalized.get(HEADER_MM_ABI.lower())
    if protocol == PROTOCOL_VERSION_MM:
        value = validate_mm_abi_header(mm_abi)
        if expected_mm_abi is None:
            raise ValueError(
                f"{PROTOCOL_VERSION_MM} requires a locally computed "
                f"{HEADER_MM_ABI} fingerprint on the cloud"
            )
        if enforce_mm_abi_match and value != expected_mm_abi:
            raise ValueError(f"{HEADER_MM_ABI} fingerprint does not match this cloud")
        # Honest negotiation: advertise the cloud's own identity. Echoing the
        # edge value would make an unchecked negotiation look successful.
        return expected_mm_abi
    if mm_abi is not None:
        raise ValueError(f"{HEADER_MM_ABI} requires protocol {PROTOCOL_VERSION_MM}")
    return None


def _sse_data(obj: Any) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def _build_chunk(
    request_id: str, model: str, choices: list, usage: dict | None = None
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def create_lwd_cloud_control_app(
    bridge: LwdCloudControlBridge,
    processor_fingerprint: bytes | None = None,
    *,
    enforce_mm_abi_match: bool = True,
    probe_timeout: float = 25.0,
    pd_config: Any = None,
    stage_role: str = "",
    pd_model_names: tuple[str, ...] = (),
) -> Any:
    """FastAPI control-plane app; successful responses use OpenAI SSE.

    Error paths still use JSONResponse, which gateways handle standardly.
    ``processor_fingerprint`` is the cloud's local processor identity and is
    used for v2 negotiation only when ``enforce_mm_abi_match`` is enabled.
    ``probe_timeout`` is this process's hard probe-result wait limit, derived
    from the edge ``probe_timeout`` minus headroom (see the constant above).
    """
    app = FastAPI()
    expected_mm_abi = (
        mm_abi_header_value(processor_fingerprint)
        if processor_fingerprint is not None
        else None
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Gateway health check; not part of edge-cloud negotiation."""
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            headers = dict(request.headers)
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)
        from vllm.v1.lwd_control.pd.cloud_http import (
            lwd_pd_envelope_from_body,
            lwd_pd_http_response,
            lwd_validate_pd_model_identity,
        )

        try:
            envelope = lwd_pd_envelope_from_body(body)
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if envelope is not None:
            if pd_config is None:
                return JSONResponse({"error": "PD is not configured"}, status_code=400)
            try:
                lwd_validate_pd_model_identity(
                    envelope,
                    processor_fingerprint,
                    pd_model_names,
                    enforce_mm_abi_match=enforce_mm_abi_match,
                )
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            return await lwd_pd_http_response(
                bridge.pd, body, envelope, stage_role, pd_config
            )
        try:
            mm_abi_header = _negotiate_lwd_protocol_headers(
                headers,
                expected_mm_abi,
                enforce_mm_abi_match=enforce_mm_abi_match,
            )
            parse_headers: Mapping[str, str] = headers
            if mm_abi_header is not None:
                # v2 has the same wire shape as v1. After validating protocol
                # and ABI, parse with the v1 parser (which accepts only the v1
                # protocol marker).
                parse_headers = {
                    k: v
                    for k, v in headers.items()
                    if k.lower() != HEADER_PROTOCOL.lower()
                }
                parse_headers[HEADER_PROTOCOL] = PROTOCOL_VERSION
            manifest = LwdPrefixManifest.from_openai_request(parse_headers, body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        # Multi-edge namespace wrapping: use the advertised
        # X-Edge-Cloud-Edge-Id to wrap manifest.request_id as cloud-unique
        # "e{edge_id}-{req_id}", matching the data-plane _lwd_dispatch wrap so
        # the coordinator reservation key equals the data-plane request key.
        # ASGI/Starlette header keys are lowercase and must be read as such;
        # otherwise this falls back to "0" and the reservation/request keys
        # diverge, breaking claim and prefix validation.
        headers_lc = {k.lower(): v for k, v in headers.items()}
        try:
            edge_id = int(headers_lc.get(HEADER_EDGE_ID.lower(), "0"))
        except (TypeError, ValueError) as exc:
            return JSONResponse(
                {"error": f"invalid {HEADER_EDGE_ID}: {exc}"}, status_code=400
            )
        if edge_id < 0:
            return JSONResponse({"error": "negative edge_id"}, status_code=400)
        manifest = dataclass_replace(
            manifest, request_id=wrap_req_id(edge_id, manifest.request_id)
        )

        try:
            logger.debug(
                "[LWD][ctrl] handler entering probe req=%s edge=%d",
                manifest.request_id,
                edge_id,
            )
            result = await bridge.probe(manifest, timeout=probe_timeout)
            logger.debug(
                "[LWD][ctrl] handler probe resolved req=%s hit=%d",
                manifest.request_id,
                result.hit_tokens,
            )
        except TimeoutError as exc:
            # This limit gives the edge headroom, so normally the cloud answers
            # first (200/503). Reaching it means the engine loop did not consume
            # the probe within the derived window under data-plane load. The
            # edge has probably failed open, making this result observational.
            # Log one warning without a traceback; frequent occurrences indicate
            # control-plane latency degradation and warrant checking step time.
            logger.warning(
                "[LWD][ctrl] probe timeout req=%s edge=%d: %s "
                "(engine loop did not answer in time; edge already "
                "failed-open)",
                manifest.request_id,
                edge_id,
                exc,
            )
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[LWD][ctrl] handler probe raised")
            return JSONResponse({"error": str(exc)}, status_code=503)

        # Probe results use SSE delta.content to survive gateway DTO conversion;
        # headers remain for direct debugging/observability. Usage travels in
        # the final chunk of this stream for gateway accounting. Echo the
        # original unwrapped request ID to the edge for consistency validation.
        display_request_id = result.request_id
        if display_request_id.startswith(f"e{edge_id}-"):
            display_request_id = display_request_id[len(f"e{edge_id}-") :]
        model = body.get("model", "edge-cloud-prefix")

        # Echo the negotiated protocol so the edge can verify that the cloud
        # did not silently downgrade to v1.
        protocol = (
            PROTOCOL_VERSION_MM if mm_abi_header is not None else PROTOCOL_VERSION
        )
        control_message = result.to_control_message(protocol, mm_abi_header)
        # Use the display form in the control message.
        control_message["request_id"] = display_request_id

        async def _events_inner() -> AsyncIterator[str]:
            # 1. Probe control message in delta.content, preserved by DTOs.
            yield _sse_data(
                _build_chunk(
                    display_request_id,
                    model,
                    [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": json.dumps(
                                    control_message, separators=(",", ":")
                                ),
                            },
                            "finish_reason": None,
                        }
                    ],
                )
            )
            # 2. Empty delta frame flushes gateway buffering (NewAPI, etc.).
            yield _sse_data(
                _build_chunk(
                    display_request_id,
                    model,
                    [{"index": 0, "delta": {}, "finish_reason": None}],
                )
            )
            # 3. Wait for usage pushed by the engine after inference finishes.
            usage: LwdUsage | None = None
            usage_task = asyncio.create_task(bridge.wait_usage(manifest.request_id))
            try:
                while not usage_task.done():
                    done, _ = await asyncio.wait(
                        {usage_task}, timeout=_USAGE_HEARTBEAT_INTERVAL
                    )
                    if not done:
                        yield ": edge-cloud-usage-pending\n\n"
                usage = usage_task.result()
            except asyncio.CancelledError:
                if not usage_task.done():
                    usage_task.cancel()
                await asyncio.gather(usage_task, return_exceptions=True)
                raise
            # 4. Final usage chunk + [DONE].
            if usage is None:
                # Without real usage, never emit a usage frame: fabricated
                # zeros would create zero-quota accounting. Close normally with
                # a comment + [DONE], avoiding an aborted-stream client_gone or
                # gateway error while making the absence of usage explicit.
                logger.error(
                    "[LWD][ctrl] usage unavailable req=%s: ending stream "
                    "without usage frame (never bill fabricated zeros)",
                    manifest.request_id,
                )
                yield ": edge-cloud-usage-unavailable\n\n"
                yield "data: [DONE]\n\n"
                return
            yield _sse_data(
                _build_chunk(
                    display_request_id,
                    model,
                    [],
                    usage=usage.to_openai_dict(),
                )
            )
            yield "data: [DONE]\n\n"

        async def events() -> AsyncIterator[str]:
            try:
                async for frame in _events_inner():
                    yield frame
            finally:
                # Remove registrations after normal completion, early edge
                # disconnect, or generator cancellation; otherwise usage waits
                # for process close() as the only fallback.
                bridge.release(manifest.request_id)

        headers_out = {
            **result.to_headers(),
            # The coordinator holds the wrapped cloud-local ID; response headers
            # must echo the unwrapped form so edge/cloud request-ID reconciliation
            # uses the same value.
            HEADER_REQUEST_ID: display_request_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        if mm_abi_header is not None:
            headers_out[HEADER_PROTOCOL] = PROTOCOL_VERSION_MM
            headers_out[HEADER_MM_ABI] = mm_abi_header
        return StreamingResponse(
            events(), media_type="text/event-stream", headers=headers_out
        )

    return app


def start_lwd_cloud_control_server(
    vllm_config: Any,
    *,
    managed_dp_ranks: tuple[int, ...] | None = None,
) -> tuple[dict[int, Any] | None, Any | None]:
    """Start the parent HTTP control service and queue bridge when enabled.

    Return ``(command_queues, event_queue)`` for child EngineCore processes via
    ``CoreEngineProcManager``; return ``(None, None)`` when coordination is
    disabled or the role is not cloud. command_queues maps the actual locally
    managed global DP ranks to queues. External-LB services do not create or
    wait for queues belonging to engines in another process. event_queue is
    shared by the managed engines (events carry their global engine_index).

    Uvicorn and the bridge run on background threads; call this before spawning
    EngineCoreProc.
    """
    lwd_config = getattr(vllm_config, "lwd_config", None)
    coord_cfg = getattr(lwd_config, "coordination", None)
    if coord_cfg is None or not coord_cfg.enabled:
        return None, None
    if lwd_config.role != "cloud":
        return None, None

    import uvicorn

    from vllm.utils.system_utils import get_mp_context

    mp_ctx = get_mp_context()
    # Each cloud-local DP engine has an independent KV pool/prefix cache, so
    # probes must fan out to every engine.
    dp_size = max(int(vllm_config.parallel_config.data_parallel_size), 1)
    if managed_dp_ranks is None:
        parallel = vllm_config.parallel_config
        count = int(getattr(parallel, "data_parallel_size_local", None) or dp_size)
        first = int(parallel.data_parallel_rank)
        managed_dp_ranks = tuple(range(first, first + count))
    command_queues = {rank: mp_ctx.Queue() for rank in managed_dp_ranks}
    event_queue = mp_ctx.Queue()
    bridge = LwdCloudControlBridge(command_queues, event_queue)
    bridge.start()
    # Uvicorn runs on a daemon thread without a shutdown channel. Register
    # atexit to stop dispatch and fail pending futures instead of relying on a
    # hard daemon-thread kill.
    atexit.register(bridge.close)
    # Compute the cloud fingerprint with the same mm_identity implementation as
    # the edge's _lwd_processor_fingerprint; use it for v2 only when enforcement
    # is enabled. Derive the probe limit from the edge timeout minus headroom so
    # the cloud answers first (200/503).
    cloud_probe_timeout = max(coord_cfg.probe_timeout - _CLOUD_PROBE_HEADROOM_S, 1.0)
    model_names = vllm_config.model_config.served_model_name
    if isinstance(model_names, str):
        model_names = (model_names,)
    app = create_lwd_cloud_control_app(
        bridge,
        compute_processor_fingerprint(vllm_config.model_config),
        enforce_mm_abi_match=coord_cfg.enforce_mm_abi_match,
        probe_timeout=cloud_probe_timeout,
        pd_config=lwd_config.pd,
        stage_role=lwd_config.stage_role,
        pd_model_names=tuple(model_names or (vllm_config.model_config.model,)),
    )
    listen_port = coord_cfg.listen_port
    if lwd_config.pd.enabled and vllm_config.parallel_config.data_parallel_external_lb:
        listen_port += int(vllm_config.parallel_config.data_parallel_rank)
    if listen_port > 65535:
        raise ValueError("[LWD] per-DP control port exceeds 65535")
    threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={
            "host": coord_cfg.listen_host,
            "port": listen_port,
            "log_level": "warning",
        },
        name="lwd-cloud-ctrl",
        daemon=True,
    ).start()
    logger.info(
        "[LWD] cloud control HTTP server listening on %s:%s "
        "(cloud dp_size=%d, %d engine command queue(s), "
        "probe_timeout=%.1fs enforce_mm_abi_match=%s)",
        coord_cfg.listen_host,
        listen_port,
        dp_size,
        len(command_queues),
        cloud_probe_timeout,
        coord_cfg.enforce_mm_abi_match,
    )
    return command_queues, event_queue
