# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Explicit-DP PD HTTP registration and stage-result futures."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

from vllm.v1.lwd_control.lwd_comm.lwd_id_adapter import wrap_req_id
from vllm.v1.lwd_control.pd.envelope import ENVELOPE_TYPE, parse_envelope


@dataclass(eq=False)
class LwdPdHttpRegistration:
    request_id: str
    attempt_id: str
    loop: asyncio.AbstractEventLoop
    registered: asyncio.Future
    exported: asyncio.Future
    result: asyncio.Future


class LwdPdCloudHttpBridge:
    """Forward only to the Proxy-selected DP; never probe or reselect."""

    def __init__(self, command_queues: dict[int, Any]) -> None:
        self.command_queues = command_queues
        self.pending: dict[tuple[str, str], list[LwdPdHttpRegistration]] = {}
        self._lock = threading.Lock()

    async def register(
        self,
        envelope: dict[str, Any],
        target_dp: int,
        kv_params: dict[str, Any] | None,
        timeout: float,
    ) -> LwdPdHttpRegistration:
        if type(target_dp) is not int or target_dp not in self.command_queues:
            raise ValueError(
                "target_dp must identify a DP managed by this control service"
            )
        loop = asyncio.get_running_loop()
        request_id = wrap_req_id(envelope["edge_id"], envelope["rid"])
        registration = LwdPdHttpRegistration(
            request_id,
            envelope["attempt_id"],
            loop,
            loop.create_future(),
            loop.create_future(),
            loop.create_future(),
        )
        key = request_id, envelope["attempt_id"]
        with self._lock:
            self.pending.setdefault(key, []).append(registration)
        self.command_queues[target_dp].put(
            {
                "type": "pd_register",
                "envelope": envelope,
                "engine_index": target_dp,
                "kv_params": kv_params,
            }
        )
        try:
            await asyncio.wait_for(asyncio.shield(registration.registered), timeout)
            return registration
        except BaseException:
            self.release(registration)
            raise

    def dispatch(self, event: dict[str, Any]) -> None:
        key = event.get("request_id"), event.get("attempt_id")
        with self._lock:
            registrations = tuple(self.pending.get(key, ()))
        for registration in registrations:
            try:
                registration.loop.call_soon_threadsafe(
                    self._deliver, registration, event
                )
            except RuntimeError:
                self.release(registration)

    @staticmethod
    def _deliver(registration: LwdPdHttpRegistration, event: dict[str, Any]) -> None:
        future = {
            "pd_register": registration.registered,
            "pd_export": registration.exported,
            "pd_result": registration.result,
        }.get(event["type"])
        if future is None or future.done():
            return
        if event.get("ok"):
            future.set_result(event)
        else:
            future.set_exception(ValueError(event.get("error", "PD stage failed")))
            # An export/result can fail before its HTTP coroutine awaits it.
            future.exception()

    def release(self, registration: LwdPdHttpRegistration) -> None:
        key = registration.request_id, registration.attempt_id
        with self._lock:
            waiters = self.pending.get(key, [])
            if registration in waiters:
                waiters.remove(registration)
            if not waiters:
                self.pending.pop(key, None)
        for future in (
            registration.registered,
            registration.exported,
            registration.result,
        ):
            if not future.done():
                future.cancel()

    def close(self) -> None:
        with self._lock:
            pending = [tuple(waiters) for waiters in self.pending.values()]
        for waiters in pending:
            for registration in tuple(waiters):
                event = {
                    "type": "pd_result",
                    "ok": False,
                    "error": "PD control service stopped",
                }
                for kind in ("pd_register", "pd_export", "pd_result"):
                    registration.loop.call_soon_threadsafe(
                        self._deliver, registration, {**event, "type": kind}
                    )


def lwd_pd_envelope_from_body(body: dict[str, Any]) -> dict[str, Any] | None:
    messages = body.get("messages")
    content = (
        messages[0].get("content") if isinstance(messages, list) and messages else None
    )
    if not isinstance(content, str):
        return None
    try:
        candidate = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(candidate, dict) or candidate.get("type") != ENVELOPE_TYPE:
        return None
    return parse_envelope(content)


def lwd_validate_pd_model_identity(
    envelope: dict[str, Any],
    processor_fingerprint: bytes | None,
    model_names: tuple[str, ...],
    *,
    enforce_mm_abi_match: bool,
) -> None:
    """Use the immutable envelope when a Proxy replaces the outer headers."""
    if enforce_mm_abi_match:
        if processor_fingerprint is None:
            raise ValueError("PD requires this cloud's local processor fingerprint")
        if envelope["input_abi"] != processor_fingerprint.hex():
            raise ValueError("PD input processor ABI does not match this cloud")
    if model_names and envelope["model_id"] not in model_names:
        raise ValueError("PD model identity is not served by this cloud")


async def lwd_pd_http_response(
    bridge: LwdPdCloudHttpBridge,
    body: dict[str, Any],
    envelope: dict[str, Any],
    stage_role: str,
    config: Any,
) -> Any:
    if not config.enabled or stage_role not in ("prefill", "decode"):
        return JSONResponse(
            {"error": "PD is not enabled on this cloud"}, status_code=400
        )
    if body.get("stage_role", stage_role) != stage_role:
        return JSONResponse(
            {"error": "PD request addressed to the wrong stage"}, status_code=400
        )
    try:
        registration = await bridge.register(
            envelope,
            body.get("target_dp"),
            body.get("kv_transfer_params"),
            config.notify_claim_timeout_s,
        )
    except (ValueError, TimeoutError) as exc:
        return JSONResponse(
            {"error": str(exc)}, status_code=400 if isinstance(exc, ValueError) else 504
        )
    common = {
        "id": envelope["rid"],
        "created": int(time.time()),
        "model": body.get("model", "lwd-pd"),
        "lwd_pd_attempt_id": envelope["attempt_id"],
    }
    if stage_role == "prefill":
        try:
            event = await asyncio.wait_for(
                asyncio.shield(registration.exported), config.export_hold_timeout_s
            )
            return JSONResponse(
                {
                    **common,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "stop",
                        }
                    ],
                    "kv_transfer_params": event["kv_transfer_params"],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )
        except (ValueError, TimeoutError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)
        finally:
            bridge.release(registration)

    async def events():
        try:
            deadline = asyncio.get_running_loop().time() + config.usage_wait_timeout_s
            while not registration.result.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("PD final result unconfirmed")
                done, _ = await asyncio.wait(
                    {registration.result}, timeout=min(10.0, remaining)
                )
                if not done:
                    yield ": lwd-pd-result-pending\n\n"
            result = registration.result.result()["result"]
            chunk = {
                **common,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": result["usage"],
                "lwd_pd_result": result,
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        except (ValueError, TimeoutError) as exc:
            error = {"error": str(exc), "lwd_pd_status": "unconfirmed"}
            yield f"data: {json.dumps(error)}\n\n"
        finally:
            bridge.release(registration)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
