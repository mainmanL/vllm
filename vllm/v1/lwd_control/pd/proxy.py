# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Required PD orchestrator: choose each stage DP once per request attempt."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .envelope import parse_envelope


@dataclass(frozen=True)
class LwdPdProxyTarget:
    cloud_id: int
    target_dp: int
    url: str
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.cloud_id, self.target_dp)
        ):
            raise ValueError("Proxy cloud_id/target_dp must be non-negative integers")
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ValueError("Proxy target url must be an HTTP(S) endpoint")
        if not isinstance(self.headers, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in self.headers.items()
        ):
            raise ValueError("Proxy target headers must contain strings")


@dataclass(frozen=True)
class LwdPdProxyConfig:
    prefill: tuple[LwdPdProxyTarget, ...]
    decode: tuple[LwdPdProxyTarget, ...]
    max_attempts: int = 4096
    max_response_bytes: int = 1024 * 1024
    request_timeout_s: float = 600.0

    def __post_init__(self) -> None:
        for targets in (self.prefill, self.decode):
            if not targets or len({(t.cloud_id, t.target_dp) for t in targets}) != len(
                targets
            ):
                raise ValueError("Proxy stage targets must be nonempty and unique")
        if {t.cloud_id for t in self.prefill} & {t.cloud_id for t in self.decode}:
            raise ValueError("P and D must be independent cloud instances")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("Proxy max_attempts must be positive")
        if type(self.max_response_bytes) is not int or self.max_response_bytes < 1024:
            raise ValueError("Proxy max_response_bytes must be at least 1024")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("Proxy request_timeout_s must be finite and positive")

    @classmethod
    def from_file(cls, path: str) -> LwdPdProxyConfig:
        raw = json.loads(Path(path).read_text())
        return cls(
            **{
                **raw,
                "prefill": tuple(LwdPdProxyTarget(**t) for t in raw["prefill"]),
                "decode": tuple(LwdPdProxyTarget(**t) for t in raw["decode"]),
            },
        )


@dataclass
class _LwdProxyAttempt:
    fingerprint: str
    expires_at: float
    prefill: LwdPdProxyTarget
    decode: LwdPdProxyTarget
    chunks: list[bytes] = field(default_factory=list)
    updated: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    response_bytes: int = 0
    done: bool = False


class LwdPdProxy:
    """Bounded attempt cache; retrying attaches to the original stage execution."""

    def __init__(self, config: LwdPdProxyConfig, client: httpx.AsyncClient) -> None:
        self.config = config
        self.client = client
        self.attempts: dict[tuple[str, int, str, str], _LwdProxyAttempt] = {}
        self._next_p = 0
        self._next_d = 0
        self._lock = asyncio.Lock()

    async def register(self, body: dict[str, Any]) -> _LwdProxyAttempt:
        try:
            content = body["messages"][0]["content"]
            envelope = parse_envelope(content)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise HTTPException(400, f"invalid PD envelope: {exc}") from exc
        key = (
            envelope["tenant_id"],
            envelope["edge_id"],
            envelope["rid"],
            envelope["attempt_id"],
        )
        async with self._lock:
            now = time.time()
            for expired_key, attempt in list(self.attempts.items()):
                if attempt.expires_at <= now:
                    if attempt.task is not None and not attempt.task.done():
                        attempt.task.cancel()
                    self.attempts.pop(expired_key)
            if existing := self.attempts.get(key):
                if existing.fingerprint != envelope["fingerprint"]:
                    raise HTTPException(
                        409, "PD attempt was reused with different semantics"
                    )
                return existing
            if len(self.attempts) >= self.config.max_attempts:
                # Never evict a live identity: a retry could otherwise rerun it.
                raise HTTPException(503, "PD Proxy attempt cache is full")
            p_target = self.config.prefill[self._next_p % len(self.config.prefill)]
            d_target = self.config.decode[self._next_d % len(self.config.decode)]
            self._next_p += 1
            self._next_d += 1
            attempt = _LwdProxyAttempt(
                envelope["fingerprint"],
                envelope["issued_at"] + envelope["ttl_s"],
                p_target,
                d_target,
            )
            self.attempts[key] = attempt
            attempt.task = asyncio.create_task(
                self._produce(attempt, copy.deepcopy(body))
            )
            return attempt

    def _append(self, attempt: _LwdProxyAttempt, chunk: bytes) -> None:
        if attempt.response_bytes + len(chunk) > self.config.max_response_bytes:
            raise ValueError("PD control response exceeded its bounded replay buffer")
        attempt.chunks.append(chunk)
        attempt.response_bytes += len(chunk)
        attempt.updated.set()

    async def _execute(self, attempt: _LwdProxyAttempt, body: dict[str, Any]) -> None:
        p_body = {
            **body,
            "target_dp": attempt.prefill.target_dp,
            "stage_role": "prefill",
            "stream": False,
            "max_tokens": 1,
        }
        p_body.pop("kv_transfer_params", None)
        response = await self.client.post(
            attempt.prefill.url,
            json=p_body,
            headers=attempt.prefill.headers,
            timeout=self.config.request_timeout_s,
        )
        response.raise_for_status()
        exported = response.json().get("kv_transfer_params")
        if not isinstance(exported, dict):
            raise ValueError("P did not return kv_transfer_params")
        d_body = {
            **body,
            "target_dp": attempt.decode.target_dp,
            "stage_role": "decode",
            "stream": True,
            "kv_transfer_params": exported,
        }
        async with self.client.stream(
            "POST",
            attempt.decode.url,
            json=d_body,
            headers=attempt.decode.headers,
            timeout=self.config.request_timeout_s,
        ) as stream:
            stream.raise_for_status()
            if "text/event-stream" not in stream.headers.get("content-type", ""):
                raise ValueError("D did not return an SSE control response")
            terminal = b""
            saw_done = False
            async for chunk in stream.aiter_bytes():
                self._append(attempt, chunk)
                terminal = (terminal + chunk)[-64:]
                saw_done = saw_done or b"data: [DONE]" in terminal
            if not saw_done:
                raise ConnectionError(
                    "D control stream ended before its terminal marker"
                )

    async def _produce(self, attempt: _LwdProxyAttempt, body: dict[str, Any]) -> None:
        try:
            remaining = max(0.0, attempt.expires_at - time.time())
            await asyncio.wait_for(self._execute(attempt, body), remaining)
        except (Exception, asyncio.CancelledError) as exc:
            # Reserve a small terminal marker in addition to the configured
            # response cap. No missing result is converted into zero usage.
            error = json.dumps(
                {
                    "error": f"PD stage failed: {type(exc).__name__}",
                    "lwd_pd_status": "unconfirmed",
                }
            )
            attempt.chunks.append(f"data: {error}\n\ndata: [DONE]\n\n".encode())
        finally:
            attempt.done = True
            attempt.updated.set()

    async def events(self, attempt: _LwdProxyAttempt) -> AsyncIterator[bytes]:
        cursor = 0
        while True:
            while cursor < len(attempt.chunks):
                chunk = attempt.chunks[cursor]
                cursor += 1
                yield chunk
            if attempt.done:
                return
            attempt.updated.clear()
            if cursor < len(attempt.chunks) or attempt.done:
                continue
            await attempt.updated.wait()

    async def close(self) -> None:
        tasks = [a.task for a in self.attempts.values() if a.task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def create_app(
    config: LwdPdProxyConfig, client: httpx.AsyncClient | None = None
) -> FastAPI:
    owned_client = client is None
    upstream = client or httpx.AsyncClient()
    proxy = LwdPdProxy(config, upstream)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await proxy.close()
        if owned_client:
            await upstream.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.pd_proxy = proxy

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> StreamingResponse:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "PD request body must be an object")
        attempt = await proxy.register(body)
        return StreamingResponse(
            proxy.events(attempt),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()
    uvicorn.run(
        create_app(LwdPdProxyConfig.from_file(args.config)),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
