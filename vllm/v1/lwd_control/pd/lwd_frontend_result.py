# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Frontend-owned immutable results until the edge engine accepts ownership."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

LwdFrontendReason = Literal[
    "complete", "stop_string", "cancelled", "disconnected", "timeout", "error"
]


@dataclass(frozen=True)
class LwdFrontendResult:
    request_id: str
    attempt_id: str
    accepted_tokens: int
    frontend_reason: LwdFrontendReason
    cancel_execution: bool

    @property
    def key(self) -> tuple[str, str]:
        return self.request_id, self.attempt_id


class LwdFrontendResultInbox:
    """Freezing is synchronous, so RequestState can be freed after this returns."""

    def __init__(self, retention_s: float, max_records: int = 4096) -> None:
        self.retention_s = retention_s
        self.max_records = max_records
        self.records: dict[tuple[str, str], LwdFrontendResult] = {}
        self.pending: dict[tuple[str, str], float] = {}
        self.confirmed: dict[tuple[str, str], float] = {}
        self.unconfirmed: set[tuple[str, str]] = set()
        self.handoff_failed: set[tuple[str, str]] = set()

    def freeze(self, result: LwdFrontendResult) -> None:
        if (
            not result.attempt_id
            or type(result.accepted_tokens) is not int
            or result.accepted_tokens < 0
        ):
            raise ValueError("PD frontend result requires an attempt and a real K")
        if previous := self.records.get(result.key):
            if previous != result:
                raise ValueError("conflicting immutable PD frontend result")
            return
        now = time.monotonic()
        self.expire(now)
        if len(self.records) >= self.max_records:
            raise RuntimeError(
                "PD frontend result inbox is full; result not handed off"
            )
        self.records[result.key] = result
        self.pending[result.key] = now + self.retention_s

    def acknowledge(self, result: LwdFrontendResult) -> None:
        if self.records.get(result.key) != result:
            raise ValueError("PD local receipt does not match the frozen result")
        self.pending.pop(result.key, None)
        self.unconfirmed.discard(result.key)
        self.handoff_failed.discard(result.key)
        self.confirmed[result.key] = time.monotonic() + self.retention_s

    def expire(self, now: float | None = None) -> list[LwdFrontendResult]:
        now = time.monotonic() if now is None else now
        expired = [
            key
            for key, deadline in self.pending.items()
            if deadline <= now and key not in self.unconfirmed
        ]
        self.unconfirmed.update(expired)
        for key, deadline in list(self.confirmed.items()):
            if deadline <= now:
                self.confirmed.pop(key)
                self.records.pop(key, None)
        # Unconfirmed records keep their real K; exhausting the bound fails
        # the local handoff rather than silently replacing a missing result.
        return [self.records[key] for key in expired]


class LwdFrontendResultHandoff:
    """Local utility retries; the edge engine owns remote ResultCommit retries."""

    def __init__(
        self,
        inbox: LwdFrontendResultInbox,
        utility: Callable[..., Awaitable[Any]],
        logger: Any,
        *,
        timeout_s: float = 1.0,
    ) -> None:
        self.inbox = inbox
        self.utility = utility
        self.logger = logger
        self.timeout_s = timeout_s
        self._lock = asyncio.Lock()

    async def flush(self, request_ids: set[str] | None = None) -> None:
        async with self._lock:
            for record in self.inbox.expire():
                self.logger.error(
                    "[LWD] PD result locally unconfirmed: request=%s attempt=%s K=%d",
                    record.request_id,
                    record.attempt_id,
                    record.accepted_tokens,
                )
            candidates = [
                self.inbox.records[key]
                for key in tuple(self.inbox.pending)
                if key not in self.inbox.unconfirmed
                and (request_ids is None or key[0] in request_ids)
            ]
            # A lost utility receipt must not serialize N one-second waits
            # on the output loop. Background retries also have a bounded batch.
            selected = candidates if request_ids is not None else candidates[:32]
            await asyncio.gather(*(self._deliver(record) for record in selected))

    async def _deliver(self, record: LwdFrontendResult) -> None:
        try:
            accepted = await asyncio.wait_for(
                self.utility(
                    "lwd_pd_finalize",
                    record.request_id,
                    record.attempt_id,
                    record.accepted_tokens,
                    record.frontend_reason,
                    record.cancel_execution,
                ),
                self.timeout_s,
            )
            if accepted is False:
                raise RuntimeError("edge engine rejected PD result handoff")
        except Exception as exc:
            if record.key not in self.inbox.handoff_failed:
                self.logger.warning(
                    "[LWD] PD local result handoff unconfirmed; retaining K: "
                    "request=%s attempt=%s error=%s",
                    record.request_id,
                    record.attempt_id,
                    type(exc).__name__,
                )
            self.inbox.handoff_failed.add(record.key)
            deadline = self.inbox.pending.pop(record.key, None)
            if deadline is not None:
                self.inbox.pending[record.key] = deadline
            return
        self.inbox.acknowledge(record)

    async def run(self) -> None:
        while True:
            await self.flush()
            await asyncio.sleep(0.25)
