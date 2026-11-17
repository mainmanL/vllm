# SPDX-License-Identifier: Apache-2.0
"""Engine-owned, bounded result retries; never retains execution resources."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LwdPdResultCommit,
    LwdPdResultReceipt,
)

from .results import validate_result


@dataclass
class _LwdPendingCommit:
    connection: Any
    message: LwdPdResultCommit
    deadline: float
    next_send: float
    receipt: dict[str, Any] | None = None


class LwdPdCommitOutbox:
    """Call from the engine thread; the caller drives a nonblocking timer."""

    def __init__(self, retry_interval: float = 1.0, capacity: int = 8192) -> None:
        if retry_interval <= 0 or capacity <= 0:
            raise ValueError("invalid PD retry bounds")
        self.retry_interval = retry_interval
        self.capacity = capacity
        self._pending: dict[tuple[str, str], _LwdPendingCommit] = {}

    def put(
        self, connection: Any, message: LwdPdResultCommit, now: float, timeout: float
    ) -> None:
        key = (message.request_id, message.attempt_id)
        existing = self._pending.get(key)
        if existing is not None:
            if existing.message != message or existing.connection != connection:
                raise ValueError("conflicting PD frontend finalization")
            return
        if len(self._pending) >= self.capacity:
            raise RuntimeError("PD result outbox capacity exhausted")
        if timeout <= 0:
            raise ValueError("PD result timeout must be positive")
        self._pending[key] = _LwdPendingCommit(connection, message, now + timeout, now)

    def due(self, now: float) -> list[tuple[Any, LwdPdResultCommit]]:
        messages = []
        for pending in self._pending.values():
            if pending.receipt is None and pending.next_send <= now < pending.deadline:
                messages.append((pending.connection, pending.message))
                pending.next_send = now + self.retry_interval
        return messages

    def acknowledge(self, receipt: LwdPdResultReceipt) -> bool:
        pending = self._pending.get((receipt.request_id, receipt.attempt_id))
        if pending is None or pending.message.binding_id != receipt.binding_id:
            return False
        result = receipt.result
        validate_result(result)
        commit = pending.message
        if (
            result.get("attempt_id"),
            result.get("binding_id"),
            result.get("accepted_tokens"),
            result.get("frontend_reason"),
        ) != (
            commit.attempt_id,
            commit.binding_id,
            commit.accepted_tokens,
            commit.frontend_reason,
        ):
            raise ValueError("PD receipt does not acknowledge the committed result")
        if pending.receipt is not None and pending.receipt != result:
            raise ValueError("conflicting PD result receipt")
        pending.receipt = copy.deepcopy(result)
        return True

    def expire(self, now: float) -> list[LwdPdResultCommit]:
        """Remove expired records, returning only unconfirmed commits to report."""
        unconfirmed = []
        for key, pending in list(self._pending.items()):
            if pending.deadline <= now:
                if pending.receipt is None:
                    unconfirmed.append(pending.message)
                del self._pending[key]
        return unconfirmed

    @property
    def pending(self) -> bool:
        return any(p.receipt is None for p in self._pending.values())
