# SPDX-License-Identifier: Apache-2.0
"""Bounded, nonblocking control sends with FIFO ordering per peer."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any


class LwdPdControlOutbox:
    """A blocked peer cannot stall the model thread or another peer's sends."""

    def __init__(
        self,
        send: Callable[[Any, Any], bool],
        *,
        capacity: int = 8192,
        timeout: float = 3.0,
    ) -> None:
        if capacity < 1 or timeout <= 0:
            raise ValueError("invalid PD control outbox bounds")
        self.send = send
        self.capacity = capacity
        self.timeout = timeout
        self.queues: dict[Any, deque[tuple[float, Any]]] = {}
        self.count = 0

    def put(self, peer: Any, message: Any, now: float) -> None:
        if peer not in self.queues and self.send(peer, message):
            return
        if self.count >= self.capacity:
            raise RuntimeError("PD control outbound capacity exhausted")
        self.queues.setdefault(peer, deque()).append((now + self.timeout, message))
        self.count += 1

    def flush(self, now: float) -> None:
        for peer, messages in list(self.queues.items()):
            for _ in range(8):
                if not messages:
                    del self.queues[peer]
                    break
                deadline, message = messages[0]
                if now >= deadline:
                    raise RuntimeError("PD control peer stalled past the send deadline")
                if not self.send(peer, message):
                    break
                messages.popleft()
                self.count -= 1
            if not messages:
                self.queues.pop(peer, None)
