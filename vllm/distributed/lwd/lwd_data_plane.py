# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Process-wide entry point and backend registration for LWD transfers."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import logger

from .lwd_transfer_future import LwdTransferFuture
from .lwd_transfer_queue import LwdBaseTransferQueue
from .lwd_types import ConnectionId, LwdChannelType, LwdTransferRequest

if TYPE_CHECKING:
    from vllm.config.lwd import LwdConfig

LwdQueueFactory = Callable[[ConnectionId, LwdChannelType, str], LwdBaseTransferQueue]
_queue_factory: LwdQueueFactory | None = None


def lwd_set_data_plane_queue_factory(factory: LwdQueueFactory) -> None:
    """Select the process backend during worker initialization."""
    global _queue_factory
    _queue_factory = factory


def lwd_init_payload_scrambler(lwd_config: LwdConfig | None) -> None:
    """Initialize transport obfuscation from the shared deployment config."""
    if lwd_config is None:
        return
    scramble = lwd_config.topology.feature_ctrl.scramble
    if not scramble.enable:
        return
    if not scramble.device_id or not scramble.custom_factor:
        raise ValueError(
            "[LWD] feature_ctrl.scramble.enable requires both "
            "device_id and custom_factor"
        )
    from .lwd_payload_scrambler import lwd_get_payload_scrambler

    lwd_get_payload_scrambler(
        device_id=scramble.device_id,
        custom_factor=scramble.custom_factor,
    )


class LwdDataPlane:
    """Process-wide singleton. All public methods are thread-safe."""

    _instance: LwdDataPlane | None = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> LwdDataPlane:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._queues: dict[
            tuple[ConnectionId, LwdChannelType, str], LwdBaseTransferQueue
        ] = {}
        self._pending_skips: dict[
            tuple[ConnectionId, LwdChannelType, str], set[int]
        ] = {}
        self._lock = threading.Lock()
        self._shutting_down = False
        self._closed = False
        self._failed_connections: set[ConnectionId] = set()

    def submit_send(self, request: LwdTransferRequest) -> LwdTransferFuture:
        assert request.op == "send", "submit_send requires op='send'"
        return self._submit(request)

    def submit_recv(self, request: LwdTransferRequest) -> LwdTransferFuture:
        assert request.op == "recv", "submit_recv requires op='recv'"
        return self._submit(request)

    def poll_completions(self) -> int:
        with self._lock:
            queues = list(self._queues.values())
        completed = 0
        for queue in queues:
            completed += len(queue.reap())
        return completed

    def skip_seqno(
        self,
        channel_type: LwdChannelType,
        seqno: int,
        connection_id: ConnectionId,
        op: str = "recv",
    ) -> None:
        key = (connection_id, channel_type, op)
        with self._lock:
            queue = self._queues.get(key)
            if queue is None:
                self._pending_skips.setdefault(key, set()).add(seqno)
                return
        queue.skip_seqno(seqno)

    def abort_connection(self, connection: ConnectionId, error: BaseException) -> None:
        with self._lock:
            self._failed_connections.add(connection)
            keys = [key for key in self._queues if key[0] == connection]
            queues = [self._queues.pop(key) for key in keys]
            for key in list(self._pending_skips):
                if key[0] == connection:
                    self._pending_skips.pop(key, None)
        for queue in queues:
            queue.abort(error)
        logger.warning("[LWD] aborted data-plane connection %s: %r", connection, error)

    def shutdown(self, timeout: float | None = None) -> None:
        with self._lock:
            if self._shutting_down:
                return
            self._shutting_down = True
            self._closed = True
            queues = list(self._queues.values())
        try:
            for queue in queues:
                queue.shutdown(timeout=timeout)
        finally:
            with self._lock:
                self._queues.clear()
                self._pending_skips.clear()
                self._failed_connections.clear()
                self._shutting_down = False
        logger.info("[lwd-comm] service shut down")

    def _submit(self, request: LwdTransferRequest) -> LwdTransferFuture:
        with self._lock:
            if self._shutting_down or self._closed:
                raise RuntimeError("lwd-comm service is shut down")
            if request.connection in self._failed_connections:
                raise RuntimeError(
                    f"lwd-comm connection {request.connection} has failed"
                )
            key = (request.connection, request.channel, request.op)
            queue = self._queues.get(key)
            if queue is None:
                if _queue_factory is None:
                    raise RuntimeError("LWD data-plane backend is not registered")
                queue = _queue_factory(request.connection, request.channel, request.op)
                self._queues[key] = queue
                for seqno in sorted(self._pending_skips.pop(key, ())):
                    queue.skip_seqno(seqno)
        return queue.submit(request)


def lwd_get_data_plane() -> LwdDataPlane:
    return LwdDataPlane.instance()
