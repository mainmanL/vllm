# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Device-independent FIFO sequencing for the LWD data plane."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any

import torch

from vllm.logger import logger
from vllm.v1.lwd_comm.data_comm.ucx_channel import LwdHostDoneEvent

from .lwd_transfer_future import LwdTransferFuture
from .lwd_types import ConnectionId, LwdChannelType, LwdTransferRequest


def lwd_scramble_seed_for(request: LwdTransferRequest) -> int | None:
    """Derive a flight seed when the process-wide scrambler is enabled."""
    from .lwd_payload_scrambler import (
        lwd_get_payload_scrambler,
        lwd_payload_scrambler_initialized,
    )

    if not lwd_payload_scrambler_initialized():
        return None
    return lwd_get_payload_scrambler().seed_for(
        channel=request.channel.value,
        seqno=request.seqno,
        num_elements=request.num_elements,
    )


class LwdBaseTransferQueue:
    """One FIFO of pending requests for one connection, channel and op."""

    def __init__(
        self, dp_idx: int | ConnectionId, channel_type: LwdChannelType, op: str
    ) -> None:
        self.connection = (
            dp_idx if isinstance(dp_idx, ConnectionId) else ConnectionId(0, 0, dp_idx)
        )
        self.dp_idx = self.connection.dp_idx
        self.channel_type = channel_type
        self.op = op
        self._pending: deque[LwdTransferFuture] = deque()
        self._lock = threading.Lock()
        self._submission_lock = threading.Lock()
        self._next_seqno: int | None = None
        self._held: dict[int, tuple[LwdTransferRequest, LwdTransferFuture]] = {}
        self._skipped: set[int] = set()
        self._abort_error: BaseException | None = None

    def _execute(
        self,
        request: LwdTransferRequest,
        predecessor: LwdTransferFuture | None,
        into: LwdTransferFuture | None = None,
    ) -> LwdTransferFuture:
        raise NotImplementedError

    def _new_future(
        self,
        request: LwdTransferRequest,
        *,
        event: Any = None,
        tensor: torch.Tensor | None = None,
    ) -> LwdTransferFuture:
        if event is None:
            event = LwdHostDoneEvent()
            event.set()
        return LwdTransferFuture(request, [], event, tensor=tensor)

    def skip_seqno(self, seqno: int) -> None:
        with self._submission_lock:
            if self._next_seqno is None:
                self._next_seqno = 0
            if seqno < self._next_seqno:
                return
            held = self._held.pop(seqno, None)
            if held is not None:
                held[1]._finalize(
                    error=RuntimeError(
                        f"seqno {seqno} aborted on channel {self.channel_type.value}"
                    )
                )
            self._skipped.add(seqno)
            self._advance_past_skipped()

    def _advance_past_skipped(self) -> None:
        while self._next_seqno in self._skipped:
            self._skipped.discard(self._next_seqno)
            self._next_seqno += 1
            while self._next_seqno in self._held:
                request, future = self._held.pop(self._next_seqno)
                self._execute_next(request, into=future)
                self._next_seqno += 1

    def submit(self, request: LwdTransferRequest) -> LwdTransferFuture:
        if request.op != self.op:
            raise RuntimeError(
                f"op mismatch on {self.channel_type.value} FIFO: channel "
                f"carries {self.op!r}, got {request.op!r}"
            )
        self._finalize_many(self._reap())
        if request.op == "send":
            request = self._snapshot_send(request)
        with self._submission_lock:
            if self._abort_error is not None:
                raise RuntimeError(
                    f"channel {self.channel_type.value} {self.op} is aborted "
                    f"for connection {self.connection}"
                ) from self._abort_error
            if request.seqno is not None:
                return self._submit_sequenced(request)
            return self._execute_next(request)

    @staticmethod
    def _snapshot_send(request: LwdTransferRequest) -> LwdTransferRequest:
        assert request.tensor is not None, "send requires tensor"
        owned = request.tensor.detach().clone()
        seed = lwd_scramble_seed_for(request)
        if seed is not None:
            from .lwd_payload_scrambler import lwd_get_payload_scrambler

            lwd_get_payload_scrambler().scramble(owned, seed=seed)
        aux_owned = (
            request.aux_tensor.detach().clone()
            if request.aux_tensor is not None
            else None
        )
        return replace(request, tensor=owned, aux_tensor=aux_owned)

    def _submit_sequenced(self, request: LwdTransferRequest) -> LwdTransferFuture:
        seqno = request.seqno
        assert seqno is not None
        if self._next_seqno is None:
            self._next_seqno = 0
        self._advance_past_skipped()
        if seqno in self._skipped or seqno < self._next_seqno:
            logger.warning(
                "[lwd-comm] dropping skipped/duplicate seqno %d on %s %s (next=%d)",
                seqno,
                self.channel_type.value,
                self.op,
                self._next_seqno,
            )
            future = self._new_future(request)
            future._finalize(
                error=RuntimeError(
                    f"seqno {seqno} was skipped on {self.channel_type.value}"
                )
            )
            return future
        if seqno > self._next_seqno:
            if seqno in self._held:
                return self._held[seqno][1]
            future = LwdTransferFuture.deferred(request)
            self._held[seqno] = (request, future)
            return future
        future = self._execute_next(request)
        self._next_seqno += 1
        self._advance_past_skipped()
        while self._next_seqno in self._held:
            held_request, held_future = self._held.pop(self._next_seqno)
            self._execute_next(held_request, into=held_future)
            self._next_seqno += 1
            self._advance_past_skipped()
        return future

    def _execute_next(
        self,
        request: LwdTransferRequest,
        into: LwdTransferFuture | None = None,
    ) -> LwdTransferFuture:
        with self._lock:
            predecessor = self._pending[-1] if self._pending else None
        future = self._execute(request, predecessor, into=into)
        with self._lock:
            self._pending.append(future)
        return future

    def reap(self) -> list[LwdTransferFuture]:
        finalized = self._reap()
        self._finalize_many(finalized)
        return finalized

    def abort(self, error: BaseException) -> None:
        with self._submission_lock:
            self._abort_error = error
            for _request, future in self._held.values():
                future._finalize(error=error)
            self._held.clear()
            with self._lock:
                pending = list(self._pending)
                self._pending.clear()
            for future in pending:
                future._finalize(error=error)
            self._skipped.clear()

    def shutdown(self, timeout: float | None = None) -> list[LwdTransferFuture]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._submission_lock:
            for _request, future in self._held.values():
                future._finalize(
                    error=RuntimeError(
                        f"{self.channel_type.value} shut down with a held request"
                    )
                )
            self._held.clear()
            with self._lock:
                pending = list(self._pending)
            for future in pending:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                future.wait(timeout=remaining)
            finalized = self._reap()
            self._finalize_many(finalized)
            return finalized

    def _reap(self) -> list[LwdTransferFuture]:
        done: list[LwdTransferFuture] = []
        with self._lock:
            while self._pending and self._pending[0].done():
                done.append(self._pending.popleft())
        return done

    @staticmethod
    def _finalize_many(futures: list[LwdTransferFuture]) -> None:
        for future in futures:
            future._finalize()

    @staticmethod
    def _validate_stage_segments(request: LwdTransferRequest) -> None:
        if request.stage_segments is None:
            return
        for dst, src_offset, numel in request.stage_segments:
            if src_offset < 0 or numel < 0 or src_offset + numel > request.num_elements:
                raise ValueError(
                    f"invalid stage segment offset={src_offset} numel={numel} "
                    f"for recv num_elements={request.num_elements}"
                )
            if dst.numel() != numel:
                raise ValueError(
                    f"stage destination numel {dst.numel()} != segment numel {numel}"
                )
