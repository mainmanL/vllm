# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD UCX data-plane accelerator assembly layer.

Generic host-host transport (the vLLM repo's ``vllm.v1.lwd_comm.data_comm``
UHC1 channel family) plus accelerator device orchestration (D2H/H2D) and future/callback
wiring, on top of ``LwdTransferFuture``.

Layering and responsibilities:
  * transport: LwdUcxChannel -- seqno is the tag, zero frame header, zero
    software CRC; the receive block pool is built in. Transport-layer
    supervise/reconnect is kept, but ``on_peer_restart`` is NOT wired (no seqno
    reset, no re-dispatch): a peer process restart means the whole deployment
    restarts.
  * send: sender thread + library-pool staging -- ``ch.acquire_staging()`` takes
    a pool block (pool_staging: the channel's pool blocks are not attached to
    the wildcard, they serve purely as D2H staging), one D2H copy writes
    directly into it, then ``ch.send(tag, blk.addr)``; the completion callback
    releases the staging back. Pool depth = in-flight cap = backpressure.
  * recv: ``ch.recv(tag, cb)`` callback (library IO thread) submits an async
    H2D + descramble and records an accelerator event, then ``future._bind``; with
    stage_segments it further D2H-copies per segment into each request's pinned
    buffer and records staged_event.
  * aux: packed contiguously (payload then aux) into ONE message; the receiver
    splits it back into two H2D copies sized by the buffers the control plane
    already knows.
  * skip: aborted seqnos get a pre-armed discard recv; peer-restart reset is
    not wired.

Deployment: both peers must use the same transport; UCX 1.22 + pinned TLS. The
RMA-put large-frame path stays off (LwdUcxConfig.put_thresh default 0). Block
capacity is per direction and comes from the topology
(``deployment.data_plane_host.up_buffer_size`` / ``down_buffer_size``):
identical on both peers and verified per link at handshake.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm.distributed.lwd.lwd_types import LwdChannelType
from vllm.logger import logger
from vllm.v1.lwd_comm.data_comm.ucx_channel import (
    LwdHostDoneEvent,
    LwdUcxConfig,
    LwdUcxListener,
)
from vllm.v1.lwd_comm.data_comm.ucx_port import LwdUcxTransportPort, open_transport

# Blocks per channel: absorbs the early frames and provides send backpressure.
BLOCKS_PER_CHANNEL = 4
# Early frames that arrived before their receive was pre-posted. The old
# branch allowed 4096; leaving this at LwdUcxConfig's default caps the store
# at BLOCKS_PER_CHANNEL and silently drops the oldest frame on overflow.
_EARLY_MAX = 4096
_HS_TIMEOUT_S = 600.0


# Reserved tag for frames without a seqno (production request streams all
# carry seqnos; only defends the queue's non-sequenced path — at most one
# such frame in flight at a time).
_NO_SEQNO_TAG = 1 << 60


@dataclass(frozen=True)
class LwdUcxChannelIoConfig:
    """Static settings of one (channel, op) accelerator IO object."""

    op: Literal["send", "recv"]
    device: torch.device
    up_block_mb: int
    down_block_mb: int
    host: str = ""
    port: int = 0
    serve: bool = False
    token: int = 0
    listener: LwdUcxListener | None = None
    transport_mode: Literal["embedded", "daemon"] = "embedded"
    traffic_class: int | None = None
    sl: int | None = None


class LwdBaseUcxChannelIo:
    """One (channel, op) LWD IO: LwdUcxChannel + sender thread + recv H2D.

    Roles: the edge connects (host, port); the cloud serves (serve=True,
    host, port) — i.e. the RECEIVER of a (channel, op) link serves.
    """

    def __init__(
        self,
        channel: LwdChannelType,
        stream: Any,
        config: LwdUcxChannelIoConfig,
    ) -> None:
        self._channel = channel
        self._op = config.op
        self._device = config.device
        self._stream = stream
        self.tag = f"{channel.value}/{config.op}"
        self._stopped = False
        if not config.token:
            raise RuntimeError(
                f"{self.tag}: UCX handshake token is 0; ucx_channels must "
                "pass a non-zero token derived from (dp_idx, channel, op)"
            )
        # Per-direction block capacity (see the file header); identical on
        # both peers and verified per link at handshake.
        _is_up = channel is LwdChannelType.UP
        _block_mb = int(config.up_block_mb if _is_up else config.down_block_mb)
        _block_env = (
            "data_plane_host.up_buffer_size"
            if _is_up
            else "data_plane_host.down_buffer_size"
        )
        if _block_mb <= 0:
            raise RuntimeError(
                f"{self.tag}: {_block_env} must be positive, got {_block_mb}MiB"
            )
        self._block_bytes = _block_mb << 20
        self._block_env = _block_env
        cfg = LwdUcxConfig(
            block_mb=self._block_bytes >> 20,
            n_blocks=BLOCKS_PER_CHANNEL,
            early_max=_EARLY_MAX,
            hs_timeout_s=_HS_TIMEOUT_S,
            token=config.token,
            # Send-only link: pool blocks are not attached to the
            # wildcard; they serve purely as D2H staging (cycled via
            # acquire_staging/release_staging).
            pool_staging=(config.op == "send"),
            traffic_class=config.traffic_class,
            sl=config.sl,
        )
        # Transport executor: embedded = in-process direct LwdUcxChannel
        # (default); daemon = dedicated transport daemon (ucx_daemon +
        # shm ring). The factory assembles by mode; this layer only faces
        # the LwdUcxTransportPort protocol.
        self._ch: LwdUcxTransportPort = open_transport(
            cfg,
            mode=config.transport_mode,
            serve=config.serve,
            host=config.host,
            port=config.port,
            listener=config.listener,
        )
        # No on_peer_restart registration (U1): transport-layer reconnect
        # is kept, but recovery is not promised.
        # Send direction: task queue + sender thread (staging blocks come
        # from the library pool via ch.acquire_staging; this layer never
        # allocates its own memory/MR).
        self._sq: queue.Queue[tuple | None] = queue.Queue()
        self._send_thread: threading.Thread | None = None
        if config.op == "send":
            self._send_thread = threading.Thread(
                target=self._sender_loop, name=f"lwd-ucx-send-{self.tag}", daemon=True
            )
            self._send_thread.start()
        # skip arming (only meaningful on the recv side)
        self._skip_armed: set[int] = set()
        self._skip_lock = threading.Lock()
        # Dedicated copy stream for segmented D2H (built at recv-side
        # construction to avoid creating streams on the lib IO thread)
        self._copy_stream: Any | None = (
            torch.Stream(device=config.device) if config.op == "recv" else None
        )
        # First-frame delimitation traces (error level, once per IO each)
        self._tr_submit_send = True
        self._tr_send_done = True
        self._tr_d2h = True
        self._tr_submit_recv = True
        self._tr_recv_done = True

    # ------------------------------------------------------------------ #
    # Queue call surface                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _nbytes(tensor: Any) -> int:
        # symbolic-meta-ok: host-side frame size in bytes, not a meta shape
        return tensor.numel() * tensor.element_size()

    @classmethod
    def _frame_nbytes(cls, tensor: Any, aux_tensor: Any) -> int:
        aux_bytes = cls._nbytes(aux_tensor) if aux_tensor is not None else 0
        return cls._nbytes(tensor) + aux_bytes

    def _raise_if_frame_too_large(
        self, seqno: int | None, tensor: Any, aux_tensor: Any
    ) -> None:
        """Fail at submit time, before queueing a frame the peer cannot hold."""
        total = self._frame_nbytes(tensor, aux_tensor)
        if total <= self._block_bytes:
            return
        raise RuntimeError(
            f"[lwd-ucx] {self.tag} seqno={seqno}: frame "
            f"{total / 2**20:.1f}MiB ({total} B) > "
            f"{self._block_env}={self._block_bytes >> 20}MiB; "
            f"increase {self._block_env} on both edge and cloud "
            "(or reduce max_num_batched_tokens/chunk size)"
        )

    def submit_send(
        self,
        seqno: int | None,
        tensor: Any,
        aux_tensor: Any,
        req: Any,
        future: Any,
    ) -> None:
        # Fast fail on the edge: a frame larger than the negotiated block
        # can never reach the cloud. Queueing it would leave the request
        # awaiting forever because _do_send only finalizes the future.
        self._raise_if_frame_too_large(seqno, tensor, aux_tensor)
        # The caller enqueues the snapshot clone asynchronously on the
        # producer stream, while the sender thread D2Hs on this channel
        # stream. Order the D2H after that producer work or it can read
        # half-written bytes under back-to-back submits.
        producer = torch.accelerator.current_stream()
        self._stream.wait_stream(producer)
        if self._tr_submit_send:
            self._tr_submit_send = False
            logger.error(
                "[lwd-ucx-trace] %s SUBMIT_SEND seqno=%s payload=%dB aux=%dB sq=%d",
                self.tag,
                seqno,
                self._nbytes(tensor),
                (self._nbytes(aux_tensor) if aux_tensor is not None else 0),
                self._sq.qsize(),
            )
        self._sq.put((seqno, tensor, aux_tensor, future))

    def submit_recv(
        self,
        seqno: int | None,
        req: Any,
        buffer: Any,
        aux_buffer: Any,
        future: Any,
        *,
        post_h2d: Any = None,
    ) -> None:
        tag = seqno if seqno is not None else _NO_SEQNO_TAG
        if self._tr_submit_recv:
            self._tr_submit_recv = False
            logger.error(
                "[lwd-ucx-trace] %s SUBMIT_RECV seqno=%s tag=%d numel=%s "
                "aux_numel=%s stage=%s",
                self.tag,
                seqno,
                tag,
                buffer.numel(),  # symbolic-meta-ok: log-only size, not a meta shape
                aux_buffer.numel()
                if aux_buffer is not None
                else 0,  # symbolic-meta-ok: log-only size
                req.stage_segments is not None,
            )
        self._ch.recv(tag, cb=self._recv_cb(buffer, aux_buffer, future, req, post_h2d))

    def add_skipped(self, seqno: int) -> None:
        """Silently drop this seqno's frame on the recv side: a pre-armed
        discard recv (dedup)."""
        if seqno is None or self._op != "recv":
            return
        with self._skip_lock:
            if seqno in self._skip_armed:
                return
            self._skip_armed.add(seqno)
        # daemon mode goes through recv_discard (occupies no recv block,
        # produces no completion); embedded falls back to the generic path
        # (a cb that discards).
        discard = getattr(self._ch, "recv_discard", None)
        if discard is not None:
            discard(seqno)
        else:
            self._ch.recv(seqno, cb=lambda op: None)

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ch.wait_ready(timeout=timeout)

    def shutdown(self, timeout: float = 2.0) -> None:
        self._stopped = True
        if self._send_thread is not None:
            self._sq.put(None)
            self._send_thread.join(timeout=timeout)
        self._ch.close(timeout=timeout)
        with self._skip_lock:
            self._skip_armed.clear()

    # ------------------------------------------------------------------ #
    # Send: pool block (acquire_staging) -> direct D2H -> ch.send        #
    # ------------------------------------------------------------------ #
    def _sender_loop(self) -> None:
        torch.accelerator.set_device_index(self._device)
        while True:
            task = self._sq.get()
            if task is None:
                break
            seqno, tensor, aux_tensor, future = task
            try:
                self._do_send(seqno, tensor, aux_tensor, future)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[lwd-ucx-trace] %s SEND_FAILED seqno=%s err=%r",
                    self.tag,
                    seqno,
                    exc,
                )
                with contextlib.suppress(Exception):
                    future._finalize(error=exc)

    def _do_send(
        self, seqno: int | None, tensor: Any, aux_tensor: Any, future: Any
    ) -> None:
        payload_bytes = self._nbytes(tensor)
        aux_bytes = self._nbytes(aux_tensor) if aux_tensor is not None else 0
        total = payload_bytes + aux_bytes
        if total > self._block_bytes:
            # Defensive: submit_send already rejects this; keep the sender
            # path consistent if a frame is enqueued by another caller.
            self._raise_if_frame_too_large(seqno, tensor, aux_tensor)
            return
        # Library-pool staging block (pool depth = in-flight cap, i.e.
        # backpressure; blocks until one frees — the completion callback
        # releases it back; registration/rcache handled by the library).
        stg = self._ch.acquire_staging()
        sent = False
        try:
            tag = seqno if seqno is not None else _NO_SEQNO_TAG
            t0 = time.monotonic()
            with torch.inference_mode(), torch.stream(self._stream):
                mv = memoryview(stg.buf)
                dst = torch.frombuffer(mv[:payload_bytes], dtype=tensor.dtype)
                dst.copy_(tensor.reshape(dst.shape))
                if aux_bytes:
                    adst = torch.frombuffer(
                        mv[payload_bytes:total], dtype=aux_tensor.dtype
                    )
                    adst.copy_(aux_tensor.reshape(adst.shape))
            if self._tr_d2h:
                self._tr_d2h = False
                logger.error(
                    "[lwd-ucx-trace] %s SEND_D2H seqno=%s dur=%.1fms "
                    "(copy returned means host-readable; no such line = "
                    "stuck in D2H or staging acquire)",
                    self.tag,
                    seqno,
                    (time.monotonic() - t0) * 1000,
                )
            event = LwdHostDoneEvent()
            # bind BEFORE send: the handle must be in place before the
            # completion callback (IO thread) sets the event.
            future._bind(handles=[], done_event=event)
            self._ch.send(
                tag, stg.addr, total, ref=stg, cb=self._send_done(event, stg, future)
            )
            sent = True
        finally:
            if not sent:
                self._ch.release_staging(stg)

    def _send_done(
        self, event: LwdHostDoneEvent, stg: Any, future: Any
    ) -> Callable[[Any], None]:
        def cb(op) -> None:  # noqa: ANN001 - LwdUcxOp (lib IO thread)
            if self._tr_send_done:
                self._tr_send_done = False
                logger.error(
                    "[lwd-ucx-trace] %s SEND_DONE tag=%d size=%s err=%r",
                    self.tag,
                    op.tag,
                    op.size,
                    op.error,
                )
            # op completion = data handed over: return the staging block
            # to the library pool.
            self._ch.release_staging(stg)
            event.set()
            if op.error is not None:
                with contextlib.suppress(Exception):
                    future._finalize(error=op.error)

        return cb

    # ------------------------------------------------------------------ #
    # Recv: lib IO thread callback -> H2D + descramble + staging -> bind  #
    # ------------------------------------------------------------------ #
    def _copy_stream_or_create(self) -> Any:
        if self._copy_stream is None:
            self._copy_stream = torch.Stream(device=self._device)
        return self._copy_stream

    def _recv_cb(
        self,
        buffer: Any,
        aux_buffer: Any,
        future: Any,
        req: Any,
        post_h2d: Any = None,
    ) -> Callable[[Any], None]:
        stream = self._stream
        tag_str = self.tag

        def cb(op) -> None:  # noqa: ANN001 - LwdUcxOp (lib IO thread)
            torch.accelerator.set_device_index(self._device)
            if self._tr_recv_done:
                self._tr_recv_done = False
                logger.error(
                    "[lwd-ucx-trace] %s RECV_DONE tag=%d len=%s err=%r",
                    tag_str,
                    op.tag,
                    len(op.data) if op.data is not None else None,
                    op.error,
                )
            if op.error is not None:
                with contextlib.suppress(Exception):
                    future._finalize(error=op.error)
                return
            data = op.data
            try:
                payload_len = self._nbytes(buffer)
                aux_len = self._nbytes(aux_buffer) if aux_buffer is not None else 0
                if payload_len + aux_len != len(data):
                    err = ConnectionError(
                        f"payload size mismatch on {tag_str}: got {len(data)}B "
                        f"expect {payload_len}+{aux_len}"
                    )
                    future._finalize(error=err)
                    self._ch.fail(err)
                    return
                with torch.inference_mode(), torch.stream(stream):
                    mv = memoryview(data)
                    buffer.copy_(
                        torch.frombuffer(mv[:payload_len], dtype=buffer.dtype),
                        non_blocking=True,
                    )
                    if aux_buffer is not None:
                        aux_buffer.copy_(
                            torch.frombuffer(mv[payload_len:], dtype=aux_buffer.dtype),
                            non_blocking=True,
                        )
                    # The HCCL queue descrambles after _bridge_and_record;
                    # UCX must run the same logic after H2D and before
                    # recording done_event.
                    if post_h2d is not None:
                        post_h2d(buffer, req)
                    done_event = torch.Event()
                    done_event.record()
                staged_event = None
                if req.stage_segments is not None:
                    staged_event = self._stage_to_buffers(buffer, req, done_event)
                future._bind(
                    handles=[],
                    done_event=done_event,
                    tensor=buffer,
                    keepalive=(data,),
                    aux_tensor=aux_buffer,
                    staged_event=staged_event,
                )
                # daemon mode: data is a shm-block view — register the H2D
                # completion event so the port reaper can return the recv
                # block (embedded's LwdUcxOp has no such method; skip).
                recycle = getattr(op, "mark_recycle", None)
                if recycle is not None:
                    recycle(done_event)
            except Exception as exc:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    future._finalize(error=exc)

        return cb

    def _stage_to_buffers(self, buffer: Any, req: Any, done_event: Any) -> Any:
        """D2H-copy the main frame per stage_segment into each request's
        pinned buffer.

        Same semantics as the HCCL queue's staging path: the copy stream
        first waits done_event (descramble included), segments stay FIFO
        on one stream, staged_event is recorded after the last segment.
        """
        copy_stream = self._copy_stream_or_create()
        copy_stream.wait_event(done_event)
        with torch.stream(copy_stream):
            for dst, src_off, n in req.stage_segments:
                dst.copy_(buffer[src_off : src_off + n], non_blocking=True)
            staged_event = torch.Event()
            staged_event.record()
        return staged_event
