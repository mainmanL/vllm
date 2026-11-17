# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Engine-thread ownership of PD registrations, execution and results."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec

from vllm.lwd_logger import lwd_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestType, FinishReason
from vllm.v1.lwd_control.lwd_comm.lwd_id_adapter import unwrap_req_id, wrap_req_id
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdC2eNotify,
    LwdPdBindNotify,
    LwdPdQuotaNotify,
    LwdPdResultReceipt,
    LwdPdStageDoneNotify,
    LwdRequestNotify,
)
from vllm.v1.lwd_control.pd.cloud_scheduler import LwdPdDecodeAdmission
from vllm.v1.lwd_control.pd.control_outbox import LwdPdControlOutbox
from vllm.v1.lwd_control.pd.envelope import (
    ENVELOPE_VERSION,
    compute_handoff_plan,
    envelope_to_manifest,
    model_needs_truncate,
    range_digest,
    validate_handoff_plan,
)
from vllm.v1.lwd_control.pd.results import LwdPdResultLedger
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
        LwdCloudEngineCoreMixin,
    )
    from vllm.v1.request import Request


@dataclass
class LwdPdCloudStage:
    request_id: str
    envelope: dict[str, Any]
    binding_id: str
    peer: LwdPeer
    request: Request | None
    p_end: int
    i_end: int
    r_pos: int
    hit_tokens: int = 0
    claimed: bool = False
    terminal: bool = False
    error: str = ""
    created: float = field(default_factory=time.monotonic)
    finished: float = 0.0
    export: dict[str, Any] | None = None
    results: LwdPdResultLedger | None = None
    published: bool = False
    ranges: tuple[int, int, str] = (0, 0, "")
    source_p_binding: str = ""
    source_export_id: str = ""
    output_authorized: int = 0
    input_consumed: int = 0
    terminal_notified: bool = False
    input_windows: dict[int, tuple[int, int]] = field(default_factory=dict)
    computed_end: int = 0
    last_window_id: int = 0

    @property
    def attempt_id(self) -> str:
        return self.envelope["attempt_id"]


class LwdPdCloudRuntime:
    """Small stage records outlive native Request/KV and transport resources."""

    def __init__(self, engine: LwdCloudEngineCoreMixin) -> None:
        self.engine = engine
        self.stage_role = engine.lwd_config.stage_role
        self.config = engine.lwd_config.pd
        self.contexts: dict[tuple[str, str], LwdPdCloudStage] = {}
        self.active: dict[str, LwdPdCloudStage] = {}
        self.retired: dict[tuple[str, str], float] = {}
        self.range_routes: dict[tuple[Any, int, str], tuple[str, str]] = {}
        self._closed = threading.Event()
        self.control_outbox = LwdPdControlOutbox(engine._lwd_send)
        engine.scheduler.lwd_pd_runtime = self
        engine.scheduler.lwd_pd_scheduling.fail = self.fail

    def send(self, peer: LwdPeer, message: Any) -> None:
        self.control_outbox.put(peer, message, time.monotonic())

    def start(self) -> None:
        def wake_deadlines() -> None:
            interval = min(1.0, self.config.notify_claim_timeout_s / 2)
            while not self._closed.wait(interval):
                # Timers only wake the engine; all pins/results are mutated by
                # sweep on its owning thread, including while no model runs.
                self.engine.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

        threading.Thread(
            target=wake_deadlines, name="lwd-pd-deadlines", daemon=True
        ).start()

    def close(self) -> None:
        self._closed.set()
        reservations = getattr(self.engine.scheduler, "lwd_prefix_reservations", None)
        for ctx in self.contexts.values():
            if reservations is not None:
                reservations.release(ctx.request_id)
            if not ctx.terminal:
                kind = "pd_export" if self.stage_role == "prefill" else "pd_result"
                self.event(kind, ctx, ok=False, error="PD engine shut down")

    def event(self, kind: str, ctx: LwdPdCloudStage, **fields: Any) -> None:
        self.engine.cloud_control_event_queue.put(
            {
                "type": kind,
                "request_id": ctx.request_id,
                "attempt_id": ctx.attempt_id,
                "engine_index": self.engine._lwd_engine_index,
                **fields,
            }
        )

    def command(self, command: dict[str, Any]) -> None:
        envelope = command["envelope"]
        request_id = wrap_req_id(int(envelope["edge_id"]), envelope["rid"])
        try:
            ctx = self.register(
                envelope, command.get("kv_params"), command["engine_index"]
            )
            self.bind(ctx)
            self.event("pd_register", ctx, ok=True, binding_id=ctx.binding_id)
            if ctx.export is not None:
                self.event("pd_export", ctx, ok=True, kv_transfer_params=ctx.export)
            self.publish_result(ctx, repeat=True)
        except Exception as exc:
            lwd_logger.warning("[pd] registration failed req=%s: %s", request_id, exc)
            self.engine.cloud_control_event_queue.put(
                {
                    "type": "pd_register",
                    "request_id": request_id,
                    "attempt_id": envelope["attempt_id"],
                    "ok": False,
                    "error": str(exc),
                }
            )

    def register(
        self, envelope: dict[str, Any], kv_params: dict[str, Any] | None, dp_idx: int
    ) -> LwdPdCloudStage:
        engine = self.engine
        if self.stage_role not in ("prefill", "decode"):
            raise ValueError("PD cloud has no configured stage role")
        if dp_idx != engine._lwd_engine_index:
            raise ValueError("PD registration addressed to another DP rank")
        request_id = wrap_req_id(int(envelope["edge_id"]), envelope["rid"])
        key = request_id, envelope["attempt_id"]
        if time.time() >= float(envelope["issued_at"]) + float(envelope["ttl_s"]):
            raise ValueError("PD execution envelope expired")
        if key in self.retired:
            raise ValueError("PD execution attempt has already expired")
        existing = self.contexts.get(key)
        if existing is not None:
            if existing.envelope["fingerprint"] != envelope["fingerprint"]:
                raise ValueError("conflicting PD registration fingerprint")
            if existing.error:
                raise ValueError(existing.error)
            return existing
        if len(self.contexts) + len(self.retired) >= 4096:
            raise ValueError("PD registration/result retention capacity exhausted")
        if request_id in self.active and not self.active[request_id].terminal:
            raise ValueError("another PD attempt is executing with this request id")
        if not self.previous_attempt_drained(request_id):
            raise ValueError(
                "previous PD attempt still has execution, cleanup "
                "or transfers in flight"
            )
        n, m = (
            int(envelope["original_prompt_tokens"]),
            int(envelope["user_max_output_tokens"]),
        )
        if n < 1 or m < 1:
            raise ValueError("PD prompt/output lengths must be positive")
        peer = LwdPeer(int(envelope["edge_id"]), 0)
        if peer not in engine._lwd_edges:
            raise ValueError("PD edge has not registered its control connection")
        manifest = envelope_to_manifest(envelope, request_id)
        if manifest.block_size != engine._lwd_hash_block_size:
            raise ValueError("PD manifest and cloud hash block sizes differ")
        truncate = model_needs_truncate(
            engine.vllm_config.model_config,
            engine.scheduler.kv_cache_config.kv_cache_groups,
        )
        p_end, i_end, r_pos = compute_handoff_plan(n, truncate)
        required = p_end + 1 if self.stage_role == "prefill" else n + m
        if required > engine.scheduler.max_model_len:
            raise ValueError("PD stage input/output length exceeds the model budget")
        handoff = self._validate_handoff(envelope, kv_params, (p_end, i_end, r_pos))
        sampling = envelope["sampling"]
        values = {
            key: sampling.get(key, default)
            for key, default in LWD_SAMPLING_DEFAULTS.items()
        }
        notify = LwdRequestNotify(
            request_id,
            n,
            block_hashes=list(manifest.full_block_hashes),
            eos_token_id=sampling.get("eos_token_id"),
            skip_reading_prefix_cache=bool(
                sampling.get("skip_reading_prefix_cache", False)
            ),
            structured_outputs=(
                envelope.get("structured_outputs", "")
                if self.stage_role == "decode"
                else ""
            ),
            **values,
        )
        request = engine._lwd_build_request(notify)
        binding_id = (
            f"{self.stage_role}-{engine.lwd_config.instance_id}-{dp_idx}-"
            f"{uuid.uuid4().hex}"
        )
        self._configure_request(request, envelope, kv_params, binding_id, r_pos)
        ctx = LwdPdCloudStage(
            request_id, envelope, binding_id, peer, request, p_end, i_end, r_pos
        )
        if self.stage_role == "prefill" and engine._lwd_prefix_enabled:
            ctx.hit_tokens = engine.scheduler.lwd_prefix_reservations.reserve(
                request, max_hit_tokens=p_end - 1
            )
        if self.stage_role == "decode":
            ctx.source_p_binding = handoff["source_p_binding"]
            ctx.source_export_id = handoff["export_id"]
            ctx.hit_tokens = int(handoff.get("hit_tokens", 0))
            ctx.results = LwdPdResultLedger(
                n, ctx.hit_tokens, ctx.attempt_id, binding_id
            )
        self.contexts[key] = ctx
        self.active[request_id] = ctx
        return ctx

    def previous_attempt_drained(self, request_id: str) -> bool:
        scheduler = self.engine.scheduler
        if (
            request_id in scheduler.requests
            or self.engine._lwd_ledger.request_seqnos.get(request_id)
        ):
            return False
        if request_id in getattr(scheduler, "finished_req_ids", ()):
            return False
        finished = getattr(scheduler, "finished_req_ids_dict", None) or {}
        if any(request_id in request_ids for request_ids in finished.values()):
            return False
        if request_id in getattr(scheduler, "_lwd_abort_pending", ()):
            return False
        if any(
            request_id in notify.req_ids
            for notify in getattr(scheduler, "pending_c2e", ())
        ):
            return False
        for _future, scheduled, _execute in (
            getattr(self.engine, "batch_queue", ()) or ()
        ):
            batch_meta = getattr(
                getattr(scheduled, "lwd_batch", None), "batch_meta", None
            )
            if (
                request_id in scheduled.num_scheduled_tokens
                or request_id in scheduled.finished_req_ids
                or request_id in getattr(scheduled, "lwd_pd_cloud_identities", {})
                or request_id in getattr(batch_meta, "aborted_req_ids", ())
                or request_id in getattr(batch_meta, "aborted_pd_bindings", {})
            ):
                return False
        return True

    def _validate_handoff(
        self,
        envelope: dict[str, Any],
        params: dict[str, Any] | None,
        plan: tuple[int, int, int],
    ) -> dict[str, Any]:
        if self.stage_role == "prefill":
            return {}
        if not params or not isinstance(params.get("lwd_handoff"), dict):
            raise ValueError("PD decode requires the P export handoff")
        handoff = params["lwd_handoff"]
        if (
            handoff.get("version") != ENVELOPE_VERSION
            or handoff.get("rid") != envelope["rid"]
            or handoff.get("attempt_id") != envelope["attempt_id"]
            or handoff.get("fingerprint") != envelope["fingerprint"]
            or int(handoff.get("n_tokens", -1)) != envelope["original_prompt_tokens"]
        ):
            raise ValueError("PD handoff does not identify this execution attempt")
        received = tuple(int(handoff[key]) for key in ("p_end", "i_end", "r_pos"))
        validate_handoff_plan(envelope["original_prompt_tokens"], *received)
        if received != plan:
            raise ValueError("PD source and target model handoff plans differ")
        for key in (
            "remote_engine_id",
            "remote_request_id",
            "remote_host",
            "remote_port",
        ):
            if params.get(key) in (None, ""):
                raise ValueError(f"PD export is missing {key}")
        if not handoff.get("source_p_binding") or not handoff.get("export_id"):
            raise ValueError("PD handoff is missing its source binding/export")
        return handoff

    def _configure_request(
        self,
        request: Request,
        envelope: dict[str, Any],
        params: dict[str, Any] | None,
        binding_id: str,
        r_pos: int,
    ) -> None:
        request.lwd_pd_stage_role = self.stage_role
        request.lwd_pd_binding_id = binding_id
        request.lwd_pd_attempt_id = envelope["attempt_id"]
        request.lwd_pd_tail_start = r_pos
        request.lwd_pd_tail_end = envelope["original_prompt_tokens"]
        request.lwd_pd_suppress_user_output = self.stage_role == "prefill"
        if self.stage_role == "prefill":
            # The connector owns the one state-model truncation, after this
            # N-row meta placeholder was built. Never truncate a second time.
            skip_cache = bool(
                envelope["sampling"].get("skip_reading_prefix_cache", False)
            )
            request.sampling_params = SamplingParams(
                max_tokens=1,
                min_tokens=1,
                ignore_eos=True,
                skip_reading_prefix_cache=skip_cache,
            )
            request.skip_reading_prefix_cache = skip_cache
            request.max_tokens = 1
            request.kv_transfer_params = {
                "do_remote_prefill": False,
                "do_remote_decode": True,
            }
        else:
            request.kv_transfer_params = dict(params or {})
            request.kv_transfer_params.update(
                do_remote_prefill=True, do_remote_decode=False
            )
            request.skip_reading_prefix_cache = True
            request.sampling_params.skip_reading_prefix_cache = True

    def bind(self, ctx: LwdPdCloudStage) -> None:
        native_cap = int(self.engine.scheduler.max_num_scheduled_tokens)
        window_cap = min(self.config.max_window_tokens or native_cap, native_cap)
        if window_cap < 1 or self.config.max_inflight_windows < 1:
            raise ValueError("PD input window and receive credit must permit progress")
        self.send(
            ctx.peer,
            LwdPdBindNotify(
                request_id=unwrap_req_id(ctx.request_id),
                stage_role=self.stage_role,
                binding_id=ctx.binding_id,
                attempt_id=ctx.attempt_id,
                fingerprint=ctx.envelope["fingerprint"],
                cloud_id=self.engine.lwd_config.instance_id,
                dp_idx=self.engine._lwd_engine_index,
                hit_tokens=ctx.hit_tokens,
                p_end=ctx.p_end,
                i_end=ctx.i_end,
                r_pos=ctx.r_pos,
                max_window_tokens=window_cap,
                max_inflight_windows=self.config.max_inflight_windows,
                recv_windows=self.config.max_inflight_windows,
                source_p_binding=ctx.source_p_binding,
                source_export_id=ctx.source_export_id,
                ttl_s=self.config.request_ttl_s,
            ),
        )

    def claim(self, peer: LwdPeer, notify: LwdRequestNotify) -> None:
        ctx = self.match(notify.request_id, notify.attempt_id, notify.binding_id)
        if ctx is None or ctx.peer != peer:
            raise ValueError("PD claim has no matching stage registration")
        if ctx.terminal or ctx.claimed:
            return
        if (
            notify.stage_role != self.stage_role
            or notify.fingerprint != ctx.envelope["fingerprint"]
        ):
            raise ValueError("PD claim stage/fingerprint mismatch")
        if time.monotonic() - ctx.created > self.config.notify_claim_timeout_s:
            self.fail(ctx.request_id, "PD registration claim timed out")
            return
        if self.stage_role == "prefill" and notify.prefix_hit_tokens != ctx.hit_tokens:
            raise ValueError("PD claim changed its frozen prefix hit")
        request = ctx.request
        assert request is not None
        request.lwd_mrope_positions = notify.mrope_positions
        request.lwd_prefix_nonce = notify.prefix_nonce
        if self.stage_role == "decode":
            self.engine.scheduler.lwd_pd_scheduling.register(
                LwdPdDecodeAdmission(
                    ctx.request_id,
                    ctx.binding_id,
                    ctx.attempt_id,
                    ctx.r_pos,
                    ctx.envelope["original_prompt_tokens"],
                    ctx.output_authorized,
                )
            )
        if request.use_structured_output:
            self.engine.structured_output_manager.grammar_init(request)
        ctx.claimed = True
        self.engine._lwd_ledger.request_routes[ctx.request_id] = peer
        self.engine._lwd_ledger.prompt_lens[ctx.request_id] = request.num_prompt_tokens
        self.engine._lwd_ledger.request_nonces[ctx.request_id] = notify.prefix_nonce
        self.engine.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))

    def match(
        self, request_id: str, attempt_id: str, binding_id: str
    ) -> LwdPdCloudStage | None:
        ctx = self.contexts.get((request_id, attempt_id))
        return ctx if ctx is not None and ctx.binding_id == binding_id else None

    def validate_range(self, connection: Any, item: Any, seqno: int) -> int:
        ctx = self.match(item.request_id, item.attempt_id, item.binding_id)
        if ctx is None or connection != self.engine._lwd_connection_of(ctx.peer):
            raise ValueError("PD input range has no matching connection/stage binding")
        n = ctx.envelope["original_prompt_tokens"]
        lower, upper = (
            (ctx.r_pos, n)
            if self.stage_role == "decode"
            else (ctx.hit_tokens, ctx.p_end)
        )
        if not lower <= item.offset < item.offset + item.num_tokens <= upper:
            raise ValueError("PD input range exceeds its stage plan")
        if item.is_refill:
            raise ValueError("PD refill/recovery is unsupported")
        native_cap = int(self.engine.scheduler.max_num_scheduled_tokens)
        window_cap = min(self.config.max_window_tokens or native_cap, native_cap)
        if item.num_tokens > window_cap:
            raise ValueError("PD input window exceeds the advertised stage budget")
        if len(ctx.input_windows) >= self.config.max_inflight_windows:
            raise ValueError("PD input windows exceeded their receive credit")
        previous = ctx.input_windows.get(item.window_id)
        geometry = item.offset, item.offset + item.num_tokens
        if previous is not None and previous != geometry:
            raise ValueError("conflicting PD input window geometry")
        if item.window_id != ctx.last_window_id + 1:
            raise ValueError("PD input window sequence is not contiguous")
        ctx.last_window_id = item.window_id
        ctx.input_windows[item.window_id] = geometry
        self.range_routes[connection, seqno, item.request_id] = (
            ctx.attempt_id,
            ctx.binding_id,
        )
        return n

    def note_arrival(
        self, connection: Any, seqno: int, request_id: str, end: int
    ) -> bool:
        identity = self.range_routes.get((connection, seqno, request_id))
        if identity is None:
            return False
        ctx = self.match(request_id, *identity)
        if ctx is None:
            return False
        if ctx.terminal:
            self.return_input_credit(ctx, end)
        return self.active.get(request_id) is ctx and not ctx.terminal

    def complete_seqno(self, connection: Any, seqno: int) -> None:
        for key in list(self.range_routes):
            if key[0] == connection and key[1] == seqno:
                self.range_routes.pop(key)

    def quota(self, peer: LwdPeer, notify: Any) -> None:
        ctx = self.match(notify.request_id, notify.attempt_id, notify.binding_id)
        if ctx is None or ctx.peer != peer or ctx.terminal or notify.kind != "output":
            return
        if notify.watermark <= ctx.output_authorized:
            return
        record = self.engine.scheduler.lwd_pd_scheduling.records.get(ctx.request_id)
        if record is not None:
            record.output_quota += notify.watermark - ctx.output_authorized
        ctx.output_authorized = notify.watermark

    def fail(self, request_id: str, reason: str) -> None:
        ctx = self.active.get(request_id)
        if ctx is None or ctx.terminal:
            return
        ctx.error = reason
        self._abort_worker(ctx)
        self.engine.scheduler.finish_requests(
            [request_id], RequestStatus.FINISHED_ERROR
        )
        self._terminal(ctx, int(FinishReason.ERROR))

    def abort(self, request_id: str, attempt_id: str, binding_id: str) -> bool:
        ctx = self.match(request_id, attempt_id, binding_id)
        if ctx is None:
            return False
        if not ctx.terminal:
            self._abort_worker(ctx)
            self.engine.scheduler.finish_requests(
                [request_id], RequestStatus.FINISHED_ABORTED
            )
            self._terminal(ctx, int(FinishReason.ABORT))
        return True

    def _abort_worker(self, ctx: LwdPdCloudStage) -> None:
        scheduler = self.engine.scheduler
        pending = getattr(scheduler, "_lwd_aborted_pd_bindings", None)
        if pending is None:
            pending = scheduler._lwd_aborted_pd_bindings = {}
        pending[ctx.request_id] = (ctx.binding_id, ctx.attempt_id)
        scheduler._lwd_drop_request(ctx.request_id)

    def _terminal(self, ctx: LwdPdCloudStage, finish_reason: int) -> None:
        if ctx.terminal:
            return
        ctx.terminal, ctx.finished = True, time.monotonic()
        reservations = getattr(self.engine.scheduler, "lwd_prefix_reservations", None)
        if reservations is not None:
            reservations.release(ctx.request_id)
        record = self.engine.scheduler.lwd_pd_scheduling.records.get(ctx.request_id)
        if record is not None and record.binding_id == ctx.binding_id:
            record.cancelled = True
        ctx.request = None
        # Result retention must not retain the original manifest chain,
        # grammar JSON or sampling objects after execution is released.
        ctx.envelope = {
            key: ctx.envelope[key]
            for key in (
                "rid",
                "attempt_id",
                "fingerprint",
                "edge_id",
                "tenant_id",
                "original_prompt_tokens",
                "user_max_output_tokens",
                "model_id",
                "input_abi",
                "issued_at",
                "ttl_s",
            )
        }
        if self.stage_role == "prefill":
            if ctx.export is None:
                self.event(
                    "pd_export",
                    ctx,
                    ok=False,
                    error=ctx.error or "P finished without export",
                )
            self.send(
                ctx.peer,
                LwdPdStageDoneNotify(
                    request_id=unwrap_req_id(ctx.request_id),
                    binding_id=ctx.binding_id,
                    attempt_id=ctx.attempt_id,
                    export_id=ctx.source_export_id,
                    ok=ctx.export is not None,
                    error=ctx.error,
                ),
            )
        else:
            assert ctx.results is not None
            try:
                ctx.results.finish(FinishReason(finish_reason).name.lower())
            except ValueError as exc:
                ctx.error = f"PD final result protocol error: {exc}"
                self.event("pd_result", ctx, ok=False, error=ctx.error)
                return
            self.publish_result(ctx)

    def on_outputs(self, outputs: Any, scheduled: Any = None) -> None:
        identities = getattr(scheduled, "lwd_pd_scheduled_ranges", {})
        cloud_identities = getattr(scheduled, "lwd_pd_cloud_identities", {})

        def context(request_id: str) -> LwdPdCloudStage | None:
            identity = cloud_identities.get(request_id) or identities.get(request_id)
            if identity is not None:
                return self.match(request_id, identity[0], identity[1])
            if scheduled is not None:
                # A cleanup-only or late output without a batch identity must
                # never terminate a newer attempt sharing the canonical id.
                return None
            return self.active.get(request_id)

        step_contexts = {
            id(ctx): ctx
            for request_id in identities
            if (ctx := context(request_id)) is not None
        }
        for ctx in step_contexts.values():
            if ctx.results is not None:
                end = ctx.results.final_g
                start = ctx.ranges[1] if end is None else end
                ctx.ranges = (
                    start,
                    start,
                    range_digest(ctx.binding_id, start, start, []),
                )
        for batch in (outputs or {}).values():
            for output in batch.outputs:
                ctx = context(output.request_id)
                if ctx is None:
                    continue
                if self.stage_role == "prefill" and output.kv_transfer_params:
                    self._export(ctx, output.kv_transfer_params)
                elif ctx.results is not None and not ctx.terminal:
                    ctx.ranges = ctx.results.append(list(output.new_token_ids))
                if output.finish_reason is not None:
                    self._terminal(ctx, int(output.finish_reason))
            for request_id in batch.finished_requests or ():
                ctx = context(request_id)
                if ctx is not None and not ctx.terminal:
                    self._terminal(ctx, int(FinishReason.ABORT))

    def capture_schedule(self, out: Any) -> None:
        out.lwd_pd_cloud_identities = {
            request_id: (ctx.attempt_id, ctx.binding_id)
            for request_id in self.engine.scheduler.requests
            if (ctx := self.active.get(request_id)) is not None and not ctx.terminal
        }
        starts = {
            data.req_id: data.num_computed_tokens for data in out.scheduled_new_reqs
        }
        cached = out.scheduled_cached_reqs
        starts.update(zip(cached.req_ids, cached.num_computed_tokens))
        out.lwd_pd_scheduled_ranges = {
            request_id: (ctx.attempt_id, ctx.binding_id, starts[request_id] + rows)
            for request_id, rows in out.num_scheduled_tokens.items()
            if (ctx := self.active.get(request_id)) is not None and request_id in starts
        }

    def completed(self, out: Any) -> None:
        for request_id, (attempt, binding, end) in getattr(
            out, "lwd_pd_scheduled_ranges", {}
        ).items():
            ctx = self.match(request_id, attempt, binding)
            if ctx is not None:
                ctx.computed_end = max(ctx.computed_end, end)
                self.return_input_credit(ctx, ctx.computed_end)

    def return_input_credit(self, ctx: LwdPdCloudStage, end: int) -> None:
        consumed = [
            window for window, (_, limit) in ctx.input_windows.items() if limit <= end
        ]
        if not consumed:
            return
        for window in consumed:
            ctx.input_windows.pop(window)
        ctx.input_consumed += len(consumed)
        self.send(
            ctx.peer,
            LwdPdQuotaNotify(
                request_id=unwrap_req_id(ctx.request_id),
                attempt_id=ctx.attempt_id,
                binding_id=ctx.binding_id,
                kind="recv",
                watermark=ctx.input_consumed,
            ),
        )

    def _export(self, ctx: LwdPdCloudStage, params: dict[str, Any]) -> None:
        if ctx.terminal or ctx.export is not None:
            return
        export_id = str(params.get("remote_request_id") or ctx.request_id)
        handoff = {
            "version": ENVELOPE_VERSION,
            "rid": ctx.envelope["rid"],
            "attempt_id": ctx.attempt_id,
            "fingerprint": ctx.envelope["fingerprint"],
            "source_p_binding": ctx.binding_id,
            "export_id": export_id,
            "n_tokens": ctx.envelope["original_prompt_tokens"],
            "p_end": ctx.p_end,
            "i_end": ctx.i_end,
            "r_pos": ctx.r_pos,
            "hit_tokens": ctx.hit_tokens,
            "input_abi": ctx.envelope["input_abi"],
        }
        ctx.export = {**params, "lwd_handoff": handoff}
        ctx.source_export_id = export_id
        self.event("pd_export", ctx, ok=True, kv_transfer_params=ctx.export)

    def annotate(
        self,
        notify: LwdC2eNotify,
        physical_rows: list[int],
        metadata: dict[str, Any] | None = None,
        *,
        diagnostic: bool = False,
    ) -> LwdC2eNotify:
        if metadata:
            if any(
                len(metadata.get(key, ())) != len(notify.req_ids)
                for key in ("attempt_ids", "binding_ids", "row_counts")
            ):
                raise ValueError("PD DOWN immutable batch metadata is misaligned")
            contexts = [
                self.match(request_id, attempt, binding)
                for request_id, attempt, binding in zip(
                    notify.req_ids, metadata["attempt_ids"], metadata["binding_ids"]
                )
            ]
            physical_rows = metadata["row_counts"]
        else:
            if not diagnostic:
                raise ValueError(
                    "PD DOWN requires immutable batch identities and physical rows"
                )
            contexts = [self.active.get(request_id) for request_id in notify.req_ids]
        if any(ctx is None or ctx.results is None for ctx in contexts):
            raise ValueError("PD user DOWN frame contains a non-decode stage")
        return msgspec.structs.replace(
            notify,
            attempt_ids=[ctx.attempt_id for ctx in contexts],
            binding_ids=[ctx.binding_id for ctx in contexts],
            valid_ranges=[[ctx.ranges[0], ctx.ranges[1]] for ctx in contexts],
            row_counts=physical_rows,
            final_g=[
                ctx.results.final_g if ctx.results.final_g is not None else -1
                for ctx in contexts
            ],
            range_digests=[ctx.ranges[2] for ctx in contexts],
        )

    def ensure_terminals(self, notifies: list[LwdC2eNotify]) -> None:
        if self.stage_role != "decode":
            return
        covered = {
            (request_id, attempt, binding)
            for notify in notifies
            for request_id, attempt, binding, reason in zip(
                notify.req_ids,
                notify.attempt_ids,
                notify.binding_ids,
                notify.finish_reasons,
            )
            if reason != -1
        }
        for ctx in self.active.values():
            if not ctx.terminal or ctx.terminal_notified:
                continue
            identity = ctx.request_id, ctx.attempt_id, ctx.binding_id
            if identity not in covered:
                end = ctx.results.final_g
                notify = LwdC2eNotify(
                    hidden_num_elements=0,
                    top_id_ths=[[]],
                    num_accepted_tokens=[0],
                    req_ids=[ctx.request_id],
                    finish_reasons=[
                        int(FinishReason.ERROR)
                        if ctx.error
                        else int(FinishReason.ABORT)
                    ],
                    attempt_ids=[ctx.attempt_id],
                    binding_ids=[ctx.binding_id],
                    valid_ranges=[[end, end]],
                    row_counts=[0],
                    final_g=[end],
                    range_digests=[range_digest(ctx.binding_id, end, end, [])],
                )
                notifies.append(notify)
            ctx.terminal_notified = True

    def commit(self, peer: LwdPeer, notify: Any) -> None:
        ctx = self.match(notify.request_id, notify.attempt_id, notify.binding_id)
        if ctx is None or ctx.peer != peer or ctx.results is None:
            return
        try:
            ctx.results.commit(
                notify.accepted_tokens, notify.frontend_reason, notify.verified_tokens
            )
        except ValueError as exc:
            lwd_logger.warning(
                "[pd] invalid result commit req=%s: %s", ctx.request_id, exc
            )
            self.event("pd_result", ctx, ok=False, error=str(exc))
            self.fail(ctx.request_id, "PD result protocol conflict")
            return
        if not ctx.terminal:
            self.abort(ctx.request_id, ctx.attempt_id, ctx.binding_id)
        self.publish_result(ctx, repeat=True)

    def publish_result(self, ctx: LwdPdCloudStage, *, repeat: bool = False) -> None:
        result = ctx.results.result if ctx.results is not None else None
        if result is None or (ctx.published and not repeat):
            return
        ctx.published = True
        self.send(
            ctx.peer,
            LwdPdResultReceipt(
                request_id=unwrap_req_id(ctx.request_id),
                attempt_id=ctx.attempt_id,
                binding_id=ctx.binding_id,
                result=result,
            ),
        )
        self.event("pd_result", ctx, ok=True, result=result)

    def sweep(self) -> None:
        now = time.monotonic()
        self.control_outbox.flush(now)
        wall_now = time.time()
        for key, deadline in list(self.retired.items()):
            if deadline <= wall_now:
                self.retired.pop(key)
        for key, ctx in list(self.contexts.items()):
            if not ctx.terminal:
                limit = (
                    self.config.request_ttl_s
                    if ctx.claimed
                    else self.config.notify_claim_timeout_s
                )
                if now - ctx.created > limit:
                    self.fail(ctx.request_id, "PD execution/claim deadline expired")
            elif now - ctx.finished > self.config.result_hold_timeout_s:
                if ctx.results is not None and ctx.results.result is None:
                    self.event(
                        "pd_result", ctx, ok=False, error="PD result unconfirmed"
                    )
                self.contexts.pop(key)
                if self.active.get(ctx.request_id) is ctx:
                    self.active.pop(ctx.request_id)
                expiry = float(ctx.envelope["issued_at"]) + float(ctx.envelope["ttl_s"])
                if expiry > wall_now:
                    self.retired[key] = expiry
                records = self.engine.scheduler.lwd_pd_scheduling.records
                record = records.get(ctx.request_id)
                if record is not None and record.binding_id == ctx.binding_id:
                    records.pop(ctx.request_id)
