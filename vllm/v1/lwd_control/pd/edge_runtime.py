# SPDX-License-Identifier: Apache-2.0
"""Edge engine PD orchestration; routing is accepted only from stage BINDs."""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

import aiohttp
import msgspec

from vllm.config.lwd import LwdConnectionId
from vllm.lwd_logger import lwd_logger
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import _iter_sse_data
from vllm.v1.lwd_control.control_edge_scheduler.lwd_probe_runtime import LwdProbeRuntime
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_PD_VERSION,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdPdBindNotify,
    LwdPdQuotaNotify,
    LwdPdResultCommit,
    LwdPdResultReceipt,
    LwdPdStageDoneNotify,
)
from vllm.v1.request import FinishReason

from .control_outbox import LwdPdControlOutbox
from .edge_state import LwdPdEdgeBinding, LwdPdEdgeState
from .envelope import build_envelope, envelope_content
from .outbox import LwdPdCommitOutbox
from .results import validate_result


class LwdPdEdgeRuntime:
    """All mutable state is engine-thread owned; HTTP only enqueues outcomes."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.scheduler = engine.scheduler
        self.config = engine.lwd_config.pd
        self.states: dict[str, LwdPdEdgeState] = {}
        self.outbox = LwdPdCommitOutbox()
        self.control_outbox = LwdPdControlOutbox(engine._lwd_send)
        self.http = LwdProbeRuntime(
            max_inflight=self.scheduler.max_num_running_reqs * 2
        )
        self.done: queue.Queue[tuple[str, str, Any]] = queue.Queue()
        self.frames: dict[Any, tuple[int, bytes]] = {}
        self.scheduler.lwd_pd_runtime = self
        self.stopped = threading.Event()
        self.timer = threading.Thread(
            target=self._wake_timers, name="lwd-pd-results", daemon=True
        )
        self.timer.start()

    def _wake_timers(self) -> None:
        while not self.stopped.wait(0.5):
            self.engine.input_queue.put_nowait(
                (EngineCoreRequestType.WAKEUP, "lwd_pd_tick")
            )

    @property
    def pending(self) -> bool:
        return any(
            state.tail_pending and not state.cancelled for state in self.states.values()
        )

    def validate_clouds(self) -> None:
        sizes = set()
        roles = set()
        for params in self.engine._lwd_cloud_params.values():
            if params.pd_version != LWD_PD_VERSION or params.stage_role not in (
                "prefill",
                "decode",
            ):
                raise ValueError("PD requires compatible stage-aware cloud peers")
            sizes.add(params.hash_block_size)
            roles.add(params.stage_role)
        if roles != {"prefill", "decode"} or len(sizes) != 1 or min(sizes) <= 0:
            raise ValueError("PD requires P/D peers sharing the prefix hash ABI")
        self.hash_block_size = sizes.pop()

    def add(self, request: Any) -> None:
        rid = request.request_id
        if rid in self.states:
            raise ValueError("PD internal request id is retained by another attempt")
        if len(self.states) >= self.outbox.capacity:
            raise RuntimeError("PD result retention capacity exhausted")
        self.scheduler._lwd_validate_request(request)
        self.scheduler._lwd_compute_mrope_positions(request)
        attempt = (request.sampling_params.extra_args or {}).get("lwd_pd_attempt_id")
        if not isinstance(attempt, str) or not attempt:
            raise ValueError("PD requires a frontend-owned execution attempt")
        coordination = self.engine.lwd_config.coordination
        url = coordination.control_url
        if not url:
            raise ValueError(
                "PD requires coordination.control_url pointing to its Proxy"
            )
        manager = self.engine._lwd_probe_manager
        client = manager._client(0, self.hash_block_size, url)
        manifest = client.build_manifest(
            rid, list(request.prompt_token_ids), manager._media_items(request)
        )
        model = self.engine.vllm_config.model_config
        model_name = model.served_model_name or model.model
        if isinstance(model_name, list):
            model_name = model_name[0]
        envelope = build_envelope(
            request_id=rid,
            attempt_id=attempt,
            edge_id=self.scheduler._lwd_edge_id_of(request),
            tenant_id=self.config.tenant_id or coordination.consumer_id,
            original_prompt_tokens=request.num_prompt_tokens,
            user_max_output_tokens=request.sampling_params.max_tokens,
            sampling_params=request.sampling_params,
            model_id=model_name,
            input_abi=manager._fingerprint().hex(),
            digest_block_size=manifest.block_size,
            digest_chain=manifest.full_block_hashes,
            digest_tail_hash=manifest.tail_hash,
            ttl_s=self.config.request_ttl_s,
        )
        now = time.monotonic()
        state = LwdPdEdgeState(
            rid,
            attempt,
            envelope["fingerprint"],
            request.num_prompt_tokens,
            now + self.config.request_ttl_s,
            now + self.config.p_bind_timeout_s,
            request,
            envelope,
        )
        self.states[rid] = state
        self.engine._lwd_probe_chains[rid] = list(manifest.full_block_hashes)
        headers, body = client.build_control_request(
            manifest, model_name, multimodal=bool(request.mm_features)
        )
        body["messages"] = [{"role": "user", "content": envelope_content(envelope)}]
        future = self.http.submit(self._exchange(url, headers, body))

        def completed(future: Any) -> None:
            try:
                outcome = future.result()
            except Exception as exc:
                outcome = exc
            self.done.put((rid, attempt, outcome))
            self.engine.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

        future.add_done_callback(completed)
        if request.abort_immediately:
            self.fail(state, "request aborted before PD registration")

    async def _exchange(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, Any]:
        session = await self.http.ensure_session()
        timeout = aiohttp.ClientTimeout(
            total=self.config.request_ttl_s,
            sock_connect=self.engine.lwd_config.coordination.connect_timeout,
        )
        result = None
        async with session.post(
            url, headers=headers, json=body, timeout=timeout
        ) as response:
            del body
            response.raise_for_status()
            async for payload in _iter_sse_data(response.content):
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                candidate = event.get("lwd_pd_result")
                if candidate is not None:
                    if result is not None and result != candidate:
                        raise ValueError("conflicting PD HTTP results")
                    result = candidate
                    if event.get("usage") != result.get("usage"):
                        raise ValueError("PD HTTP usage differs from its result")
        if result is None:
            raise ValueError("PD control stream ended without a confirmed result")
        return result

    def handle(self, connection: Any, msg: Any) -> bool:
        if not isinstance(
            msg,
            (
                LwdPdBindNotify,
                LwdPdStageDoneNotify,
                LwdPdQuotaNotify,
                LwdPdResultReceipt,
            ),
        ):
            return False
        state = self.states.get(msg.request_id)
        if state is None or state.attempt_id != msg.attempt_id:
            return True
        try:
            if isinstance(msg, LwdPdBindNotify):
                self._bind(state, connection, msg)
            else:
                binding = state.match(connection, msg.attempt_id, msg.binding_id)
                if binding is None:
                    raise ValueError("PD control event has an unrelated binding")
                if isinstance(msg, LwdPdStageDoneNotify):
                    if binding.notify.stage_role != "prefill" or not msg.ok:
                        raise ValueError(msg.error or "invalid PD prefill completion")
                    if state.p_done and state.p_done != msg.export_id:
                        raise ValueError("conflicting PD export identity")
                    decode = state.bindings.get("decode")
                    if decode and decode.notify.source_export_id != msg.export_id:
                        raise ValueError("PD export differs from decode source")
                    state.p_done = msg.export_id
                    state.bind_deadline = (
                        time.monotonic() + self.config.d_bind_timeout_s
                    )
                elif isinstance(msg, LwdPdQuotaNotify):
                    if msg.kind != "recv":
                        raise ValueError("unexpected PD receive quota kind")
                    binding.release(msg.watermark)
                elif self.outbox.acknowledge(msg):
                    self._result(state, msg.result, http=False)
        except (ValueError, RuntimeError) as exc:
            self.fail(state, str(exc))
        return True

    def _bind(
        self, state: LwdPdEdgeState, connection: Any, notify: LwdPdBindNotify
    ) -> None:
        expected = LwdConnectionId(
            self.engine.lwd_config.instance_id, notify.cloud_id, notify.dp_idx
        )
        if connection != expected or connection not in self.engine._lwd_clouds:
            raise ValueError("PD BIND connection differs from its registered target")
        params = self.engine._lwd_cloud_params[connection]
        if params.stage_role != notify.stage_role:
            raise ValueError("PD BIND role differs from its configured cloud")
        if not state.bind(connection, notify):
            return
        binding = state.bindings[notify.stage_role]
        request = state.request
        if state.cancelled or request is None:
            state.tail_pending = False
            self._abort_binding(state, binding)
            self._commit(state)
            return
        if notify.stage_role == "prefill":
            request.num_computed_tokens = notify.hit_tokens
            request.lwd_pd_upload_end = notify.p_end
            request.lwd_pd_stage_role = "prefill"
            self.scheduler.lwd_admit_request(request, connection)
        else:
            old = self.scheduler._lwd_account_of(state.request_id)
            if old is not None:
                old.note_request_settled(
                    state.request_id, state.request_id not in old.prefill_requests
                )
            self.scheduler._lwd_route[state.request_id] = connection
            self.scheduler._lwd_accounts[connection].note_prefill_admitted(
                state.request_id
            )
            request.lwd_pd_stage_role = "decode"
            request.lwd_pd_upload_end = request.num_computed_tokens
        request.lwd_pd_attempt_id = state.attempt_id
        request.lwd_pd_binding_id = notify.binding_id
        request.lwd_pd_fingerprint = state.fingerprint
        request.lwd_pd_hit_tokens = notify.hit_tokens
        client = self.engine._lwd_coord_clients.get(connection)
        if client is not None:
            client.send_first_req()
        self.engine._lwd_notify_request(connection, request)
        if notify.stage_role == "prefill":
            self.engine._lwd_enqueue_request(request)
        else:
            binding.output_watermark = self.config.d_output_quota_rows or max(
                self.scheduler.max_num_scheduled_tokens,
                1 + getattr(params, "num_speculative_tokens", 0),
            )
            self._send_quota(state, binding)
            state.envelope = None
            self.engine._lwd_probe_chains.pop(state.request_id, None)
            self._commit(state)

    def _send_quota(self, state: LwdPdEdgeState, binding: LwdPdEdgeBinding) -> None:
        self.engine._lwd_send_to_cloud(
            binding.connection,
            LwdPdQuotaNotify(
                state.request_id,
                state.attempt_id,
                binding.notify.binding_id,
                "output",
                binding.output_watermark,
            ),
        )

    def decorate_range(self, request_id: str, item: Any) -> Any:
        state = self.states[request_id]
        role = "decode" if "decode" in state.bindings else "prefill"
        binding = state.bindings[role]
        if state.cancelled or binding.available_windows <= 0:
            raise ValueError("PD input scheduled without receive credit")
        binding.sent_windows += 1
        return msgspec.structs.replace(
            item,
            attempt_id=state.attempt_id,
            binding_id=binding.notify.binding_id,
            window_id=binding.sent_windows,
        )

    def upload_limit(self, request: Any) -> int:
        state = self.states.get(request.request_id)
        if (
            state is None
            or state.cancelled
            or "prefill" not in state.bindings
            or "decode" in state.bindings
        ):
            return 0
        binding = state.bindings["prefill"]
        if binding.available_windows <= 0:
            return 0
        remaining = binding.notify.p_end - request.num_computed_tokens
        return max(0, min(remaining, binding.notify.max_window_tokens or remaining))

    def upload_budget(self, connection: Any) -> int:
        return self.engine._lwd_cloud_params[connection].max_num_batched_tokens

    def authorize_frame(self, connection: Any, notify: LwdC2eNotify) -> bool:
        """Deduplicate BEFORE receiving the physical tensor; never receive twice."""
        count = len(notify.req_ids)
        aligned = (
            notify.attempt_ids,
            notify.binding_ids,
            notify.valid_ranges,
            notify.row_counts,
            notify.final_g,
            notify.range_digests,
            notify.num_accepted_tokens,
            notify.finish_reasons,
        )
        if not count or any(len(values) != count for values in aligned):
            raise ValueError("PD DOWN frame is missing mandatory row identities")
        for index, rid in enumerate(notify.req_ids):
            start, end = notify.valid_ranges[index]
            if (
                not 0 <= start <= end
                or end - start != notify.num_accepted_tokens[index]
                or notify.row_counts[index] < end - start
            ):
                raise ValueError("invalid PD DOWN geometry")
            if (notify.finish_reasons[index] != -1) != (notify.final_g[index] >= 0):
                raise ValueError("PD execution terminal must carry final G")
            state = self.states.get(rid)
            if state is not None and state.attempt_id == notify.attempt_ids[index]:
                binding = state.match(
                    connection, notify.attempt_ids[index], notify.binding_ids[index]
                )
                if binding is None or binding.notify.stage_role != "decode":
                    raise ValueError("PD DOWN does not belong to the decode binding")
        if notify.hidden_num_elements > 0:
            encoded = msgspec.msgpack.encode(notify)
            previous = self.frames.get(connection)
            if previous is not None and notify.down_seqno <= previous[0]:
                if notify.down_seqno == previous[0] and encoded != previous[1]:
                    raise ValueError("conflicting PD DOWN frame replay")
                return False
            if (
                notify.down_seqno < 0
                or previous is not None
                and notify.down_seqno != previous[0] + 1
            ):
                raise ValueError("gap in PD DOWN frame sequence")
            self.frames[connection] = notify.down_seqno, encoded
        return True

    def verify_row(self, notify: LwdC2eNotify, index: int, tokens: list[int]) -> bool:
        state = self.states.get(notify.req_ids[index])
        if (
            state is None
            or state.attempt_id != notify.attempt_ids[index]
            or state.output is None
        ):
            return False
        try:
            accepted = state.output.accept(
                tuple(notify.valid_ranges[index]),
                tokens,
                notify.range_digests[index],
                notify.row_counts[index],
                notify.final_g[index],
            )
            binding = state.bindings["decode"]
            binding.output_watermark += notify.row_counts[index]
            if not state.cancelled and notify.final_g[index] < 0:
                self._send_quota(state, binding)
            return accepted and not state.cancelled
        except ValueError as exc:
            self.fail(state, str(exc))
            return False

    def finalize(
        self, request_id: str, attempt_id: str, accepted: int, reason: str, cancel: bool
    ) -> bool:
        state = self.states.get(request_id)
        if state is None or state.attempt_id != attempt_id:
            raise ValueError("PD finalization has no retained execution attempt")
        state.freeze(accepted, reason, cancel)
        self._commit(state)
        if cancel:
            self.engine.abort_requests([request_id])
        return True

    def _commit(self, state: LwdPdEdgeState) -> None:
        binding = state.bindings.get("decode")
        if state.frontend_commit is None or binding is None:
            return
        accepted, reason, _cancel = state.frontend_commit
        self.outbox.put(
            binding.connection,
            LwdPdResultCommit(
                state.request_id,
                state.attempt_id,
                binding.notify.binding_id,
                accepted,
                reason,
                state.frontend_verified_tokens,
            ),
            time.monotonic(),
            self.config.result_hold_timeout_s,
        )

    def abort(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            state = self.states.get(rid)
            if state is None:
                continue
            state.cancelled = True
            state.tail_pending = False
            for binding in state.bindings.values():
                self._abort_binding(state, binding)
            self.release(rid)

    def _abort_binding(self, state: LwdPdEdgeState, binding: LwdPdEdgeBinding) -> None:
        self.engine._lwd_send_to_cloud(
            binding.connection,
            LwdAbortNotify(
                state.request_id,
                attempt_id=state.attempt_id,
                binding_id=binding.notify.binding_id,
            ),
        )

    def release(self, request_id: str) -> None:
        state = self.states.get(request_id)
        if state is not None:
            state.execution_done = True
            state.request = None
            state.envelope = None

    def fail(self, state: LwdPdEdgeState, error: str) -> None:
        if state.error:
            return
        state.error = error
        lwd_logger.error("[LWD] PD request %s failed: %s", state.request_id, error)
        request = state.request
        self.engine.abort_requests([state.request_id])
        if request is not None:
            output = EngineCoreOutput(
                state.request_id,
                [],
                finish_reason=FinishReason.ERROR,
                lwd_pd_attempt_id=state.attempt_id,
            )
            self.engine.output_queue.put_nowait(
                (
                    request.client_index,
                    EngineCoreOutputs(
                        outputs=[output], finished_requests={state.request_id}
                    ),
                )
            )

    def _result(
        self, state: LwdPdEdgeState, result: dict[str, Any], *, http: bool
    ) -> None:
        validate_result(result)
        binding = state.bindings.get("decode")
        if (
            binding is None
            or result.get("attempt_id") != state.attempt_id
            or result.get("binding_id") != binding.notify.binding_id
        ):
            raise ValueError("PD result has an unrelated identity")
        if (
            state.frontend_commit is None
            or result["prompt_tokens"] != state.n_tokens
            or result["cached_tokens"] != binding.notify.hit_tokens
            or (result["accepted_tokens"], result["frontend_reason"])
            != state.frontend_commit[:2]
        ):
            raise ValueError("PD result differs from the frontend commitment")
        if (
            state.output.final_g is not None
            and result["generated_tokens"] != state.output.final_g
        ):
            raise ValueError("PD result G differs from the verified stream terminal")
        other = state.receipt if http else state.http_result
        if other is not None and other != result:
            raise ValueError("PD HTTP and receipt results differ")
        if http:
            state.http_result = result
        else:
            state.receipt = result

    def tick(self) -> None:
        while not self.done.empty():
            rid, attempt, outcome = self.done.get_nowait()
            state = self.states.get(rid)
            if state is None or state.attempt_id != attempt:
                continue
            try:
                if isinstance(outcome, Exception):
                    raise outcome
                self._result(state, outcome, http=True)
            except Exception as exc:
                self.fail(state, str(exc))
        now = time.monotonic()
        self.control_outbox.flush(now)
        for connection, message in self.outbox.due(now):
            self.engine._lwd_send(self.engine._lwd_clouds[connection], message)
        for message in self.outbox.expire(now):
            lwd_logger.error(
                "[LWD] PD result unconfirmed req=%s attempt=%s",
                message.request_id,
                message.attempt_id,
            )
        for rid, state in list(self.states.items()):
            if not state.execution_done and (
                now >= state.deadline
                or not state.bindings
                and now >= state.bind_deadline
                or state.p_done
                and "decode" not in state.bindings
                and now >= state.bind_deadline
            ):
                self.fail(state, "PD stage or request deadline expired")
            if now >= state.deadline:
                if state.frontend_commit is None or state.receipt is None:
                    lwd_logger.error(
                        "[LWD] PD expired without confirmed frontend result req=%s", rid
                    )
                del self.states[rid]

    def close(self) -> None:
        self.stopped.set()
        self.http.stop()
