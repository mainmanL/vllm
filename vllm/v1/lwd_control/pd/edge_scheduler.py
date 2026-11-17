# SPDX-License-Identifier: Apache-2.0
"""Build a decode tail upload without rewinding the original edge request."""

from __future__ import annotations

from typing import Any

from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdRangeItem, LwdRangeNotify


def lwd_pd_schedule_tail(runtime: Any) -> SchedulerOutput | None:
    scheduler = runtime.scheduler
    for state in runtime.states.values():
        if not state.tail_pending or state.cancelled or state.request is None:
            continue
        binding = state.bindings["decode"]
        if binding.available_windows <= 0:
            continue
        request = state.request
        start = binding.notify.r_pos
        token_ids = list(request.prompt_token_ids[start : state.n_tokens])
        # The supported handoff always consumes exactly the last prompt row.
        if len(token_ids) != 1:
            raise ValueError("PD decode tail differs from its handoff plan")
        account = scheduler._lwd_accounts[binding.connection]
        item = runtime.decorate_range(
            state.request_id,
            LwdRangeItem(
                state.request_id,
                start,
                len(token_ids),
                has_mrope=state.request_id in scheduler._lwd_mrope_positions_dict,
            ),
        )
        seqno = account.take_seqno()
        scheduler.lwd_send_to_cloud(
            binding.connection, LwdRangeNotify(seqno=seqno, items=[item])
        )
        account.register_batch(seqno, {state.request_id})
        output = SchedulerOutput.make_empty()
        output.scheduled_new_reqs = [
            NewRequestData(
                req_id=state.request_id,
                prompt_token_ids=None,
                mm_features=list(request.mm_features),
                sampling_params=None,
                pooling_params=None,
                block_ids=tuple(),
                num_computed_tokens=start,
                lora_request=None,
                lwd_mrope_positions=scheduler.lwd_mrope_positions(state.request_id),
                lwd_pd_stage_role="decode",
                lwd_pd_attempt_id=state.attempt_id,
                lwd_pd_binding_id=binding.notify.binding_id,
                lwd_pd_tail_start=start,
                lwd_pd_tail_end=state.n_tokens,
            )
        ]
        output.lwd_batch = LwdBatch(
            LwdBatchType.LWD_EMBED,
            seqno,
            LwdEmbedBatch(
                req_ids=[state.request_id],
                token_ids=[token_ids],
                prompt_offsets=[start],
                has_mrope=[item.has_mrope],
                connection_id=binding.connection,
                retain_mm_req_ids=[state.request_id],
            ),
        )
        state.tail_pending = False
        return output
    return None
