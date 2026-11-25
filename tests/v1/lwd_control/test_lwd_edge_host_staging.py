# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config.lwd import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
    LwdCloudDpAccount,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import LwdBaseEngineCore
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdAbortNotify,
    LwdRequestNotify,
)
from vllm.v1.lwd_control.lwd_model.lwd_model_family import LwdMropeModelFamily

DP0 = LwdConnectionId(0, 0, 0)
DP1 = LwdConnectionId(0, 0, 1)


def _scheduler() -> LwdEdgeScheduler:
    scheduler = object.__new__(LwdEdgeScheduler)
    scheduler._lwd_accounts = {
        DP0: LwdCloudDpAccount(DP0),
        DP1: LwdCloudDpAccount(DP1),
    }
    scheduler._lwd_aborted_req_ids = {DP0: [], DP1: []}
    scheduler._lwd_route = {}
    scheduler.lwd_model_family = object.__new__(LwdMropeModelFamily)
    scheduler.lwd_model_family._positions = {}
    scheduler._lwd_pending_refills = ()
    scheduler.lwd_has_pending = None
    scheduler.unembed_notify_queue = ()
    scheduler.requests = {}
    return scheduler


def test_placeholder_output_consumes_only_addressed_dp_watermark() -> None:
    scheduler = _scheduler()
    scheduler._lwd_accounts[DP0].pending_batches = {0: {"a"}, 1: {"b"}}
    scheduler._lwd_accounts[DP1].pending_batches = {0: {"c"}}
    output = SimpleNamespace(lwd_up_send_watermarks={DP0: 0})

    result = scheduler.update_from_output(SimpleNamespace(lwd_batch=None), output)

    assert result == {}
    assert scheduler._lwd_accounts[DP0].pending_chunks() == 1
    assert scheduler._lwd_accounts[DP1].pending_chunks() == 1


def test_pending_up_send_keeps_empty_steps_alive_until_watermark_arrives() -> None:
    scheduler = _scheduler()
    account = scheduler._lwd_accounts[DP0]
    account.register_batch(0, {"a"})
    scheduler._lwd_embed_dispatchable = lambda: False

    assert scheduler.has_requests()
    output = scheduler._lwd_schedule_unembed()
    assert output.lwd_batch is None

    scheduler.update_from_output(
        output,
        SimpleNamespace(lwd_up_send_watermarks={DP0: 0}),
    )

    assert account.pending_chunks() == 0
    assert not scheduler.has_requests()


def test_announced_refill_survives_settlement_until_watermark() -> None:
    scheduler = _scheduler()
    account = scheduler._lwd_accounts[DP0]
    account.register_batch(3, {"r1"})
    scheduler._lwd_route["r1"] = DP0
    scheduler._lwd_announced_refills = {(DP0, 3)}

    scheduler._lwd_settle_request("r1")

    assert account.pending_batches == {3: {"r1"}}
    account.note_send_watermark(3)
    assert account.pending_batches == {}


def test_first_token_keeps_pending_but_finish_releases_request() -> None:
    scheduler = _scheduler()
    account = scheduler._lwd_accounts[DP0]
    account.register_batch(0, {"r1"})
    scheduler._lwd_route["r1"] = DP0
    scheduler.requests["r1"] = SimpleNamespace(is_finished=lambda: False)

    assert scheduler.lwd_edge_deliver_tokens("r1", [7], finished=False)
    assert account.pending_chunks() == 1

    scheduler._lwd_settle_request("r1")
    assert account.pending_chunks() == 0


def test_abort_returns_original_routes_and_queues_worker_notice() -> None:
    scheduler = _scheduler()
    for request_id, connection in (("r1", DP1), ("r2", DP0)):
        scheduler._lwd_route[request_id] = connection

    routed = scheduler.lwd_edge_abort(["r1", "r2", "unknown"])

    assert routed == [(DP1, "r1"), (DP0, "r2")]
    assert scheduler._lwd_aborted_req_ids == {DP0: ["r2"], DP1: ["r1"]}
    assert scheduler._lwd_route == {}
    assert scheduler.has_requests()

    notice = scheduler._lwd_schedule_abort_notice()

    assert notice is not None and notice.lwd_batch is not None
    assert notice.lwd_batch.seqno == -1
    assert notice.lwd_batch.batch_meta.aborted_req_ids == ["r2"]
    assert scheduler._lwd_aborted_req_ids == {DP0: [], DP1: ["r1"]}


def test_range_leaves_mrope_bytes_empty_and_full_prompt_is_read_only() -> None:
    class Positions:
        def contiguous(self):
            return self

        def numpy(self):
            return self

        def tobytes(self):
            return b"full-prompt"

    scheduler = _scheduler()
    scheduler.lwd_model_family._positions["r1"] = Positions()
    scheduler.requests["r1"] = SimpleNamespace(
        num_computed_tokens=5,
        prompt_token_ids=[0, 1, 2, 3, 4],
        lwd_prefix_nonce=11,
    )

    built = scheduler._lwd_build_range_items({"r1": 2})

    assert built is not None
    items, _, _, _, _ = built
    assert items[0].has_mrope
    assert items[0].prefix_nonce == 11
    assert items[0].mrope_positions == b""
    assert scheduler.lwd_mrope_positions("r1") == b"full-prompt"


def test_prefill_cap_reserves_spec_tokens_only_for_decode_requests() -> None:
    scheduler = _scheduler()
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_running_reqs = 16
    scheduler._lwd_accounts[DP0].num_speculative_tokens = 4
    scheduler._lwd_chunk_unit = 128
    scheduler._lwd_route = {"r1": DP0, "r2": DP0}
    scheduler.running = [
        SimpleNamespace(
            request_id="r1",
            num_prompt_tokens=8,
            num_computed_tokens=0,
            is_finished=lambda: False,
        ),
        SimpleNamespace(
            request_id="r2",
            num_prompt_tokens=8,
            num_computed_tokens=8,
            is_finished=lambda: False,
        ),
    ]
    scheduler.skipped_waiting = []
    scheduler.waiting = []
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="test"))
    )

    _, _, cap = scheduler._lwd_pick_prefill_batch(DP0)

    # Only the DECODE-phase request reserves spec tokens; the in-flight
    # PREFILL request does not.
    assert cap == 128 - 4


def test_engine_sends_abort_to_original_dp(monkeypatch) -> None:
    engine = object.__new__(LwdEdgeEngineCore)
    engine.scheduler = SimpleNamespace(
        lwd_edge_abort=lambda _: [(DP1, "r1")],
        requests={},
    )
    engine._lwd_probe_manager = SimpleNamespace(abort=lambda _: None)
    engine._lwd_probe_chains = {}
    sent = []
    engine._lwd_send_to_cloud = lambda dp_idx, msg: sent.append((dp_idx, msg))
    monkeypatch.setattr(LwdBaseEngineCore, "abort_requests", lambda *_: None)

    engine.abort_requests(["r1"])

    assert sent == [(DP1, LwdAbortNotify("r1"))]


def test_engine_request_notify_uses_scheduler_full_mrope_bytes() -> None:
    engine = object.__new__(LwdEdgeEngineCore)
    engine.scheduler = SimpleNamespace(
        lwd_mrope_positions=lambda _: b"full-prompt",
        lwd_model_family=SimpleNamespace(
            request_metadata=lambda _: SimpleNamespace(image_placeholder_spans=[])
        ),
    )
    engine._lwd_probe_chains = {}
    engine.lwd_config = SimpleNamespace(coordination=None)
    sent = []
    engine._lwd_send_to_cloud = lambda dp_idx, msg: sent.append((dp_idx, msg))
    sampling_params = SimpleNamespace(
        **dict(LWD_SAMPLING_DEFAULTS),
        eos_token_id=99,
        structured_outputs=None,
    )
    request = SimpleNamespace(
        request_id="r1",
        num_prompt_tokens=1,
        num_computed_tokens=0,
        block_hashes=[],
        sampling_params=sampling_params,
        structured_output_request=None,
        skip_reading_prefix_cache=False,
    )

    engine._lwd_notify_request(DP0, request)

    assert len(sent) == 1
    dp_idx, message = sent[0]
    assert dp_idx == DP0
    assert isinstance(message, LwdRequestNotify)
    assert message.mrope_positions == b"full-prompt"
