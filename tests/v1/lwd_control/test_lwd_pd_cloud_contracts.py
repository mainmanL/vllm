# SPDX-License-Identifier: Apache-2.0
"""CPU contracts of PD admission, frozen cache ownership and explicit routing.

Load the real control modules without importing the optional vLLM kernels.
Only the native scheduler boundary is represented by a small behavioral fake.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import queue
import sys
import types
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_SOURCE = Path(__file__).resolve().parents[3] / "vllm"


@pytest.fixture
def modules(monkeypatch):
    for package in (
        "vllm",
        "vllm.v1",
        "vllm.v1.lwd_control",
        "vllm.v1.lwd_control.pd",
        "vllm.v1.lwd_control.lwd_comm",
        "vllm.v1.lwd_control.control_cloud_scheduler",
    ):
        module = types.ModuleType(package)
        module.__path__ = [str(_SOURCE.joinpath(*package.split(".")[1:]))]
        monkeypatch.setitem(sys.modules, package, module)
    logger = types.ModuleType("vllm.lwd_logger")
    logger.lwd_logger = logging.getLogger("pd-contracts")
    monkeypatch.setitem(sys.modules, logger.__name__, logger)
    ordinary_logger = types.ModuleType("vllm.logger")
    ordinary_logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, ordinary_logger.__name__, ordinary_logger)
    envs = types.ModuleType("vllm.envs")
    monkeypatch.setitem(sys.modules, envs.__name__, envs)

    class Status(IntEnum):
        WAITING = 0
        WAITING_FOR_REMOTE_KVS = 1
        RUNNING = 2
        FINISHED_ERROR = 3
        FINISHED_ABORTED = 4

    class Finish(IntEnum):
        STOP = 0
        LENGTH = 1
        ABORT = 2
        ERROR = 3

    request = types.ModuleType("vllm.v1.request")
    request.RequestStatus = Status
    monkeypatch.setitem(sys.modules, request.__name__, request)
    engine = types.ModuleType("vllm.v1.engine")
    engine.FinishReason = Finish
    engine.EngineCoreRequestType = SimpleNamespace(ADD="add", ABORT="abort")
    monkeypatch.setitem(sys.modules, engine.__name__, engine)
    sampling = types.ModuleType("vllm.sampling_params")
    sampling.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, sampling.__name__, sampling)
    names = [
        "lwd_comm.lwd_messages",
        "lwd_comm.lwd_link_protocol",
        "lwd_comm.lwd_id_adapter",
        "lwd_comm.lwd_prefix",
        "lwd_comm.lwd_mm_identity",
        "pd.envelope",
        "pd.results",
        "pd.control_outbox",
        "pd.cloud_scheduler",
        "pd.cloud_runtime",
        "pd.cloud_http",
        "control_cloud_scheduler.lwd_prefix_cache",
        "control_cloud_scheduler.lwd_cloud_control",
    ]
    imported = {}
    for name in names:
        full_name = f"vllm.v1.lwd_control.{name}"
        monkeypatch.delitem(sys.modules, full_name, raising=False)
        imported[name] = importlib.import_module(full_name)
    yield SimpleNamespace(
        **{name.rsplit(".", 1)[-1]: module for name, module in imported.items()},
        Status=Status,
        Finish=Finish,
    )
    for name in names:
        sys.modules.pop(f"vllm.v1.lwd_control.{name}", None)


def _engine(modules, role="prefill"):
    scheduler = SimpleNamespace(
        max_model_len=1024,
        max_num_scheduled_tokens=8192,
        requests={},
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        lwd_prefix_reservations=Mock(),
        lwd_pd_scheduling=SimpleNamespace(fail=None, records={}),
        finish_requests=Mock(),
        _lwd_drop_request=Mock(),
    )
    scheduler.lwd_prefix_reservations.reserve.return_value = 128
    peer = modules.lwd_link_protocol.LwdPeer(7, 0)

    def build(notify):
        return SimpleNamespace(
            request_id=notify.request_id,
            num_prompt_tokens=notify.num_prompt_tokens,
            sampling_params=SimpleNamespace(skip_reading_prefix_cache=False),
            skip_reading_prefix_cache=notify.skip_reading_prefix_cache,
            block_hashes=list(notify.block_hashes),
            use_structured_output=False,
        )

    config = SimpleNamespace(
        enabled=True,
        notify_claim_timeout_s=30,
        request_ttl_s=300,
        result_hold_timeout_s=300,
        max_window_tokens=0,
        max_inflight_windows=4,
        d_output_quota_rows=0,
    )
    engine = SimpleNamespace(
        lwd_config=SimpleNamespace(stage_role=role, pd=config, instance_id=11),
        scheduler=scheduler,
        _lwd_engine_index=2,
        _lwd_hash_block_size=128,
        _lwd_prefix_enabled=True,
        _lwd_edges={peer},
        cloud_control_event_queue=queue.Queue(),
        input_queue=queue.Queue(),
        _lwd_send_until_queued=Mock(),
        _lwd_build_request=build,
        _lwd_ledger=SimpleNamespace(
            request_routes={}, prompt_lens={}, request_nonces={}, request_seqnos={}
        ),
        vllm_config=SimpleNamespace(model_config=object()),
    )
    engine._lwd_send = lambda peer, message: (
        engine._lwd_send_until_queued(peer, message) or True
    )
    return engine, peer


def _envelope(
    modules, *, input_abi="abi", model_id="m", attempt_id="attempt", skip_cache=False
):
    return modules.envelope.build_envelope(
        request_id="r",
        attempt_id=attempt_id,
        edge_id=7,
        tenant_id="t",
        original_prompt_tokens=257,
        user_max_output_tokens=16,
        sampling_params=SimpleNamespace(
            eos_token_id=None, skip_reading_prefix_cache=skip_cache
        ),
        model_id=model_id,
        input_abi=input_abi,
        digest_block_size=128,
        digest_chain=[bytes(32), bytes([1]) * 32],
        digest_tail_hash=bytes([2]) * 32,
        ttl_s=300,
    )


def test_p_registration_pins_stage_bound_and_claim_reuses_snapshot(
    modules, monkeypatch
):
    engine, peer = _engine(modules)
    runtime = modules.cloud_runtime.LwdPdCloudRuntime(engine)
    monkeypatch.setattr(
        modules.cloud_runtime, "model_needs_truncate", lambda *args: True
    )
    envelope = _envelope(modules)
    ctx = runtime.register(envelope, None, 2)
    engine.scheduler.lwd_prefix_reservations.reserve.assert_called_once_with(
        ctx.request, max_hit_tokens=255
    )
    assert (ctx.hit_tokens, ctx.p_end, ctx.r_pos) == (128, 256, 256)
    assert runtime.register(envelope, None, 2) is ctx
    notify = modules.lwd_messages.LwdRequestNotify(
        ctx.request_id,
        257,
        attempt_id=ctx.attempt_id,
        binding_id=ctx.binding_id,
        stage_role="prefill",
        fingerprint=envelope["fingerprint"],
        prefix_hit_tokens=128,
    )
    runtime.claim(peer, notify)
    runtime.claim(peer, notify)
    assert engine.input_queue.qsize() == 1
    assert engine.input_queue.get()[1][0] is ctx.request
    assert engine.scheduler.lwd_prefix_reservations.reserve.call_count == 1
    wrong = modules.lwd_messages.LwdRequestNotify(
        ctx.request_id,
        257,
        attempt_id="old",
        binding_id=ctx.binding_id,
        stage_role="prefill",
    )
    with pytest.raises(ValueError, match="matching stage"):
        runtime.claim(peer, wrong)


def test_p_cancel_before_claim_releases_pin_and_never_adds_request(
    modules, monkeypatch
):
    engine, _ = _engine(modules)
    runtime = modules.cloud_runtime.LwdPdCloudRuntime(engine)
    monkeypatch.setattr(
        modules.cloud_runtime, "model_needs_truncate", lambda *args: False
    )
    ctx = runtime.register(_envelope(modules), None, 2)
    assert runtime.abort(ctx.request_id, ctx.attempt_id, ctx.binding_id)
    engine.scheduler.lwd_prefix_reservations.release.assert_called_once_with(
        ctx.request_id
    )
    assert ctx.terminal and engine.input_queue.empty()
    assert not runtime.abort(ctx.request_id, "old-attempt", ctx.binding_id)


def test_fine_grained_snapshot_uses_p_end_bound_and_survives_owner_release(modules):
    blocks = [SimpleNamespace(is_null=False)]
    coordinator = SimpleNamespace(
        find_longest_cache_hit=Mock(return_value=((blocks,), 128, 0))
    )
    manager = SimpleNamespace(coordinator=coordinator, block_pool=Mock())
    reservations = modules.lwd_prefix_cache.LwdPrefixReservations(manager)
    request = SimpleNamespace(
        request_id="r",
        num_prompt_tokens=257,
        block_hashes=[b"h"],
        skip_reading_prefix_cache=False,
    )
    assert reservations.reserve(request, max_hit_tokens=255) == 128
    coordinator.find_longest_cache_hit.assert_called_once_with(
        [b"h"], max_cache_hit_length=255
    )
    manager.block_pool.touch.assert_called_once_with(blocks)
    coordinator.find_longest_cache_hit.side_effect = AssertionError(
        "snapshot was queried twice"
    )
    assert reservations.snapshot(request) == (((blocks[0],),), 128)
    reservations.release("r")
    reservations.release("r")
    manager.block_pool.free_blocks.assert_called_once_with(blocks)


def test_http_skip_prefix_cache_is_respected_before_bind(modules, monkeypatch):
    engine, _peer = _engine(modules)
    monkeypatch.setattr(
        modules.cloud_runtime, "model_needs_truncate", lambda *args: False
    )
    manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            find_longest_cache_hit=Mock(
                side_effect=AssertionError("disabled prefix cache was queried")
            )
        ),
        block_pool=Mock(),
    )
    reservations = modules.lwd_prefix_cache.LwdPrefixReservations(manager)
    engine.scheduler.lwd_prefix_reservations = reservations
    runtime = modules.cloud_runtime.LwdPdCloudRuntime(engine)
    ctx = runtime.register(_envelope(modules, skip_cache=True), None, 2)
    assert ctx.hit_tokens == 0
    assert ctx.request.skip_reading_prefix_cache
    assert ctx.request.sampling_params.skip_reading_prefix_cache
    assert reservations.snapshot(ctx.request) == ((), 0)
    manager.coordinator.find_longest_cache_hit.assert_not_called()


def test_cloud_pd_send_backpressure_never_sleeps_and_expires_closed(
    modules, monkeypatch
):
    engine, _peer = _engine(modules)
    blocked, healthy = "blocked", "healthy"
    engine._lwd_send = Mock(side_effect=lambda peer, message: peer != blocked)
    runtime = modules.cloud_runtime.LwdPdCloudRuntime(engine)
    monkeypatch.setattr("time.sleep", lambda seconds: pytest.fail("model thread slept"))
    runtime.send(blocked, "bind")
    runtime.send(healthy, "receipt")
    assert runtime.control_outbox.count == 1
    deadline = runtime.control_outbox.queues[blocked][0][0]
    runtime.control_outbox.flush(deadline - 1)
    assert runtime.control_outbox.count == 1
    with pytest.raises(RuntimeError, match="send deadline"):
        runtime.control_outbox.flush(deadline)


@pytest.mark.parametrize("tail_first", [False, True])
def test_decode_import_progresses_without_tail_then_joins_existing_mtp(
    modules, tail_first
):
    Status = modules.Status
    request = SimpleNamespace(
        request_id="new",
        status=Status.WAITING,
        num_preemptions=0,
        num_computed_tokens=0,
        is_finished=lambda: False,
        lwd_window_cap=None,
    )
    running = SimpleNamespace(
        request_id="old",
        status=Status.RUNNING,
        num_preemptions=0,
        num_computed_tokens=260,
        is_finished=lambda: False,
    )
    scheduler = SimpleNamespace(
        requests={"new": request, "old": running},
        failed_recving_kv_req_ids=set(),
        finished_recving_kv_req_ids=set(),
        max_num_scheduled_tokens=16,
        num_spec_tokens=3,
        _lwd_compute_decode_consumption=lambda ids: 4 * len(ids),
    )
    calls = []

    def native(ids):
        calls.append(ids)
        counts = {}
        if "new" in ids:
            if request.status == Status.WAITING and request.num_computed_tokens == 0:
                request.status = Status.WAITING_FOR_REMOTE_KVS
            else:
                assert request.lwd_window_cap == 1
                counts["new"] = 1
        if "old" in ids:
            counts["old"] = 4
        return SimpleNamespace(
            num_scheduled_tokens=counts, scheduled_spec_decode_tokens={"old": [1, 2, 3]}
        )

    scheduler._lwd_schedule = native
    scheduler._update_waiting_for_remote_kv = lambda req: setattr(
        req, "num_computed_tokens", 256
    )
    admission = modules.cloud_scheduler.LwdPdCloudScheduling(scheduler)
    record = modules.cloud_scheduler.LwdPdDecodeAdmission(
        "new", "b", "a", 256, 257, 100
    )
    admission.register(record)
    admission.register(
        modules.cloud_scheduler.LwdPdDecodeAdmission(
            "old",
            "c",
            "d",
            256,
            257,
            100,
            kv_ready=True,
            tail_ready=True,
            tail_done=True,
        )
    )
    if tail_first:
        admission.note_arrival("new", 257)
    admission.schedule()
    assert calls[-1] == ["old", "new"] and not record.kv_ready
    scheduler.finished_recving_kv_req_ids.add("new")
    admission.schedule()
    assert record.kv_ready and not record.tail_done
    if not tail_first:
        assert calls[-1] == ["old"]
        admission.note_arrival("new", 257)
        admission.schedule()
    assert calls[-1] == ["old", "new"]
    assert scheduler.num_spec_tokens == 3 and request.lwd_window_cap is None
    assert record.tail_inflight and not record.tail_done
    admission.consumed({"new"}, {"new": ("old-attempt", "b")})
    assert not record.tail_done
    admission.consumed({"new"}, {"new": ("a", "b")})
    assert record.tail_done
    # Runtime physical rows can be fewer than SO speculative slots. Only
    # actual transport rows spend authorization after the async completion.
    old_record = admission.records["old"]
    before = old_record.output_quota
    completed = SimpleNamespace(lwd_pd_reserved_rows={"old": ("d", "c", 4)})
    admission.reconcile_rows(
        completed,
        SimpleNamespace(
            lwd_down_carriers=[(None, ["old"], 64, 5, "connection")],
            lwd_pd_down_metadata={
                ("connection", 5): {
                    "attempt_ids": ["d"],
                    "binding_ids": ["c"],
                    "row_counts": [2],
                }
            },
        ),
    )
    assert old_record.output_quota == before + 2


def test_pd_preemption_fails_without_reimport_or_refill(modules):
    request = SimpleNamespace(num_preemptions=1, is_finished=lambda: False)
    scheduler = SimpleNamespace(requests={"r": request})
    admission = modules.cloud_scheduler.LwdPdCloudScheduling(scheduler)
    admission.register(
        modules.cloud_scheduler.LwdPdDecodeAdmission("r", "b", "a", 2, 3, 10)
    )
    admission.fail = Mock()
    assert admission._collect() == ([], [], [])
    admission.fail.assert_called_once_with("r", "PD preemption recovery is unsupported")


@pytest.mark.asyncio
async def test_http_bridge_preserves_proxy_dp_and_attempt_result(modules):
    queues = {2: queue.Queue(), 4: queue.Queue()}
    bridge = modules.cloud_http.LwdPdCloudHttpBridge(queues)
    envelope = _envelope(modules)
    task = asyncio.create_task(bridge.register(envelope, 2, None, 1))
    await asyncio.sleep(0)
    assert queues[2].get_nowait()["engine_index"] == 2 and queues[4].empty()
    bridge.dispatch(
        {
            "type": "pd_register",
            "request_id": "e7-r",
            "attempt_id": "attempt",
            "ok": True,
        }
    )
    registration = await task
    bridge.dispatch(
        {
            "type": "pd_result",
            "request_id": "e7-r",
            "attempt_id": "old",
            "ok": True,
            "result": {"bad": True},
        }
    )
    await asyncio.sleep(0)
    assert not registration.result.done()
    expected = {"usage": {"completion_tokens": 0}, "status": "cancelled"}
    bridge.dispatch(
        {
            "type": "pd_result",
            "request_id": "e7-r",
            "attempt_id": "attempt",
            "ok": True,
            "result": expected,
        }
    )
    assert (await registration.result)["result"] is expected
    bridge.release(registration)
    assert not bridge.pending
    for target in (None, 0, True):
        with pytest.raises(ValueError, match="target_dp"):
            await bridge.register(envelope, target, None, 1)
    assert queues[2].empty() and queues[4].empty()


@pytest.mark.asyncio
async def test_http_export_to_decode_c2e_commit_receipt_share_one_result(
    modules, monkeypatch
):
    import json

    monkeypatch.setattr(
        modules.cloud_runtime, "model_needs_truncate", lambda *args: True
    )
    envelope = _envelope(modules)

    def setup(role, dp):
        engine, peer = _engine(modules, role)
        engine._lwd_engine_index = dp
        engine.lwd_config.pd.export_hold_timeout_s = 1
        engine.lwd_config.pd.usage_wait_timeout_s = 1
        engine.scheduler.lwd_pd_scheduling = (
            modules.cloud_scheduler.LwdPdCloudScheduling(engine.scheduler)
        )
        runtime = modules.cloud_runtime.LwdPdCloudRuntime(engine)
        commands = queue.Queue()
        bridge = modules.cloud_http.LwdPdCloudHttpBridge({dp: commands})
        return engine, peer, runtime, bridge, commands

    def dispatch(engine, bridge):
        while not engine.cloud_control_event_queue.empty():
            bridge.dispatch(engine.cloud_control_event_queue.get_nowait())

    async def begin_http(engine, runtime, bridge, commands, body):
        task = asyncio.create_task(
            modules.cloud_http.lwd_pd_http_response(
                bridge, body, envelope, runtime.stage_role, engine.lwd_config.pd
            )
        )
        await asyncio.sleep(0)
        runtime.command(commands.get_nowait())
        dispatch(engine, bridge)
        await asyncio.sleep(0)
        return task

    p, peer, p_runtime, p_bridge, commands = setup("prefill", 2)
    p_http = await begin_http(p, p_runtime, p_bridge, commands, {"target_dp": 2})
    p_ctx = p_runtime.active["e7-r"]
    p_runtime.claim(
        peer,
        modules.lwd_messages.LwdRequestNotify(
            "e7-r",
            257,
            attempt_id=envelope["attempt_id"],
            binding_id=p_ctx.binding_id,
            fingerprint=envelope["fingerprint"],
            stage_role="prefill",
            prefix_hit_tokens=128,
        ),
    )
    assert p.input_queue.qsize() == 1
    p_runtime.on_outputs(
        {
            0: SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        request_id="e7-r",
                        kv_transfer_params={
                            "remote_engine_id": "P-DP2",
                            "remote_request_id": "export-r",
                            "remote_host": "127.0.0.1",
                            "remote_port": 9001,
                        },
                        new_token_ids=[999],
                        finish_reason=modules.Finish.LENGTH,
                    )
                ],
                finished_requests={"e7-r"},
            )
        }
    )
    dispatch(p, p_bridge)
    p_response = await p_http
    exported = json.loads(p_response.body)
    assert p_ctx.request is None
    assert not {"sampling", "structured_outputs", "digest"}.intersection(p_ctx.envelope)
    assert exported["usage"]["completion_tokens"] == 0
    handoff = exported["kv_transfer_params"]
    assert handoff["lwd_handoff"]["p_end"] == 256
    assert handoff["lwd_handoff"]["hit_tokens"] == 128

    d, peer, d_runtime, d_bridge, commands = setup("decode", 4)
    d_http = await begin_http(
        d,
        d_runtime,
        d_bridge,
        commands,
        {"target_dp": 4, "kv_transfer_params": handoff},
    )
    d_ctx = d_runtime.active["e7-r"]
    d_runtime.claim(
        peer,
        modules.lwd_messages.LwdRequestNotify(
            "e7-r",
            257,
            attempt_id=envelope["attempt_id"],
            binding_id=d_ctx.binding_id,
            fingerprint=envelope["fingerprint"],
            stage_role="decode",
        ),
    )
    d_response = await d_http
    d_runtime.on_outputs(
        {
            0: SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        request_id="e7-r",
                        kv_transfer_params=None,
                        new_token_ids=[8, 9, 10],
                        finish_reason=modules.Finish.LENGTH,
                    )
                ],
                finished_requests={"e7-r"},
            )
        }
    )
    notify = d_runtime.annotate(
        modules.lwd_messages.LwdC2eNotify(
            32, [[0]], [3], ["e7-r"], finish_reasons=[1], down_seqno=0
        ),
        [4],
        {
            "attempt_ids": [d_ctx.attempt_id],
            "binding_ids": [d_ctx.binding_id],
            "row_counts": [4],
        },
    )
    assert notify.valid_ranges == [[0, 3]] and notify.row_counts == [4]
    assert notify.final_g == [3] and notify.range_digests[0]
    assert not d_ctx.published
    commit = modules.lwd_messages.LwdPdResultCommit(
        "e7-r", d_ctx.attempt_id, d_ctx.binding_id, 2, "stop_string", 3
    )
    d_runtime.commit(peer, commit)
    result = d_ctx.results.result
    d_runtime.commit(peer, commit)
    receipts = [
        call.args[1]
        for call in d._lwd_send_until_queued.call_args_list
        if isinstance(call.args[1], modules.lwd_messages.LwdPdResultReceipt)
    ]
    assert len(receipts) == 2 and receipts[0].result == receipts[1].result == result
    assert result["usage"]["completion_tokens"] == 2
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 128
    dispatch(d, d_bridge)
    frames = [frame async for frame in d_response.body_iterator]
    chunks = [json.loads(frame[6:]) for frame in frames if frame.startswith("data: {")]
    assert chunks[0]["lwd_pd_result"] == result
    assert chunks[0]["usage"] == result["usage"]
    assert frames[-1] == "data: [DONE]\n\n"


def test_commit_before_decode_claim_explicitly_confirms_zero_output(
    modules, monkeypatch
):
    monkeypatch.setattr(
        modules.cloud_runtime, "model_needs_truncate", lambda *args: True
    )
    p, peer = _engine(modules)
    p_runtime = modules.cloud_runtime.LwdPdCloudRuntime(p)
    envelope = _envelope(modules)
    p_ctx = p_runtime.register(envelope, None, 2)
    p_runtime._export(
        p_ctx,
        {
            "remote_engine_id": "P",
            "remote_request_id": "r",
            "remote_host": "127.0.0.1",
            "remote_port": 9001,
        },
    )
    d, peer = _engine(modules, "decode")
    d_runtime = modules.cloud_runtime.LwdPdCloudRuntime(d)
    ctx = d_runtime.register(envelope, p_ctx.export, 2)
    d_runtime.commit(
        peer,
        modules.lwd_messages.LwdPdResultCommit(
            "e7-r", ctx.attempt_id, ctx.binding_id, 0, "cancelled", 0
        ),
    )
    assert ctx.results.result["generated_tokens"] == 0
    assert ctx.results.result["accepted_tokens"] == 0
    assert ctx.results.result["status"] == "cancelled"
    assert d.input_queue.empty() and ctx.terminal


@pytest.mark.asyncio
async def test_real_cloud_asgi_validates_frozen_identity_and_explicit_dp(
    modules, monkeypatch
):
    import json

    import httpx

    monkeypatch.setattr(
        modules.cloud_runtime, "model_needs_truncate", lambda *args: False
    )
    engine, _peer = _engine(modules)
    engine.lwd_config.pd.export_hold_timeout_s = 1
    runtime = modules.cloud_runtime.LwdPdCloudRuntime(engine)
    commands = queue.Queue()
    bridge = modules.lwd_cloud_control.LwdCloudControlBridge(
        {2: commands}, engine.cloud_control_event_queue
    )
    bridge.start()
    fingerprint = bytes([9]) * 32
    app = modules.lwd_cloud_control.create_lwd_cloud_control_app(
        bridge,
        fingerprint,
        pd_config=engine.lwd_config.pd,
        stage_role="prefill",
        pd_model_names=("m",),
    )

    def body(envelope, **extra):
        return {
            "model": "m",
            "messages": [{"role": "user", "content": json.dumps(envelope)}],
            **extra,
        }

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://cloud"
        ) as client:
            invalid = await client.post(
                "/v1/chat/completions", json=body(_envelope(modules), target_dp=2)
            )
            assert invalid.status_code == 400 and "ABI" in invalid.json()["error"]
            invalid = await client.post(
                "/v1/chat/completions",
                json=body(
                    _envelope(modules, input_abi=fingerprint.hex(), model_id="another"),
                    target_dp=2,
                ),
            )
            assert (
                invalid.status_code == 400
                and "model identity" in invalid.json()["error"]
            )
            envelope = _envelope(modules, input_abi=fingerprint.hex())
            invalid = await client.post("/v1/chat/completions", json=body(envelope))
            assert invalid.status_code == 400 and commands.empty()
            response_task = asyncio.create_task(
                client.post("/v1/chat/completions", json=body(envelope, target_dp=2))
            )
            for _ in range(100):
                if not commands.empty():
                    break
                await asyncio.sleep(0.001)
            runtime.command(commands.get_nowait())
            ctx = runtime.active["e7-r"]
            runtime._export(
                ctx,
                {
                    "remote_engine_id": "P-DP2",
                    "remote_request_id": "r",
                    "remote_host": "127.0.0.1",
                    "remote_port": 9,
                },
            )
            response = await response_task
            assert response.status_code == 200
            assert (
                response.json()["kv_transfer_params"]["lwd_handoff"]["hit_tokens"]
                == 128
            )
            assert commands.empty()
    finally:
        bridge.close()
        bridge._dispatch_thread.join(timeout=1)


def test_late_cancelled_range_does_not_activate_reused_request_binding(
    modules, monkeypatch
):
    monkeypatch.setattr(
        modules.cloud_runtime, "model_needs_truncate", lambda *args: True
    )
    engine, _peer = _engine(modules)
    engine._lwd_connection_of = lambda peer: (peer.instance_id, 11, 2)
    runtime = modules.cloud_runtime.LwdPdCloudRuntime(engine)
    old = runtime.register(_envelope(modules), None, 2)
    runtime.abort(old.request_id, old.attempt_id, old.binding_id)
    new = runtime.register(_envelope(modules, attempt_id="new"), None, 2)
    item = modules.lwd_messages.LwdRangeItem(
        "e7-r",
        255,
        1,
        attempt_id=old.attempt_id,
        binding_id=old.binding_id,
        window_id=1,
    )
    connection = engine._lwd_connection_of(old.peer)
    assert runtime.validate_range(connection, item, 8) == 257
    assert not runtime.note_arrival(connection, 8, "e7-r", 256)
    assert not new.input_windows and not new.terminal
    quota = engine._lwd_send_until_queued.call_args.args[1]
    assert (quota.kind, quota.binding_id, quota.watermark) == (
        "recv",
        old.binding_id,
        1,
    )


@pytest.mark.parametrize(
    ("configured", "expected"), [(0, 8192), (128, 128), (16384, 8192)]
)
def test_bind_advertises_effective_cloud_window_budget(
    modules, monkeypatch, configured, expected
):
    monkeypatch.setattr(
        modules.cloud_runtime, "model_needs_truncate", lambda *args: False
    )
    engine, _peer = _engine(modules)
    engine.lwd_config.pd.max_window_tokens = configured
    runtime = modules.cloud_runtime.LwdPdCloudRuntime(engine)
    ctx = runtime.register(_envelope(modules), None, 2)
    runtime.bind(ctx)
    bind = engine._lwd_send_until_queued.call_args.args[1]
    assert bind.max_window_tokens == expected
    assert bind.recv_windows == engine.lwd_config.pd.max_inflight_windows


def test_pd_single_receive_credit_advances_inside_large_alignment_block(
    modules, monkeypatch
):
    """Run real arrival selection and native split with window128/credit1/block512."""
    import __future__

    import ast
    import itertools
    from dataclasses import dataclass

    def load_function(path, name, namespace):
        tree = ast.parse(path.read_text())
        selected = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
        )
        exec(
            compile(
                ast.Module(body=[selected], type_ignores=[]),
                str(path),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        return namespace[name]

    namespace = {
        "dataclass": dataclass,
        "__name__": __name__,
        "itertools": itertools,
        "parse_edge_id": modules.lwd_id_adapter.parse_edge_id,
    }
    cloud_source = (
        _SOURCE / "v1/lwd_control/control_cloud_scheduler/lwd_cloud_scheduler.py"
    )
    load_function(cloud_source, "_LwdPrefillWindow", namespace)
    collect = load_function(cloud_source, "_lwd_collect_prefill_windows", namespace)
    native_namespace = {}
    load_function(
        _SOURCE / "v1/kv_cache_interface.py",
        "get_mamba_prefill_checkpoint_position",
        native_namespace,
    )
    split = load_function(
        _SOURCE / "v1/core/sched/scheduler.py",
        "_mamba_block_aligned_split",
        native_namespace,
    )
    monkeypatch.setattr(
        modules.cloud_runtime, "model_needs_truncate", lambda *args: True
    )
    engine, _peer = _engine(modules)
    engine.lwd_config.pd.max_window_tokens = 128
    engine.lwd_config.pd.max_inflight_windows = 1
    engine._lwd_prefix_enabled = False
    engine._lwd_connection_of = lambda peer: (peer.instance_id, 11, 2)
    runtime = modules.cloud_runtime.LwdPdCloudRuntime(engine)
    ctx = runtime.register(_envelope(modules), None, 2)
    request = ctx.request
    request.num_prompt_tokens = request.num_tokens = 256
    request.num_computed_tokens = request.num_preemptions = 0
    request.shared_prefix_boundary = 0
    request.is_finished = lambda: False
    scheduler = SimpleNamespace(
        running=[request],
        waiting=[],
        skipped_waiting=[],
        _lwd_prefill_group_cursor=0,
        _lwd_dropped=set(),
        _lwd_prefix_enabled=False,
        lwd_recv_rows={"e7-r": 128},
    )
    native = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=512),
        use_eagle_block_drop=False,
        hash_block_size=128,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=False,
        mamba_fine_grained_prefix_cache=False,
    )
    for window_id in (1, 2):
        start = (window_id - 1) * 128
        item = modules.lwd_messages.LwdRangeItem(
            "e7-r",
            start,
            128,
            binding_id=ctx.binding_id,
            attempt_id=ctx.attempt_id,
            window_id=window_id,
        )
        runtime.validate_range(engine._lwd_connection_of(ctx.peer), item, window_id)
        scheduler.lwd_recv_rows["e7-r"] = start + 128
        windows = collect(scheduler, 8192)
        assert [(window.start, window.num_rows) for window in windows] == [(start, 128)]
        # The genuine native method permits sub-block private state progress;
        # it does not round this received window to zero at a 512-row boundary.
        assert split(native, request, windows[0].num_rows) == 128
        request.num_computed_tokens += 128
        runtime.return_input_credit(ctx, request.num_computed_tokens)
        assert not ctx.input_windows
        returned = engine._lwd_send_until_queued.call_args.args[1]
        assert returned.kind == "recv" and returned.watermark == window_id
    assert request.num_computed_tokens == ctx.p_end


def test_reused_id_waits_for_native_cleanup_and_unidentified_late_output_is_ignored(
    modules, monkeypatch
):
    monkeypatch.setattr(
        modules.cloud_runtime, "model_needs_truncate", lambda *args: False
    )
    engine, _peer = _engine(modules)
    runtime = modules.cloud_runtime.LwdPdCloudRuntime(engine)
    old = runtime.register(_envelope(modules), None, 2)
    runtime.abort(old.request_id, old.attempt_id, old.binding_id)
    envelope = _envelope(modules, attempt_id="new")
    engine.scheduler.finished_req_ids = {"e7-r"}
    with pytest.raises(ValueError, match="cleanup"):
        runtime.register(envelope, None, 2)
    engine.scheduler.finished_req_ids.clear()
    engine.scheduler.finished_req_ids_dict = {0: {"e7-r"}}
    with pytest.raises(ValueError, match="cleanup"):
        runtime.register(envelope, None, 2)
    engine.scheduler.finished_req_ids_dict.clear()
    engine.batch_queue = [
        (
            None,
            SimpleNamespace(num_scheduled_tokens={}, finished_req_ids={"e7-r"}),
            None,
        )
    ]
    with pytest.raises(ValueError, match="cleanup"):
        runtime.register(envelope, None, 2)
    engine.batch_queue.clear()
    engine.batch_queue = [
        (
            None,
            SimpleNamespace(
                num_scheduled_tokens={},
                finished_req_ids=set(),
                lwd_batch=SimpleNamespace(
                    batch_meta=SimpleNamespace(aborted_req_ids=["e7-r"])
                ),
            ),
            None,
        )
    ]
    with pytest.raises(ValueError, match="cleanup"):
        runtime.register(envelope, None, 2)
    engine.batch_queue.clear()
    new = runtime.register(envelope, None, 2)
    cleanup = {0: SimpleNamespace(outputs=[], finished_requests={"e7-r"})}
    runtime.on_outputs(
        cleanup, SimpleNamespace(lwd_pd_scheduled_ranges={}, lwd_pd_cloud_identities={})
    )
    assert not new.terminal
    runtime.on_outputs(
        cleanup,
        SimpleNamespace(
            lwd_pd_scheduled_ranges={},
            lwd_pd_cloud_identities={"e7-r": (old.attempt_id, old.binding_id)},
        ),
    )
    assert not new.terminal


def test_cancel_before_queued_native_add_does_not_resurrect_stage(modules, monkeypatch):
    import __future__

    import ast

    monkeypatch.setattr(
        modules.cloud_runtime, "model_needs_truncate", lambda *args: False
    )
    engine, peer = _engine(modules)
    runtime = modules.cloud_runtime.LwdPdCloudRuntime(engine)
    envelope = _envelope(modules)
    ctx = runtime.register(envelope, None, 2)
    runtime.claim(
        peer,
        modules.lwd_messages.LwdRequestNotify(
            "e7-r",
            257,
            stage_role="prefill",
            attempt_id=ctx.attempt_id,
            binding_id=ctx.binding_id,
            fingerprint=envelope["fingerprint"],
            prefix_hit_tokens=ctx.hit_tokens,
        ),
    )
    queued_request = engine.input_queue.get_nowait()[1][0]
    runtime.abort(ctx.request_id, ctx.attempt_id, ctx.binding_id)
    source = _SOURCE / "v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py"
    tree = ast.parse(source.read_text())
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LwdCloudEngineCoreMixin"
    )
    add = next(
        node
        for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "add_request"
    )
    native = Mock()
    namespace = {"Native": type("Native", (), {"add_request": native})}
    shell = ast.ClassDef(
        name="Cloud",
        bases=[ast.Name(id="Native", ctx=ast.Load())],
        keywords=[],
        body=[add],
        decorator_list=[],
        type_params=[],
    )
    compiled = ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[]))
    exec(
        compile(
            compiled, str(source), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    cloud = namespace["Cloud"]()
    cloud._lwd_pd_runtime = runtime
    cloud.add_request(queued_request)
    native.assert_not_called()
    assert ctx.terminal and ctx.request is None
