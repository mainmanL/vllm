# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Run the real edge runtime and tail scheduler with fake engine boundaries."""

from __future__ import annotations

import ast
import importlib
import logging
import queue
import sys
import time
import types
from collections import deque
from concurrent.futures import Future
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm"


@pytest.fixture
def modules(monkeypatch):
    for package in (
        "vllm",
        "vllm.config",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.lwd_control",
        "vllm.v1.lwd_control.pd",
        "vllm.v1.lwd_control.lwd_comm",
        "vllm.v1.lwd_control.control_edge_scheduler",
    ):
        module = types.ModuleType(package)
        module.__path__ = [str(SOURCE.joinpath(*package.split(".")[1:]))]
        monkeypatch.setitem(sys.modules, package, module)

    def stub(name, **fields):
        module = types.ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)

    class Finish(IntEnum):
        STOP = 0
        LENGTH = 1
        ABORT = 2
        ERROR = 3

    class ProbeRuntime:
        def __init__(self, **kwargs):
            self.bodies = []

        def submit(self, coroutine):
            self.bodies.append(coroutine.cr_frame.f_locals["body"])
            coroutine.close()
            return Future()

        def stop(self):
            pass

    stub("vllm.lwd_logger", lwd_logger=logging.getLogger("pd-edge-test"))
    stub(
        "vllm.v1.engine",
        EngineCoreOutput=lambda rid, ids, **kw: SimpleNamespace(
            request_id=rid, new_token_ids=ids, **kw
        ),
        EngineCoreOutputs=SimpleNamespace,
        EngineCoreRequestType=SimpleNamespace(WAKEUP="wake"),
    )
    stub("vllm.v1.request", FinishReason=Finish)
    stub("vllm.config.ec_manager_config", EncoderCacheManagerMetadata=object)
    stub(
        "vllm.multimodal.utils",
        strip_covered_mm_data=lambda features, *a, **kw: features,
    )
    stub(
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client",
        _iter_sse_data=object(),
    )
    stub(
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_probe_runtime",
        LwdProbeRuntime=ProbeRuntime,
    )
    imported = {}
    names = (
        "vllm.config.lwd",
        "vllm.v1.core.sched.output",
        "vllm.v1.lwd_control.lwd_comm.lwd_messages",
        "vllm.v1.lwd_control.lwd_comm.lwd_prefix",
        "vllm.v1.lwd_control.pd.envelope",
        "vllm.v1.lwd_control.pd.output",
        "vllm.v1.lwd_control.pd.results",
        "vllm.v1.lwd_control.pd.edge_state",
        "vllm.v1.lwd_control.pd.outbox",
        "vllm.v1.lwd_control.pd.edge_runtime",
        "vllm.v1.lwd_control.pd.edge_scheduler",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_picker",
    )
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
        imported[name] = importlib.import_module(name)
    bundle = SimpleNamespace(
        runtimes=[],
        config=imported[names[0]],
        output=imported[names[1]],
        messages=imported[names[2]],
        prefix=imported[names[3]],
        envelope=imported[names[4]],
        runtime=imported[names[9]],
        scheduler=imported[names[10]],
        account=imported[names[11]],
        picker=imported[names[12]],
    )
    yield bundle
    for runtime in bundle.runtimes:
        runtime.close()


def scheduler_method(name: str, **scope):
    path = SOURCE / "v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py"
    original = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef)
    )
    method = next(
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


def engine_runtime(modules):
    connection = modules.config.LwdConnectionId
    connections = [
        connection(0, cloud, dp)
        for cloud, size in [(0, 2), (1, 4)]
        for dp in range(size)
    ]
    accounts = {c: modules.account.LwdCloudDpAccount(c) for c in connections}
    scheduler = SimpleNamespace(
        max_num_running_reqs=16,
        max_num_scheduled_tokens=128,
        _lwd_accounts=accounts,
        _lwd_route={},
        _lwd_connections=connections,
        _lwd_last_engine=-1,
        _lwd_mrope_positions_dict={"r": b"positions"},
        _lwd_validate_request=Mock(),
        _lwd_compute_mrope_positions=Mock(),
        _lwd_edge_id_of=lambda request: 0,
        requests={},
        waiting=[],
        skipped_waiting=[],
        running=[],
        lwd_mrope_positions=lambda rid: b"positions",
        lwd_send_to_cloud=Mock(),
    )
    route = scheduler_method(
        "_lwd_route_request", lwd_logger=logging.getLogger("pd-edge-test")
    )
    scheduler.lwd_admit_request = lambda request, preferred: route(
        scheduler, request.request_id, 0, preferred
    )
    scheduler._lwd_account_of = lambda rid: accounts.get(scheduler._lwd_route.get(rid))
    coordination = SimpleNamespace(
        control_url="http://proxy/v1/chat/completions",
        consumer_id="tenant",
        connect_timeout=1,
    )
    config = SimpleNamespace(
        p_bind_timeout_s=30,
        d_bind_timeout_s=30,
        request_ttl_s=300,
        result_hold_timeout_s=60,
        d_output_quota_rows=0,
        tenant_id="",
    )
    hasher = modules.prefix.LwdPrefixHasher(bytes(32), 128, bytes(32))
    client = SimpleNamespace(
        build_manifest=hasher.build_manifest,
        build_control_request=lambda *args, **kw: ({}, {"stream": True}),
    )
    engine = SimpleNamespace(
        scheduler=scheduler,
        lwd_config=SimpleNamespace(pd=config, coordination=coordination, instance_id=0),
        _lwd_clouds={c: object() for c in connections},
        _lwd_cloud_params={
            c: SimpleNamespace(
                pd_version=modules.messages.LWD_PD_VERSION,
                stage_role="prefill" if c.cloud_id == 0 else "decode",
                hash_block_size=128,
            )
            for c in connections
        },
        _lwd_probe_manager=SimpleNamespace(
            _client=lambda *args: client,
            _media_items=lambda r: (),
            _fingerprint=lambda: bytes(32),
        ),
        _lwd_probe_chains={},
        _lwd_coord_clients={},
        input_queue=queue.Queue(),
        output_queue=queue.Queue(),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(served_model_name=["model"], model="model")
        ),
        _lwd_notify_request=Mock(),
        _lwd_send_to_cloud=Mock(),
        _lwd_send=Mock(),
    )
    runtime = modules.runtime.LwdPdEdgeRuntime(engine)
    modules.runtimes.append(runtime)
    runtime.validate_clouds()

    def enqueue(request):
        scheduler.requests[request.request_id] = request
        scheduler.waiting.append(request)

    engine._lwd_enqueue_request = Mock(side_effect=enqueue)
    engine.abort_requests = Mock(side_effect=runtime.abort)
    request = SimpleNamespace(
        request_id="r",
        num_prompt_tokens=257,
        prompt_token_ids=list(range(257)),
        sampling_params=SimpleNamespace(
            extra_args={"lwd_pd_attempt_id": "attempt"}, max_tokens=8, eos_token_id=None
        ),
        mm_features=[],
        abort_immediately=False,
        client_index=0,
        num_computed_tokens=0,
    )
    runtime.add(request)
    return engine, runtime, request


def bind(modules, runtime, role, dp, *, hit=128, state_model=True):
    state = runtime.states["r"]
    p_end, i_end, r_pos = modules.envelope.compute_handoff_plan(257, state_model)
    notify = modules.messages.LwdPdBindNotify(
        "r",
        "attempt",
        role,
        f"binding-{role}",
        state.fingerprint,
        cloud_id=0 if role == "prefill" else 1,
        dp_idx=dp,
        hit_tokens=hit,
        p_end=p_end,
        i_end=i_end,
        r_pos=r_pos,
        max_window_tokens=64,
        max_inflight_windows=2,
        recv_windows=2,
        source_p_binding="binding-prefill" if role == "decode" else "",
        source_export_id="export" if role == "decode" else "",
    )
    connection = modules.config.LwdConnectionId(0, notify.cloud_id, dp)
    runtime._bind(state, connection, notify)
    return connection, notify


def test_add_builds_real_parseable_envelope_and_waits_for_proxy_bind(modules):
    engine, runtime, request = engine_runtime(modules)
    encoded = runtime.http.bodies[0]["messages"][0]["content"]
    envelope = modules.envelope.parse_envelope(encoded)
    assert envelope["tenant_id"] == "tenant"
    assert envelope["attempt_id"] == "attempt"
    assert envelope["original_prompt_tokens"] == 257
    assert len(envelope["digest"]["chain"]) == 2 and envelope["digest"]["tail"]
    assert envelope["input_abi"] == bytes(32).hex()
    assert not engine.scheduler._lwd_route
    engine._lwd_enqueue_request.assert_not_called()
    assert request.num_computed_tokens == 0


@pytest.mark.parametrize("state_model", [False, True])
def test_bind_p1_d3_preserves_separate_routes_caps_and_tail(modules, state_model):
    engine, runtime, request = engine_runtime(modules)
    p, p_notify = bind(modules, runtime, "prefill", 1, state_model=state_model)
    assert engine.scheduler._lwd_route["r"] == p
    assert request.num_computed_tokens == 128
    assert runtime.upload_limit(request) == 64
    runtime._bind(runtime.states["r"], p, p_notify)
    assert engine.scheduler._lwd_accounts[p].prefill_load == 1
    request.num_computed_tokens = p_notify.p_end
    d, _ = bind(modules, runtime, "decode", 3, state_model=state_model)
    assert engine.scheduler._lwd_route["r"] == d
    assert runtime.states["r"].bindings["prefill"].connection == p
    assert engine.scheduler._lwd_accounts[p].load == 0
    before = request.num_computed_tokens
    output = modules.scheduler.lwd_pd_schedule_tail(runtime)
    batch = output.lwd_batch.batch_meta
    assert batch.connection_id == d
    assert batch.prompt_offsets == [256] and batch.token_ids == [[256]]
    assert request.num_computed_tokens == before
    assert output.scheduled_new_reqs[0].lwd_mrope_positions == b"positions"
    assert output.scheduled_new_reqs[0].lwd_pd_binding_id == "binding-decode"
    sent_connection, range_notify = engine.scheduler.lwd_send_to_cloud.call_args.args
    assert sent_connection == d and range_notify.items[0].offset == 256
    assert range_notify.items[0].attempt_id == "attempt"
    assert modules.scheduler.lwd_pd_schedule_tail(runtime) is None


def frame(modules, *, seq=0, start=0, tokens=(11, 12), final=-1, attempt="attempt"):
    end = start + len(tokens)
    return modules.messages.LwdC2eNotify(
        hidden_num_elements=24,
        top_id_ths=[[0] * len(tokens)],
        num_accepted_tokens=[len(tokens)],
        req_ids=["r"],
        finish_reasons=[1 if final >= 0 else -1],
        down_seqno=seq,
        attempt_ids=[attempt],
        binding_ids=["binding-decode"],
        valid_ranges=[[start, end]],
        row_counts=[3],
        final_g=[final],
        range_digests=[
            modules.envelope.range_digest("binding-decode", start, end, tokens)
        ],
    )


def decoded_runtime(modules):
    engine, runtime, request = engine_runtime(modules)
    bind(modules, runtime, "prefill", 1)
    request.num_computed_tokens = 256
    d, _ = bind(modules, runtime, "decode", 3)
    return engine, runtime, d


def test_down_frame_received_once_and_only_verified_rows_reach_k(modules):
    engine, runtime, d = decoded_runtime(modules)
    notify = frame(modules, final=2)
    assert runtime.authorize_frame(d, notify)
    assert not runtime.authorize_frame(d, notify)
    assert runtime.verify_row(notify, 0, [11, 12])
    assert not runtime.verify_row(notify, 0, [11, 12])
    assert runtime.states["r"].output.verified_tokens == 2
    altered = msgspec.structs.replace(notify, range_digests=["f" * 64])
    with pytest.raises(ValueError, match="conflicting.*replay"):
        runtime.authorize_frame(d, altered)
    assert runtime.finalize("r", "attempt", 1, "stop_string", False)
    assert runtime.finalize("r", "attempt", 1, "stop_string", False)
    commits = runtime.outbox.due(time.monotonic())
    assert len(commits) == 1 and commits[0][0] == d
    assert commits[0][1].accepted_tokens == 1
    assert commits[0][1].verified_tokens == 2
    with pytest.raises(ValueError, match="conflicting"):
        runtime.finalize("r", "attempt", 0, "stop_string", False)


def test_late_attempt_and_corrupt_rows_cannot_reactivate_or_advance_output(modules):
    engine, runtime, d = decoded_runtime(modules)
    stale = frame(modules, attempt="previous")
    assert runtime.authorize_frame(d, stale)  # physical stale frames still drain
    assert not runtime.verify_row(stale, 0, [11, 12])
    assert runtime.states["r"].output.verified_tokens == 0
    current = frame(modules, seq=1)
    assert runtime.authorize_frame(d, current)
    assert not runtime.verify_row(current, 0, [99, 12])
    assert runtime.states["r"].output.verified_tokens == 0
    assert runtime.states["r"].cancelled
    engine.abort_requests.assert_called_once_with(["r"])
    assert runtime.states["r"].request is None


def test_wrong_bind_dp_is_rejected_and_no_local_load_balance_fallback(modules):
    engine, runtime, _ = engine_runtime(modules)
    p, notify = bind(modules, runtime, "prefill", 1)
    with pytest.raises(ValueError, match="registered target"):
        runtime._bind(
            runtime.states["r"], modules.config.LwdConnectionId(0, 0, 0), notify
        )
    assert engine.scheduler._lwd_route["r"] == p
    old = msgspec.structs.replace(notify, attempt_id="previous", binding_id="old")
    assert runtime.handle(p, old)
    assert runtime.states["r"].bindings["prefill"].notify == notify


def test_finalize_rpc_keeps_result_after_abort_and_late_bind_does_not_execute(modules):
    engine, runtime, request = engine_runtime(modules)
    p, _ = bind(modules, runtime, "prefill", 1)
    assert runtime.finalize("r", "attempt", 0, "cancelled", True)
    assert runtime.states["r"].request is None
    # A D bind already in flight is aborted, never enqueued or tail-scheduled.
    bind(modules, runtime, "decode", 3)
    assert runtime.states["r"].frontend_commit == (0, "cancelled", True)
    assert modules.scheduler.lwd_pd_schedule_tail(runtime) is None
    assert engine._lwd_enqueue_request.call_count == 1
    assert runtime.outbox.pending
    with pytest.raises(ValueError, match="retained execution attempt"):
        runtime.finalize("r", "previous", 0, "cancelled", True)


def test_tail_and_existing_down_frames_alternate_without_starvation(
    modules, monkeypatch
):
    engine, runtime, d = decoded_runtime(modules)
    scheduler = engine.scheduler
    scheduler.lwd_pd_runtime = runtime
    scheduler._lwd_pd_uploaded_last = False
    scheduler.unembed_notify_queue = deque([object(), object()])
    scheduler._lwd_schedule_unembed = lambda: (
        "down",
        scheduler.unembed_notify_queue.popleft(),
    )
    method = scheduler_method("_lwd_schedule_step")
    first = method(scheduler)
    assert first.lwd_batch.batch_meta.connection_id == d
    # More ready tails arrive while earlier DOWN work is still queued.
    runtime.states["r"].tail_pending = True
    assert method(scheduler)[0] == "down"
    second = method(scheduler)
    assert second.lwd_batch.batch_meta.connection_id == d
    assert method(scheduler)[0] == "down"


def test_decode_default_credit_covers_its_own_spec_width_with_smaller_edge_budget(
    modules,
):
    engine, runtime, request = engine_runtime(modules)
    engine.scheduler.max_num_scheduled_tokens = 1
    target = modules.config.LwdConnectionId(0, 1, 3)
    engine._lwd_cloud_params[target].num_speculative_tokens = 3
    _p, notify = bind(modules, runtime, "prefill", 1)
    request.num_computed_tokens = notify.p_end
    bind(modules, runtime, "decode", 3)
    assert runtime.states["r"].bindings["decode"].output_watermark == 4
    quota = engine._lwd_send_to_cloud.call_args.args[1]
    assert quota.kind == "output" and quota.watermark == 4


def test_upload_batch_obeys_selected_cloud_budget_across_requests(modules):
    import copy

    engine, runtime, request = engine_runtime(modules)
    connection, first = bind(modules, runtime, "prefill", 1)
    engine._lwd_cloud_params[connection].max_num_batched_tokens = 32
    second = copy.deepcopy(request)
    second.request_id = "second"
    second.sampling_params.extra_args = {"lwd_pd_attempt_id": "second-attempt"}
    runtime.add(second)
    second_state = runtime.states["second"]
    runtime._bind(
        second_state,
        connection,
        msgspec.structs.replace(
            first,
            request_id="second",
            attempt_id=second_state.attempt_id,
            binding_id="second-binding",
            fingerprint=second_state.fingerprint,
        ),
    )
    scheduler = engine.scheduler
    scheduler.lwd_pd_runtime = runtime
    scheduler._lwd_chunk_unit = 128
    scheduler._lwd_reqs_in_phase = lambda phase, requests: requests
    scheduler._lwd_dp_of = lambda rid: scheduler._lwd_route[rid]
    for item in scheduler.requests.values():
        item.status = 0
    pick = scheduler_method(
        "_lwd_pick_prefill_batch",
        LwdPrefillCandidate=modules.picker.LwdPrefillCandidate,
        lwd_pick_prefill_batch=modules.picker.lwd_pick_prefill_batch,
        lwd_grammar_ready=modules.picker.lwd_grammar_ready,
        LwdReqPhase=SimpleNamespace(PREFILL="prefill"),
        RequestStatus=SimpleNamespace(WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR=-1),
    )
    picked, amounts, cap = pick(scheduler, connection)
    assert cap == 32 and sum(amounts.values()) == 32
    assert picked == ["r"]
