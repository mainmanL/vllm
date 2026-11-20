# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Result ownership, real frontend K, and local handoff ordering without NPU."""

from __future__ import annotations

import ast
import asyncio
import enum
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

VLLM = Path(__file__).resolve().parents[3] / "vllm"
_path = VLLM / "v1/lwd_control/pd/lwd_frontend_result.py"
_spec = importlib.util.spec_from_file_location("frontend_result_under_test", _path)
assert _spec and _spec.loader
result_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = result_module
_spec.loader.exec_module(result_module)
Inbox = result_module.LwdFrontendResultInbox
Result = result_module.LwdFrontendResult
Handoff = result_module.LwdFrontendResultHandoff


class Finish(enum.IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4


def load_methods(relative: str, cls_name: str, names: set[str], scope: dict):
    path = VLLM / relative
    tree = ast.parse(path.read_text())
    original = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    selected = [
        n
        for n in original.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    cls = ast.ClassDef(
        name=cls_name, bases=[], keywords=[], body=selected, decorator_list=[]
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), scope)
    return scope[cls_name]


class Detokenizer:
    def __init__(self, tokens=(), stop=False):
        self.tokens = list(tokens)
        self.stop = stop

    def num_output_tokens(self):
        return len(self.tokens)

    def update(self, tokens, terminated):
        self.tokens.extend(tokens)
        return "stop" if self.stop else None


def processor(tokens=(1, 2), *, stop=False):
    cls = load_methods(
        "v1/engine/output_processor.py",
        "OutputProcessor",
        {
            "_freeze_pd_result",
            "abort_requests",
            "process_outputs",
            "_finish_request",
            "propagate_error",
        },
        {
            "LwdFrontendResult": Result,
            "FinishReason": Finish,
            "OutputProcessorOutput": SimpleNamespace,
        },
    )
    obj = cls()
    obj.lwd_pd_results = Inbox(60.0)
    state = SimpleNamespace(
        request_id="r",
        external_req_id="external",
        lwd_pd_attempt_id="attempt",
        detokenizer=Detokenizer(tokens, stop),
        logprobs_processor=Mock(),
        routed_experts_chunks=[],
        sampling_mask_chunks=[],
        is_prefilling=False,
        queue=None,
        lora_name=None,
        parent_req=None,
        streaming_input=False,
        make_request_output=Mock(return_value=None),
    )
    obj.request_states = {"r": state}
    obj.external_req_ids = defaultdict(list, {"external": ["r"]})
    obj.parent_requests = {}
    obj.lora_states = Mock()
    obj._update_admission_stats = Mock()
    obj._update_stats_from_output = Mock()
    obj._update_stats_from_finished = Mock()
    obj.tracing_enabled = False
    return obj, state


def output(*, attempt="attempt", finish=Finish.LENGTH, tokens=(3,)):
    return SimpleNamespace(
        request_id="r",
        lwd_pd_attempt_id=attempt,
        new_token_ids=list(tokens),
        pooling_output=None,
        finish_reason=finish,
        finished=finish is not None,
        stop_reason=None,
        kv_transfer_params=None,
        ec_transfer_params=None,
        routed_experts=None,
        spec_decode_metrics=None,
        new_sampling_mask=None,
    )


@pytest.mark.parametrize(
    "finish,stop,reason",
    [
        (Finish.LENGTH, False, "complete"),
        (Finish.STOP, True, "stop_string"),
        (Finish.ERROR, False, "error"),
        (Finish.ABORT, False, "cancelled"),
    ],
)
def test_finished_request_freezes_real_k_before_cleanup(finish, stop, reason):
    obj, _ = processor(stop=stop)
    obj.process_outputs([output(finish=finish)])
    assert not obj.request_states
    assert obj.lwd_pd_results.records["r", "attempt"] == Result(
        "r", "attempt", 3, reason, False
    )


def test_abort_freezes_existing_token_count_before_removing_external_mapping():
    obj, _ = processor(tokens=(10, 20, 30, 40))
    assert obj.abort_requests(["external"], False, "disconnected") == ["r"]
    assert not obj.request_states and not obj.external_req_ids
    assert obj.lwd_pd_results.records["r", "attempt"] == Result(
        "r", "attempt", 4, "disconnected", True
    )


def test_stale_attempt_cannot_supply_tokens_to_a_reused_request_id():
    obj, state = processor()
    obj.process_outputs([output(attempt="previous", finish=None)])
    assert state.detokenizer.num_output_tokens() == 2
    assert not obj.lwd_pd_results.records
    with pytest.raises(RuntimeError, match="missing.*attempt"):
        obj.process_outputs([output(attempt=None)])


def test_unknown_frontend_count_is_not_replaced_with_zero():
    obj, state = processor()
    state.detokenizer = None
    with pytest.raises(RuntimeError, match="authoritative token count"):
        obj.abort_requests(["external"], False)
    assert obj.request_states["r"] is state
    assert obj.external_req_ids["external"] == ["r"]
    assert not obj.lwd_pd_results.records


def test_local_handoff_failure_retains_immutable_k_for_idempotent_retry():
    async def run():
        obj, _ = processor(tokens=(1, 2, 3))
        obj.abort_requests(["external"], False)
        record = obj.lwd_pd_results.records["r", "attempt"]
        calls = []

        async def utility(*args):
            calls.append(args)
            if len(calls) == 1:
                raise ConnectionError("local utility temporarily unavailable")
            return True

        handoff = Handoff(obj.lwd_pd_results, utility, Mock())
        await handoff.flush()
        assert record.key in obj.lwd_pd_results.pending
        assert record.key in obj.lwd_pd_results.handoff_failed
        assert not obj.request_states
        await handoff.flush()
        assert not obj.lwd_pd_results.pending
        assert (
            calls[0]
            == calls[1]
            == ("lwd_pd_finalize", "r", "attempt", 3, "cancelled", True)
        )
        obj.lwd_pd_results.freeze(record)
        with pytest.raises(ValueError, match="conflicting"):
            obj.lwd_pd_results.freeze(Result("r", "attempt", 0, "cancelled", True))

    asyncio.run(run())


def test_async_abort_hands_result_to_engine_before_aborting_execution():
    async def run():
        obj, _ = processor(tokens=(1, 2))
        events = []

        async def utility(*args):
            events.append(("finalize", args[3]))
            assert not obj.request_states
            return True

        async def abort(ids):
            events.append(("abort", ids))

        cls = load_methods(
            "v1/engine/async_llm.py",
            "AsyncLLM",
            {"abort"},
            {"as_list": list, "logger": Mock()},
        )
        llm = cls()
        llm.output_processor = obj
        llm.engine_core = SimpleNamespace(abort_requests_async=abort)
        llm._lwd_pd_handoff = Handoff(obj.lwd_pd_results, utility, Mock())
        llm.log_requests = False
        await llm.abort("external", frontend_reason="timeout")
        assert events == [("finalize", 2), ("abort", ["r"])]
        assert obj.lwd_pd_results.records["r", "attempt"].frontend_reason == "timeout"

    asyncio.run(run())


def test_expired_local_handoff_is_explicitly_unconfirmed_and_keeps_k():
    inbox = Inbox(1.0, max_records=1)
    record = Result("r", "a", 7, "error", True)
    inbox.freeze(record)
    assert inbox.expire(inbox.pending[record.key] + 1) == [record]
    assert record.key in inbox.unconfirmed
    assert inbox.records[record.key].accepted_tokens == 7
    with pytest.raises(RuntimeError, match="inbox is full"):
        inbox.freeze(Result("other", "b", 1, "complete", False))
