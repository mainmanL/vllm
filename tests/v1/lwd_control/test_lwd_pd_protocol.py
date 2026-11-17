# SPDX-License-Identifier: Apache-2.0
"""CPU protocol tests using real serializers, hashing and lifecycle code."""

from __future__ import annotations

import importlib
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def pd(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "vllm"
    for name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.lwd_control",
        "vllm.v1.lwd_control.pd",
        "vllm.v1.lwd_control.lwd_comm",
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(source.joinpath(*name.split(".")[1:]))]
        monkeypatch.setitem(sys.modules, name, module)
    result = {}
    for name in (
        "lwd_comm.lwd_messages",
        "lwd_comm.lwd_prefix",
        "pd.envelope",
        "pd.results",
        "pd.output",
        "pd.outbox",
        "pd.edge_state",
    ):
        fullname = f"vllm.v1.lwd_control.{name}"
        monkeypatch.delitem(sys.modules, fullname, raising=False)
        result[name.rsplit(".", 1)[-1]] = importlib.import_module(fullname)
    return SimpleNamespace(**result)


def envelope(pd, **kwargs):
    values = dict(
        request_id="request",
        attempt_id="attempt",
        edge_id=7,
        tenant_id="tenant",
        original_prompt_tokens=9,
        user_max_output_tokens=20,
        sampling_params=None,
        model_id="model",
        input_abi="abi",
        digest_block_size=4,
        digest_chain=[b"a" * 32, b"b" * 32],
        digest_tail_hash=b"c" * 32,
        ttl_s=60,
    )
    values.update(kwargs)
    return pd.envelope.build_envelope(**values)


def test_envelope_binds_attempt_constraints_and_expiry(pd):
    original = envelope(pd)
    assert pd.envelope.parse_envelope(json.dumps(original)) == original
    for field, value in (
        ("attempt_id", "other"),
        ("ttl_s", 600),
        ("issued_at", time.time() + 10),
        ("structured_outputs", '{"json": {}}'),
    ):
        changed = dict(original, **{field: value})
        with pytest.raises(ValueError, match="fingerprint"):
            pd.envelope.parse_envelope(json.dumps(changed))
    with pytest.raises(ValueError, match="expired"):
        pd.envelope.parse_envelope(json.dumps(original), now=original["issued_at"] + 61)


@pytest.mark.parametrize(
    "n,stateful,plan",
    [(9, False, (9, 9, 8)), (9, True, (8, 8, 8)), (1, False, (1, 1, 0))],
)
def test_handoff_boundary(pd, n, stateful, plan):
    assert pd.envelope.compute_handoff_plan(n, stateful) == plan
    pd.envelope.validate_handoff_plan(n, *plan)


def test_stateful_one_token_and_arbitrary_plan_rejected(pd):
    with pytest.raises(ValueError):
        pd.envelope.compute_handoff_plan(1, True)
    with pytest.raises(ValueError):
        pd.envelope.validate_handoff_plan(9, 7, 7, 7)


@pytest.mark.parametrize("terminal_first", [False, True])
def test_result_order_independent_and_usage_uses_frontend_prefix(pd, terminal_first):
    ledger = pd.results.LwdPdResultLedger(100, 7, "a", "d")
    ledger.append([10, 11, 12, 13])
    if terminal_first:
        assert ledger.finish("length") is None
        result = ledger.commit(2, "stop_string", 4)
    else:
        assert ledger.commit(2, "stop_string", 4) is None
        result = ledger.finish("length")
    assert result["generated_tokens"] == 4
    assert result["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 2,
        "total_tokens": 102,
        "prompt_tokens_details": {"cached_tokens": 7},
    }
    assert ledger.commit(2, "stop_string", 4) == result
    with pytest.raises(ValueError, match="conflicting"):
        ledger.commit(3, "stop_string", 4)
    result["usage"]["completion_tokens"] = 99
    assert ledger.result["usage"]["completion_tokens"] == 2


def test_unknown_g_not_zero_and_cancel_before_forward_is_explicit(pd):
    ledger = pd.results.LwdPdResultLedger(9, 0, "a", "d")
    assert ledger.commit(0, "cancelled", 0) is None
    assert ledger.final_g is None
    result = ledger.finish("abort")
    assert result["generated_tokens"] == 0
    assert result["status"] == "cancelled"


@pytest.mark.parametrize(
    "k,reason,verified",
    [(2, "complete", 2), (4, "cancelled", 4), (3, "error", 2), (True, "error", 3)],
)
def test_invalid_counts_never_publish_a_result(pd, k, reason, verified):
    ledger = pd.results.LwdPdResultLedger(9, 0, "a", "d")
    ledger.append([1, 2, 3])
    ledger.finish("stop")
    with pytest.raises(ValueError):
        ledger.commit(k, reason, verified)
    assert ledger.result is None
    assert ledger.commit(3, "complete", 3)["status"] == "completed"


def test_output_digest_gap_overlap_replay_and_physical_rows(pd):
    stream = pd.output.LwdPdOutputStream("a", "d")
    digest = pd.envelope.range_digest("d", 0, 2, [11, 12])
    assert stream.accept((0, 2), [11, 12], digest, physical_rows=4)
    assert not stream.accept((0, 2), [11, 12], digest, physical_rows=4)
    for interval, tokens, signature in [
        ((0, 2), [11, 13], digest),
        ((3, 4), [14], pd.envelope.range_digest("d", 3, 4, [14])),
        ((1, 3), [12, 13], pd.envelope.range_digest("d", 1, 3, [12, 13])),
    ]:
        with pytest.raises(ValueError):
            stream.accept(interval, tokens, signature, 4)
    assert stream.verified_tokens == 2
    terminal = pd.envelope.range_digest("d", 2, 2, [])
    assert stream.accept((2, 2), [], terminal, 0, final_g=2)
    assert stream.final_g == 2


def test_zero_live_padding_can_close_same_endpoint(pd):
    stream = pd.output.LwdPdOutputStream("a", "d")
    digest = pd.envelope.range_digest("d", 0, 0, [])
    assert stream.accept((0, 0), [], digest, 1)
    assert stream.accept((0, 0), [], digest, 0, 0)
    assert not stream.accept((0, 0), [], digest, 0, 0)


def test_result_retries_identity_receipt_and_expiry(pd):
    wire = pd.lwd_messages
    box = pd.outbox.LwdPdCommitOutbox(retry_interval=1, capacity=1)
    commit = wire.LwdPdResultCommit("r", "a", "d", 2, "stop_string", 4)
    box.put((7, 9, 3), commit, 10, 5)
    assert box.due(10) == [((7, 9, 3), commit)]
    assert not box.due(10.5)
    assert box.due(11)
    assert not box.acknowledge(wire.LwdPdResultReceipt("r", "old", "d", {}))
    ledger = pd.results.LwdPdResultLedger(9, 1, "a", "d")
    ledger.append([1, 2, 3, 4])
    ledger.finish("length")
    result = ledger.commit(2, "stop_string", 4)
    receipt = wire.LwdPdResultReceipt("r", "a", "d", result)
    assert box.acknowledge(receipt)
    assert box.acknowledge(receipt)
    assert not box.pending
    assert not box.due(12)
    assert not box.expire(15)
    box.put((7, 9, 3), commit, 16, 1)
    assert box.expire(17) == [commit]


def binding(pd, role, **kwargs):
    values = dict(
        request_id="r",
        attempt_id="a",
        stage_role=role,
        binding_id=role,
        fingerprint="f",
        cloud_id=1,
        dp_idx=1,
        hit_tokens=3,
        p_end=9,
        i_end=9,
        r_pos=8,
        max_inflight_windows=2,
        recv_windows=2,
        source_p_binding="prefill" if role == "decode" else "",
        source_export_id="export" if role == "decode" else "",
    )
    values.update(kwargs)
    return pd.lwd_messages.LwdPdBindNotify(**values)


def test_asymmetric_bindings_and_cumulative_credits(pd):
    state = pd.edge_state.LwdPdEdgeState("r", "a", "f", 9, 100, 10, object(), {})
    p = binding(pd, "prefill")
    assert state.bind((7, 1, 1), p)
    assert not state.bind((7, 1, 1), p)
    d = binding(pd, "decode", cloud_id=2, dp_idx=3)
    assert state.bind((7, 2, 3), d)
    with pytest.raises(ValueError, match="conflicting"):
        state.bind((7, 2, 0), d)
    credit = state.bindings["prefill"]
    credit.sent_windows = 2
    assert credit.available_windows == 0
    credit.release(1)
    credit.release(1)
    assert credit.available_windows == 1
    with pytest.raises(ValueError):
        credit.release(3)


@pytest.mark.parametrize(
    "changes",
    [{"hit_tokens": 4}, {"source_p_binding": "wrong"}, {"source_export_id": ""}],
)
def test_decode_cannot_change_frozen_handoff(pd, changes):
    state = pd.edge_state.LwdPdEdgeState("r", "a", "f", 9, 100, 10, object(), {})
    state.bind((7, 1, 1), binding(pd, "prefill"))
    with pytest.raises(ValueError):
        state.bind((7, 2, 3), binding(pd, "decode", **changes))


def test_finalization_freezes_verified_count_before_late_tokens(pd):
    state = pd.edge_state.LwdPdEdgeState("r", "a", "f", 9, 100, 10, object(), {})
    state.bind((7, 1, 1), binding(pd, "prefill"))
    state.bind((7, 2, 3), binding(pd, "decode"))
    state.freeze(0, "cancelled", True)
    state.output.verified_tokens = 2
    state.freeze(0, "cancelled", True)
    assert state.frontend_verified_tokens == 0
    with pytest.raises(ValueError, match="conflicting"):
        state.freeze(1, "cancelled", True)


def test_required_pd_wire_fields_and_roundtrip(pd):
    import msgspec

    wire = pd.lwd_messages
    message = binding(pd, "decode")
    assert wire.lwd_decode_cloud_to_edge(wire.lwd_encode_message(message)) == message
    raw = msgspec.to_builtins(message)
    del raw["attempt_id"]
    with pytest.raises(msgspec.ValidationError):
        wire.lwd_decode_cloud_to_edge(msgspec.msgpack.encode(raw))


def test_result_receipt_rejects_changed_usage_and_digest(pd):
    ledger = pd.results.LwdPdResultLedger(9, 1, "a", "d")
    ledger.append([1])
    ledger.finish("stop")
    result = ledger.commit(1, "complete", 1)
    pd.results.validate_result(result)
    result["usage"]["total_tokens"] = 9
    with pytest.raises(ValueError, match="digest or usage"):
        pd.results.validate_result(result)


def test_old_padding_endpoint_cannot_terminate_a_later_prefix(pd):
    stream = pd.output.LwdPdOutputStream("a", "d")
    empty = pd.envelope.range_digest("d", 0, 0, [])
    stream.accept((0, 0), [], empty, 1)
    stream.accept((0, 1), [10], pd.envelope.range_digest("d", 0, 1, [10]), 1)
    with pytest.raises(ValueError):
        stream.accept((0, 0), [], empty, 0, 0)
    assert stream.final_g is None


def test_control_backpressure_preserves_peer_fifo_without_blocking_other_peers(pd):
    from vllm.v1.lwd_control.pd.control_outbox import LwdPdControlOutbox

    sent = []
    blocked = {"a"}

    def send(peer, message):
        if peer in blocked:
            return False
        sent.append((peer, message))
        return True

    outbox = LwdPdControlOutbox(send)
    outbox.put("a", "claim", 0)
    outbox.put("a", "range", 0)
    outbox.put("b", "result", 0)
    assert sent == [("b", "result")]
    outbox.flush(0.5)
    assert outbox.count == 2
    blocked.clear()
    outbox.flush(1)
    assert sent == [("b", "result"), ("a", "claim"), ("a", "range")]
    assert outbox.count == 0


def test_control_backpressure_is_bounded_and_fails_explicitly(pd):
    from vllm.v1.lwd_control.pd.control_outbox import LwdPdControlOutbox

    outbox = LwdPdControlOutbox(lambda peer, message: False, capacity=1, timeout=3)
    outbox.put("a", "range", 0)
    with pytest.raises(RuntimeError, match="capacity"):
        outbox.put("a", "next", 0)
    with pytest.raises(RuntimeError, match="deadline"):
        outbox.flush(3)
