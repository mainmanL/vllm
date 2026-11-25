"""LwdControlDiag 测试:开关语义、旁路载荷提取、spec 观测累计。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.lwd_control.lwd_control_diag import (
    LwdCloudAdmissionMeter,
    LwdControlDiag,
    LwdEdgeDispatchMeter,
)


def test_switches_strict_one_and_live_read(monkeypatch):
    """开关严格 == "1",按调用现读(进程内改 env 即时生效)。"""
    monkeypatch.delenv("VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_LWD_DISABLE_DOWN", raising=False)
    assert LwdControlDiag.token_passthrough_enabled() is False
    assert LwdControlDiag.disable_down_enabled() is False
    # 非 "1" 一律 False
    monkeypatch.setenv("VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE", "true")
    assert LwdControlDiag.token_passthrough_enabled() is False
    # 现读语义 + OR 关系
    monkeypatch.setenv("VLLM_ASCEND_LWD_DISABLE_DOWN", "1")
    assert LwdControlDiag.disable_down_enabled() is True
    assert LwdControlDiag.token_passthrough_enabled() is True
    monkeypatch.setenv("VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE", "1")
    assert LwdControlDiag.token_passthrough_enabled() is True


def _outputs(rows: dict[str, list[int]], finished=()):
    """伪 engine_core_outputs:{0: EngineCoreOutputs(outputs, finished)}。"""
    outputs = [
        SimpleNamespace(request_id=rid, new_token_ids=toks, finish_reason=None)
        for rid, toks in rows.items()
    ]
    return {0: SimpleNamespace(outputs=outputs, finished_requests=list(finished))}


def test_bypass_token_payload_aligned_rows():
    """SKIP_SAMPLE 载荷:按 c2e 行集对齐,缺席行补空。"""
    outputs = _outputs({"a": [1, 2], "b": [3]})
    payload = LwdControlDiag.bypass_token_payload(["a", "zzz"], outputs)
    assert payload == [[1, 2], []]


def test_disable_down_rows_pads_finished():
    """DISABLE_DOWN 行集:本步 token + finished 补空行;空步为空。"""
    outputs = _outputs({"a": [7]}, finished=["b"])
    rows = LwdControlDiag.disable_down_rows(outputs)
    assert rows == {"a": [7], "b": []}
    assert LwdControlDiag.disable_down_rows(_outputs({})) == {}


def test_spec_observation_accumulates_native_formula():
    """spec 观测按原生口径累计:draft += len(spec_ids),
    accepted += len(generated) - 1;未知/已终结请求跳过。"""
    diag = LwdControlDiag()
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"a": [1, 2, 3], "gone": [9], "fin": [9]}
    )
    model_output = SimpleNamespace(
        sampled_token_ids=[[0, 4, 5, 6]],
        req_id_to_index={"a": 0},
    )
    requests = {
        "a": SimpleNamespace(is_finished=lambda: False),
        "fin": SimpleNamespace(is_finished=lambda: True),
    }
    diag.observe_spec_acceptance(scheduler_output, model_output, requests)
    # a: draft 3, accepted = len([0,4,5,6]) - 1 = 3;gone 未知/fin 已终结不计
    assert diag._spec_draft_tokens == 3
    assert diag._spec_accepted == 3
    # 无 spec token 的步:零开销返回
    diag.observe_spec_acceptance(
        SimpleNamespace(scheduled_spec_decode_tokens={}), model_output, requests
    )
    assert diag._spec_draft_tokens == 3


def test_bypass_unembed_rows_gating(monkeypatch):
    """直通判定:UNEMBED 批 + 开关开 + 通告携带 token 才出直通行集;
    只出 (req_ids, 旁路 token) 数据,执行适配器归边引擎。"""
    try:
        from vllm.v1.core.sched.output import LwdBatchType
    except Exception:  # pragma: no cover - py<3.10 轻量环境
        pytest.skip("sched.output 不可导入(需 Python>=3.10)")

    batch = SimpleNamespace(
        batch_type=LwdBatchType.LWD_UNEMBED,
        batch_meta=SimpleNamespace(req_ids=["a"], token_ids=[[1, 2]]),
    )
    monkeypatch.delenv("VLLM_ASCEND_LWD_DISABLE_DOWN", raising=False)
    assert LwdControlDiag.bypass_unembed_rows(batch) is None
    assert LwdControlDiag.bypass_unembed_rows(None) is None
    monkeypatch.setenv("VLLM_ASCEND_LWD_DISABLE_DOWN", "1")
    assert LwdControlDiag.bypass_unembed_rows(batch) == (["a"], [[1, 2]])
    # 无旁路 token 的 UNEMBED 批(正常批)不旁路
    empty = SimpleNamespace(
        batch_type=LwdBatchType.LWD_UNEMBED,
        batch_meta=SimpleNamespace(req_ids=["a"], token_ids=[]),
    )
    assert LwdControlDiag.bypass_unembed_rows(empty) is None


def test_edge_dispatch_meter_reports_gate_reasons_and_snapshot(caplog):
    meter = LwdEdgeDispatchMeter()
    meter.note_gate(
        "edge0-cloud0-dp0",
        pending=10,
        max_pending=10,
        running=8,
        max_running=8,
        resume=False,
        cap=0,
        free_seq=0,
        decode_debt=8192,
    )
    meter.note_pick("edge0-cloud0-dp0", 0)

    with caplog.at_level("INFO", logger="vllm.lwd_meter"):
        meter._report()

    messages = [record.getMessage() for record in caplog.records]
    assert (
        "[lwd][edge-gate] 1 gate checks: seat=1 pending=1 cap0=1 free_seq0=1; "
        "picks=1 empty=1"
    ) in messages
    assert (
        "[lwd][edge-gate-detail] edge0-cloud0-dp0 pending=10/10 running=8/8 "
        "resume=False cap=0 free_seq=0 decode_debt=8192 picked=0"
    ) in messages


def test_cloud_admission_meter_reports_latest_running_and_waiting(caplog):
    meter = LwdCloudAdmissionMeter()
    meter.note_step(
        room=100,
        admitted=64,
        num_windows=1,
        blocked=(0, 0),
        queue_depth=3,
        reserved=0,
        running=7,
        waiting=3,
    )
    meter.note_step(
        room=100,
        admitted=0,
        num_windows=0,
        blocked=(0, 0),
        queue_depth=1,
        reserved=0,
        running=9,
        waiting=1,
    )

    with caplog.at_level("INFO", logger="vllm.lwd_meter"):
        meter._report()

    message = caplog.records[-1].getMessage()
    assert "avg queue=2.0" in message
    assert "running=9 waiting=1" in message
