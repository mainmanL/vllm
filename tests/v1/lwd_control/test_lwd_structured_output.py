# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""F3 structured-output tests: wire roundtrip and LWD seam wiring.

- LwdRequestNotify gains structured_outputs (JSON blob) and
  LWD_MESSAGE_VERSION bumps to 4, so a mixed-version deployment fails
  the link handshake instead of a v3 cloud silently dropping the
  constraint.
- The edge packs SamplingParams.structured_outputs with
  lwd_encode_structured_outputs (plain dataclass dict, private fields
  included); the cloud rebuilds it with _lwd_decode_structured_outputs,
  backfilling _backend (init=False; grammar_init reads it to pick the
  backend) and dropping _backend_was_auto (frontend bookkeeping bit).
- Guard-site scans (no model build needed): the edge no longer rejects
  structured requests, its prefill picker defers candidates in
  WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR until the grammar compiles, the
  cloud compiles explicitly before the native ADD (wire-rebuilt
  requests bypass preprocess_add_request) and refuses to start with
  skip_tokenizer_init.

lwd_messages / lwd_handshake need only msgspec (a vllm/__init__ stub
replicates the directory conftest when torch is missing); the cloud
engine and scheduler chains need torch and skip in light environments.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = REPO_ROOT / "vllm"
EDGE_SCHEDULER_FILE = (
    VLLM_DIR / "v1" / "lwd_control" / "control_edge_scheduler" / "lwd_edge_scheduler.py"
)
EDGE_ENGINE_FILE = (
    VLLM_DIR / "v1" / "lwd_control" / "control_edge_scheduler" / "lwd_edge_engine.py"
)
CLOUD_ENGINE_FILE = (
    VLLM_DIR
    / "v1"
    / "lwd_control"
    / "control_cloud_scheduler"
    / "lwd_cloud_engine.py"
)

# Light environments (no torch) cannot import vllm/__init__ (its env
# machinery needs torch). tests/v1/lwd_control/conftest.py stubs the
# package; replicate that stub so this file also works when the
# directory conftest is skipped (e.g. --noconftest on a light host).
if importlib.util.find_spec("torch") is None and "vllm" not in sys.modules:
    stub = types.ModuleType("vllm")
    stub.__path__ = [str(VLLM_DIR)]
    sys.modules["vllm"] = stub

try:
    import msgspec

    from vllm.v1.lwd_control.lwd_comm.lwd_handshake import LwdCloudHandshake
    from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import (
        LWD_LINK_PROTOCOL_VERSION,
        LwdLinkAccept,
        LwdLinkHello,
        LwdLinkReject,
        lwd_edge_identity,
    )
    from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
        LWD_MESSAGE_VERSION,
        LWD_SAMPLING_FIELDS,
        LwdRequestNotify,
        lwd_encode_structured_outputs,
    )

    _MESSAGES_IMPORTABLE = True
except Exception:  # pragma: no cover - light environments without msgspec
    _MESSAGES_IMPORTABLE = False

requires_messages = pytest.mark.skipif(
    not _MESSAGES_IMPORTABLE, reason="lwd_messages/lwd_handshake unavailable (msgspec)"
)

try:
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
        LwdCloudEngineCoreMixin,
        _lwd_decode_structured_outputs,
    )

    _CLOUD_CHAIN_IMPORTABLE = True
except Exception:  # pragma: no cover - light environments (see module docstring)
    _CLOUD_CHAIN_IMPORTABLE = False

requires_cloud_chain = pytest.mark.skipif(
    not _CLOUD_CHAIN_IMPORTABLE,
    reason="cloud engine / sampling_params chain unavailable (needs torch)",
)

try:
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
        LwdEdgeScheduler,
    )
    from vllm.v1.request import RequestStatus

    _SCHEDULER_CHAIN_IMPORTABLE = True
except Exception:  # pragma: no cover - light environments (see module docstring)
    _SCHEDULER_CHAIN_IMPORTABLE = False

requires_scheduler_chain = pytest.mark.skipif(
    not _SCHEDULER_CHAIN_IMPORTABLE,
    reason="edge scheduler chain unavailable (needs torch)",
)


def _between(text: str, start: str, end: str) -> str:
    """Source slice from `start` up to the next `end` (fail-fast indexes)."""
    i = text.index(start)
    j = text.index(end, i + len(start))
    return text[i:j]


@requires_messages
def test_request_notify_carries_structured_outputs():
    """The field exists, defaults for plain requests, and stays off the
    scalar sampling table (it needs JSON decode, not a plain getattr)."""
    assert LWD_MESSAGE_VERSION == 4
    assert "structured_outputs" in LwdRequestNotify.__struct_fields__
    assert LwdRequestNotify(request_id="r", num_prompt_tokens=1).structured_outputs == ""
    assert "structured_outputs" not in dict(LWD_SAMPLING_FIELDS)
    # old (v3) wire payloads lack the field: decode fills the default, so
    # the bump is a handshake guard, not a decode-compatibility break
    legacy = msgspec.msgpack.encode({"request_id": "r", "num_prompt_tokens": 4})
    assert msgspec.msgpack.decode(legacy, type=LwdRequestNotify).structured_outputs == ""


@requires_messages
def test_handshake_rejects_old_message_version():
    """A v3 edge must be rejected by a v4 cloud, not silently served with
    the constraint dropped."""
    cloud = LwdCloudHandshake(instance_id=1, dp_rank=0, dp_size=1)

    def hello(message_version: int) -> LwdLinkHello:
        return LwdLinkHello(
            link_version=LWD_LINK_PROTOCOL_VERSION,
            message_version=message_version,
            instance_id=1,
            dp_rank=0,
            dp_size=1,
            session_id="s-1",
        )

    identity = lwd_edge_identity(1, 0, "s-1")
    decision = cloud.on_hello(identity, hello(LWD_MESSAGE_VERSION - 1))
    assert isinstance(decision.reply, LwdLinkReject)
    assert "message version" in decision.reply.reason

    decision = cloud.on_hello(identity, hello(LWD_MESSAGE_VERSION))
    assert isinstance(decision.reply, LwdLinkAccept)


@requires_messages
def test_encode_serializes_dataclass_with_private_fields():
    """The wire form must carry init=False fields: asdict includes them,
    which is what lets the cloud backfill _backend (it has no frontend to
    re-derive it). None means a plain request and travels as ""."""
    assert lwd_encode_structured_outputs(None) == ""

    @dataclass
    class _StandInParams:
        regex: str | None = None
        _backend: str | None = field(default=None, init=False)
        _backend_was_auto: bool = field(default=False, init=False)

    stand_in = _StandInParams(regex="[0-9]+")
    stand_in._backend = "xgrammar"
    stand_in._backend_was_auto = True
    wire = json.loads(lwd_encode_structured_outputs(stand_in))
    assert wire == {
        "regex": "[0-9]+",
        "_backend": "xgrammar",
        "_backend_was_auto": True,
    }


@requires_cloud_chain
def test_roundtrip_preserves_backend_and_drops_was_auto():
    params = StructuredOutputsParams(json={"type": "object"})
    params._backend = "xgrammar"
    params._backend_was_auto = True

    decoded = _lwd_decode_structured_outputs(
        lwd_encode_structured_outputs(params)
    )
    assert decoded is not None
    assert decoded.json == {"type": "object"}
    assert decoded._backend == "xgrammar"
    assert decoded._backend_was_auto is False
    assert _lwd_decode_structured_outputs("") is None


@requires_cloud_chain
def test_wire_rebuild_sampling_params_roundtrip():
    params = StructuredOutputsParams(regex="[0-9]+")
    params._backend = "xgrammar"
    notify = LwdRequestNotify(
        request_id="req-1",
        num_prompt_tokens=4,
        structured_outputs=lwd_encode_structured_outputs(params),
    )
    sp = LwdCloudEngineCoreMixin._lwd_build_sampling_params(notify)
    assert sp.structured_outputs is not None
    assert sp.structured_outputs.regex == "[0-9]+"
    assert sp.structured_outputs._backend == "xgrammar"


@pytest.mark.parametrize(
    "payload",
    [
        "{not-json",  # not JSON at all
        # two constraints at once: the rebuild re-validates mutual exclusion
        # even though the edge frontend should never pack this
        '{"json": {"type": "object"}, "regex": "[0-9]"}',
    ],
)
@requires_cloud_chain
def test_wire_rebuild_rejects_malformed_payload(payload):
    notify = LwdRequestNotify(
        request_id="req-9", num_prompt_tokens=4, structured_outputs=payload
    )
    with pytest.raises(ValueError, match="req-9"):
        LwdCloudEngineCoreMixin._lwd_build_sampling_params(notify)


@requires_scheduler_chain
def test_grammar_ready_gate():
    """Plain and promoted requests pass; only grammar-pending structured
    requests are deferred (read-only, the native loop keeps the
    promotion)."""
    gate = LwdEdgeScheduler._lwd_grammar_ready
    assert (
        gate(
            SimpleNamespace(
                status=RequestStatus.WAITING, structured_output_request=None
            )
        )
        is True
    )
    assert (
        gate(
            SimpleNamespace(
                status=RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
                structured_output_request=SimpleNamespace(grammar=None),
            )
        )
        is False
    )
    assert (
        gate(
            SimpleNamespace(
                status=RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
                structured_output_request=SimpleNamespace(grammar=object()),
            )
        )
        is True
    )


def test_edge_admission_no_longer_rejects_structured_output():
    text = EDGE_SCHEDULER_FILE.read_text(encoding="utf-8")
    assert "does not support structured output" not in text


def test_picker_defers_grammar_compilation():
    text = EDGE_SCHEDULER_FILE.read_text(encoding="utf-8")
    assert "[*self.skipped_waiting, *self.waiting]" in text
    assert "and self._lwd_grammar_ready(r)" in text
    gate = _between(text, "def _lwd_grammar_ready", "def _lwd_pick_prefill_batch")
    assert "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR" in gate
    assert "structured_req.grammar" in gate


def test_edge_packs_structured_outputs_on_wire():
    text = EDGE_ENGINE_FILE.read_text(encoding="utf-8")
    assert "structured_outputs=lwd_encode_structured_outputs(" in text


def test_cloud_compiles_grammar_before_native_add():
    text = CLOUD_ENGINE_FILE.read_text(encoding="utf-8")
    handler = _between(
        text, "def _lwd_handle_request_notify", "def _lwd_handle_abort_notify"
    )
    assert "if request.use_structured_output:" in handler
    assert (
        handler.index("self.structured_output_manager.grammar_init(request)")
        < handler.index("put_nowait((EngineCoreRequestType.ADD")
    )


def test_cloud_rebuild_wires_structured_outputs():
    text = CLOUD_ENGINE_FILE.read_text(encoding="utf-8")
    builder = _between(
        text, "def _lwd_build_sampling_params", "def step_with_batch_queue"
    )
    assert "_lwd_decode_structured_outputs(" in builder
    assert "structured_outputs=structured_outputs," in builder


def test_cloud_decode_handles_private_fields():
    text = CLOUD_ENGINE_FILE.read_text(encoding="utf-8")
    decoder = _between(
        text, "def _lwd_decode_structured_outputs", "class LwdCloudEngineCoreMixin"
    )
    assert 'data.pop("_backend", None)' in decoder
    assert 'data.pop("_backend_was_auto", None)' in decoder
    assert "params._backend = backend" in decoder


def test_cloud_startup_requires_tokenizer():
    text = CLOUD_ENGINE_FILE.read_text(encoding="utf-8")
    assert "if vllm_config.model_config.skip_tokenizer_init:" in text
