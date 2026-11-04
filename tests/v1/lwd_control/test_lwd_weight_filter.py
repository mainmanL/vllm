# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load-time weight filter tests: skip-key derivation + tensor-name prefix
matching + pipeline wiring guards.

lwd_weight_filter shares LWD_MODULE_OWNERSHIP with the build-time guards
(single source of truth): the edge drops transformer layers / the final
norm / MTP, the cloud drops the visual tower. This file locks, table
driven:

- key derivation: parse_skip_keys(edge/cloud) mirrors the ownership table;
- tensor-name matching: both the HF layout (``model.layers.3.``) and the
  DeepSeek native layout (no leading ``model``, ``layers.3.``/``norm.``)
  hit; matching is anchored at the start of the name, so names inside the
  visual tower (``visual.merger.norm.``) are not mis-pruned by the
  ``norm`` rule; tid2eid (Hash MoE routing tables, read by path on the
  edge) is exempted and kept;
- a None role (non-LWD deployment) never filters: zero behavior change.

In light environments (no torch, see conftest) the model_loader chain is
not importable and the table-driven cases skip per the family convention;
the source-scan guards still run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = REPO_ROOT / "vllm"

try:
    from vllm.model_executor.model_loader.lwd_weight_filter import (
        lwd_should_skip_weight,
        parse_skip_keys,
    )

    _FILTER_IMPORTABLE = True
except Exception:  # pragma: no cover - light environments (see conftest)
    _FILTER_IMPORTABLE = False

requires_filter = pytest.mark.skipif(
    not _FILTER_IMPORTABLE,
    reason="lwd_weight_filter chain unavailable (needs torch and the model_loader chain)",
)


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("edge", frozenset(("model.layers", "model.norm", "model.mtp"))),
        ("cloud", frozenset(("visual",))),
    ],
)
@requires_filter
def test_parse_skip_keys_mirrors_ownership_table(role, expected):
    """Key derivation = the table modules the role does not own; the two
    roles' skip keys are disjoint (no module can be built by neither)."""
    skip_keys = parse_skip_keys(role)
    assert skip_keys == expected
    other = parse_skip_keys("cloud" if role == "edge" else "edge")
    assert skip_keys.isdisjoint(other)


@pytest.mark.parametrize(
    ("role", "name", "expected"),
    [
        # --- edge: transformer layers / final norm / MTP dropped before read ---
        ("edge", "model.layers.0.mlp.gate_proj.weight", True),
        ("edge", "layers.29.mlp.w1.weight", True),  # DeepSeek native layout
        ("edge", "model.norm.weight", True),
        ("edge", "norm.weight", True),  # native layout
        ("edge", "model.mtp.fc.weight", True),
        ("edge", "mtp.eh_proj.weight", True),  # native layout
        # --- edge: embed / lm_head / visual tower kept ---
        ("edge", "model.embed_tokens.weight", False),
        ("edge", "embed.weight", False),  # native layout
        ("edge", "lm_head.weight", False),
        ("edge", "head.weight", False),  # native layout
        ("edge", "visual.blk.0.attn.qkv.weight", False),
        ("edge", "model.visual.merger.l1.qkv.weight", False),
        # --- tid2eid exemption: Hash MoE routing tables must stay on the edge ---
        ("edge", "model.layers.3.mlp.gate.tid2eid", False),
        ("edge", "layers.7.ffn.gate.tid2eid.weight", False),
        # --- anchored matching: names not starting with a key are not pruned ---
        ("edge", "norms.weight", False),
        ("edge", "layers_extra.0.attn.weight", False),
        ("edge", "visual.merger.norm.weight", False),  # in-tower norm not pruned
        # --- cloud: the whole visual tree dropped ---
        ("cloud", "visual.blk.0.attn.qkv.weight", True),
        ("cloud", "visual.merger.deepstack.l1.norm.weight", True),
        # --- cloud: layers / norm / embed / lm_head / MTP kept ---
        ("cloud", "model.layers.0.self_attn.q_proj.weight", False),
        ("cloud", "model.norm.weight", False),
        ("cloud", "model.embed_tokens.weight", False),
        ("cloud", "lm_head.weight", False),
        ("cloud", "model.mtp.fc.weight", False),
        # --- None role: non-LWD deployments filter nothing ---
        (None, "model.layers.0.mlp.gate_proj.weight", False),
        (None, "visual.blk.0.attn.qkv.weight", False),
        (None, "model.norm.weight", False),
    ],
)
@requires_filter
def test_lwd_should_skip_weight_table(role, name, expected):
    """Table driven: both naming layouts hit, anchored boundary
    counterexamples, the tid2eid exemption, and None role pass-through."""
    assert lwd_should_skip_weight(name, role) is expected


def test_weight_filter_keeps_models_utils_import_lazy():
    """Import-cycle guard: lwd_weight_filter must not import models.utils at
    module level (models.utils imports model_loader back; a top-level
    import closes the cycle)."""
    src = (
        VLLM_DIR
        / "model_executor"
        / "model_loader"
        / "lwd_weight_filter.py"
    ).read_text()
    assert not re.search(r"^from vllm\.|^import vllm\.", src, re.M), (
        "module-level vllm import in lwd_weight_filter reintroduces the "
        "models.utils <-> model_loader import cycle"
    )
    assert "from vllm.model_executor.models.utils import LWD_MODULE_OWNERSHIP" in src


def test_skip_name_pipeline_wired():
    """Pipeline wiring guard: the iterator signature takes skip_name, the
    single _skip check runs before reading from disk, and default_loader
    derives the predicate from the role and passes it through."""
    weight_utils = (
        VLLM_DIR / "model_executor" / "model_loader" / "weight_utils.py"
    ).read_text()
    assert "skip_name: Callable[[str], bool] | None = None" in weight_utils
    assert "def _skip(name: str) -> bool:" in weight_utils
    assert weight_utils.count("_skip(name)") == 3  # eager / torchao / default paths
    # the EP filter still merges through the same decision point, never
    # bypassing skip_name
    assert "or should_skip_weight(" in weight_utils

    default_loader = (
        VLLM_DIR / "model_executor" / "model_loader" / "default_loader.py"
    ).read_text()
    assert 'skip_name: Callable[[str], bool] | None = None' in default_loader
    assert "skip_name=skip_name" in default_loader
    assert "parse_skip_keys(role)" in default_loader
    assert "partial(lwd_should_skip_weight, role=role)" in default_loader
    # lwd_role is imported lazily (inside get_all_weights), not at module level
    assert "from vllm.model_executor.models.utils import lwd_role" in default_loader
