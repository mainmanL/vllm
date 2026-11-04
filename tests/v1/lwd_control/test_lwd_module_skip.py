# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""F1 build-time pruning tests: ownership-table semantics + all guard sites.

- ``lwd_skips_module`` is a pure function: dotted-path and prefixed-path
  suffix matching, unlisted modules pass through, and a None role (non-LWD
  deployment) means zero behavior change -- all locked by table-driven
  cases.
- The 5 model guard sites (deepseek_v2 norm, qwen3_5 norm/visual x2,
  qwen3_vl visual) are locked by source-text scan assertions (no model
  needs to be built).
- ``lwd_role()`` must return None in a plain process without a
  vllm_config -- the precondition for every guard taking the original
  path in non-LWD deployments.

The models chain (vllm.model_executor.models.utils) needs torch; in light
environments its import fails and only the source-scan cases run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = REPO_ROOT / "vllm"
MODELS_DIR = VLLM_DIR / "model_executor" / "models"

try:
    from vllm.model_executor.models.utils import (
        LWD_MODULE_OWNERSHIP,
        lwd_role,
        lwd_skips_module,
    )

    _MODELS_UTILS_IMPORTABLE = True
except Exception:  # pragma: no cover - light environments (see conftest)
    _MODELS_UTILS_IMPORTABLE = False

requires_models_utils = pytest.mark.skipif(
    not _MODELS_UTILS_IMPORTABLE,
    reason="models.utils chain unavailable (needs torch)",
)

BOTH = frozenset(("edge", "cloud"))
EDGE_ONLY = frozenset(("edge",))
CLOUD_ONLY = frozenset(("cloud",))


@requires_models_utils
def test_ownership_table_locks_role_split():
    """Lock the ownership table: who builds what, said once (F2 and the
    load-time filter both derive from it)."""
    assert LWD_MODULE_OWNERSHIP == {
        "model.embed_tokens": BOTH,
        "lm_head": BOTH,
        "model.layers": CLOUD_ONLY,
        "model.norm": CLOUD_ONLY,
        "visual": EDGE_ONLY,
        "model.mtp": CLOUD_ONLY,
    }


@pytest.mark.parametrize(
    ("module_path", "role", "want"),
    [
        # dotted paths: split wholesale per the ownership table
        ("model.norm", "edge", True),
        ("model.norm", "cloud", False),
        ("model.layers", "edge", True),
        ("model.layers", "cloud", False),
        ("visual", "cloud", True),
        ("visual", "edge", False),
        ("model.mtp", "edge", True),
        # prefixed paths: under a WeightsMapper-style prefix (e.g. Qwen3-VL's
        # language_model.) the suffix match keeps the semantics
        ("language_model.model.norm", "edge", True),
        ("language_model.model.norm", "cloud", False),
        ("model.visual", "cloud", True),
        # built by both roles: never pruned
        ("model.embed_tokens", "edge", False),
        ("model.embed_tokens", "cloud", False),
        ("lm_head", "edge", False),
        ("lm_head", "cloud", False),
        # unlisted modules: never pruned
        ("model.rotary_emb", "edge", False),
        ("language_model.lm_head", "cloud", False),
        # None role = non-LWD deployment: everything passes, original
        # semantics unchanged
        ("model.norm", None, False),
        ("visual", None, False),
        ("model.layers", None, False),
        # a bare "norm" is not "model.norm": keys are minimal ownership
        # units, no wildcard suffixes
        ("norm", "edge", False),
    ],
)
@requires_models_utils
def test_lwd_skips_module(module_path, role, want):
    assert lwd_skips_module(module_path, role) is want


@requires_models_utils
def test_lwd_role_is_none_outside_lwd_deployment():
    """``lwd_role()`` in a plain process (no vllm_config) must be None."""
    assert lwd_role() is None


@pytest.mark.parametrize(
    ("model_file", "site_text"),
    [
        # deepseek_v2: final norm guard (edge does not build it, cloud does)
        (
            "deepseek_v2.py",
            'if get_pp_group().is_last_rank and not lwd_skips_module(',
        ),
        # qwen3_5: norm guard of the same shape; both visual towers (dense
        # + MoE) must be guarded
        (
            "qwen3_5.py",
            'if get_pp_group().is_last_rank and not lwd_skips_module(',
        ),
        (
            "qwen3_5.py",
            'if lwd_skips_module(maybe_prefix(prefix, "visual"), lwd_role()):',
        ),
        (
            "qwen3_vl.py",
            'if lwd_skips_module(maybe_prefix(prefix, "visual"), lwd_role()):',
        ),
        # qwen3_vl: deepstack buffers follow the whole visual tree (with the
        # tower as a placeholder use_deepstack is off, no buffers built)
        (
            "qwen3_vl.py",
            "and not isinstance(self.visual, (StageMissingLayer, PPMissingLayer))",
        ),
    ],
)
def test_guard_sites_present(model_file, site_text):
    """All 5 guard sites present: migration completeness locked by source
    scan."""
    source = (MODELS_DIR / model_file).read_text()
    assert site_text in source, f"{model_file} missing guard: {site_text}"
    # both qwen3_5 visual guards must be present
    if model_file == "qwen3_5.py" and "visual" in site_text:
        assert source.count(site_text) == 2
