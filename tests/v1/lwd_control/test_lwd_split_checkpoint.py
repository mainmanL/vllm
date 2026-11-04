# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""F2 checkpoint-split script tests: role-rule derivation + vendored table
+ end-to-end split planning.

``split_lwd_checkpoint.py`` derives its per-role keep prefixes from the
same LWD_MODULE_OWNERSHIP table as the build-time guards and the
load-time weight filter (single source of truth), with a vendored copy as
the fallback for hosts without a vllm checkout. This file locks:

- the vendored copy mirrors LWD_MODULE_OWNERSHIP whenever the vllm import
  chain is available, so the two cannot drift silently;
- ROLE_RULES expand the ownership table through the naming variants (HF
  ``model.*`` / VL-wrapped ``model.language_model.*`` / DeepSeek native
  without the ``model.`` prefix) plus the per-role extra keep rules;
- ``_dedup_subsumed`` drops prefixes already covered by a shorter kept one
  (the cloud keeps ``model.language_model.*`` wholesale, so the wrapped
  per-module variants would always report zero matches);
- the tid2eid name pattern matches exactly the DeepSeek Hash-MoE routing
  tables the edge reads straight from its shards;
- an end-to-end run on a synthetic checkpoint (safetensors written with
  pure stdlib): --dry-run plans without writing, a real split keeps the
  role's tensors byte-identical, and the metadata rules copy the
  tokenizer/config files.

The script itself is pure stdlib and always importable; only the
vendored-table comparison needs the vllm import chain (case-level skip
without it).
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "lwd" / "split_lwd_checkpoint.py"

_spec = importlib.util.spec_from_file_location("split_lwd_checkpoint", SCRIPT_PATH)
split_lwd_checkpoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split_lwd_checkpoint)

try:
    from vllm.model_executor.models.utils import LWD_MODULE_OWNERSHIP

    _OWNERSHIP_IMPORTABLE = True
except Exception:  # pragma: no cover - light environments (see conftest)
    _OWNERSHIP_IMPORTABLE = False

requires_ownership = pytest.mark.skipif(
    not _OWNERSHIP_IMPORTABLE,
    reason="models.utils chain unavailable (needs torch)",
)


@requires_ownership
def test_vendored_ownership_mirrors_vllm_table():
    """The vendored fallback must equal LWD_MODULE_OWNERSHIP exactly, or
    offline hosts would silently split against a stale table."""
    assert dict(split_lwd_checkpoint._VENDORED_OWNERSHIP) == dict(LWD_MODULE_OWNERSHIP)


def test_role_rules_derive_from_ownership_table():
    """ROLE_RULES = per-role extras + naming variants of the modules the
    role owns, subsumed prefixes dropped."""
    rules = split_lwd_checkpoint._build_role_rules(
        split_lwd_checkpoint._VENDORED_OWNERSHIP
    )
    # module-level ROLE_RULES follows the same derivation (against whichever
    # table _load_ownership() picked; the vendored test above pins the two
    # tables together)
    assert split_lwd_checkpoint.ROLE_RULES == rules

    edge, cloud = rules["edge"], rules["cloud"]

    # edge: embed / un-embed / visual variants + hc_head extras
    for prefix in (
        "model.embed_tokens.",
        "embed_tokens.",
        "embed.",
        "model.language_model.embed_tokens.",
        "lm_head.",
        "model.head.",
        "head.",
        "model.visual.",
        "visual.",
        "hc_head",
        "model.hc_head",
    ):
        assert prefix in edge, prefix
    # the edge never keeps transformer layers / the final norm / MTP
    for prefix in ("model.layers.", "layers.", "model.norm.", "norm.", "model.mtp.", "mtp."):
        assert prefix not in edge, prefix

    # cloud: whole VL text stack + layers + norm + MTP + embed/lm_head + extras
    for prefix in (
        "model.language_model.",
        "model.layers.",
        "layers.",
        "model.norm.",
        "norm.",
        "model.mtp.",
        "mtp.",
        "model.embed_tokens.",
        "embed.",
        "lm_head.",
        "head.",
        "hc_head",
        "model.hc_head",
    ):
        assert prefix in cloud, prefix
    # deduped away: model.language_model. covers the wrapped per-module variants
    assert "model.language_model.embed_tokens." not in cloud
    # the cloud never keeps the visual tower
    for prefix in ("model.visual.", "visual."):
        assert prefix not in cloud, prefix


def test_dedup_subsumed_drops_covered_prefixes():
    """Covered prefixes (rules that would always report zero matches) are
    dropped; near-identical names are not conflated."""
    dedup = split_lwd_checkpoint._dedup_subsumed
    assert dedup(
        ("model.language_model.", "model.language_model.embed_tokens.", "lm_head.")
    ) == ("lm_head.", "model.language_model.")
    # "embed." covers embed.weight while "embed_tokens." is a different
    # name: both must survive
    assert dedup(("embed.", "embed_tokens.")) == ("embed.", "embed_tokens.")


def test_tid2eid_pattern_matches_only_hash_moe_tables():
    """The edge's one name-level exception: DeepSeek Hash-MoE routing
    tables, in both the native and model.-prefixed layouts."""
    pattern = split_lwd_checkpoint._TID2EID_PATTERN
    assert pattern.match("layers.3.mlp.gate.tid2eid")
    assert pattern.match("model.layers.3.mlp.gate.tid2eid")
    assert pattern.match("layers.7.ffn.gate.tid2eid.weight")
    # anything else must not match
    assert pattern.match("model.layers.3.mlp.gate.weight") is None
    assert pattern.match("visual.layers.0.mlp.gate.tid2eid") is None
    assert pattern.match("model.layers.3.mlp.experts.0.gate.tid2eid") is None


def test_role_name_patterns_wiring():
    """Only the edge needs name-level exceptions; the cloud builds every
    layer, so plain prefix rules cover tid2eid there."""
    assert split_lwd_checkpoint.ROLE_NAME_PATTERNS["edge"] == (
        split_lwd_checkpoint._TID2EID_PATTERN,
    )
    assert split_lwd_checkpoint.ROLE_NAME_PATTERNS["cloud"] == ()


# ---------------------------------------------------------------------------
# End-to-end runs on a synthetic checkpoint (pure stdlib safetensors writer)
# ---------------------------------------------------------------------------

# (name, shape, fill byte) -- distinct fills make byte-level copy errors visible
SYNTHETIC_TENSORS = [
    ("model.embed_tokens.weight", (8, 4), 0x11),
    ("model.layers.0.self_attn.q_proj.weight", (4, 4), 0x22),
    ("model.layers.0.mlp.gate.tid2eid", (4,), 0x33),
    ("model.norm.weight", (4,), 0x44),
    ("lm_head.weight", (4, 8), 0x55),
    ("model.visual.blk.0.attn.qkv.weight", (4, 4), 0x66),
]

# edge keeps embed / tid2eid / lm_head / visual; cloud keeps everything but visual
EXPECTED_KEPT = {
    "edge": {
        "model.embed_tokens.weight",
        "model.layers.0.mlp.gate.tid2eid",
        "lm_head.weight",
        "model.visual.blk.0.attn.qkv.weight",
    },
    "cloud": {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate.tid2eid",
        "model.norm.weight",
        "lm_head.weight",
    },
}


def _write_safetensors(path: Path) -> None:
    """Write a minimal safetensors file with pure stdlib (F32 tensors)."""
    header: dict[str, dict] = {}
    blob = bytearray()
    for name, shape, fill in SYNTHETIC_TENSORS:
        nbytes = 4
        for dim in shape:
            nbytes *= dim
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [len(blob), len(blob) + nbytes],
        }
        blob += bytes([fill]) * nbytes
    raw = json.dumps(header).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + blob)


def _make_synthetic_checkpoint(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    _write_safetensors(src / "model.safetensors")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    (src / "config.json").write_text("{}", encoding="utf-8")
    return src


@pytest.mark.parametrize("role", ("edge", "cloud"))
def test_dry_run_plans_expected_tensors(tmp_path, monkeypatch, capsys, role):
    """--dry-run writes nothing and plans keep/drop per the role rules,
    with tid2eid kept on the edge through the name pattern."""
    src = _make_synthetic_checkpoint(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["split_lwd_checkpoint.py", "--src", str(src), "--role", role, "--dry-run"],
    )
    assert split_lwd_checkpoint.main() == 0
    out = capsys.readouterr().out

    kept = len(EXPECTED_KEPT[role])
    dropped = len(SYNTHETIC_TENSORS) - kept
    assert f"=> keep {kept} tensor(s) of {len(SYNTHETIC_TENSORS)}; drop {dropped}" in out
    # both roles copy the tokenizer/config files
    assert "[meta] will copy 2 non-weight entries" in out
    assert "tokenizer.json" in out
    assert "config.json" in out
    # no --dst given, nothing must appear on disk
    assert list(tmp_path.glob("dst*")) == []


@pytest.mark.parametrize("role", ("edge", "cloud"))
def test_split_writes_role_slice(tmp_path, monkeypatch, capsys, role):
    """A real single-process split: the role slice keeps exactly the
    planned tensors, byte-identical to the source, with metadata copied
    and no stale index for a single shard."""
    src = _make_synthetic_checkpoint(tmp_path)
    dst = tmp_path / f"dst-{role}"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_lwd_checkpoint.py",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--role",
            role,
            "--jobs",
            "1",
            "--no-progress",
        ],
    )
    assert split_lwd_checkpoint.main() == 0
    out = capsys.readouterr().out
    assert "[verify] OK:" in out
    assert "[guard] source read-only verified" in out

    # single shard, no companion safetensors -> one shard file, no index
    shard_files = sorted(path.name for path in dst.glob("*.safetensors"))
    assert shard_files == ["model-00001-of-00001.safetensors"]
    assert not (dst / "model.safetensors.index.json").exists()
    # metadata copied for both roles
    assert (dst / "tokenizer.json").read_text(encoding="utf-8") == "{}"
    assert (dst / "config.json").read_text(encoding="utf-8") == "{}"

    # the slice holds exactly the planned tensors, byte-identical to source
    src_header, _, src_start = split_lwd_checkpoint.read_shard_header(
        str(src / "model.safetensors")
    )
    out_header, _, out_start = split_lwd_checkpoint.read_shard_header(
        str(dst / shard_files[0])
    )
    assert set(out_header) == EXPECTED_KEPT[role]
    with open(src / "model.safetensors", "rb") as handle:
        src_bytes = handle.read()
    with open(dst / shard_files[0], "rb") as handle:
        out_bytes = handle.read()
    for name, tensor in out_header.items():
        src_tensor = src_header[name]
        want = src_bytes[src_start + src_tensor.start : src_start + src_tensor.end]
        got = out_bytes[out_start + tensor.start : out_start + tensor.end]
        assert got == want, name
