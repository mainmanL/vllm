#!/usr/bin/env python3
"""Split a full HF safetensors checkpoint into LWD role (edge/cloud) subsets.

Background: each side of the edge-cloud LWD deployment runs only part of the
model (the ownership table ``LWD_MODULE_OWNERSHIP`` in
vllm/model_executor/models/utils.py drives both the build-time guards that
skip modules a role never runs and the load-time ``lwd_weight_filter`` that
drops placeholder weights), so each side's model directory only needs the
tensors it will actually use. HF shards interleave tensors arbitrarily, so
they cannot be handed to one side wholesale -- the split is per tensor name.

Implementation notes (tuned for very large checkpoints):

* **Zero-copy byte-level writes**: tensor bytes are moved as-is, with no
  dtype/device conversion and no materialized torch tensors. Only the
  safetensors headers are parsed (pure stdlib), a new header is recomputed,
  and data ranges are copied by byte interval (``os.copy_file_range`` when
  on the same filesystem, large read/write buffers otherwise).
* **Multiprocessing**: ``--jobs N`` parallelizes by output shard; the parent
  plans shards and aggregates progress, workers share no state. ``--jobs 1``
  never touches multiprocessing.
* **Index autodetection**: the source index may use a non-standard name such
  as ``quant_model_weights.safetensors.index.json``; without an index, shard
  headers are read one by one to build the map (no safetensors package
  needed).
* **Companion safetensors**: files like ``rot.safetensors`` /
  ``optional/quarot.safetensors`` are not weight shards but are read
  directly by path at runtime (QuaRot/EAGLE3 rotation matrices); they are
  copied to the roles that need them, and whenever such a file exists in
  the destination an index is written **even for a single shard** --
  otherwise vLLM would load them as weight files.
* **Source read-only guard (on by default)**: all files under ``--src`` are
  stat-snapshotted before and after the run; any change aborts with an
  error. The script never writes ``--src`` and rejects ``--dst`` equal to
  or containing ``--src``.

Usage (dry-run first, then the real split)::

    # inspect what the rules keep (writes nothing)
    python split_lwd_checkpoint.py --src /weight/GLM-5.2-w4a8c8 --role edge --dry-run

    # real split (defaults to a small parallelism based on CPU count)
    python split_lwd_checkpoint.py --src /weight/GLM-5.2-w4a8c8 \
        --dst /weight/GLM-5.2-w4a8c8-cloud --role cloud --max-shard-gb 8 --jobs 8

Rules (extend with --keep-prefix / --keep-name-regex). Role membership is
derived from LWD_MODULE_OWNERSHIP; the naming-variant expansion covers three
layouts: VL-wrapped (model.language_model.*), model.* (Qwen3.5 / GLM-5.x),
and DeepSeek native (layers./norm./embed./head./hc_head/mtp. without the
model. prefix).

  edge  : head.* / lm_head.* / model.head.*            (unembed: rank-replay)
          + embed.* / embed_tokens.* / model.embed*    (embed: PP first rank)
          + hc_head* (V4 top-level unconditional params, edge too)
          + model.visual.* / visual.*                   (multimodal tower)
          + name-matched <*>.layers.<i>.mlp|ffn.gate.tid2eid
          (DeepSeek-V4 Hash MoE routing tables: the edge builds no layers
           but reads these tables straight from the checkpoint shards;
           missing them fails startup with "Missing tid2eid tables")
          (the edge only embeds / unembeds; transformer layers and the
           final norm are never built)
  cloud : layers.* / model.layers.* / model.language_model.* (all layers)
          + norm.* + embed* + head.*/lm_head.* + hc_head*
          + mtp.* / model.mtp.* (with speculative decoding; MTP expressed
            as extra layer indices is covered by layers./model.layers. too)
          (the cloud never builds the visual tower)

Non-weight entries are copied per role (--metadata auto, the default; rules
support * wildcards). Directories are skipped unless explicitly named by a
rule (e.g. "optional") or --metadata all is given. The edge serves requests
directly, so tokenizer/processor/chat_template files are all copied; the
cloud compiles structured-output grammars (tokenizer/vocab needed) but keeps
mm_encoder_remote forced, so only config.json, configuration.json, the
quantization description (*quant*description*.json), chat_template.jinja
and tokenizer entries are copied. QuaRot rotation matrices
(rot.safetensors / optional) are copied to both sides.

Output: tensors are packed into model-XXXXX-of-YYYYY.safetensors shards of
at most --max-shard-gb; an index (model.safetensors.index.json) is written
whenever there is more than one shard or a companion safetensors file is
present.

Progress (percentage / GiB written / rate / ETA + finished shards) is shown
while writing: one in-place refreshed line on a terminal, one line every 10
seconds when redirected to a log file; disable with --no-progress. Progress
goes to stderr only so stdout stays grep-able.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import multiprocessing as mp
import os
import re
import shutil
import struct
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

# vLLM only recognizes this index name (SAFE_WEIGHTS_INDEX_NAME in
# vllm/model_executor/model_loader/default_loader.py); local directories are
# filtered through it, dropping any *.safetensors not listed in the index.
INDEX_NAME = "model.safetensors.index.json"
# The source index may use an arbitrary prefix (e.g.
# quant_model_weights.safetensors.index.json).
INDEX_SUFFIX = ".safetensors.index.json"

# Checkpoint naming variants per LWD_MODULE_OWNERSHIP entry. Three layouts:
#   1) VL-wrapped       model.language_model.* / model.visual.*  (Qwen3-VL)
#   2) model.*          model.layers. / model.norm. / ...        (Qwen3.5, GLM-5.x)
#   3) DeepSeek native  layers. / norm. / embed. / head. / mtp.  (no model. prefix;
#      see _make_deepseek_v4_weights_mapper: layers.->model.layers.,
#      embed.->model.embed_tokens., norm.->model.norm.,
#      head.weight->lm_head.weight, mtp.->model.mtp.)
_NAME_VARIANTS: dict[str, tuple[str, ...]] = {
    "model.embed_tokens": (
        "model.language_model.embed_tokens.",  # VL-wrapped layout
        "model.embed_tokens.",
        "embed_tokens.",
        "model.embed.",  # DeepSeek native: embed.weight -> embed_tokens.weight
        "embed.",
    ),
    "lm_head": (
        "lm_head.",
        "model.head.",
        "head.",  # DeepSeek native: head.weight -> lm_head.weight
    ),
    "model.layers": (
        "model.layers.",
        "layers.",  # DeepSeek native layout
    ),
    "model.norm": (
        "model.norm.",
        "norm.",  # DeepSeek native layout
    ),
    "visual": (
        "model.visual.",  # GLM-style VL layout
        "visual.",  # Qwen3-VL top-level tower
    ),
    "model.mtp": (
        "model.mtp.",
        "mtp.",  # DeepSeek native layout
    ),
}

# Keep prefixes that are not module ownership but still required per role:
# DeepSeek-V4's top-level hc_head parameters are created unconditionally in
# the model code (both roles must receive them or startup reports missing
# weights); the cloud keeps the whole VL-wrapped text stack, which subsumes
# the per-module language_model variants. Towers named vision_tower /
# vision_model / mm_projector etc. are appended via --keep-prefix.
_EXTRA_KEEP_RULES: dict[str, tuple[str, ...]] = {
    "edge": ("hc_head", "model.hc_head"),
    "cloud": ("model.language_model.", "hc_head", "model.hc_head"),
}

# Vendored copy of LWD_MODULE_OWNERSHIP for offline hosts without a vllm
# checkout. Keep in sync with vllm/model_executor/models/utils.py; the unit
# test (tests/v1/lwd_control/test_lwd_split_checkpoint.py) asserts the two
# are equal so they cannot drift silently.
_VENDORED_OWNERSHIP: dict[str, frozenset[str]] = {
    "model.embed_tokens": frozenset(("edge", "cloud")),
    "lm_head": frozenset(("edge", "cloud")),
    "model.layers": frozenset(("cloud",)),
    "model.norm": frozenset(("cloud",)),
    "visual": frozenset(("edge",)),
    "model.mtp": frozenset(("cloud",)),
}


def _load_ownership() -> dict[str, frozenset[str]]:
    """LWD_MODULE_OWNERSHIP from vllm, or the vendored copy without vllm."""
    try:
        from vllm.model_executor.models.utils import LWD_MODULE_OWNERSHIP

        return dict(LWD_MODULE_OWNERSHIP)
    except Exception:  # noqa: BLE001 - offline hosts have no vllm checkout
        return dict(_VENDORED_OWNERSHIP)


def _dedup_subsumed(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """Drop prefixes already covered by a shorter kept prefix (e.g. the
    cloud keeps model.language_model.* wholesale, so the wrapped per-module
    variants would always report zero matches)."""
    kept: list[str] = []
    for prefix in sorted(set(prefixes)):
        if not any(prefix.startswith(shorter) for shorter in kept):
            kept.append(prefix)
    return tuple(kept)


def _build_role_rules(ownership: dict[str, frozenset[str]]) -> dict[str, tuple[str, ...]]:
    """Keep prefixes per role, derived from the ownership table."""
    rules: dict[str, tuple[str, ...]] = {}
    for role in sorted({r for owners in ownership.values() for r in owners}):
        prefixes = list(_EXTRA_KEEP_RULES.get(role, ()))
        for module_path, owners in ownership.items():
            if role in owners:
                prefixes.extend(_NAME_VARIANTS[module_path])
        rules[role] = _dedup_subsumed(tuple(prefixes))
    return rules


# Checkpoint name prefixes each role must keep. Same single source of truth
# as the build-time guards and the load-time weight filter.
ROLE_RULES: dict[str, tuple[str, ...]] = _build_role_rules(_load_ownership())

# DeepSeek-V4 Hash MoE routing tables. The edge builds no layers but reads
# these tables straight from the checkpoint shards (vllm-ascend's
# worker/lwd_edge_worker.py:load_model -> worker/lwd_hash/lwd_hash_tables.py),
# so they must stay with the edge slice, otherwise edge startup fails with
# "Missing tid2eid tables for Hash MoE layers [...]".
# The cloud needs no exception: it builds every layer, so these tables are
# ordinary parameters covered by layers./model.layers.
_TID2EID_PATTERN = re.compile(
    r"^(?:model\.)?layers\.\d+\.(?:ffn|mlp)\.gate\.tid2eid(?:\.weight)?$"
)

# Keep-by-full-tensor-name patterns that prefix rules cannot express.
ROLE_NAME_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "edge": (_TID2EID_PATTERN,),
    "cloud": (),
}

# Known "companion safetensors": not model weight shards, but read directly
# by path at runtime.
#   rot.safetensors / *quarot*.safetensors -- QuaRot rotation matrices,
#   loaded by vllm-ascend's patch_draft_quarot.py, which takes the relative
#   path from optional.quarot.rotation_map.global_rotation of
#   quant_model_description.json and load_file()s it (only needed with the
#   EAGLE3 drafter). With an index, any *.safetensors absent from the index
#   is auto-detected as a companion; without an index these patterns are
#   the fallback.
COMPANION_SAFETENSORS = ("rot.safetensors", "*quarot*.safetensors", "*rotation*.safetensors")

SKIP_FILE_EXACT = {"README.md", "LICENSE", ".gitattributes", "gitattributes"}
SKIP_FILE_PREFIX = (".git", "__pycache__", INDEX_NAME)

# Non-weight entries each side copies ("auto" policy); * wildcards supported
# (fnmatch). Files and companion safetensors follow the rules; **directories
# are skipped by default** and only copied recursively when a rule names
# them explicitly (e.g. "optional"), or with --metadata all.
METADATA_RULES: dict[str, tuple[str, ...]] = {
    # The edge serves requests: copy everything; directories are not copied
    # by default, so QuaRot's optional/ must be named explicitly.
    "edge": ("*", "optional"),
    "cloud": (
        "config.json",
        "configuration.json",
        "*quant*description*.json",
        # Only for compatibility with startup scripts that pass
        # "--chat-template <model_dir>/chat_template.jinja"; the cloud never
        # renders templates (requests are all served edge-side), so this can
        # be dropped when the launch command does not pass --chat-template.
        "chat_template.jinja",
        # tokenizer/vocab: cloud-side structured-output grammar compilation
        # loads a tokenizer (StructuredOutputManager -> XgrammarBackend).
        # mm_encoder_remote stays forced, no multimodal processor is built,
        # so processor files are not copied.
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "vocab.*",
        "tokenizer.model",
        # QuaRot rotation matrices (for the EAGLE3 drafter; the drafter is
        # cloud-side).
        "rot.safetensors",
        "optional",
    ),
}

COPY_CHUNK = 8 * 2**20  # fallback copy buffer
PROGRESS_STEP = 32 * 2**20  # worker reports progress every this many bytes


class TensorInfo:
    """One tensor's metadata from the safetensors header (offsets relative
    to the data region start)."""

    __slots__ = ("dtype", "shape", "start", "end")

    def __init__(self, dtype: str, shape, start: int, end: int) -> None:
        self.dtype = dtype
        self.shape = list(shape)
        self.start = start
        self.end = end

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def read_shard_header(path: str) -> tuple[dict[str, TensorInfo], dict | None, int]:
    """Parse the safetensors header (pure stdlib, never reads tensor data).

    Layout: ``[8-byte little-endian u64: header length N][N bytes JSON
    header][data region]``.

    Returns:
        (tensor name -> TensorInfo, ``__metadata__`` or None, absolute
        offset of the data region start)
    """
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a valid safetensors file (too short)")
        (header_len,) = struct.unpack("<Q", raw)
        header = json.loads(handle.read(header_len))
    tensors: dict[str, TensorInfo] = {}
    metadata = header.get("__metadata__")
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        tensors[name] = TensorInfo(info["dtype"], info["shape"], start, end)
    return tensors, metadata, 8 + header_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", required=True, help="full model directory (with *.safetensors)")
    parser.add_argument("--dst", help="output directory (different from --src; may be omitted with --dry-run)")
    parser.add_argument("--role", required=True, choices=sorted(ROLE_RULES))
    parser.add_argument(
        "--max-shard-gb",
        type=float,
        default=5.0,
        help="output shard size limit (default 5GB; larger means fewer shards and a smaller index)",
    )
    parser.add_argument("--keep-prefix", action="append", default=[], help="extra keep prefixes (repeatable)")
    parser.add_argument(
        "--keep-name-regex",
        action="append",
        default=[],
        help="extra full-tensor-name regexes (repeatable; for cases prefix rules cannot express, e.g. keeping only some in-layer params)",
    )
    parser.add_argument(
        "--metadata",
        choices=("auto", "all"),
        default="auto",
        help="non-weight entries: auto=per-role rules; all=copy everything (incl. directories; for clouds still running unpatched vllm)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel writer processes; 0=auto (min(4, CPU count)), 1=single process (easier to debug)",
    )
    parser.add_argument(
        "--no-verify-src-readonly",
        action="store_true",
        help="disable the before/after read-only snapshot check of --src (on by default)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable write progress (default: one refreshed line on a terminal; one line every 10s when redirected)",
    )
    parser.add_argument("--dry-run", action="store_true", help="only print the keep/drop listing, write nothing")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Source directory inventory
# ---------------------------------------------------------------------------
def find_index_file(src: str) -> str | None:
    """Find the index file in the source directory (its name may be
    quant_model_weights.safetensors.index.json etc.)."""
    if os.path.isfile(os.path.join(src, INDEX_NAME)):
        return INDEX_NAME
    candidates = sorted(
        name
        for name in os.listdir(src)
        if name.endswith(INDEX_SUFFIX) and os.path.isfile(os.path.join(src, name))
    )
    return candidates[0] if candidates else None


def list_safetensors(src: str) -> list[str]:
    return sorted(name for name in os.listdir(src) if name.endswith(".safetensors"))


def load_weight_map(src: str) -> tuple[dict[str, str], str | None]:
    """Return (tensor name -> shard file name, index file name or None).

    Uses the index (any ``*.safetensors.index.json``) when present;
    otherwise reads shard headers one by one, without needing the
    safetensors package.
    """
    index_name = find_index_file(src)
    if index_name is not None:
        with open(os.path.join(src, index_name), encoding="utf-8") as handle:
            return json.load(handle)["weight_map"], index_name
    weight_map: dict[str, str] = {}
    for name in list_safetensors(src):
        tensors, _, _ = read_shard_header(os.path.join(src, name))
        for key in tensors:
            weight_map[key] = name
    return weight_map, None


def classify_safetensors(
    src: str, weight_map: dict[str, str], index_name: str | None
) -> tuple[set[str], list[str]]:
    """Split ``*.safetensors`` into "weight shards" and "companion files".

    Returns:
        (set of weight shard file names, list of companion safetensors names)
    """
    files = list_safetensors(src)
    if index_name is not None:
        weights = set(weight_map.values())
        return weights, [name for name in files if name not in weights]
    # No index: fall back to known patterns so rot.safetensors is never
    # mistaken for a weight shard.
    companions = [
        name
        for name in files
        if any(fnmatch.fnmatch(name, pattern) for pattern in COMPANION_SAFETENSORS)
    ]
    return {name for name in files if name not in companions}, companions


def select_keys(
    weight_map: dict[str, str],
    prefixes: tuple[str, ...],
    patterns: tuple[tuple[str, "re.Pattern[str]"], ...] = (),
) -> tuple[dict[str, list[str]], int]:
    """Group the tensors to keep; returns {rule label: [tensor names]} and
    the total number of matches.

    Prefix rules match first; unmatched names go through full-name regexes
    (for parameters like the DeepSeek-V4 Hash MoE routing tables that live
    under layers.<i> but must stay with the edge).
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    matched = 0
    for key in weight_map:
        for prefix in prefixes:
            if key.startswith(prefix):
                grouped[prefix].append(key)
                matched += 1
                break
        else:
            for label, pattern in patterns:
                if pattern.match(key):
                    grouped[label].append(key)
                    matched += 1
                    break
    return grouped, matched


def collect_tensor_info(
    src: str, weight_files: set[str], keep: set[str], weight_map: dict[str, str]
) -> tuple[dict[str, tuple[str, TensorInfo]], dict | None, dict[str, int]]:
    """Read the headers of the source shards we need.

    Returns:
        ({tensor name: (shard, TensorInfo)}, first shard's __metadata__,
        {shard: absolute offset of the data region start})

    Note: TensorInfo offsets are relative to **that shard's data region**;
    copying must add the data-region start. The per-shard data-region start
    is returned here so plan_shards() can convert to absolute offsets.
    """
    wanted_per_shard: dict[str, list[str]] = defaultdict(list)
    # Sorted for reproducibility: keep is a set whose iteration order
    # follows PYTHONHASHSEED; without sorting the same input could pack
    # shards differently across runs (boundary/order drift).
    for key in sorted(keep):
        wanted_per_shard[weight_map[key]].append(key)

    info: dict[str, tuple[str, TensorInfo]] = {}
    metadata: dict | None = None
    data_start: dict[str, int] = {}
    for shard in sorted(wanted_per_shard):
        if shard not in weight_files:
            raise ValueError(f"shard holding a wanted tensor is not in the index: {shard}")
        tensors, shard_meta, shard_data_start = read_shard_header(os.path.join(src, shard))
        data_start[shard] = shard_data_start
        if metadata is None:
            metadata = shard_meta
        missing = [key for key in wanted_per_shard[shard] if key not in tensors]
        if missing:
            raise ValueError(f"{len(missing)} wanted tensors not found in {shard}, e.g. {missing[:3]}")
        for key in wanted_per_shard[shard]:
            info[key] = (shard, tensors[key])
    return info, metadata, data_start


def gib(num_bytes: int) -> float:
    """Bytes -> GiB."""
    return num_bytes / 2**30


def fmt_duration(seconds: float) -> str:
    """Seconds -> "m:ss" / "h:mm:ss"."""
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class Progress:
    """Write-progress display (percentage + bytes written + rate + ETA).

    Modes:
      bar   -- terminal: refresh one line in place (default on a tty)
      lines -- non-tty (redirected/logged): one line every interval seconds,
               so the log is not flooded
      off   -- no output

    Progress goes to stderr only; stdout stays grep-able / redirectable.
    """

    def __init__(
        self,
        total_bytes: int,
        label: str,
        mode: str = "auto",
        stream=None,
        interval: float | None = None,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        if mode == "auto":
            is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
            mode = "bar" if is_tty else "lines"
        self.mode = mode
        self.total = max(int(total_bytes), 1)
        self.label = label
        self.interval = interval if interval is not None else (0.3 if mode == "bar" else 10.0)
        self.start = time.monotonic()
        self.last_update = 0.0
        self.done = 0
        self.extra = ""
        self.printed = False

    def update(self, delta_bytes: int, **extra: object) -> None:
        self.done += max(int(delta_bytes), 0)
        if extra:
            self.extra = " ".join(f"{key}={value}" for key, value in extra.items())
        if self.mode == "off":
            return
        now = time.monotonic()
        finished = self.done >= self.total
        if not finished and now - self.last_update < self.interval:
            return
        self.last_update = now
        elapsed = max(now - self.start, 1e-6)
        ratio = min(self.done / self.total, 1.0)
        speed = self.done / elapsed
        eta = (self.total - self.done) / speed if speed > 0 else 0.0
        text = (
            f"{self.label} {ratio * 100:5.1f}% "
            f"({gib(self.done):.1f}/{gib(self.total):.1f} GiB, "
            f"{gib(speed):.2f} GiB/s, ETA {fmt_duration(eta)})"
        )
        if self.extra:
            text += f" {self.extra}"
        if self.mode == "bar":
            print(f"\r{text}", end="\n" if finished else "", file=self.stream, flush=True)
        else:
            print(text, file=self.stream, flush=True)
        self.printed = True

    def close(self) -> None:
        """Finish up (bar mode emits a newline so later output starts fresh)."""
        if self.mode == "bar" and self.printed and self.done < self.total:
            print(file=self.stream, flush=True)
        self.printed = False


# ---------------------------------------------------------------------------
# Shard planning and writing
# ---------------------------------------------------------------------------
def plan_shards(
    info: dict[str, tuple[str, TensorInfo]],
    data_start: dict[str, int],
    max_shard_bytes: int,
) -> list[dict]:
    """Pack the kept tensors into output shards, in source-shard order.

    Tensors from the same source shard follow header order, so adjacent
    intervals merge into contiguous copies when writing.
    entry = (tensor name, source shard, dtype, shape, **absolute offset in
    the source file**, nbytes).
    """
    entries_by_shard: dict[str, list[str]] = defaultdict(list)
    order: dict[str, int] = {}
    for key, (shard, _) in info.items():
        if shard not in order:
            order[shard] = len(order)
        entries_by_shard[shard].append(key)

    shards: list[dict] = []
    current: list[tuple] = []
    current_bytes = 0
    for shard in sorted(entries_by_shard, key=lambda name: order[name]):
        base = data_start[shard]
        for key in entries_by_shard[shard]:
            _, tensor = info[key]
            if current and current_bytes + tensor.nbytes > max_shard_bytes:
                shards.append({"entries": current})
                current, current_bytes = [], 0
            # Plain tuples (no custom class) pickle well for worker processes.
            current.append(
                (key, shard, tensor.dtype, list(tensor.shape), base + tensor.start, tensor.nbytes)
            )
            current_bytes += tensor.nbytes
    if current:
        shards.append({"entries": current})
    return shards


def _write_all(fd: int, data: bytes, at: int) -> None:
    """Write the whole buffer at absolute offset ``at`` (os.write may be partial)."""
    os.lseek(fd, at, os.SEEK_SET)
    view = memoryview(data)
    written = 0
    while written < len(view):
        written += os.write(fd, view[written:])


def _copy_range(src_fd: int, dst_fd: int, count: int, src_off: int, dst_off: int) -> None:
    """Copy [src_off, src_off+count) from the source fd to dst_off on the target.

    ``os.copy_file_range`` is preferred (near zero-copy within one
    filesystem); cross-filesystem cases fall back to large-buffer
    read/write. Both paths use explicit offsets, never the fd positions.
    """
    remaining = count
    src_pos, dst_pos = src_off, dst_off
    if hasattr(os, "copy_file_range"):
        try:
            while remaining > 0:
                copied = os.copy_file_range(src_fd, dst_fd, remaining, src_pos, dst_pos)
                if copied <= 0:
                    break
                remaining -= copied
                src_pos += copied
                dst_pos += copied
            if remaining == 0:
                return
        except OSError:
            pass  # cross-filesystem etc.: fall back to plain read/write
    # Fallback path uses lseek + read/write (no preadv/pwritev on Windows)
    buf = bytearray(min(COPY_CHUNK, max(count, 1)))
    view = memoryview(buf)
    remaining = count
    while remaining > 0:
        want = min(remaining, len(buf))
        os.lseek(src_fd, src_pos, os.SEEK_SET)
        got = os.read(src_fd, want)
        if not got:
            raise OSError("source file ended early (truncated?)")
        _write_all(dst_fd, bytes(got), dst_pos)
        src_pos += len(got)
        dst_pos += len(got)
        remaining -= len(got)


def write_one_shard(job: dict) -> tuple[str, int, int]:
    """Worker: write one output shard under its temp name; return
    (temp name, total bytes including header, tensor count)."""
    entries: list[tuple] = job["entries"]
    src_dir: str = job["src_dir"]
    temp_path: str = job["temp_path"]

    # 1) Recompute the header (data offsets restart from 0)
    header: dict[str, dict] = {}
    layout: list[tuple[str, int, int, int]] = []  # (src shard, src off, len, dst off)
    offset = 0
    for key, shard, dtype, shape, start, nbytes in entries:
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        layout.append((shard, start, nbytes, offset))
        offset += nbytes
    if job.get("metadata"):
        header["__metadata__"] = job["metadata"]
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)  # data area must be 8-byte aligned
    data_offset = 8 + len(raw)

    # 2) Write the header, then copy data per interval (adjacent intervals
    #    are contiguous and merge naturally)
    dst_fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    src_fd = -1
    current_src = None
    queue = job.get("queue")
    pending_report = 0
    try:
        _write_all(dst_fd, struct.pack("<Q", len(raw)) + raw, 0)
        for shard, src_off, length, dst_off in layout:
            if shard != current_src:
                if src_fd >= 0:
                    os.close(src_fd)
                # Open read-only: the source dir is never written.
                src_fd = os.open(os.path.join(src_dir, shard), os.O_RDONLY)
                current_src = shard
            _copy_range(src_fd, dst_fd, length, src_off, data_offset + dst_off)
            pending_report += length
            if queue is not None and pending_report >= PROGRESS_STEP:
                queue.put(pending_report)
                pending_report = 0
    except BaseException:
        if src_fd >= 0:
            os.close(src_fd)
        os.close(dst_fd)
        if os.path.isfile(temp_path):
            os.remove(temp_path)  # drop the half-written shard (still -of-XXXXX)
        raise
    if src_fd >= 0:
        os.close(src_fd)
    os.close(dst_fd)
    if queue is not None and pending_report:
        queue.put(pending_report)
    return os.path.basename(temp_path), data_offset + offset, len(entries)


def write_slices(
    src: str,
    dst: str,
    shards: list[dict],
    metadata: dict | None,
    jobs: int,
    progress: "Progress | None" = None,
) -> list[str]:
    """Write all output shards in parallel (or single process), then rename
    them to their final names once every shard has succeeded."""
    for index, shard in enumerate(shards, start=1):
        shard["temp_path"] = os.path.join(dst, f"model-{index:05d}-of-XXXXX.safetensors")
        shard["src_dir"] = src
        shard["metadata"] = metadata

    done_count = 0
    manager = None
    queue = None
    try:
        if jobs > 1:
            try:
                # mp.Queue can only be inherited at process creation; passing
                # one through ProcessPoolExecutor.submit fails. Use a Manager
                # proxy queue instead (progress reports only, tiny volume).
                manager = mp.Manager()
                queue = manager.Queue()
                for shard in shards:
                    shard["queue"] = queue
                with ProcessPoolExecutor(max_workers=jobs) as pool:
                    futures = {pool.submit(write_one_shard, shard): shard for shard in shards}
                    pending = set(futures)
                    while pending:
                        done, pending = wait(pending, timeout=0.3, return_when=FIRST_COMPLETED)
                        while True:
                            try:
                                delta = queue.get_nowait()
                            except Exception:
                                break
                            if progress is not None:
                                progress.update(delta, shards=done_count)
                        for future in sorted(done, key=lambda f: futures[f]["temp_path"]):
                            future.result()
                            done_count += 1
                            if progress is not None:
                                progress.update(0, shards=done_count)
            except (OSError, PermissionError, RuntimeError, ImportError) as exc:
                # Restricted environments (containers/seccomp/sandboxes that
                # ban anonymous pipes) cannot start workers: fall back to a
                # single process. Semantics and results are identical, only
                # slower.
                print(
                    f"[warn] multiprocess write unavailable "
                    f"({type(exc).__name__}: {exc}); falling back to a single process.",
                    file=sys.stderr,
                )
                if manager is not None:
                    manager.shutdown()
                manager, queue = None, None
                done_count = 0
                for shard in shards:
                    shard.pop("queue", None)
                    write_one_shard(shard)
                    done_count += 1
                    if progress is not None:
                        progress.update(0, shards=done_count)
        else:
            for shard in shards:
                write_one_shard(shard)
                done_count += 1
                if progress is not None:
                    progress.update(0, shards=done_count)
    except BaseException:
        if progress is not None:
            progress.close()
        # Shards written this run are incomplete, so the whole set is
        # unusable; remove them, keep only the copied metadata files.
        removed = 0
        for shard in shards:
            path = shard["temp_path"]
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
        print(
            f"[clean] write failed; removed {removed} shard(s) written this run "
            f"(only metadata files remain in the target; safe to rerun)",
            file=sys.stderr,
        )
        raise
    finally:
        if manager is not None:
            manager.shutdown()

    total = len(shards)
    final_names: list[str] = []
    for index in range(1, total + 1):
        temp_name = f"model-{index:05d}-of-XXXXX.safetensors"
        final_name = f"model-{index:05d}-of-{total:05d}.safetensors"
        os.replace(os.path.join(dst, temp_name), os.path.join(dst, final_name))
        final_names.append(final_name)
    if progress is not None:
        progress.update(max(progress.total - progress.done, 0), shards=total)
        progress.close()
    return final_names


def write_index(dst: str, shards: list[dict], shard_names: list[str], companion_present: bool) -> None:
    """Write ``model.safetensors.index.json``.

    * <= 1 shard and no companion safetensors in the target dir: delete any
      stale index (it would filter out our freshly built shards);
    * otherwise (including "one shard but rot.safetensors exists"): the
      index **must** be written, or vLLM would read the companion
      safetensors as weight files.
    """
    index_path = os.path.join(dst, INDEX_NAME)
    if len(shard_names) <= 1 and not companion_present:
        if os.path.isfile(index_path):
            os.remove(index_path)
        return

    weight_map: dict[str, str] = {}
    total_size = 0
    for shard, name in zip(shards, shard_names):
        total_size += os.path.getsize(os.path.join(dst, name))
        for entry in shard["entries"]:
            weight_map[entry[0]] = name
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            handle,
            indent=2,
        )
    expected = {entry[0] for shard in shards for entry in shard["entries"]}
    missing = expected - set(weight_map)
    if missing:
        raise RuntimeError(
            f"{len(missing)} tensor(s) missing from the new shards, "
            f"e.g. {sorted(missing)[:5]}"
        )


def verify_output(dst: str, shard_names: list[str], info: dict[str, tuple[str, TensorInfo]]) -> int:
    """Read back the output shard headers and require the key set plus every
    tensor's dtype/shape/nbytes to match the plan."""
    seen: dict[str, tuple[str, list[int], int]] = {}
    for name in shard_names:
        tensors, _, _ = read_shard_header(os.path.join(dst, name))
        for key, tensor in tensors.items():
            seen[key] = (tensor.dtype, tensor.shape, tensor.nbytes)
    expected = {key: (t.dtype, t.shape, t.nbytes) for key, (_, t) in info.items()}
    if set(seen) != set(expected):
        print(
            f"[verify] FAIL: planned {len(expected)} tensor(s), found {len(seen)}",
            file=sys.stderr,
        )
        return 4
    for key, want in expected.items():
        if seen[key] != want:
            print(f"[verify] FAIL: {key} metadata mismatch {seen[key]} != {want}", file=sys.stderr)
            return 4
    return 0


# ---------------------------------------------------------------------------
# Non-weight entries / source read-only verification
# ---------------------------------------------------------------------------
def _metadata_candidates(
    src: str, weight_files: set[str], companions: list[str]
) -> tuple[list[str], list[str]]:
    """Candidate non-weight entries -> (files, dirs).

    Files = plain files + companion safetensors; dirs are returned separately
    (not copied by default, only when a rule names them).
    """
    files = list(companions)
    dirs: list[str] = []
    for name in sorted(os.listdir(src)):
        if name in weight_files or name in companions:
            continue
        if name.endswith(".safetensors") or name.startswith(SKIP_FILE_PREFIX):
            continue
        if name in SKIP_FILE_EXACT:
            continue
        (dirs if os.path.isdir(os.path.join(src, name)) else files).append(name)
    return sorted(files), dirs


def plan_metadata(
    src: str, role: str, mode: str, weight_files: set[str], companions: list[str]
) -> tuple[list[str], list[str]]:
    """Return (entries to copy, entries deliberately skipped).

    Files (including companion safetensors) follow the per-role rules
    (``*`` wildcards supported); directories are not copied by default,
    unless a rule **explicitly names** them (``"*"`` does not count, so
    ``--metadata auto`` cannot drag whole directories along) or mode is
    ``all``.
    """
    allowed = ("*",) if mode == "all" else METADATA_RULES[role]
    files, dirs = _metadata_candidates(src, weight_files, companions)

    def keep_file(name: str) -> bool:
        return any(fnmatch.fnmatch(name, rule) for rule in allowed)

    def keep_dir(name: str) -> bool:
        if mode == "all":
            return True
        return any(rule != "*" and fnmatch.fnmatch(name, rule) for rule in allowed)

    wanted = [name for name in files if keep_file(name)]
    wanted += [name for name in dirs if keep_dir(name)]
    skipped = [name for name in files + dirs if name not in wanted]
    return wanted, skipped


def copy_metadata(src: str, dst: str, names: list[str]) -> list[str]:
    """Copy non-weight entries; directories are copied recursively (e.g. optional/)."""
    copied = []
    for name in names:
        source_path = os.path.join(src, name)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, os.path.join(dst, name), dirs_exist_ok=True)
            copied.append(name + "/")
            continue
        if not os.path.isfile(source_path):
            continue
        shutil.copy2(source_path, os.path.join(dst, name))
        copied.append(name)
    return copied


def snapshot_tree(root: str) -> dict[str, tuple[int, int, int]]:
    """relpath -> (size, mtime_ns, inode); used to prove the source dir is untouched."""
    snapshot: dict[str, tuple[int, int, int]] = {}
    for current, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(current, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            snapshot[os.path.relpath(path, root)] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    return snapshot


def main() -> int:
    args = parse_args()
    prefixes = tuple(args.keep_prefix) + ROLE_RULES[args.role]
    patterns: list[tuple[str, "re.Pattern[str]"]] = [
        (f"re:{pattern.pattern}", pattern) for pattern in ROLE_NAME_PATTERNS[args.role]
    ]
    for raw in args.keep_name_regex:
        try:
            patterns.append((f"re:{raw}", re.compile(raw)))
        except re.error as exc:
            print(f"invalid --keep-name-regex: {raw} ({exc})", file=sys.stderr)
            return 2
    patterns_tuple = tuple(patterns)

    # Relative paths resolve against the current working directory; ~ is
    # expanded and symlinks resolved, so the "dst must not equal/contain
    # src" check below cannot be bypassed through a symlink.
    src = os.path.realpath(os.path.expanduser(args.src))
    if not os.path.isdir(src):
        print(f"source dir not found: {src}", file=sys.stderr)
        return 2

    # Source read-only guard (on by default): snapshot now, compare at the end.
    guard_snapshot = None if args.no_verify_src_readonly else snapshot_tree(src)

    weight_map, index_name = load_weight_map(src)
    weight_files, companions = classify_safetensors(src, weight_map, index_name)
    grouped, matched = select_keys(weight_map, prefixes, patterns_tuple)

    total_bytes = sum(os.path.getsize(os.path.join(src, name)) for name in weight_files)
    print(f"[in ] {len(weight_map)} tensor(s), {gib(total_bytes):.2f} GiB on disk")
    print(f"[plan] source index: {index_name or '(no index; reading shard headers one by one)'}")
    if companions:
        print(f"[plan] companion safetensors (not weight shards): {companions}")
    print(f"[rule] keep prefixes: {list(prefixes)}")
    if patterns_tuple:
        print(f"[rule] keep name patterns: {[label for label, _ in patterns_tuple]}")

    labels = prefixes + tuple(label for label, _ in patterns_tuple)
    width = max([40, *(len(label) for label in labels)])
    print("\n  matched tensors per rule:")
    kept = 0
    for label in labels:
        keys = grouped.get(label, [])
        if not keys:
            print(f"    {label:{width}s} 0 tensor(s)  <-- no match; check the naming layout / whether it should be kept")
            continue
        kept += len(keys)
        print(f"    {label:{width}s} {len(keys)} tensor(s)")
    print(f"\n  => keep {kept} tensor(s) of {len(weight_map)}; drop {len(weight_map) - kept}")

    dropped_groups: dict[str, int] = defaultdict(int)
    keep = {key for keys in grouped.values() for key in keys}
    for key in weight_map:
        if key not in keep:
            dropped_groups[".".join(key.split(".")[:3])] += 1
    print("\n  dropped tensors grouped by leading components (top 15):")
    for name, count in sorted(dropped_groups.items(), key=lambda item: -item[1])[:15]:
        print(f"    {name:48s} {count}")

    if matched == 0:
        print(
            "\nno tensor matched any keep rule; use --keep-prefix/--keep-name-regex "
            "to specify the correct rules.",
            file=sys.stderr,
        )
        return 3

    # Read headers -> exact sizes + shard plan (done for dry-run too, so the
    # [size] line is accurate)
    try:
        info, metadata, data_start = collect_tensor_info(src, weight_files, keep, weight_map)
    except (OSError, ValueError) as exc:
        print(f"\n  [size] cannot compute the kept size: {exc}", file=sys.stderr)
        return 2
    keep_size = sum(tensor.nbytes for _, tensor in info.values())
    print(
        f"\n  [size] will write {gib(keep_size):.2f} GiB of {gib(total_bytes):.2f} GiB source "
        f"({keep_size / max(total_bytes, 1) * 100:.1f}%)"
    )

    max_shard_bytes = int(args.max_shard_gb * 2**30)
    shards = plan_shards(info, data_start, max_shard_bytes)
    print(f"  [plan] {len(shards)} output shard(s), <= {args.max_shard_gb} GiB each")

    meta_copy, meta_skip = plan_metadata(src, args.role, args.metadata, weight_files, companions)
    print(f"\n[meta] will copy {len(meta_copy)} non-weight entries: {meta_copy}")
    if meta_skip:
        print(f"[meta] deliberately skipped {len(meta_skip)}: {meta_skip}")
    if args.metadata == "auto" and args.role == "cloud":
        print(
            "[meta] cloud keeps only what the vLLM LWD changes require "
            "(skip_tokenizer_init + mm_encoder_remote); if the cloud still "
            "runs an unpatched vllm, add --metadata all"
        )
    if args.dry_run:
        return 0

    if not args.dst:
        print("--dst is required unless --dry-run", file=sys.stderr)
        return 2
    dst = os.path.realpath(os.path.expanduser(args.dst))
    if dst == src or src.startswith(dst + os.sep):
        print("for safety, --dst must not equal or contain --src", file=sys.stderr)
        return 2

    # Free-space precheck: fail here rather than filling the disk on shard N
    # and leaving half-written files behind.
    if keep_size:
        probe = dst
        while not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        free = shutil.disk_usage(probe).free
        need = keep_size + 256 * 2**20  # 256 MiB headroom
        print(f"\n[space] target {probe} has {free / 2**30:.2f} GiB free, need {need / 2**30:.2f} GiB")
        if free < need:
            print(
                "insufficient space: use a --dst with room (another disk/mount), "
                "or clean up the target disk first.",
                file=sys.stderr,
            )
            return 5

    jobs = args.jobs if args.jobs > 0 else min(4, os.cpu_count() or 1)
    os.makedirs(dst, exist_ok=True)
    copied = copy_metadata(src, dst, meta_copy)
    print(f"\n[copy] {len(copied)} metadata entrie(s): {copied}")
    print(f"[write] {len(shards)} shard(s), jobs={jobs}, byte-level copy (no materialized tensors)")

    progress = Progress(
        keep_size,
        "[write]",
        mode="off" if (args.no_progress or not keep_size) else "auto",
    )
    shard_names = write_slices(src, dst, shards, metadata, jobs, progress=progress)
    companion_in_dst = any(name in set(companions) for name in meta_copy)
    write_index(dst, shards, shard_names, companion_in_dst)

    code = verify_output(dst, shard_names, info)
    if code != 0:
        return code
    total_written = sum(os.path.getsize(os.path.join(dst, name)) for name in shard_names)
    print(f"[verify] OK: {len(info)} tensor(s), {gib(total_written):.2f} GiB in {dst}")

    if guard_snapshot is not None:
        after = snapshot_tree(src)
        if after != guard_snapshot:
            changed = sorted(set(after) ^ set(guard_snapshot))[:5]
            details = [f"{name}: {guard_snapshot.get(name)} -> {after.get(name)}" for name in changed]
            print(
                f"[guard] FAIL: source dir {src} changed during this run "
                f"(never allowed): {details}",
                file=sys.stderr,
            )
            return 6
        print(f"[guard] source read-only verified ({len(after)} file snapshot(s) unchanged)")

    print(
        "[next ] start this role from the split dir; if it still reports "
        "'Following weights were not initialized', add the listed tensors"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
