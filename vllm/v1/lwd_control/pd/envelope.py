# SPDX-License-Identifier: Apache-2.0
"""Immutable PD request semantics, independent of HTTP routing and tensors."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import struct
import time
from collections.abc import Mapping, Sequence
from typing import Any

ENVELOPE_TYPE = "lwd_pd_control"
ENVELOPE_VERSION = "lwd-pd-v2"
DIGEST_ALGO = "hmac-sha256-chain"
DIGEST_DOMAIN = "edge-cloud-prefix"
_SEMANTIC_FIELDS = (
    "rid",
    "attempt_id",
    "edge_id",
    "tenant_id",
    "original_prompt_tokens",
    "user_max_output_tokens",
    "sampling",
    "model_id",
    "input_abi",
    "digest",
    "structured_outputs",
    "issued_at",
    "ttl_s",
)


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def request_fingerprint(fields: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_dumps(fields).encode()).hexdigest()


def encode_digest(value: bytes) -> str:
    if len(value) != 32:
        raise ValueError("PD digest must contain 32 bytes")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def decode_digest(value: str) -> bytes:
    raw = value.encode("ascii")
    result = base64.b64decode(
        raw + b"=" * (-len(raw) % 4), altchars=b"-_", validate=True
    )
    if len(result) != 32:
        raise ValueError("PD digest must contain 32 bytes")
    return result


def sampling_whitelist(params: Any) -> dict[str, Any]:
    from vllm.v1.lwd_control.lwd_comm.lwd_messages import LWD_SAMPLING_FIELDS

    if params is None:
        return {}
    result = {
        name: getattr(params, name, default) for name, default in LWD_SAMPLING_FIELDS
    }
    result["stop_token_ids"] = list(result["stop_token_ids"] or [])
    result["eos_token_id"] = params.eos_token_id
    result["skip_reading_prefix_cache"] = bool(
        getattr(params, "skip_reading_prefix_cache", False)
    )
    return result


def build_envelope(
    *,
    request_id: str,
    attempt_id: str,
    edge_id: int,
    tenant_id: str,
    original_prompt_tokens: int,
    user_max_output_tokens: int,
    sampling_params: Any,
    model_id: str,
    input_abi: str,
    digest_block_size: int,
    digest_chain: Sequence[bytes],
    digest_tail_hash: bytes | None = None,
    media_digests: Sequence[Mapping[str, Any]] = (),
    ttl_s: float,
    issued_at: float | None = None,
) -> dict[str, Any]:
    from vllm.v1.lwd_control.lwd_comm.lwd_messages import lwd_encode_structured_outputs

    structured = getattr(sampling_params, "structured_outputs", None)
    semantic = {
        "rid": request_id,
        "attempt_id": attempt_id,
        "edge_id": edge_id,
        "tenant_id": tenant_id,
        "original_prompt_tokens": original_prompt_tokens,
        "user_max_output_tokens": user_max_output_tokens,
        "sampling": sampling_whitelist(sampling_params),
        "model_id": model_id,
        "input_abi": input_abi,
        "issued_at": time.time() if issued_at is None else issued_at,
        "ttl_s": ttl_s,
        "structured_outputs": lwd_encode_structured_outputs(structured)
        if structured is not None
        else "",
        "digest": {
            "algo": DIGEST_ALGO,
            "domain": DIGEST_DOMAIN,
            "block_size": digest_block_size,
            "chain": [encode_digest(d) for d in digest_chain],
            "tail": encode_digest(digest_tail_hash) if digest_tail_hash else None,
            "media": list(media_digests),
        },
    }
    envelope = {
        "type": ENVELOPE_TYPE,
        "version": ENVELOPE_VERSION,
        **semantic,
        "fingerprint": request_fingerprint(semantic),
    }
    return parse_envelope(canonical_dumps(envelope))


def envelope_content(envelope: Mapping[str, Any]) -> str:
    return canonical_dumps(envelope)


def parse_envelope(content: Any, *, now: float | None = None) -> dict[str, Any]:
    if not isinstance(content, str):
        raise ValueError("PD envelope must be a JSON string")
    try:
        envelope = json.loads(content)
        if envelope["type"] != ENVELOPE_TYPE or envelope["version"] != ENVELOPE_VERSION:
            raise ValueError("unsupported PD envelope protocol")
        semantic = {key: envelope[key] for key in _SEMANTIC_FIELDS}
        expected = request_fingerprint(semantic)
        if not hmac.compare_digest(envelope["fingerprint"], expected):
            raise ValueError("PD envelope fingerprint mismatch")
        for key in ("rid", "attempt_id", "tenant_id"):
            if not isinstance(envelope[key], str) or not envelope[key]:
                raise ValueError(f"PD {key} must be nonempty")
        for key, minimum in (
            ("edge_id", 0),
            ("original_prompt_tokens", 1),
            ("user_max_output_tokens", 1),
        ):
            value = envelope[key]
            if type(value) is not int or value < minimum:
                raise ValueError(f"invalid PD {key}")
        ttl = envelope["ttl_s"]
        if type(ttl) not in (int, float) or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("PD ttl_s must be finite and positive")
        issued = envelope["issued_at"]
        if type(issued) not in (int, float) or not math.isfinite(issued) or issued <= 0:
            raise ValueError("PD issued_at must be finite and positive")
        current = time.time() if now is None else now
        if current >= issued + ttl or issued > current + 30:
            raise ValueError("PD envelope expired or issued in the future")
        digest = envelope["digest"]
        if digest["algo"] != DIGEST_ALGO or digest["domain"] != DIGEST_DOMAIN:
            raise ValueError("unsupported PD digest protocol")
        envelope_to_manifest(envelope, envelope["rid"])
        if not isinstance(envelope["sampling"], dict):
            raise ValueError("PD sampling must be an object")
        structured = envelope["structured_outputs"]
        if not isinstance(structured, str):
            raise ValueError("PD structured_outputs must be a string")
        if structured:
            json.loads(structured)
    except (KeyError, TypeError, UnicodeError) as exc:
        raise ValueError(f"malformed PD envelope: {exc}") from exc
    return envelope


def envelope_to_manifest(envelope: Mapping[str, Any], request_id: str) -> Any:
    from vllm.v1.lwd_control.lwd_comm.lwd_prefix import LwdPrefixManifest

    digest = envelope["digest"]
    return LwdPrefixManifest(
        request_id=request_id,
        prompt_tokens=envelope["original_prompt_tokens"],
        block_size=digest["block_size"],
        full_block_hashes=tuple(decode_digest(d) for d in digest["chain"]),
        tail_hash=decode_digest(digest["tail"]) if digest["tail"] else None,
    )


def model_needs_truncate(model_config: Any, groups: Any) -> bool:
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    ratios = getattr(model_config.hf_config, "compress_ratios", None)
    if isinstance(ratios, (list, tuple, dict)):
        return True
    for group in groups or ():
        spec = group.kv_cache_spec
        specs = (
            spec.kv_cache_specs.values()
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )
        if any(isinstance(item, MambaSpec) for item in specs):
            return True
    return False


def compute_handoff_plan(n_tokens: int, need_truncate: bool) -> tuple[int, int, int]:
    if n_tokens < 1 or (need_truncate and n_tokens == 1):
        raise ValueError("no valid PD handoff for this prompt length")
    p_end = n_tokens - int(need_truncate)
    return p_end, p_end, n_tokens - 1


def validate_handoff_plan(n_tokens: int, p_end: int, i_end: int, r_pos: int) -> None:
    if (p_end, i_end, r_pos) not in (
        compute_handoff_plan(n_tokens, False),
        *((compute_handoff_plan(n_tokens, True),) if n_tokens > 1 else ()),
    ):
        raise ValueError("invalid PD handoff boundaries")


def range_digest(
    binding_id: str, start: int, end: int, token_ids: Sequence[int]
) -> str:
    if not binding_id or not 0 <= start <= end or len(token_ids) != end - start:
        raise ValueError("invalid PD output interval")
    material = bytearray(binding_id.encode())
    material.extend(struct.pack(">QQ", start, end))
    for token in token_ids:
        if type(token) is not int or not 0 <= token <= 0xFFFFFFFF:
            raise ValueError("PD token id must fit uint32")
        material.extend(struct.pack(">I", token))
    return hashlib.sha256(material).hexdigest()
