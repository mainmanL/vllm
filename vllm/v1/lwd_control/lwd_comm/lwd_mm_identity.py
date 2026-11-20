# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Processor fingerprints and MM-ABI identities for multimodal hash chaining.

The fingerprint contributes HMAC material to media-covered blocks (hash domains
0x04/0x05) and travels with v2 requests as
``X-Edge-Cloud-MM-ABI: mm1:<base64url(fingerprint)>`` for cloud validation.
Edge hashing and cloud validation must use identical inputs, so both share this
module. Wire encoding remains centralized in ``lwd_prefix``.

**Fingerprint inputs deliberately exclude local model/tokenizer paths**: the
same artifact is often mounted at different paths on the edge and cloud.
Including paths would make one deployment produce different fingerprints and
reject valid image requests when ``enforce_mm_abi_match`` is enabled. Stable
identity comes from revision, code_revision, and config hashes.

Text-only requests are unaffected: fingerprints apply only to media blocks.
"""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from vllm import envs
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    HEADER_MM_ABI,
    PROTOCOL_VERSION_MM,
    decode_digest,
    encode_digest,
)

MM_ABI_VERSION = "mm1"

_FILE_HASH_CHUNK_SIZE = 1024 * 1024
# Versions of packages that determine media placeholders and content digests.
_PROCESSOR_SOFTWARE_PACKAGES = (
    "vllm",
    "vllm-ascend",
    "transformers",
    "pillow",
)
# MultiModalConfig fields affecting preprocessing, placeholders, and digests.
_MM_CONFIG_FIELDS = (
    "mm_processor_kwargs",
    "media_io_kwargs",
    "limit_per_prompt",
    "interleave_mm_strings",
    "video_pruning_rate",
)
# Node-local resource locations do not contribute to deployment identity.
_ARTIFACT_LOCATION_PLACEHOLDER = "<model-artifact>"


def mm_abi_header_value(fingerprint: bytes) -> str:
    """Render a processor fingerprint as the ``X-Edge-Cloud-MM-ABI`` value."""
    return f"{MM_ABI_VERSION}:{encode_digest(fingerprint)}"


def validate_mm_abi_header(value: str | None) -> str:
    """Validate an ``X-Edge-Cloud-MM-ABI`` header value, returned unchanged."""
    if value is None:
        raise ValueError(f"missing required header {HEADER_MM_ABI}")
    declared, separator, encoded = value.partition(":")
    if not separator or declared != MM_ABI_VERSION or not encoded:
        raise ValueError(
            f"{HEADER_MM_ABI} must be '{MM_ABI_VERSION}:<base64url fingerprint>'"
        )
    # decode_digest also validates the 32-byte length.
    decode_digest(encoded)
    return value


def negotiate_lwd_protocol(
    protocol: str | None,
    mm_abi: str | None,
    *,
    expected_protocol: str | None = None,
    expected_mm_abi: str | None = None,
    enforce_mm_abi_match: bool = True,
) -> str | None:
    """Validate a protocol stack and return the local MM identity, if any."""
    if expected_protocol is not None and protocol != expected_protocol:
        raise ValueError(
            f"cloud did not acknowledge {expected_protocol} (returned {protocol!r})"
        )
    if protocol == PROTOCOL_VERSION_MM:
        value = validate_mm_abi_header(mm_abi)
        if expected_mm_abi is None:
            raise ValueError(
                f"{PROTOCOL_VERSION_MM} requires a locally computed "
                f"{HEADER_MM_ABI} fingerprint on the cloud"
            )
        if enforce_mm_abi_match and value != expected_mm_abi:
            raise ValueError(f"{HEADER_MM_ABI} fingerprint does not match this cloud")
        return expected_mm_abi
    if mm_abi is not None:
        raise ValueError(f"{HEADER_MM_ABI} requires protocol {PROTOCOL_VERSION_MM}")
    return None


def compute_processor_fingerprint(model_config: Any) -> bytes:
    """Return a 32-byte digest identifying the multimodal deployment ABI.

    The digest covers parsed model/processor configuration, preprocessing
    package versions, and the media digest algorithm. It detects common
    deployment drift but does not prove equivalent media execution, matching
    the existing limitation of upstream ``mm_hash``. Missing attributes are
    represented as None so both sides hash the same schema."""
    mm_config = getattr(model_config, "multimodal_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    payload: dict[str, Any] = {
        "revision": getattr(model_config, "revision", None),
        "code_revision": getattr(model_config, "code_revision", None),
        "tokenizer_revision": getattr(model_config, "tokenizer_revision", None),
        "hf_config_commit": getattr(hf_config, "_commit_hash", None),
        "model_type": getattr(hf_text_config, "model_type", None),
        "architecture": getattr(model_config, "_architecture", None),
        # vLLM's config hash covers graph choices that are easy to omit from
        # a manual field list (dtype, quantization, MM attention, TP, FP8).
        "model_config_hash": _config_hash(
            model_config,
            neutralized_fields={
                "model": _ARTIFACT_LOCATION_PLACEHOLDER,
                "model_weights": _ARTIFACT_LOCATION_PLACEHOLDER,
                "tokenizer": _ARTIFACT_LOCATION_PLACEHOLDER,
            },
            scrub_hf_locations=True,
        ),
        "multimodal_config_hash": _config_hash(
            mm_config,
            neutralized_fields={
                # Identify calibration weights by content, not local path.
                "mm_encoder_fp8_scale_path": _file_content_identity(
                    getattr(mm_config, "mm_encoder_fp8_scale_path", None)
                )
            },
        ),
        "hf_image_processor_config": getattr(
            model_config, "hf_image_processor_config", None
        ),
        "mm_hasher_algorithm": getattr(envs, "VLLM_MM_HASHER_ALGORITHM", None),
        "software_versions": {
            package: _package_version(package)
            for package in _PROCESSOR_SOFTWARE_PACKAGES
        },
    }
    for field in _MM_CONFIG_FIELDS:
        value = getattr(mm_config, field, None)
        if value is None:
            # mm_processor_kwargs / media_io_kwargs are InitVars and may be
            # stored only on ModelConfig. Fall back when MultiModalConfig is
            # absent so both sides hash the same values.
            value = getattr(model_config, field, None)
        payload[field] = value
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _config_hash(
    config: Any,
    *,
    neutralized_fields: dict[str, Any] | None = None,
    scrub_hf_locations: bool = False,
) -> str | None:
    """Hash config after neutralizing node-local resource locations."""
    if config is None:
        return None
    stable_config = copy.copy(config)
    for field, value in (neutralized_fields or {}).items():
        if hasattr(stable_config, field):
            setattr(stable_config, field, value)
    if scrub_hf_locations:
        for field in ("hf_config", "hf_text_config"):
            nested = getattr(stable_config, field, None)
            if nested is None:
                continue
            stable_nested = copy.deepcopy(nested)
            _scrub_name_or_path(stable_nested, set())
            setattr(stable_config, field, stable_nested)
    compute_hash = getattr(stable_config, "compute_hash", None)
    if not callable(compute_hash):
        return None
    return str(compute_hash())


def _scrub_name_or_path(value: Any, seen: set[int]) -> None:
    """Recursively replace HF ``_name_or_path`` values with a placeholder."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, dict):
        if "_name_or_path" in value:
            value["_name_or_path"] = _ARTIFACT_LOCATION_PLACEHOLDER
        for nested in value.values():
            _scrub_name_or_path(nested, seen)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            _scrub_name_or_path(nested, seen)
        return
    attributes = getattr(value, "__dict__", None)
    if not isinstance(attributes, dict):
        return
    if "_name_or_path" in attributes:
        attributes["_name_or_path"] = _ARTIFACT_LOCATION_PLACEHOLDER
    for nested in attributes.values():
        _scrub_name_or_path(nested, seen)


def _file_content_identity(path: Any) -> str | None:
    """Identify a graph input by bytes rather than local path."""
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(_FILE_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
