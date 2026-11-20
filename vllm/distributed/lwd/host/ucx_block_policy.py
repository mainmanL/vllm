# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD UCX UP block policy.

Pure standard-library policy shared by the runtime UCX data plane and the
standalone tooling. Keep this module free of torch/vllm imports.
"""

from __future__ import annotations

from dataclasses import dataclass

FALLBACK_UP_BLOCK_MB = 128
MIN_UP_BLOCK_MB = 128
BLOCK_ALIGN_MB = 32
BF16_BYTES = 2
DSV4_MODEL_TYPE = "deepseek_v4"

AUTO_SOURCE = "auto"
EXPLICIT_SOURCE = "explicit"
FALLBACK_SOURCE = "fallback"


class LwdUcxBlockPolicyError(ValueError):
    """Invalid inputs for UCX block sizing."""


@dataclass(frozen=True)
class LwdUcxBlockPlan:
    """Resolved UCX UP block and the inputs/warnings behind it."""

    up_block_mb: int
    source: str
    auto_up_block_mb: int
    raw_mib: float
    per_token_elements: int
    model_type: str
    max_num_batched_tokens: int
    warnings: tuple[str, ...] = ()


def _is_known_model(model_type: str) -> bool:
    return model_type.startswith("deepseek_v4") or model_type.startswith("qwen")


def _align_up_block_mb(raw_bytes: int) -> int:
    align_bytes = BLOCK_ALIGN_MB << 20
    aligned = ((raw_bytes + align_bytes - 1) // align_bytes) * BLOCK_ALIGN_MB
    return max(MIN_UP_BLOCK_MB, aligned)


def _per_token_elements(
    *,
    model_type: str,
    hidden_size: int,
    num_hash_layers: int,
    num_experts_per_tok: int,
    num_engram_layers: int = 0,
    num_engram_cols: int = 0,
) -> int:
    if model_type.startswith("deepseek_v4"):
        if (
            min(
                num_hash_layers, num_experts_per_tok, num_engram_layers, num_engram_cols
            )
            < 0
        ):
            raise LwdUcxBlockPolicyError(
                "DeepSeek V4 hash fields must be non-negative: "
                f"num_hash_layers={num_hash_layers}, "
                f"num_experts_per_tok={num_experts_per_tok}, "
                f"num_engram_layers={num_engram_layers}, "
                f"num_engram_cols={num_engram_cols}"
            )
        return hidden_size + 4 * (
            num_hash_layers * num_experts_per_tok + num_engram_layers * num_engram_cols
        )
    return hidden_size


def resolve(
    *,
    model_type: str,
    hidden_size: int | None,
    num_hash_layers: int = 0,
    num_experts_per_tok: int = 0,
    max_num_batched_tokens: int,
    explicit_up_block_mb: int | None = None,
    fallback_up_block_mb: int = FALLBACK_UP_BLOCK_MB,
    num_engram_layers: int = 0,
    num_engram_cols: int = 0,
) -> LwdUcxBlockPlan:
    """Resolve the UP block from model metadata and the batch-token cap.

    Rules:
    * known model + no explicit value -> 32MiB aligned auto value;
    * known model + explicit value -> explicit value, warning if below auto;
    * unknown model -> explicit value if given, else default 128MiB;
    * known model missing hidden_size -> error.

    A frame larger than the resolved block is rejected by
    ``LwdBaseUcxChannelIo.submit_send`` at runtime.
    """
    if max_num_batched_tokens <= 0:
        raise LwdUcxBlockPolicyError(
            f"max_num_batched_tokens must be positive, got {max_num_batched_tokens}"
        )
    if explicit_up_block_mb is not None and explicit_up_block_mb <= 0:
        raise LwdUcxBlockPolicyError(
            f"explicit UP block must be positive, got {explicit_up_block_mb}"
        )
    if fallback_up_block_mb <= 0:
        raise LwdUcxBlockPolicyError(
            f"fallback UP block must be positive, got {fallback_up_block_mb}"
        )

    normalized_type = str(model_type or "").strip().lower()
    warnings: list[str] = []

    if not _is_known_model(normalized_type):
        if explicit_up_block_mb is not None:
            warnings.append(
                f"unknown model_type={normalized_type!r}; using explicit "
                f"up_buffer_size={explicit_up_block_mb}"
            )
            return LwdUcxBlockPlan(
                up_block_mb=explicit_up_block_mb,
                source=EXPLICIT_SOURCE,
                auto_up_block_mb=0,
                raw_mib=0.0,
                per_token_elements=0,
                model_type=normalized_type,
                max_num_batched_tokens=max_num_batched_tokens,
                warnings=tuple(warnings),
            )
        warnings.append(
            f"unknown model_type={normalized_type!r}; fallback "
            f"up_buffer_size={fallback_up_block_mb}; large frames will hit "
            "submit_send fast-fail"
        )
        return LwdUcxBlockPlan(
            up_block_mb=fallback_up_block_mb,
            source=FALLBACK_SOURCE,
            auto_up_block_mb=0,
            raw_mib=0.0,
            per_token_elements=0,
            model_type=normalized_type,
            max_num_batched_tokens=max_num_batched_tokens,
            warnings=tuple(warnings),
        )

    if hidden_size is None or hidden_size <= 0:
        raise LwdUcxBlockPolicyError(
            f"model_type={normalized_type!r} needs a positive hidden_size, "
            f"got {hidden_size!r}"
        )

    per_token = _per_token_elements(
        model_type=normalized_type,
        hidden_size=int(hidden_size),
        num_hash_layers=int(num_hash_layers),
        num_experts_per_tok=int(num_experts_per_tok),
        num_engram_layers=num_engram_layers,
        num_engram_cols=num_engram_cols,
    )
    raw_bytes = max_num_batched_tokens * per_token * BF16_BYTES
    raw_mib = raw_bytes / float(1 << 20)
    auto_mb = _align_up_block_mb(raw_bytes)

    if explicit_up_block_mb is not None:
        if explicit_up_block_mb < auto_mb:
            warnings.append(
                f"explicit up_buffer_size={explicit_up_block_mb} < "
                f"auto_required={auto_mb}MiB; large frames will hit "
                "submit_send fast-fail"
            )
        return LwdUcxBlockPlan(
            up_block_mb=explicit_up_block_mb,
            source=EXPLICIT_SOURCE,
            auto_up_block_mb=auto_mb,
            raw_mib=raw_mib,
            per_token_elements=per_token,
            model_type=normalized_type,
            max_num_batched_tokens=max_num_batched_tokens,
            warnings=tuple(warnings),
        )

    return LwdUcxBlockPlan(
        up_block_mb=auto_mb,
        source=AUTO_SOURCE,
        auto_up_block_mb=auto_mb,
        raw_mib=raw_mib,
        per_token_elements=per_token,
        model_type=normalized_type,
        max_num_batched_tokens=max_num_batched_tokens,
        warnings=tuple(warnings),
    )
