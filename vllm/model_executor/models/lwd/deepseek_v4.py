# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4-family dedicated edge builders."""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from .lwd_edge_model import LwdEdgeModel


def build_deepseek_v4_edge_model(vllm_config: VllmConfig) -> LwdEdgeModel:
    from vllm.models.deepseek_v4.nvidia.model import (
        _make_deepseek_v4_weights_mapper,
    )

    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config
    embed_tokens = VocabParallelEmbedding(
        config.vocab_size,
        config.hidden_size,
        quant_config=quant_config,
        prefix="model.embed_tokens",
    )
    lm_head = ParallelLMHead(
        config.vocab_size,
        config.hidden_size,
        prefix="lm_head",
    )
    expert_dtype = getattr(config, "expert_dtype", "fp4")
    return LwdEdgeModel(
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        vocab_size=config.vocab_size,
        hf_to_vllm_mapper=_make_deepseek_v4_weights_mapper(expert_dtype),
    )


def build_deepseek_v41_edge_model(vllm_config: VllmConfig) -> LwdEdgeModel:
    from vllm.models.deepseek_v41.nvidia.model import (
        _linear_scale_param_name,
        _make_deepseek_v4_weights_mapper,
    )

    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config
    embed_tokens = VocabParallelEmbedding(
        config.vocab_size,
        config.hidden_size,
        quant_config=quant_config,
        prefix="model.embed_tokens",
    )
    lm_head = ParallelLMHead(
        config.vocab_size,
        config.hidden_size,
        prefix="lm_head",
    )
    expert_dtype = getattr(config, "expert_dtype", "fp4")
    return LwdEdgeModel(
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        vocab_size=config.vocab_size,
        hf_to_vllm_mapper=_make_deepseek_v4_weights_mapper(
            expert_dtype,
            _linear_scale_param_name(vllm_config, expert_dtype),
        ),
    )
