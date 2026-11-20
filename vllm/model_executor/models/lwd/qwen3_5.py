# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 multimodal model surface for LWD edge execution."""

from __future__ import annotations

from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.model_executor.models.utils import maybe_prefix
from vllm.tokenizers.registry import cached_tokenizer_from_config

from .lwd_edge_model import LwdEdgeModel


class LwdQwen35EdgeModel(Qwen3_5ForConditionalGeneration):
    """Reuse Qwen3.5 multimodal behavior without constructing decoder layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        self.visual = Qwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

        text_config = vllm_config.model_config.hf_text_config
        embed_tokens = VocabParallelEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
        )
        lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "language_model.lm_head"),
        )
        if text_config.tie_word_embeddings:
            lm_head = lm_head.tie_weights(embed_tokens)
        self.language_model = LwdEdgeModel(
            embed_tokens=embed_tokens,
            lm_head=lm_head,
            vocab_size=text_config.vocab_size,
        )


def build_qwen35_edge_model(vllm_config: VllmConfig) -> LwdQwen35EdgeModel:
    return LwdQwen35EdgeModel(vllm_config=vllm_config)
