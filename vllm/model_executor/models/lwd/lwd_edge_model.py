# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated model surface used by the LWD edge."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class LwdEdgeModel(nn.Module):
    """Only the modules required for edge embed and rank-replay."""

    def __init__(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        vocab_size: int,
        hf_to_vllm_mapper: WeightsMapper | None = None,
    ) -> None:
        super().__init__()
        self.model = LwdEdgeBackbone(embed_tokens)
        self.lm_head = lm_head
        self.logits_processor = LogitsProcessor(vocab_size)
        self.hf_to_vllm_mapper = hf_to_vllm_mapper or WeightsMapper()
        self.fall_back_to_pt_during_load = False
        self.allow_patterns_overrides = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings:
            raise ValueError("This LWD edge model has no multimodal builder")
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class LwdEdgeBackbone(nn.Module):
    def __init__(self, embed_tokens: nn.Module) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
