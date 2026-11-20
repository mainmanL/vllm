# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model loader that constructs only the LWD edge module surface."""

from __future__ import annotations

from collections.abc import Callable

import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from .deepseek_v4 import build_deepseek_v4_edge_model, build_deepseek_v41_edge_model
from .qwen3_5 import build_qwen35_edge_model

LwdEdgeBuilder = Callable[[VllmConfig], torch.nn.Module]
_BUILDERS: dict[str, LwdEdgeBuilder] = {
    "DeepseekV4ForCausalLM": build_deepseek_v4_edge_model,
    "DeepseekV41ForCausalLM": build_deepseek_v41_edge_model,
    "Qwen3_5ForConditionalGeneration": build_qwen35_edge_model,
}


class LwdEdgeModelLoader(BaseModelLoader):
    """Delegate weight IO while replacing full-model construction."""

    def __init__(self, load_config: LoadConfig, vllm_config: VllmConfig) -> None:
        super().__init__(load_config)
        self._delegate = get_model_loader(load_config)
        self._vllm_config = vllm_config

    def create_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> torch.nn.Module:
        del prefix
        architecture = model_config.architecture
        try:
            builder = _BUILDERS[architecture]
        except KeyError:
            raise ValueError(
                f"No dedicated LWD edge builder for {architecture}"
            ) from None
        return builder(vllm_config)

    def download_model(self, model_config: ModelConfig) -> None:
        self._delegate.download_model(model_config)

    def load_weights(self, model: torch.nn.Module, model_config: ModelConfig) -> None:
        self._delegate.load_weights(model, model_config)
