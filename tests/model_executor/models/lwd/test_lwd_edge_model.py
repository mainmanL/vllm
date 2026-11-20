# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.models.lwd.deepseek_v4 import (
    build_deepseek_v4_edge_model,
    build_deepseek_v41_edge_model,
)
from vllm.model_executor.models.lwd.lwd_edge_model_loader import LwdEdgeModelLoader


@pytest.fixture(autouse=True)
def single_tensor_parallel_rank():
    with (
        patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.model_executor.layers.logits_processor.get_current_vllm_config",
            return_value=SimpleNamespace(model_config=None),
        ),
    ):
        yield


def _config(architecture: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            hf_config=SimpleNamespace(
                vocab_size=8,
                hidden_size=4,
                expert_dtype="fp4",
            ),
        ),
        quant_config=None,
    )


@pytest.mark.parametrize(
    "architecture,builder",
    [
        ("DeepseekV4ForCausalLM", build_deepseek_v4_edge_model),
        ("DeepseekV41ForCausalLM", build_deepseek_v41_edge_model),
    ],
)
def test_edge_model_contains_only_embed_and_lm_head(architecture, builder) -> None:
    model = builder(_config(architecture))
    module_names = {name for name, _ in model.named_modules()}

    assert "model.embed_tokens" in module_names
    assert "lm_head" in module_names
    assert not any(
        marker in name
        for name in module_names
        for marker in ("layers", "norm", "hc_head", "mtp", "moe")
    )


def test_edge_model_loads_native_embed_and_head_weights() -> None:
    model = build_deepseek_v4_edge_model(_config("DeepseekV4ForCausalLM"))
    embed = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    head = torch.arange(100, 132, dtype=torch.float32).reshape(8, 4)

    loaded = model.load_weights([("embed.weight", embed), ("head.weight", head)])

    assert loaded == {"model.embed_tokens.weight", "lm_head.weight"}
    assert torch.equal(model.model.embed_tokens.weight[:8], embed)
    assert torch.equal(model.lm_head.weight[:8], head)


def test_loader_rejects_architecture_without_dedicated_builder() -> None:
    loader = object.__new__(LwdEdgeModelLoader)

    with pytest.raises(ValueError, match="No dedicated LWD edge builder"):
        loader.create_model(
            _config("Qwen3VLForConditionalGeneration"),
            _config("Qwen3VLForConditionalGeneration").model_config,
        )


@pytest.mark.parametrize(
    "builder",
    [build_deepseek_v4_edge_model, build_deepseek_v41_edge_model],
)
def test_edge_lm_head_matches_native_unquantized_construction(builder) -> None:
    config = _config("DeepseekV4ForCausalLM")
    quant_config = object()
    config.quant_config = quant_config

    with (
        patch(
            "vllm.model_executor.models.lwd.deepseek_v4.VocabParallelEmbedding",
            return_value=torch.nn.Embedding(8, 4),
        ) as embed,
        patch(
            "vllm.model_executor.models.lwd.deepseek_v4.ParallelLMHead",
            return_value=torch.nn.Linear(4, 8, bias=False),
        ) as head,
    ):
        builder(config)

    assert embed.call_args.kwargs["quant_config"] is quant_config
    assert "quant_config" not in head.call_args.kwargs
