# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models.lwd.lwd_edge_model_loader import LwdEdgeModelLoader
from vllm.model_executor.models.lwd.qwen3_5 import build_qwen35_edge_model
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration


class _FakeVision(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.spatial_merge_size = 1
        self.dtype = torch.float32


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="Qwen3_5ForConditionalGeneration",
            hf_config=SimpleNamespace(vision_config=SimpleNamespace(out_hidden_size=4)),
            hf_text_config=SimpleNamespace(
                vocab_size=8,
                hidden_size=4,
                tie_word_embeddings=False,
            ),
            multimodal_config=SimpleNamespace(
                mm_encoder_tp_mode="tensor",
                is_multimodal_pruning_enabled=lambda: False,
                video_pruning_rate=0.0,
            ),
        ),
        quant_config=None,
    )


@pytest.fixture(autouse=True)
def _single_tensor_parallel_rank():
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


def _build_model():
    with (
        patch(
            "vllm.model_executor.models.lwd.qwen3_5.Qwen3_VisionTransformer",
            return_value=_FakeVision(),
        ),
        patch(
            "vllm.model_executor.models.lwd.qwen3_5.cached_tokenizer_from_config",
            return_value=None,
        ),
    ):
        return build_qwen35_edge_model(_config())


def test_loader_creates_qwen35_without_decoder() -> None:
    loader = object.__new__(LwdEdgeModelLoader)

    with (
        patch(
            "vllm.model_executor.models.lwd.qwen3_5.Qwen3_VisionTransformer",
            return_value=_FakeVision(),
        ),
        patch(
            "vllm.model_executor.models.lwd.qwen3_5.cached_tokenizer_from_config",
            return_value=None,
        ),
    ):
        model = loader.create_model(_config(), _config().model_config)

    module_names = {name for name, _ in model.named_modules()}
    assert "visual" in module_names
    assert "language_model.model.embed_tokens" in module_names
    assert "language_model.lm_head" in module_names
    assert not any(
        marker in name
        for name in module_names
        for marker in ("layers", "norm", "hc_head", "mtp", "mamba")
    )


def test_qwen35_loads_native_visual_embed_and_head_weights() -> None:
    model = _build_model()
    visual = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    embed = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    head = torch.arange(100, 132, dtype=torch.float32).reshape(8, 4)

    loaded = model.load_weights(
        [
            ("model.visual.weight", visual),
            ("model.language_model.embed_tokens.weight", embed),
            ("lm_head.weight", head),
        ]
    )

    assert loaded == {
        "visual.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    }
    assert torch.equal(model.visual.weight, visual)
    assert torch.equal(model.language_model.model.embed_tokens.weight[:8], embed)
    assert torch.equal(model.language_model.lm_head.weight[:8], head)


def test_qwen35_multimodal_encoding_and_merge_use_native_path() -> None:
    model = _build_model()
    embed = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    model.load_weights([("model.language_model.embed_tokens.weight", embed)])
    image_embeds = torch.tensor([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]])
    image_input = {
        "type": "image_embeds",
        "image_embeds": image_embeds,
        "image_grid_thw": torch.tensor([[1, 1, 2]]),
    }

    with patch.object(
        type(model),
        "_parse_and_validate_multimodal_inputs",
        return_value={"image": image_input},
    ):
        encoded = model.embed_multimodal(image=object())
    merged = model.embed_input_ids(
        torch.tensor([0, 1, 2]),
        encoded,
        is_multimodal=torch.tensor([True, True, False]),
    )

    assert len(encoded) == 1
    assert torch.equal(encoded[0], image_embeds)
    assert torch.equal(merged[0], image_embeds[0])
    assert torch.equal(merged[1], image_embeds[1])
    assert torch.equal(merged[2], embed[2])


def test_qwen35_logits_match_native_model_with_same_weights() -> None:
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.model_config = _config().model_config
    text_config = config.model_config.hf_text_config
    text_config.num_hidden_layers = 0
    text_config.layer_types = []
    text_config.model_type = "qwen3_5_text"
    text_config.rms_norm_eps = 1e-6
    pp_group = SimpleNamespace(
        is_first_rank=True,
        is_last_rank=True,
        rank_in_group=0,
        world_size=1,
    )
    weights = [
        ("model.visual.weight", torch.arange(4, dtype=torch.float32).reshape(2, 2)),
        (
            "model.language_model.embed_tokens.weight",
            torch.arange(32, dtype=torch.float32).reshape(8, 4),
        ),
        ("lm_head.weight", torch.arange(100, 132, dtype=torch.float32).reshape(8, 4)),
    ]

    with (
        patch(
            "vllm.model_executor.models.qwen3_5.Qwen3_VisionTransformer",
            return_value=_FakeVision(),
        ),
        patch(
            "vllm.model_executor.models.qwen3_5.cached_tokenizer_from_config",
            return_value=None,
        ),
        patch.object(
            Qwen3_5ForConditionalGeneration,
            "_mark_tower_model",
            return_value=nullcontext(),
        ),
        patch.object(
            Qwen3_5ForConditionalGeneration,
            "_mark_language_model",
            return_value=nullcontext(),
        ),
        patch("vllm.model_executor.models.qwen3_5.get_pp_group", return_value=pp_group),
        patch(
            "vllm.model_executor.models.qwen3_next.get_pp_group", return_value=pp_group
        ),
        patch("vllm.distributed.parallel_state.get_pp_group", return_value=pp_group),
        set_current_vllm_config(config),
    ):
        native = Qwen3_5ForConditionalGeneration(vllm_config=config)
    native.load_weights(weights)
    edge = _build_model()
    edge.load_weights(weights)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10

    assert torch.equal(
        edge.compute_logits(hidden_states),
        native.compute_logits(hidden_states),
    )
