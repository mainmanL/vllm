# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LWD role-based module skipping at model construction time.

Design doc: docs/design/lwd_role_module_skip.md (F1, plan A). The
``LWD_MODULE_OWNERSHIP`` table and the ``lwd_skips_module`` pure function
drive per-role construction guards: the edge skips the final norm (the
cloud norm hidden states before sending them DOWN), the cloud skips the
vision tower (multimodal processing lives on the edge).
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import DeepseekV2Config
from transformers import Qwen3Config as Qwen3TextConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLVisionConfig,
)

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.models.utils import (
    LWD_MODULE_OWNERSHIP,
    PPMissingLayer,
    lwd_role,
    lwd_skips_module,
)
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig


def _lwd_config(role: str | None) -> SimpleNamespace | None:
    return SimpleNamespace(role=role) if role is not None else None


def _make_model_config(config: Any) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=config,
        hf_text_config=config,
        dtype=torch.float32,
        max_model_len=128,
        is_diffusion=False,
        head_dtype=None,
        is_mm_prefix_lm=False,
        multimodal_config=None,
        quantization_config=None,
        runner_type="generate",
        is_moe=True,
        logits_processors=None,
        rswa_window=None,
    )


def _build_model(model_cls: type, config: Any, role: str | None) -> torch.nn.Module:
    vllm_config = VllmConfig()
    vllm_config.model_config = _make_model_config(config)
    vllm_config.lwd_config = _lwd_config(role)
    with set_current_vllm_config(vllm_config):
        return model_cls(vllm_config=vllm_config)


def _param_names(model: torch.nn.Module) -> set[str]:
    return {name for name, _ in model.named_parameters()}


@pytest.mark.parametrize(
    ("module_path", "role", "expected_skip"),
    [
        # Dot-boundary suffix match across layouts (CausalLM and VL prefixes).
        ("model.norm", "edge", True),
        ("language_model.model.norm", "edge", True),
        ("model.norm", "cloud", False),
        # Tower-internal norms die with the visual tree, not with model.norm.
        ("visual.merger.norm", "edge", False),
        # Bare attribute names must not match the norm rule.
        ("norm", "edge", False),
        # The whole visual tree belongs to the edge. The tree dies via the
        # root cut: tower-internal module paths are never queried, they are
        # simply never built.
        ("visual", "cloud", True),
        ("model.visual", "cloud", True),
        ("language_model.visual", "cloud", True),
        ("visual.merger.proj", "cloud", False),
        ("visual", "edge", False),
        # Other towers are not covered by the rule.
        ("vision_tower", "cloud", False),
        # Both-role modules.
        ("model.embed_tokens", "edge", False),
        ("model.embed_tokens", "cloud", False),
        ("lm_head", "edge", False),
        ("lm_head", "cloud", False),
        # Transformer layers live on the cloud (edge builds zero layers).
        ("model.layers", "edge", True),
        ("model.layers", "cloud", False),
        # The MTP drafter lives on the cloud.
        ("model.mtp", "edge", True),
        ("model.mtp", "cloud", False),
    ],
)
def test_lwd_skips_module_ownership(
    module_path: str, role: str, expected_skip: bool
) -> None:
    assert lwd_skips_module(module_path, role) is expected_skip


def test_lwd_skips_module_none_role_keeps_pre_lwd_behavior() -> None:
    """Non-LWD processes (role=None) must never skip any module."""
    for module_path in ("model.norm", "visual", "model.layers", "model.mtp"):
        assert lwd_skips_module(module_path, None) is False


def test_lwd_module_ownership_entries_are_explicit() -> None:
    for suffix, owners in LWD_MODULE_OWNERSHIP.items():
        assert owners, f"{suffix} has no owning role"
        assert owners <= {"edge", "cloud"}, f"{suffix} has unknown roles"


def test_lwd_role_is_none_without_lwd_config() -> None:
    with set_current_vllm_config(VllmConfig()):
        assert lwd_role() is None


@pytest.mark.parametrize("role", ["edge", "cloud"])
def test_lwd_role_reads_current_config(role: str) -> None:
    vllm_config = VllmConfig()
    vllm_config.lwd_config = _lwd_config(role)
    with set_current_vllm_config(vllm_config):
        assert lwd_role() == role


def _make_deepseek_v2_config() -> DeepseekV2Config:
    return DeepseekV2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        first_k_dense_replace=0,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=32,
        q_lora_rank=None,
        kv_lora_rank=0,
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        v_head_dim=0,
    )


@pytest.mark.parametrize("role", ["edge", "cloud", None])
def test_deepseek_v2_norm_guard(role: str | None, dist_init: None) -> None:
    """Edge builds no final norm; cloud and non-LWD keep it."""
    model = _build_model(DeepseekV2Model, _make_deepseek_v2_config(), role)
    names = _param_names(model)
    norm_params = [n for n in names if ".norm." in n]
    layer_params = [n for n in names if ".layers." in n]
    embed_params = [n for n in names if ".embed_tokens." in n]
    if role == "edge":
        assert isinstance(model.norm, PPMissingLayer)
        assert not norm_params
        assert not layer_params
    else:
        assert not isinstance(model.norm, PPMissingLayer)
        assert norm_params
        assert layer_params
    assert embed_params


def _make_qwen3_5_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts_per_tok=1,
        num_experts=2,
        layer_types=["full_attention"],
    )


@pytest.mark.parametrize("role", ["edge", "cloud", None])
def test_qwen3_5_norm_guard(role: str | None, dist_init: None) -> None:
    """Edge builds no final norm; cloud and non-LWD keep it."""
    model = _build_model(Qwen3_5Model, _make_qwen3_5_config(), role)
    names = _param_names(model)
    norm_params = [n for n in names if ".norm." in n]
    layer_params = [n for n in names if ".layers." in n]
    embed_params = [n for n in names if ".embed_tokens." in n]
    if role == "edge":
        assert isinstance(model.norm, PPMissingLayer)
        assert not norm_params
        assert not layer_params
    else:
        assert not isinstance(model.norm, PPMissingLayer)
        assert norm_params
        assert layer_params
    assert embed_params


class _DummyVisionTower(torch.nn.Module):
    """Construction seam: stands in for the heavy vision tower internals."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)


class _DummyLanguageModel(torch.nn.Module):
    """Construction seam: stands in for the text stack built inside VL."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(16, 16)
        self.make_empty_intermediate_tensors = lambda *args, **kwargs: None


def _make_qwen3_vl_config() -> Qwen3VLConfig:
    vision_config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        num_heads=1,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        out_hidden_size=16,
        deepstack_visual_indexes=[0],
    )
    text_config = Qwen3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
    )
    return Qwen3VLConfig(vision_config=vision_config, text_config=text_config)


def _build_qwen3_vl(role: str | None) -> Qwen3VLForConditionalGeneration:
    mm_config = SimpleNamespace(
        mm_encoder_tp_mode="tensor",
        mm_encoder_only=False,
        get_limit_per_prompt=lambda modality: 1,
        get_video_pruning_spec=lambda: None,
        is_multimodal_pruning_enabled=lambda: False,
        video_pruning_rate=0.0,
    )
    config = _make_qwen3_vl_config()
    model_config = SimpleNamespace(
        hf_config=config,
        hf_text_config=config.text_config,
        dtype=torch.float32,
        max_model_len=128,
        is_diffusion=False,
        head_dtype=None,
        is_mm_prefix_lm=False,
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
        skip_tokenizer_init=True,
        quantization_config=None,
        runner_type="generate",
        is_moe=False,
        logits_processors=None,
        rswa_window=None,
        is_multimodal_model=True,
        get_model_arch_config=lambda: None,
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = model_config
    vllm_config.lwd_config = _lwd_config(role)
    with set_current_vllm_config(vllm_config):
        return Qwen3VLForConditionalGeneration(vllm_config=vllm_config)


@pytest.mark.parametrize("role", ["edge", "cloud", None])
def test_qwen3_vl_visual_guard(
    role: str | None, dist_init: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cloud builds no vision tower (and no deepstack); edge and non-LWD keep it."""
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_vl.Qwen3_VisionTransformer",
        _DummyVisionTower,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_vl.Qwen3LLMForCausalLM",
        _DummyLanguageModel,
    )
    model = _build_qwen3_vl(role)
    names = _param_names(model)
    visual_params = [n for n in names if ".visual." in n]
    if role == "cloud":
        assert isinstance(model.visual, PPMissingLayer)
        assert not visual_params
        # Deepstack lives inside the tower and dies with it.
        assert model.use_deepstack is False
    else:
        assert isinstance(model.visual, _DummyVisionTower)
        assert visual_params
        assert model.use_deepstack is True
