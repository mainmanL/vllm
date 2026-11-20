# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.lwd.lwd_payload_scrambler import LwdPayloadScrambler
from vllm.v1.lwd_comm.lwd_pinned_staging import LwdPinnedStaging
from vllm.v1.lwd_comm.lwd_up_payload import LwdUpPayloadLayout


def test_up_payload_roundtrip_preserves_large_expert_and_engram_ids() -> None:
    layout = LwdUpPayloadLayout(
        hidden_size=4,
        num_hash_layers=1,
        hash_top_k=2,
        num_engram_layers=1,
        num_engram_cols=2,
    )
    embeds = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    expert_ids = torch.tensor([[[256, 1024]], [[7, 8]]], dtype=torch.int32)
    engram_hashes = torch.tensor(
        [[[513, 1027]], [[2051, 4099]]],
        dtype=torch.int32,
    )

    payload = layout.pack(embeds, expert_ids, engram_hashes)
    actual_embeds, actual_experts, actual_engram = layout.unpack(payload, 2)

    assert payload.dtype == torch.bfloat16
    assert payload.numel() == layout.numel(2)
    assert torch.equal(actual_embeds, embeds)
    assert torch.equal(actual_experts, expert_ids)
    assert torch.equal(actual_engram, engram_hashes)


def test_up_payload_rejects_a_frame_with_the_wrong_size() -> None:
    layout = LwdUpPayloadLayout(hidden_size=4)
    payload = torch.zeros(7, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="UP payload size mismatch"):
        layout.unpack(payload, 2)


def test_scrambler_roundtrip_preserves_all_16_bit_patterns() -> None:
    scrambler = LwdPayloadScrambler(device_id="gpu-npu", custom_factor="test")
    original = torch.arange(-32768, 32768, dtype=torch.int16)
    tensor = original.view(torch.bfloat16).clone()

    scrambler.scramble(tensor, seed=0xBEEF)
    scrambler.descramble(tensor, seed=0xBEEF)

    assert torch.equal(tensor.view(torch.int16), original)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA pinned-memory staging requires CUDA"
)
def test_pinned_staging_rotates_and_grows_without_changing_values() -> None:
    ring = LwdPinnedStaging(dtype=torch.int64, min_capacity=8, device="cuda")
    pointers = []
    for value in range(4):
        view = ring.stage_to_device([value], 1)
        pointers.append(view.data_ptr())
        assert view.item() == value

    assert len(set(pointers)) == 4
    assert ring.stage_to_device(list(range(100)), 100).tolist() == list(range(100))
    assert ring.stage_to_device([7, 8], 2).tolist() == [7, 8]
