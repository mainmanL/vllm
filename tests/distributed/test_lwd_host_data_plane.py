# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.distributed.lwd.host.lwd_host_transfer_queue import LwdHostTransferQueue
from vllm.distributed.lwd.host.ucx_block_policy import resolve
from vllm.distributed.lwd.lwd_types import (
    ConnectionId,
    LwdChannelType,
    LwdTransferRequest,
)


def test_up_block_includes_engram_segments_for_deepseek_v41() -> None:
    plan = resolve(
        model_type="deepseek_v4.1",
        hidden_size=64,
        max_num_batched_tokens=4,
        num_engram_layers=2,
        num_engram_cols=3,
    )

    assert plan.per_token_elements == 88
    assert plan.raw_mib == 88 * 4 * 2 / (1 << 20)


def test_gpu_host_queue_allocates_receive_buffers_on_cuda() -> None:
    queue = LwdHostTransferQueue(
        ConnectionId(0, 0, 0), LwdChannelType.DOWN, "recv", device="cuda"
    )
    submitted = {}
    io = SimpleNamespace(
        submit_recv=lambda *args, **kwargs: submitted.update(args=args, kwargs=kwargs)
    )
    request = LwdTransferRequest(
        channel=LwdChannelType.DOWN,
        op="recv",
        num_elements=2,
        seqno=0,
    )
    devices = []
    real_empty = torch.empty

    def fake_empty(*args, **kwargs):
        devices.append(kwargs.get("device"))
        kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    with patch(
        "vllm.distributed.lwd.host.lwd_host_transfer_queue.torch.empty",
        side_effect=fake_empty,
    ):
        queue._execute_recv(request, io, None)

    assert devices == ["cuda"]
    assert submitted["args"][2].device.type == "cpu"
