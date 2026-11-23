# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.core.sched.output import LwdUnembedBatch
from vllm.v1.worker.lwd_edge.lwd_edge_gpu_model_runner import (
    LwdEdgeGPUModelRunner,
    select_token_batch,
)
from vllm.v1.worker.lwd_edge.lwd_edge_worker import LwdEdgeWorker


def test_rank_replay_uses_stable_token_id_order_for_equal_logits() -> None:
    logits = torch.tensor([[1.0, 1.0, 0.0], [0.0, 2.0, 2.0]])

    assert select_token_batch(logits, [0, 1]) == [0, 2]
    assert select_token_batch(logits, [1, 0]) == [1, 1]


def test_unembed_waits_for_receive_completion_before_lm_head() -> None:
    class IncompleteFuture:
        def __init__(self) -> None:
            self.completed = False
            self.wait_calls = 0
            self.tensor = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
            self.tensor.record_stream = lambda stream: None

        def wait_on_current_stream(self) -> None:
            pass

        def result(self) -> SimpleNamespace:
            if not self.completed:
                raise RuntimeError("receive buffer was read before completion")
            return SimpleNamespace(tensor=self.tensor)

        def wait(self) -> SimpleNamespace:
            self.wait_calls += 1
            self.completed = True
            return self.result()

    future = IncompleteFuture()
    runner = object.__new__(LwdEdgeGPUModelRunner)
    runner.worker = SimpleNamespace(
        comm_service=SimpleNamespace(submit_recv=lambda request: future)
    )
    runner.model_config = SimpleNamespace(get_hidden_size=lambda: 2)
    runner._lwd_up_send_inflight = {}
    runner._lwd_up_send_watermarks = {}
    model = SimpleNamespace(compute_logits=lambda hidden_states: hidden_states)
    batch = LwdUnembedBatch(
        req_ids=["request-a"],
        num_accept_tokens=[2],
        recv_num_elements=4,
        top_id_ths=[[0, 0]],
    )

    with (
        patch.object(runner, "get_model", return_value=model),
        patch(
            "vllm.v1.worker.lwd_edge.lwd_edge_gpu_model_runner."
            "torch.accelerator.current_stream",
            return_value=object(),
        ),
    ):
        output = runner.execute_lwd_unembed(0, batch)

    assert future.wait_calls == 1
    assert output.sampled_token_ids == [[1, 1]]


def test_gpu_worker_uses_host_ucx_without_native_warmup() -> None:
    worker = object.__new__(LwdEdgeWorker)
    worker.vllm_config = SimpleNamespace(
        lwd_config=SimpleNamespace(
            topology=SimpleNamespace(
                deployment=SimpleNamespace(
                    data_plane_comm="host",
                    data_plane_host=None,
                )
            )
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    worker.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
    worker.use_v2_model_runner = False
    worker.device = torch.device("cpu")

    with (
        patch("vllm.v1.worker.lwd_edge.lwd_edge_worker.Worker.init_device"),
        patch("vllm.v1.worker.lwd_edge.lwd_edge_worker.LwdEdgeGPUModelRunner"),
        patch("vllm.v1.worker.lwd_edge.lwd_edge_worker.lwd_init_payload_scrambler"),
        patch(
            "vllm.v1.worker.lwd_edge.lwd_edge_worker.lwd_init_host_data_plane"
        ) as init_host,
    ):
        worker.init_device()

    init_host.assert_called_once_with("cuda")
    assert worker.comm_service is not None


def test_gpu_worker_rejects_non_bf16_before_native_init() -> None:
    worker = object.__new__(LwdEdgeWorker)
    worker.vllm_config = SimpleNamespace(
        lwd_config=SimpleNamespace(
            topology=SimpleNamespace(
                deployment=SimpleNamespace(
                    data_plane_comm="host",
                    data_plane_host=None,
                )
            )
        ),
        model_config=SimpleNamespace(dtype=torch.float16),
    )
    worker.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
    worker.use_v2_model_runner = False
    worker.device = torch.device("cpu")

    with (
        patch("vllm.v1.worker.lwd_edge.lwd_edge_worker.Worker.init_device") as native,
        pytest.raises(ValueError, match="BF16"),
    ):
        worker.init_device()

    native.assert_not_called()


def test_gpu_worker_initializes_scrambler_before_host_channels() -> None:
    worker = object.__new__(LwdEdgeWorker)
    worker.vllm_config = SimpleNamespace(
        lwd_config=SimpleNamespace(
            topology=SimpleNamespace(
                deployment=SimpleNamespace(
                    data_plane_comm="host",
                    data_plane_host=None,
                ),
                feature_ctrl=SimpleNamespace(
                    scramble=SimpleNamespace(
                        enable=True,
                        device_id="gpu-npu",
                        custom_factor="test",
                    )
                ),
            )
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    worker.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
    worker.use_v2_model_runner = False
    worker.device = torch.device("cpu")

    with (
        patch("vllm.v1.worker.lwd_edge.lwd_edge_worker.Worker.init_device"),
        patch("vllm.v1.worker.lwd_edge.lwd_edge_worker.LwdEdgeGPUModelRunner"),
        patch("vllm.v1.worker.lwd_edge.lwd_edge_worker.lwd_init_host_data_plane"),
        patch(
            "vllm.v1.worker.lwd_edge.lwd_edge_worker.lwd_init_payload_scrambler"
        ) as init_scrambler,
    ):
        worker.init_device()

    init_scrambler.assert_called_once_with(worker.vllm_config.lwd_config)
