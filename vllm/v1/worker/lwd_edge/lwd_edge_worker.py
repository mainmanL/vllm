# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU edge worker for LWD prefill-only deployments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.distributed.lwd.host.lwd_host_channels import lwd_destroy_host_channels
from vllm.distributed.lwd.host.lwd_host_transfer_queue import (
    lwd_init_host_data_plane,
)
from vllm.distributed.lwd.lwd_data_plane import (
    lwd_get_data_plane,
    lwd_init_payload_scrambler,
)
from vllm.lwd_logger import lwd_logger
from vllm.v1.core.sched.output import LwdBatchType
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import CompilationTimes

from .lwd_edge_gpu_model_runner import LwdEdgeGPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class LwdEdgeWorker(Worker):
    """Dispatch LWD embed and unembed batches on one GPU."""

    def init_device(self) -> None:
        lwd_config = self.vllm_config.lwd_config
        if lwd_config.topology.deployment.data_plane_comm != "host":
            raise ValueError("GPU LWD edge requires deployment.data_plane_comm=host")
        host_plane = lwd_config.topology.deployment.data_plane_host
        if host_plane is not None and host_plane.deployment != "in_process":
            raise ValueError(
                "GPU LWD edge currently requires host in_process deployment"
            )
        parallel = self.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError(
                "GPU LWD edge requires tensor_parallel_size=1 and "
                "pipeline_parallel_size=1"
            )
        if parallel.data_parallel_size != 1:
            raise ValueError("GPU LWD edge requires data_parallel_size=1")
        if self.use_v2_model_runner:
            raise ValueError("GPU LWD edge currently requires the V1 model runner")
        if self.vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError(
                "GPU LWD edge requires BF16 model dtype, got "
                f"{self.vllm_config.model_config.dtype}"
            )

        lwd_init_payload_scrambler(lwd_config)
        super().init_device()
        self.model_runner = LwdEdgeGPUModelRunner(
            self.vllm_config,
            self.device,
            worker=self,
        )
        lwd_init_host_data_plane("cuda")
        self.comm_service = lwd_get_data_plane()
        lwd_logger.info("[lwd-edge] GPU worker ready with host UCX data plane")

    def shutdown(self) -> None:
        try:
            self.comm_service.shutdown(timeout=5.0)
            lwd_destroy_host_channels()
        finally:
            super().shutdown()

    def compile_or_warm_up_model(self) -> CompilationTimes:
        lwd_logger.info("[lwd-edge] skip full-model warmup and graph capture")
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        self.comm_service.poll_completions()
        if scheduler_output.finished_req_ids:
            self.model_runner.lwd_release_finished(
                set(scheduler_output.finished_req_ids)
            )
        lwd_batch = scheduler_output.lwd_batch
        if lwd_batch is None:
            return self.model_runner._lwd_watermarked_output()

        batch_meta = lwd_batch.batch_meta
        if lwd_batch.batch_type == LwdBatchType.LWD_EMBED:
            return self.model_runner.execute_lwd_embed(
                lwd_batch.seqno,
                batch_meta,
                scheduler_output,
            )
        if lwd_batch.batch_type == LwdBatchType.LWD_UNEMBED:
            result = self.model_runner.execute_lwd_unembed(lwd_batch.seqno, batch_meta)
            LwdControlDiag.trace_worker_missing_rows(
                lwd_batch.seqno,
                list(batch_meta.req_ids),
                list(batch_meta.num_accept_tokens),
                list(getattr(result, "req_ids", None) or []),
                [list(t) for t in (getattr(result, "sampled_token_ids", None) or [])],
            )
            return result
        return self.model_runner._lwd_watermarked_output()
