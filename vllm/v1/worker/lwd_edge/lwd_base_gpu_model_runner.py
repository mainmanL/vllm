# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Common base for the LWD edge/cloud model runners.

Owns what both data-plane runners share: the worker back-reference
(the comm service and the LWD config live on the worker) and the named
lazy staging factory that three call sites used to hand-roll.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.v1.lwd_comm.lwd_pinned_staging import LwdPinnedStaging
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.config import VllmConfig

if TYPE_CHECKING:
    from .lwd_edge_worker import LwdEdgeWorker


class LwdBaseGPUModelRunner(GPUModelRunner):
    """Shared GPU runner infrastructure for the edge/cloud LWD runners."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        worker: LwdEdgeWorker | None = None,
    ) -> None:
        super().__init__(vllm_config, device)
        self.worker = worker  # LwdWorkerBase ref (comm service lives there)
        # Named lazy staging rings; see _lwd_staging.
        self._lwd_staging_rings: dict[str, LwdPinnedStaging] = {}

    def _lwd_staging(
        self,
        name: str,
        *,
        dtype: torch.dtype,
        min_capacity: int,
        device_mirror: bool = True,
    ) -> LwdPinnedStaging:
        """Named lazy pinned-staging ring; the single home of the
        create-once pattern that edge/cloud call sites used to duplicate.

        ``device_mirror=False`` yields a pinned-only ring (D2H staging);
        the ring semantics (depth 4, capacity growth, rebuild-restart)
        live in LwdPinnedStaging and are locked by its table-driven tests.
        """
        ring = self._lwd_staging_rings.get(name)
        if ring is None:
            ring = LwdPinnedStaging(
                dtype=dtype,
                min_capacity=min_capacity,
                device=self.device if device_mirror else None,
            )
            self._lwd_staging_rings[name] = ring
        return ring
