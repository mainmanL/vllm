# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge routing tie-break: ties start at cloud ``edge_id % cloud_count``."""

from __future__ import annotations

from types import SimpleNamespace

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
    LwdCloudDpAccount,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
    _lwd_first_connection,
)


def _scheduler(edge_id: int) -> LwdEdgeScheduler:
    connections = tuple(
        LwdConnectionId(edge_id, cloud_id, dp_idx)
        for cloud_id in (0, 1)
        for dp_idx in (0, 1)
    )
    scheduler = object.__new__(LwdEdgeScheduler)
    scheduler._lwd_connections = connections
    scheduler._lwd_accounts = {
        connection_id: LwdCloudDpAccount(connection_id) for connection_id in connections
    }
    scheduler._lwd_last_engine = _lwd_first_connection(connections, edge_id) - 1
    scheduler._lwd_edge_share = False
    scheduler._lwd_instance_id = edge_id
    scheduler._lwd_route = {}
    return scheduler


def _request(edge_id: int) -> SimpleNamespace:
    return SimpleNamespace(request_id=f"req-{edge_id}", client_index=edge_id)


def test_probe_tie_staggers_cloud_and_dp_by_edge_id() -> None:
    for edge_id in range(4):
        scheduler = _scheduler(edge_id)
        hits = dict.fromkeys(scheduler._lwd_connections, 0)
        winner = scheduler.lwd_select_probe_target(_request(edge_id), hits)
        assert winner == LwdConnectionId(edge_id, edge_id % 2, edge_id // 2)


def test_route_tie_continues_after_the_last_served_connection() -> None:
    scheduler = _scheduler(1)
    scheduler._lwd_last_engine = 1  # cloud 0, dp 1 served last
    routed = scheduler._lwd_route_request("req", 1)
    assert routed == LwdConnectionId(1, 1, 0)
