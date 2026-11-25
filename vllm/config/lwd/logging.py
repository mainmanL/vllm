# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Log the resolved LWD metadata without reading files or starting execution."""

import json
import logging
from dataclasses import asdict

from .entry import LwdConfig


def _qos_line(config: LwdConfig) -> str:
    """Render the effective QoS of this deployment on one line.

    Defaults are already resolved into the topology, so the line shows what
    the plane will actually carry (the DSCP/ECN split makes the RoCE traffic
    class readable) rather than what the YAML spelled.
    """
    deployment = config.topology.deployment
    comm = deployment.data_plane_comm
    qos = config.topology.qos
    if qos is None:
        return f"comm={comm} unset (lwd marks nothing; HCCL keeps its own TC/SL)"
    data_plane = qos.data_plane
    if data_plane is None:
        data_plane_text = "data_plane=unset"
    else:
        traffic_class = data_plane.traffic_class
        traffic_class_text = (
            "ucx-default"
            if traffic_class is None
            else f"{traffic_class} (dscp={traffic_class >> 2} ecn={traffic_class & 3})"
        )
        sl = data_plane.sl
        data_plane_text = (
            f"data_plane traffic_class={traffic_class_text} "
            f"sl={'ucx-default' if sl is None else sl}"
        )
    ctrl_plane = qos.ctrl_plane
    dscp = ctrl_plane.dscp if ctrl_plane is not None else None
    ctrl_plane_text = (
        "ctrl_plane=unmarked"
        if dscp is None
        else f"ctrl_plane dscp={dscp} (tos={dscp << 2})"
    )
    return f"comm={comm} {data_plane_text} | {ctrl_plane_text}"


def log_lwd_config(logger: logging.Logger, config: LwdConfig) -> None:
    """Emit one startup record with all fields, defaults and selected connections."""
    logger.info("[LWD][qos] %s", _qos_line(config))
    # Rank-derived fields exist only in device comm (shared HCCL world);
    # host comm declares no ranks, so they are logged as None instead of
    # being derived.
    is_device = not config.topology.is_host_comm
    snapshot = {
        "metadata_only": True,
        "lwd_config": asdict(config),
        "selected_instance": asdict(config.instance),
        "connections": [
            {
                **asdict(link),
                "cloud_leader_rank": (
                    link.cloud_leader_rank if is_device and link.cloud.ranks else None
                ),
            }
            for link in config.connections
        ],
        "cloud_cards": {
            str(cloud.id): (
                sum(len(dp.ranks) for dp in cloud.dp) if is_device else None
            )
            for cloud in config.topology.clouds
        },
    }
    # Serialize the parsed snapshot only; tenant_key_file remains a path, not content.
    logger.info("[LWD][config] resolved=%s", json.dumps(snapshot, ensure_ascii=False))
