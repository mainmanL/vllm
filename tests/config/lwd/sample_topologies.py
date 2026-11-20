# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed reference layouts, independent of the production generator."""

from copy import deepcopy
from typing import Any


def _instance(
    identity: int,
    address: str,
    ranks: list[list[int]],
    ports: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "id": identity,
        "dp": [
            {
                "dp_idx": index,
                "addr": address,
                "ranks": group,
                **(
                    {
                        "ctrl_port": ports[index],
                        "coord_port": 8100 + index,
                        "coord_url": f"http://{address}:{8100 + index}",
                    }
                    if ports is not None
                    else {}
                ),
            }
            for index, group in enumerate(ranks)
        ],
    }


def _payload(
    scene: str,
    world: int,
    edges: list[dict[str, Any]],
    clouds: list[dict[str, Any]],
    links: list[tuple[int, int]],
) -> dict[str, Any]:
    payload = {
        "deployment": {
            "mode": 0,
            "scene": scene,
            "hccl_world_size": world,
            "edges_num": len(edges),
            "clouds_num": len(clouds),
        },
        "feature_ctrl": {
            "enable_early_recv": False,
            "scramble": {"enable": False, "device_id": None, "custom_factor": None},
        },
        "edges": edges,
        "clouds": clouds,
    }
    if scene == "edge_share":
        payload["instance_links"] = [
            {"edge": edge, "cloud": cloud} for edge, cloud in links
        ]
    if scene in ("cloud_share", "lwd_cluster"):
        for edge in edges:
            edge["coordination"] = {
                "enabled": True,
                "tenant_key_file": "/run/secrets/edge-cloud-tenant-key",
                "connect_timeout": 5.0,
            }
            if scene == "cloud_share":
                edge["coordination"]["consumer_id"] = "enterprise-a"
        for cloud in clouds:
            cloud["coordination"] = {"enabled": True, "listen_host": "0.0.0.0"}
    return payload


# Only repeated field names are assembled; ranks, ports and links stay explicit.
PAYLOADS: dict[str, dict[str, Any]] = {
    "s1_single_instance_1dp.yaml": _payload(
        "single_instance",
        9,
        [
            _instance(0, "10.0.0.1", [[0]]),
        ],
        [
            _instance(0, "10.1.0.1", [[1, 2, 3, 4, 5, 6, 7, 8]], [5550]),
        ],
        [(0, 0)],
    ),
    "s1_single_instance_2dp.yaml": _payload(
        "single_instance",
        9,
        [
            _instance(0, "10.0.0.1", [[0], [0]]),
        ],
        [
            _instance(0, "10.1.0.1", [[1, 2, 3, 4], [5, 6, 7, 8]], [5550, 5551]),
        ],
        [(0, 0)],
    ),
    "s2_edge_share_1dp.yaml": _payload(
        "edge_share",
        17,
        [
            _instance(0, "10.0.0.1", [[0]]),
            _instance(1, "10.0.0.1", [[0]]),
        ],
        [
            _instance(0, "10.1.0.1", [[1, 2, 3, 4, 5, 6, 7, 8]], [5550]),
            _instance(1, "10.1.0.2", [[9, 10, 11, 12, 13, 14, 15, 16]], [5550]),
        ],
        [(0, 0), (1, 1)],
    ),
    "s2_edge_share_2dp.yaml": _payload(
        "edge_share",
        17,
        [
            _instance(0, "10.0.0.1", [[0], [0]]),
            _instance(1, "10.0.0.1", [[0], [0]]),
        ],
        [
            _instance(0, "10.1.0.1", [[1, 2, 3, 4], [5, 6, 7, 8]], [5550, 5551]),
            _instance(1, "10.1.0.2", [[9, 10, 11, 12], [13, 14, 15, 16]], [5550, 5551]),
        ],
        [(0, 0), (1, 1)],
    ),
    "s3_cloud_share_1dp.yaml": _payload(
        "cloud_share",
        10,
        [
            _instance(0, "10.0.0.1", [[0]]),
            _instance(1, "10.0.0.2", [[1]]),
        ],
        [
            _instance(0, "10.1.0.1", [[2, 3, 4, 5, 6, 7, 8, 9]], [5550]),
        ],
        [(0, 0), (1, 0)],
    ),
    "s3_cloud_share_2dp.yaml": _payload(
        "cloud_share",
        10,
        [
            _instance(0, "10.0.0.1", [[0], [0]]),
            _instance(1, "10.0.0.2", [[1], [1]]),
        ],
        [
            _instance(0, "10.1.0.1", [[2, 3, 4, 5], [6, 7, 8, 9]], [5550, 5551]),
        ],
        [(0, 0), (1, 0)],
    ),
    "s4_lwd_cluster_1dp.yaml": _payload(
        "lwd_cluster",
        18,
        [
            _instance(0, "10.0.0.1", [[0]]),
            _instance(1, "10.0.0.2", [[1]]),
        ],
        [
            _instance(0, "10.1.0.1", [[2, 3, 4, 5, 6, 7, 8, 9]], [5550]),
            _instance(1, "10.1.0.2", [[10, 11, 12, 13, 14, 15, 16, 17]], [5550]),
        ],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ),
    "s4_lwd_cluster_2dp.yaml": _payload(
        "lwd_cluster",
        18,
        [
            _instance(0, "10.0.0.1", [[0], [0]]),
            _instance(1, "10.0.0.2", [[1], [1]]),
        ],
        [
            _instance(0, "10.1.0.1", [[2, 3, 4, 5], [6, 7, 8, 9]], [5550, 5551]),
            _instance(
                1, "10.1.0.2", [[10, 11, 12, 13], [14, 15, 16, 17]], [5550, 5551]
            ),
        ],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ),
}


def topology_payload(name: str) -> dict[str, Any]:
    return deepcopy(PAYLOADS[name])
