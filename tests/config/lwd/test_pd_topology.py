# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host PD topology contracts, executable without torch or an accelerator."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from lwd_config_under_test import (
    LwdConfig,
    LwdConnectionId,
    LwdTopology,
    resolve_lwd_config,
)


def pd_payload(edges: int = 1) -> dict:
    return {
        "deployment": {
            "mode": 0,
            "scene": "lwd_cluster",
            "edges_num": edges,
            "clouds_num": 2,
            "data_plane_comm": "host",
            "data_plane_host": {"up_buffer_size": 64},
            "pd": {"enabled": True},
        },
        "feature_ctrl": {},
        "edges": [
            {
                "id": edge,
                "dp": [{"dp_idx": 0, "addr": f"10.0.0.{edge + 1}"}],
                "coordination": {
                    "enabled": True,
                    "control_url": "http://10.2.0.1:8200/v1/chat/completions",
                    "control_url_map": {
                        0: "http://10.1.0.1:8100",
                        1: "http://10.1.0.2:8100",
                    },
                    "tenant_key_file": "/run/secrets/tenant",
                    "consumer_id": f"edge-{edge}",
                },
            }
            for edge in range(edges)
        ],
        "clouds": [
            {
                "id": cloud,
                "stage_role": stage,
                "dp": [
                    {
                        "dp_idx": dp,
                        "addr": f"10.1.0.{cloud + 1}",
                        "ctrl_port": 6000 + dp,
                        "data_port": 7000 + dp,
                    }
                    for dp in range(size)
                ],
                "coordination": {"enabled": True},
            }
            for cloud, stage, size in [(0, "prefill", 2), (1, "decode", 4)]
        ],
        "instance_links": [
            {"edge": edge, "cloud": cloud}
            for edge in range(edges)
            for cloud in range(2)
        ],
    }


def test_host_edge_reaches_every_asymmetric_stage_engine() -> None:
    topology = LwdTopology.from_dict(pd_payload(edges=2))
    for edge in topology.edges:
        connections = topology.connections("edge", edge.id)
        assert {c.connection_id for c in connections} == {
            LwdConnectionId(edge.id, cloud, dp)
            for cloud, size in [(0, 2), (1, 4)]
            for dp in range(size)
        }
        assert all(c.edge is edge.dp[0] for c in connections)
        assert all(c.cloud.dp_idx == c.connection_id.dp_idx for c in connections)
    assert len(topology.connections("cloud", 1)) == 8


def test_entry_exposes_stage_and_pd_limits_from_its_topology() -> None:
    raw = pd_payload()
    raw["deployment"]["pd"]["max_window_tokens"] = 128
    topology = LwdTopology.from_dict(raw)
    entry = LwdConfig.from_topology("unused.yaml", "cloud", 1, topology)
    assert entry.stage_role == "decode"
    assert entry.pd.enabled
    assert entry.pd.max_window_tokens == 128
    assert len(entry.connections) == 4


@pytest.mark.parametrize(
    "fault,match",
    [
        ("device", "requires data_plane_comm=host"),
        ("missing_decode", "separate prefill and decode"),
        ("edge_stage", "cloud instances only"),
        ("disabled", "requires deployment.pd.enabled"),
        ("no_coordination", "coordination"),
        ("no_proxy", "single Proxy control_url"),
        ("no_tenant", "tenant_id or edge consumer_id"),
        ("auto_up_buffer", "independent edge/P/D token budgets"),
        ("no_host_plane", "explicit.*up_buffer_size"),
        ("invalid_window", "greater than or equal to 1"),
        ("invalid_timeout", "greater than 0"),
    ],
)
def test_pd_rejects_invalid_deployment(fault: str, match: str) -> None:
    raw = pd_payload()
    if fault == "device":
        # Reject PD before device rank or symmetric-DP validation obscures it.
        raw["deployment"]["data_plane_comm"] = "device"
    elif fault == "missing_decode":
        raw["clouds"][1]["stage_role"] = "prefill"
    elif fault == "edge_stage":
        raw["edges"][0]["stage_role"] = "prefill"
    elif fault == "disabled":
        raw["deployment"]["pd"]["enabled"] = False
    elif fault == "no_coordination":
        raw["clouds"][1]["coordination"]["enabled"] = False
    elif fault == "no_proxy":
        raw["edges"][0]["coordination"].pop("control_url")
    elif fault == "no_tenant":
        raw["edges"][0]["coordination"].pop("consumer_id")
    elif fault == "auto_up_buffer":
        raw["deployment"]["data_plane_host"].pop("up_buffer_size")
    elif fault == "no_host_plane":
        raw["deployment"].pop("data_plane_host")
    elif fault == "invalid_window":
        raw["deployment"]["pd"]["max_inflight_windows"] = 0
    else:
        raw["deployment"]["pd"]["result_hold_timeout_s"] = 0.0
    with pytest.raises(ValueError, match=match):
        LwdTopology.from_dict(raw)


def test_non_pd_host_keeps_all_cloud_connections() -> None:
    raw = copy.deepcopy(pd_payload())
    raw["deployment"]["pd"]["enabled"] = False
    for cloud in raw["clouds"]:
        cloud["stage_role"] = ""
    topology = LwdTopology.from_dict(raw)
    assert len(topology.connections("edge", 0)) == 6


def test_non_pd_host_retains_automatic_up_buffer_sizing() -> None:
    raw = pd_payload()
    raw["deployment"]["pd"]["enabled"] = False
    raw["deployment"]["data_plane_host"].pop("up_buffer_size")
    for cloud in raw["clouds"]:
        cloud["stage_role"] = ""
    topology = LwdTopology.from_dict(raw)
    assert topology.deployment.data_plane_host.up_buffer_size is None


def _runtime_function(relative: str, name: str, **scope):
    """Run the production function without importing model/device setup."""
    path = Path(__file__).resolve().parents[3] / "vllm" / relative
    source = ast.parse(path.read_text())
    module = ast.parse("from __future__ import annotations")
    module.body.append(
        next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


def test_edge_control_plane_advertises_each_clouds_real_dp_count() -> None:
    topology = LwdTopology.from_dict(pd_payload())
    entry = LwdConfig.from_topology("unused.yaml", "edge", 0, topology)
    build = _runtime_function(
        "v1/lwd_control/lwd_comm/lwd_control_plane_factory.py",
        "_lwd_edge_spec",
        LwdEdgeControlPlaneSpec=SimpleNamespace,
        make_zmq_path=lambda protocol, addr, port: f"{protocol}://{addr}:{port}",
    )
    spec = build(entry, 0, 1)
    assert spec.cloud_dp_sizes == (2, 2, 4, 4, 4, 4)
    assert spec.connection_ids[-1] == LwdConnectionId(0, 1, 3)
    assert spec.cloud_endpoints[-1] == "tcp://10.1.0.2:6003"


@pytest.mark.parametrize(
    "role,identity,dp_size,valid",
    [
        ("edge", 0, 1, True),
        ("edge", 0, 2, False),
        ("cloud", 0, 2, True),
        ("cloud", 1, 4, True),
        ("cloud", 1, 2, False),
    ],
)
def test_host_native_dp_validation_is_instance_local(role, identity, dp_size, valid):
    topology = LwdTopology.from_dict(pd_payload())
    entry = LwdConfig.from_topology("unused.yaml", role, identity, topology)
    check = _runtime_function(
        "distributed/lwd/lwd_parallel_groups.py", "lwd_check_parallel_config"
    )
    parallel = SimpleNamespace(
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_elastic_ep=False,
        data_parallel_size=dp_size,
    )
    config = SimpleNamespace(
        lwd_config=entry, parallel_config=parallel, engram_config=None
    )
    if valid:
        check(config)
    else:
        with pytest.raises(ValueError, match="data parallel size"):
            check(config)


def test_shipped_host_p2_d4_example_and_all_process_entries_are_loadable():
    examples = Path(__file__).resolve().parents[3] / "examples/lwd"
    path = examples / "pd_host_2p4d.yaml"
    topology = LwdTopology.from_file(str(path))
    assert [len(cloud.dp) for cloud in topology.clouds] == [2, 4]
    assert len(topology.connections("edge", 0)) == 6
    for name, role, stage in [
        ("edge", "edge", ""),
        ("prefill", "cloud", "prefill"),
        ("decode", "cloud", "decode"),
    ]:
        raw = json.loads((examples / f"pd_{name}_config.json").read_text())
        raw["lwd_config"]["path"] = str(path)
        entry = resolve_lwd_config(raw)
        assert entry.pd.enabled and entry.role == role and entry.stage_role == stage
