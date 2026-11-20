# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import ast
import copy
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from lwd_config_under_test import (
    LwdAPIConfig,
    LwdConfig,
    LwdCoordinationConfig,
    resolve_lwd_config,
)

FIXTURE_NAME = "s3_cloud_share_2dp.yaml"
StartupEntry = tuple[Callable[[], None], SimpleNamespace, Mock]


def coordinated_topology(lwd_yaml_dir: Path, name: str) -> dict:
    return yaml.safe_load((lwd_yaml_dir / name).read_text())


def _startup_entry(param: tuple[str, str, str]) -> StartupEntry:
    path, class_name, method_name = param
    source = Path(__file__).resolve().parents[3] / "vllm" / path
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    # Execute the real entry method, stopping before unrelated model/device setup.
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    stop = Mock(side_effect=StopIteration)
    resolver = Mock(wraps=resolve_lwd_config)
    scope = {
        "time": time,
        "logger": Mock(),
        "log_lwd_config": Mock(),
        "resolve_lwd_config": resolver,
        "LwdAPIConfig": LwdAPIConfig,
        "current_platform": SimpleNamespace(pre_register_and_update=stop),
    }
    exec(compile(module, str(source), "exec"), scope)
    args = SimpleNamespace(
        additional_config={},
        lwd_config=None,
        model_config=SimpleNamespace(skip_tokenizer_init=False, multimodal_config=None),
        _resolve_mm_encoder_only=stop,
    )

    def run() -> None:
        with pytest.raises(StopIteration):
            scope[method_name](args)

    return run, args, resolver


@pytest.fixture(
    params=[
        ("config/vllm.py", "VllmConfig", "__post_init__"),
        ("engine/arg_utils.py", "EngineArgs", "create_engine_config"),
    ],
    ids=["VllmConfig", "EngineArgs"],
)
def startup_entry(request: pytest.FixtureRequest) -> StartupEntry:
    return _startup_entry(request.param)


@pytest.fixture
def engine_startup_entry() -> StartupEntry:
    return _startup_entry(("engine/arg_utils.py", "EngineArgs", "create_engine_config"))


@pytest.mark.parametrize("raw", [{}, {"enable_cpu_binding": True}, object()])
def test_centralized_startup_does_not_call_lwd_parser(
    startup_entry: StartupEntry, raw: object
) -> None:
    run, args, resolver = startup_entry
    args.additional_config = raw
    run()
    assert args.lwd_config is None
    resolver.assert_not_called()


@pytest.mark.parametrize("role", ["edge", "cloud"])
def test_explicit_lwd_startup_loads_selected_role(
    startup_entry: StartupEntry, role: str, lwd_yaml_dir: Path, tmp_path: Path
) -> None:
    run, args, resolver = startup_entry
    path = tmp_path / "topology.yaml"
    path.write_bytes((lwd_yaml_dir / "s1_single_instance_1dp.yaml").read_bytes())
    args.additional_config = {
        "lwd_config": {"path": str(path), "role": role, "instance_id": 0}
    }
    run()
    resolver.assert_called_once()
    if args.lwd_config is not None:
        snapshot = args.lwd_config
        assert snapshot.role == role
        path.unlink()
        run()
        assert args.lwd_config is snapshot
        resolver.assert_called_once()


@pytest.mark.parametrize("entry", [None, {}])
def test_explicit_invalid_lwd_startup_is_not_treated_as_centralized(
    startup_entry: StartupEntry, entry: object
) -> None:
    run, args, resolver = startup_entry
    args.additional_config = {"lwd_config": entry}
    with pytest.raises(ValueError):
        run()
    resolver.assert_called_once()


def test_api_attachment_without_lwd_still_fails(
    engine_startup_entry: StartupEntry,
) -> None:
    run, args, resolver = engine_startup_entry
    args.api_server_attach = "10.0.0.1:25100"
    with pytest.raises(ValueError, match="require lwd_config.path"):
        run()
    resolver.assert_not_called()


def test_snapshot_survives_handoff_without_yaml_or_key_access(
    tmp_path: Path, lwd_yaml_dir: Path
) -> None:
    path = tmp_path / "topology.yaml"
    topology = coordinated_topology(lwd_yaml_dir, FIXTURE_NAME)
    topology["edges"][1]["coordination"].update(
        tenant_key_file=str(tmp_path / "does-not-exist"),
        consumer_id="enterprise-a-edge1",
        connect_timeout=5.0,
    )
    path.write_text(yaml.safe_dump(topology, sort_keys=False))
    raw = {
        "lwd_config": {
            "path": str(path),
            "role": "edge",
            "instance_id": 1,
        },
    }
    original = copy.deepcopy(raw)
    snapshot = resolve_lwd_config(raw)
    path.unlink()
    received = pickle.loads(pickle.dumps(snapshot))
    assert received.instance.id == 1
    assert [link.connection_id.dp_idx for link in received.connections] == [0, 1]
    assert received.topology.digest
    assert received.coordination.consumer_id == "enterprise-a-edge1"
    assert received.coordination.connect_timeout == 5.0
    assert not received.api.requested
    assert raw == original


@pytest.mark.parametrize("raw", [{}, {"enable_cpu_binding": True}, None])
def test_absent_lwd_leaves_centralized_configuration_empty(raw: object) -> None:
    assert resolve_lwd_config(raw) is None
    assert resolve_lwd_config(raw, api=LwdAPIConfig()) is None
    with pytest.raises(ValueError, match="require lwd_config.path"):
        resolve_lwd_config(raw, api=LwdAPIConfig(api_server_rpc_port="10.0.0.1:25100"))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"instance_id": 2}, "Instance not found"),
        ({"instance_id": True}, "integer"),
        ({"role": "unknown"}, "role"),
        ({"scheduler": "mixed"}, "scheduler"),
    ],
)
def test_invalid_entry_fails_during_configuration(
    change: dict, match: str, lwd_yaml_dir: Path
) -> None:
    entry = {
        "path": str(lwd_yaml_dir / "s1_single_instance_1dp.yaml"),
        "role": "edge",
        "instance_id": 0,
        **change,
    }
    with pytest.raises(ValueError, match=match):
        resolve_lwd_config({"lwd_config": entry})


def test_legacy_entry_coordination_is_rejected(lwd_yaml_dir: Path) -> None:
    with pytest.raises(ValueError, match="coordination is retired from lwd_config"):
        resolve_lwd_config(
            {
                "lwd_config": {
                    "path": str(lwd_yaml_dir / "s1_single_instance_1dp.yaml"),
                    "role": "edge",
                    "instance_id": 0,
                    "coordination": {"enabled": False},
                }
            }
        )


@pytest.mark.parametrize("with_entry", [False, True])
def test_legacy_top_level_coordination_is_rejected(
    with_entry: bool,
    lwd_yaml_dir: Path,
) -> None:
    raw: dict[str, object] = {"lwd_coordination": {"enabled": False}}
    if with_entry:
        raw["lwd_config"] = {
            "path": str(lwd_yaml_dir / "s1_single_instance_1dp.yaml"),
            "role": "edge",
            "instance_id": 0,
        }
    with pytest.raises(ValueError, match="Top-level lwd_coordination is retired"):
        resolve_lwd_config(raw)


@pytest.mark.parametrize("cloud_id", [0, 1])
def test_cloud_coordination_name_defaults_to_selected_cloud(
    cloud_id: int, lwd_yaml_dir: Path, tmp_path: Path
) -> None:
    supplied = {"enabled": True}
    original = supplied.copy()
    topology = yaml.safe_load((lwd_yaml_dir / "s2_edge_share_1dp.yaml").read_text())
    topology["clouds"][cloud_id]["coordination"] = supplied
    path = tmp_path / "topology.yaml"
    path.write_text(yaml.safe_dump(topology, sort_keys=False))
    entry = {
        "path": str(path),
        "role": "cloud",
        "instance_id": cloud_id,
    }
    selected = resolve_lwd_config({"lwd_config": entry})
    assert selected.coordination.instance_id == f"cloud-{cloud_id}"
    assert supplied == original


def test_cloud_coordination_name_preserves_direct_config_override() -> None:
    coordination = LwdCoordinationConfig.from_dict(
        {"enabled": True, "instance_id": "custom-cloud"}, cloud_id=1
    )
    assert coordination.instance_id == "custom-cloud"


def test_absent_coordination_stays_disabled(lwd_yaml_dir: Path) -> None:
    selected = resolve_lwd_config(
        {
            "lwd_config": {
                "path": str(lwd_yaml_dir / "s1_single_instance_1dp.yaml"),
                "role": "cloud",
                "instance_id": 0,
            }
        }
    )
    assert selected.coordination is None


def _resolve_two_cloud_edge(topology: dict, tmp_path: Path) -> LwdConfig:
    path = tmp_path / "topology.yaml"
    path.write_text(yaml.safe_dump(topology, sort_keys=False))
    selected = resolve_lwd_config(
        {
            "lwd_config": {
                "path": str(path),
                "role": "edge",
                "instance_id": 0,
            }
        }
    )
    assert selected is not None
    return selected


def test_two_cloud_edge_projects_dp_endpoints_with_coordination_defaults(
    lwd_yaml_dir: Path, tmp_path: Path
) -> None:
    topology = coordinated_topology(lwd_yaml_dir, "s4_lwd_cluster_1dp.yaml")
    selected = _resolve_two_cloud_edge(topology, tmp_path)
    coordination = selected.coordination
    assert coordination is not None
    assert coordination.connect_timeout == 5.0
    assert coordination.probe_timeout == 25.0
    assert [connection.cloud.coord_url for connection in selected.connections] == [
        "http://10.1.0.1:8100",
        "http://10.1.0.2:8100",
    ]


def test_edge_coordination_no_longer_requires_a_url() -> None:
    coordination = LwdCoordinationConfig(
        enabled=True, tenant_key_file="/run/secrets/edge-cloud-tenant-key"
    )
    coordination.validate_role("edge")


def test_edge_coordination_still_requires_tenant_unless_placement_only() -> None:
    with pytest.raises(ValueError, match="tenant_key_file"):
        LwdCoordinationConfig(enabled=True).validate_role("edge")
    LwdCoordinationConfig(enabled=True, placement_only=True).validate_role("edge")


def test_probe_poll_budget_config_validation() -> None:
    assert LwdCoordinationConfig().probe_poll_budget == 32
    assert LwdCoordinationConfig(probe_poll_budget=2).probe_poll_budget == 2
    with pytest.raises(ValueError):
        LwdCoordinationConfig(probe_poll_budget=0)


@pytest.mark.parametrize("field", ["control_url", "control_url_map", "listen_port"])
def test_legacy_coordination_config_fields_are_rejected(field: str) -> None:
    value = "http://127.0.0.1:8100" if field == "control_url" else 8100
    if field == "control_url_map":
        value = {0: "http://127.0.0.1:8100"}
    with pytest.raises(ValueError, match="coord_port/coord_url"):
        LwdCoordinationConfig.from_dict({"enabled": True, field: value})


@pytest.mark.parametrize(
    "identity,flag", [(0, "--api-server-rpc-port"), (1, "--api-server-attach")]
)
def test_api_cli_metadata_survives_snapshot_handoff(
    tmp_path: Path, identity: int, flag: str, lwd_yaml_dir: Path
) -> None:
    parser = argparse.ArgumentParser()
    LwdAPIConfig.add_cli_args(parser)
    args = parser.parse_args([flag, "10.0.0.1:25100"])
    path = tmp_path / "topology.yaml"
    args.additional_config = {
        "lwd_config": {
            "path": str(path),
            "role": "edge",
            "instance_id": identity,
        }
    }
    # Capturing CLI options does not read YAML; snapshot construction owns that read.
    api = LwdAPIConfig.from_namespace(args)
    path.write_bytes((lwd_yaml_dir / "s2_edge_share_2dp.yaml").read_bytes())
    original = copy.deepcopy(args.additional_config)
    entry = resolve_lwd_config(args.additional_config, api=api)
    path.unlink()
    # Only round-trip our own objects, never externally supplied pickle data.
    received_entry = pickle.loads(pickle.dumps(entry))
    assert received_entry.api == api
    assert received_entry.coordination is None
    assert [c.connection_id.dp_idx for c in received_entry.connections] == [0, 1, 0, 1]
    assert received_entry.instance_id == identity
    assert args.additional_config == original
