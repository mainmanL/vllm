# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests for direct-DP cloud HTTP coordination."""

from __future__ import annotations

import asyncio
import queue
import time
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
import pytest_asyncio

from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control import (
    LwdCloudControlBridge,
    LwdCloudControlProcessor,
    create_lwd_cloud_control_app,
    start_lwd_cloud_control_server,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    HEADER_CLOUD_ID,
    HEADER_EDGE_ID,
    HEADER_ENGINE_INDEX,
    HEADER_PROBE_NONCE,
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)

BridgeFactory = Callable[
    [int],
    tuple[LwdCloudControlBridge, list[queue.Queue[Any]], queue.Queue[Any]],
]


@pytest_asyncio.fixture
async def bridge_factory() -> AsyncIterator[BridgeFactory]:
    bridges: list[LwdCloudControlBridge] = []

    def make_bridge(
        engine_count: int,
    ) -> tuple[LwdCloudControlBridge, list[queue.Queue[Any]], queue.Queue[Any]]:
        command_queues = [queue.Queue() for _ in range(engine_count)]
        event_queue: queue.Queue[Any] = queue.Queue()
        bridge = LwdCloudControlBridge(command_queues, event_queue)
        bridge.start()
        bridges.append(bridge)
        return bridge, command_queues, event_queue

    yield make_bridge

    for bridge in bridges:
        bridge.close()
        bridge._dispatch_thread.join(timeout=1)
        assert not bridge._dispatch_thread.is_alive()


def _manifest(request_id: str) -> LwdPrefixManifest:
    return LwdPrefixManifest(
        request_id=request_id,
        prompt_tokens=16,
        block_size=16,
        full_block_hashes=(bytes(range(32)),),
    )


def _result(request_id: str, hit_blocks: int, engine_index: int = 0) -> LwdProbeResult:
    return LwdProbeResult(
        request_id=request_id,
        instance_id="cloud-test",
        block_size=16,
        hit_blocks=hit_blocks,
        hit_tokens=hit_blocks * 16,
        cloud_id=0,
        engine_index=engine_index,
    )


def _usage_event(
    request_id: str,
    usage: LwdUsage,
    *,
    engine_index: int = 0,
    prefix_nonce: int = 7,
    kind: str = "usage",
) -> dict[str, Any]:
    return {
        "type": kind,
        "request_id": request_id,
        "usage": usage,
        "engine_index": engine_index,
        "prefix_nonce": prefix_nonce,
    }


async def _start_probe(
    bridge: LwdCloudControlBridge,
    command_queue: queue.Queue[Any],
    request_id: str,
    dp_idx: int = 0,
    reply_key: str | None = None,
    timeout: float = 1.0,
) -> tuple[asyncio.Task[LwdProbeResult], dict[str, Any]]:
    key = reply_key or f"{request_id}#dp{dp_idx}"
    task = asyncio.create_task(
        bridge.probe(dp_idx, _manifest(request_id), reply_key=key, timeout=timeout)
    )
    await asyncio.sleep(0)
    command = command_queue.get_nowait()
    assert command["type"] == "probe"
    assert command["reserve"] is False
    assert command["engine_index"] == dp_idx
    assert command["reply_key"] == key
    assert command["deadline"] == command["t_cmd"] + timeout
    return task, command


def _reply(
    event_queue: queue.Queue[Any],
    reply_key: str,
    engine_index: int,
    result: LwdProbeResult | None = None,
    error: str | None = None,
) -> None:
    event_queue.put(
        {
            "type": "probe",
            "request_id": reply_key,
            "engine_index": engine_index,
            "ok": result is not None,
            "result": result,
            "error": error,
        }
    )


@pytest.mark.asyncio
async def test_same_request_concurrent_dps_use_independent_reply_keys(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(2)
    probe0, _ = await _start_probe(
        bridge, command_queues[0], "shared", dp_idx=0, reply_key="shared#0"
    )
    probe1, _ = await _start_probe(
        bridge, command_queues[1], "shared", dp_idx=1, reply_key="shared#1"
    )
    assert not probe0.done() and not probe1.done()
    assert all(command_queue.empty() for command_queue in command_queues)

    _reply(event_queue, "shared#1", 1, _result("shared", 1, 1))
    _reply(event_queue, "shared#0", 0, _result("shared", 0, 0))
    assert (await probe1).engine_index == 1
    assert (await probe0).engine_index == 0
    assert bridge.registration("shared") is None


@pytest.mark.asyncio
async def test_probe_timeout_cleans_only_probe_and_drops_late_reply(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(1)
    probe_task, _ = await _start_probe(
        bridge, command_queues[0], "timeout", reply_key="timeout#0", timeout=0.01
    )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(probe_task, timeout=1)
    assert not bridge._probe_futures

    _reply(event_queue, "timeout#0", 0, _result("timeout", 1))
    await asyncio.sleep(0.15)
    assert not bridge._probe_futures


@pytest.mark.asyncio
async def test_activate_owns_usage_and_old_release_preserves_new_generation(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, _, event_queue = bridge_factory(1)
    old_registration = bridge.activate("usage", 0, 7)
    new_registration = bridge.activate("usage", 1, 8)
    assert (
        await bridge.wait_usage("usage", timeout=1, registration=old_registration)
        is None
    )
    bridge.release("usage", old_registration)
    assert bridge.registration("usage") is new_registration

    expected = LwdUsage(prompt_tokens=16, completion_tokens=4, cached_tokens=8)
    event_queue.put(
        _usage_event(
            "usage", expected, engine_index=1, prefix_nonce=8, kind="usage_start"
        )
    )
    event_queue.put(_usage_event("usage", expected, engine_index=1, prefix_nonce=8))
    assert (
        await bridge.wait_usage("usage", timeout=1, registration=new_registration)
        is expected
    )
    assert bridge.registration("usage") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("old_dp", "old_nonce"), [(0, 8), (1, 7)])
async def test_usage_rejects_mismatched_generation(
    bridge_factory: BridgeFactory, old_dp: int, old_nonce: int
) -> None:
    bridge, _, event_queue = bridge_factory(1)
    registration = bridge.activate("generation", 1, 8)
    old_usage = LwdUsage(prompt_tokens=10, completion_tokens=1, cached_tokens=0)
    expected = LwdUsage(prompt_tokens=16, completion_tokens=4, cached_tokens=8)
    waiter = asyncio.create_task(
        bridge.wait_usage("generation", timeout=1, registration=registration)
    )
    event_queue.put(
        _usage_event(
            "generation",
            old_usage,
            engine_index=old_dp,
            prefix_nonce=old_nonce,
            kind="usage_start",
        )
    )
    event_queue.put(
        _usage_event(
            "generation", old_usage, engine_index=old_dp, prefix_nonce=old_nonce
        )
    )
    await asyncio.sleep(0.05)
    assert not waiter.done()

    event_queue.put(
        _usage_event(
            "generation", expected, engine_index=1, prefix_nonce=8, kind="usage_start"
        )
    )
    event_queue.put(
        _usage_event("generation", expected, engine_index=1, prefix_nonce=8)
    )
    assert await waiter is expected


@pytest.mark.asyncio
async def test_usage_admission_timeout_releases_registration(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, _, _ = bridge_factory(1)
    registration = bridge.activate("admission", 0, 7)
    assert (
        await bridge.wait_usage("admission", timeout=0.01, registration=registration)
        is None
    )
    assert bridge.registration("admission") is None


@pytest.mark.asyncio
async def test_started_usage_wait_has_no_total_timeout(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, _, event_queue = bridge_factory(1)
    registration = bridge.activate("long", 0, 7)
    expected = LwdUsage(prompt_tokens=16, completion_tokens=4, cached_tokens=8)
    event_queue.put(_usage_event("long", expected, kind="usage_start"))
    await asyncio.sleep(0.02)
    waiter = asyncio.create_task(
        bridge.wait_usage("long", timeout=0.01, registration=registration)
    )
    await asyncio.sleep(0.02)
    assert not waiter.done()
    event_queue.put(_usage_event("long", expected))
    assert await waiter is expected


@pytest.mark.asyncio
async def test_probe_cancellation_does_not_remove_activation(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, _ = bridge_factory(1)
    registration = bridge.activate("cancel", 0, 7)
    probe_task, _ = await _start_probe(
        bridge, command_queues[0], "cancel", reply_key="cancel#0"
    )
    probe_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe_task
    assert not bridge._probe_futures
    assert bridge.registration("cancel") is registration
    bridge.release("cancel")


def test_processor_always_peeks_and_echoes_dp_reply_key() -> None:
    pending_queue: queue.Queue[Any] = queue.Queue()
    event_queue: queue.Queue[Any] = queue.Queue()
    manifest = _manifest("r")
    pending_queue.put(
        {
            "type": "probe",
            "manifest": manifest,
            "reserve": True,
            "reply_key": "r#nonce:7:0:1",
            "engine_index": 1,
        }
    )
    coordinator = Mock()
    coordinator.peek.return_value = _result("r", 1, 1)

    assert LwdCloudControlProcessor(pending_queue, event_queue).poll(coordinator) == 1

    coordinator.peek.assert_called_once_with(manifest)
    coordinator.probe.assert_not_called()
    event = event_queue.get_nowait()
    assert event["request_id"] == "r#nonce:7:0:1"
    assert event["engine_index"] == 1
    assert event["result"] is coordinator.peek.return_value


def test_processor_skips_expired_probe_without_peek() -> None:
    pending_queue: queue.Queue[Any] = queue.Queue()
    event_queue: queue.Queue[Any] = queue.Queue()
    manifest = _manifest("expired")
    now = time.monotonic()
    for index, deadline in enumerate((now - 1, now + 60)):
        pending_queue.put(
            {
                "type": "probe",
                "manifest": manifest,
                "reply_key": f"expired#{index}",
                "engine_index": 0,
                "deadline": deadline,
            }
        )
    coordinator = Mock()
    coordinator.peek.return_value = _result("expired", 0, 0)

    assert LwdCloudControlProcessor(pending_queue, event_queue).poll(coordinator) == 2

    coordinator.peek.assert_called_once_with(manifest)
    expired = event_queue.get_nowait()
    assert not expired["ok"] and expired["error"] == "probe command expired"
    assert event_queue.get_nowait()["ok"]


def test_processor_poll_budget_preserves_remaining_commands() -> None:
    pending_queue: queue.Queue[Any] = queue.Queue()
    event_queue: queue.Queue[Any] = queue.Queue()
    manifest = _manifest("budget")
    for index in range(33):
        pending_queue.put(
            {
                "type": "probe",
                "manifest": manifest,
                "reply_key": f"budget#{index}",
                "engine_index": 0,
                "deadline": time.monotonic() + 60,
            }
        )
    coordinator = Mock()
    coordinator.peek.return_value = _result("budget", 0, 0)
    processor = LwdCloudControlProcessor(pending_queue, event_queue)

    assert processor.poll(coordinator) == 32
    assert pending_queue.qsize() == 1
    assert event_queue.qsize() == 32
    assert processor.poll(coordinator) == 1
    assert pending_queue.empty()
    assert event_queue.qsize() == 33


def test_processor_custom_poll_budget_preserves_remaining_commands() -> None:
    pending_queue: queue.Queue[Any] = queue.Queue()
    event_queue: queue.Queue[Any] = queue.Queue()
    manifest = _manifest("custom-budget")
    for index in range(3):
        pending_queue.put(
            {
                "type": "probe",
                "manifest": manifest,
                "reply_key": f"custom-budget#{index}",
                "engine_index": 0,
                "deadline": time.monotonic() + 60,
            }
        )
    coordinator = Mock()
    coordinator.peek.return_value = _result("custom-budget", 0, 0)
    processor = LwdCloudControlProcessor(pending_queue, event_queue, poll_budget=2)

    assert processor.poll(coordinator) == 2
    assert pending_queue.qsize() == 1
    assert event_queue.qsize() == 2
    assert processor.poll(coordinator) == 1
    assert pending_queue.empty()
    assert event_queue.qsize() == 3


def test_probe_result_requires_and_serializes_explicit_dp_identity() -> None:
    result = _result("r", 1, 0)
    headers = result.to_headers()
    message = result.to_control_message()
    assert headers[HEADER_ENGINE_INDEX] == "0"
    assert message["engine_index"] == 0
    assert message["cloud_id"] == 0
    assert LwdProbeResult.from_headers(headers) == result
    assert LwdProbeResult.from_control_message(message) == result

    missing_engine = {
        key: value for key, value in message.items() if key != "engine_index"
    }
    with pytest.raises(ValueError, match="engine_index"):
        LwdProbeResult.from_control_message(missing_engine)
    missing_cloud = {
        key: value for key, value in headers.items() if key != HEADER_CLOUD_ID
    }
    with pytest.raises(ValueError, match="invalid integers"):
        LwdProbeResult.from_headers(missing_cloud)


def _http_request(request_id: str = "r") -> tuple[dict[str, str], dict[str, Any]]:
    manifest = _manifest(request_id)
    headers = {
        **manifest.to_headers(),
        HEADER_EDGE_ID: "2",
        HEADER_PROBE_NONCE: "7",
        HEADER_CLOUD_ID: "0",
        HEADER_ENGINE_INDEX: "0",
    }
    return headers, {"model": "test-model", "messages": manifest.to_messages()}


@pytest.mark.asyncio
async def test_probe_http_returns_control_json_and_forces_dp_identity(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(2)
    app = create_lwd_cloud_control_app(bridge, cloud_id=0, dp_idx=1)
    transport = httpx.ASGITransport(app=app)
    headers, body = _http_request()
    headers[HEADER_CLOUD_ID] = "0"
    headers[HEADER_ENGINE_INDEX] = "1"
    async with httpx.AsyncClient(
        transport=transport, base_url="http://cloud"
    ) as client:
        response_task = asyncio.create_task(
            client.post("/v1/lwd/probe", headers=headers, json=body)
        )
        while command_queues[1].empty():
            await asyncio.sleep(0)
        command = command_queues[1].get_nowait()
        assert command_queues[0].empty()
        event_queue.put(
            {
                "type": "probe",
                "request_id": command["reply_key"],
                "engine_index": 1,
                "ok": True,
                "result": _result("e2-r", 1, 0),
            }
        )
        response = await response_task

    assert response.status_code == 200
    assert response.json()["type"] == "edge_cloud_probe"
    assert response.json()["request_id"] == "r"
    assert response.json()["cloud_id"] == 0
    assert response.json()["engine_index"] == 1
    assert bridge.registration("e2-r") is None


@pytest.mark.asyncio
async def test_activate_registers_before_headers_then_streams_real_usage(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, _, event_queue = bridge_factory(1)
    app = create_lwd_cloud_control_app(bridge, cloud_id=0, dp_idx=0)
    transport = httpx.ASGITransport(app=app)
    headers, body = _http_request()
    expected = LwdUsage(prompt_tokens=16, completion_tokens=3, cached_tokens=16)
    registered: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

    async def publish_usage() -> None:
        while (registration := bridge.registration("e2-r")) is None:
            await asyncio.sleep(0)
        registered.set_result(registration)
        event_queue.put(_usage_event("e2-r", expected, kind="usage_start"))
        event_queue.put(_usage_event("e2-r", expected))

    publisher = asyncio.create_task(publish_usage())
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://cloud") as client,
        client.stream(
            "POST", "/v1/lwd/activate", headers=headers, json=body
        ) as response,
    ):
        assert response.status_code == 200
        frames = [frame async for frame in response.aiter_text()]

    await publisher
    assert await registered is not None
    assert response.headers[HEADER_ENGINE_INDEX] == "0"
    assert any('"completion_tokens":3' in frame for frame in frames)
    assert "".join(frames).endswith("data: [DONE]\n\n")
    assert bridge.registration("e2-r") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("nonce", [None, "-1", "1.5"])
async def test_http_endpoints_require_valid_probe_nonce(
    bridge_factory: BridgeFactory, nonce: str | None
) -> None:
    bridge, command_queues, _ = bridge_factory(1)
    app = create_lwd_cloud_control_app(bridge, cloud_id=0, dp_idx=0)
    transport = httpx.ASGITransport(app=app)
    headers, body = _http_request()
    if nonce is None:
        headers.pop(HEADER_PROBE_NONCE)
    else:
        headers[HEADER_PROBE_NONCE] = nonce
    async with httpx.AsyncClient(
        transport=transport, base_url="http://cloud"
    ) as client:
        for path in ("/v1/lwd/probe", "/v1/lwd/activate"):
            response = await client.post(path, headers=headers, json=body)
            assert response.status_code == 400
    assert command_queues[0].empty()
    assert bridge.registration("e2-r") is None


def test_any_dp_bind_failure_fails_whole_server_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dps = [SimpleNamespace(dp_idx=index, coord_port=8100 + index) for index in range(2)]
    vllm_config = SimpleNamespace(
        lwd_config=SimpleNamespace(
            role="cloud",
            coordination=SimpleNamespace(enabled=True, listen_host="127.0.0.1"),
            instance=SimpleNamespace(dp=dps),
        )
    )

    def fail_bind(host: str, cloud_dps: Any) -> list[Any]:
        raise OSError("second DP address is occupied")

    monkeypatch.setattr(
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control."
        "_bind_control_sockets",
        fail_bind,
    )
    with pytest.raises(OSError, match="second DP"):
        start_lwd_cloud_control_server(vllm_config)
