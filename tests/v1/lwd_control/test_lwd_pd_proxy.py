# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU HTTP integration: actual Proxy -> stage HTTP bridges, fake engine queues."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request

SOURCE = Path(__file__).resolve().parents[3] / "vllm"


@pytest.fixture
def modules(monkeypatch):
    for package in (
        "vllm",
        "vllm.v1",
        "vllm.v1.lwd_control",
        "vllm.v1.lwd_control.pd",
        "vllm.v1.lwd_control.lwd_comm",
    ):
        module = types.ModuleType(package)
        module.__path__ = [str(SOURCE.joinpath(*package.split(".")[1:]))]
        monkeypatch.setitem(sys.modules, package, module)
    imported = {}
    for suffix in (
        "lwd_comm.lwd_prefix",
        "lwd_comm.lwd_messages",
        "lwd_comm.lwd_id_adapter",
        "pd.envelope",
        "pd.results",
        "pd.cloud_http",
        "pd.proxy",
    ):
        name = f"vllm.v1.lwd_control.{suffix}"
        monkeypatch.delitem(sys.modules, name, raising=False)
        imported[suffix.rsplit(".", 1)[-1]] = importlib.import_module(name)
    return SimpleNamespace(**imported)


def body(modules, attempt="a", *, tokens=257, ttl=300):
    envelope = modules.envelope.build_envelope(
        request_id="r",
        attempt_id=attempt,
        edge_id=7,
        tenant_id="tenant",
        original_prompt_tokens=tokens,
        user_max_output_tokens=8,
        sampling_params=SimpleNamespace(eos_token_id=None),
        model_id="m",
        input_abi="abi",
        digest_block_size=128,
        digest_chain=[bytes(32)] * (tokens // 128),
        digest_tail_hash=bytes([1]) * 32 if tokens % 128 else None,
        ttl_s=ttl,
    )
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": modules.envelope.envelope_content(envelope)}
        ],
        "stream": True,
    }


class Stages:
    def __init__(self, modules):
        self.modules = modules
        self.calls = []
        self.config = SimpleNamespace(
            enabled=True,
            notify_claim_timeout_s=1,
            export_hold_timeout_s=1,
            usage_wait_timeout_s=1,
        )
        self.p = self.bridge("prefill", [0, 1])
        self.d = {dp: self.bridge("decode", [dp]) for dp in range(4)}
        self.app = FastAPI()

        @self.app.post("/p")
        async def prefill(request: Request):
            raw = await request.json()
            return await modules.cloud_http.lwd_pd_http_response(
                self.p,
                raw,
                modules.cloud_http.lwd_pd_envelope_from_body(raw),
                "prefill",
                self.config,
            )

        @self.app.post("/d/{dp}")
        async def decode(dp: int, request: Request):
            raw = await request.json()
            return await modules.cloud_http.lwd_pd_http_response(
                self.d[dp],
                raw,
                modules.cloud_http.lwd_pd_envelope_from_body(raw),
                "decode",
                self.config,
            )

    def bridge(self, stage, dps):
        bridge = self.modules.cloud_http.LwdPdCloudHttpBridge({})
        for dp in dps:
            bridge.command_queues[dp] = SimpleNamespace(
                put=lambda command, rank=dp: self.execute(bridge, stage, rank, command)
            )
        return bridge

    def execute(self, bridge, stage, dp, command):
        self.calls.append((stage, dp, command))
        envelope = command["envelope"]
        event = {
            "request_id": self.modules.lwd_id_adapter.wrap_req_id(
                envelope["edge_id"], envelope["rid"]
            ),
            "attempt_id": envelope["attempt_id"],
            "ok": True,
        }
        bridge.dispatch({**event, "type": "pd_register"})
        if stage == "prefill":
            params = {
                "source_dp": dp,
                "remote_request_id": "source-r",
                "lwd_handoff": {
                    "attempt_id": envelope["attempt_id"],
                    "fingerprint": envelope["fingerprint"],
                },
            }
            bridge.dispatch(
                {**event, "type": "pd_export", "kv_transfer_params": params}
            )
        else:
            ledger = self.modules.results.LwdPdResultLedger(
                envelope["original_prompt_tokens"],
                0,
                envelope["attempt_id"],
                "binding-d",
            )
            ledger.append([11, 12])
            ledger.finish("length")
            result = ledger.commit(2, "complete", 2)
            bridge.dispatch({**event, "type": "pd_result", "result": result})

    def proxy_config(self, max_attempts=10):
        target = self.modules.proxy.LwdPdProxyTarget
        return self.modules.proxy.LwdPdProxyConfig(
            prefill=tuple(target(0, dp, "http://cloud/p") for dp in range(2)),
            decode=tuple(target(1, dp, f"http://cloud/d/{dp}") for dp in range(4)),
            max_attempts=max_attempts,
        )


def test_proxy_p2_d4_routes_distinct_dp_and_preserves_envelope_and_kv(modules):
    async def run():
        stages = Stages(modules)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(stages.app)
        ) as upstream:
            app = modules.proxy.create_app(stages.proxy_config(), upstream)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://proxy"
            ) as client:
                originals = [body(modules, f"a{i}") for i in range(4)]
                for request in originals:
                    response = await client.post("/v1/chat/completions", json=request)
                    assert response.status_code == 200
                    assert "data: [DONE]" in response.text
                    assert '"completion_tokens": 2' in response.text
                assert [(stage, dp) for stage, dp, _ in stages.calls] == [
                    ("prefill", 0),
                    ("decode", 0),
                    ("prefill", 1),
                    ("decode", 1),
                    ("prefill", 0),
                    ("decode", 2),
                    ("prefill", 1),
                    ("decode", 3),
                ]
                for index in range(4):
                    p, d = stages.calls[2 * index], stages.calls[2 * index + 1]
                    content = originals[index]["messages"][0]["content"]
                    expected = modules.envelope.parse_envelope(content)
                    assert p[2]["envelope"] == d[2]["envelope"] == expected
                    assert d[2]["kv_params"]["source_dp"] == p[1]
                    assert (
                        d[2]["kv_params"]["lwd_handoff"]["fingerprint"]
                        == expected["fingerprint"]
                    )
            await app.state.pd_proxy.close()

    asyncio.run(run())


def test_same_attempt_http_retries_share_one_execution_and_identical_sse(modules):
    async def run():
        stages = Stages(modules)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(stages.app)
        ) as upstream:
            app = modules.proxy.create_app(stages.proxy_config(), upstream)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://proxy"
            ) as client:
                request = body(modules)
                first, retry = await asyncio.gather(
                    *(
                        client.post("/v1/chat/completions", json=request)
                        for _ in range(2)
                    )
                )
                assert first.content == retry.content
                assert len(stages.calls) == 2
                later = await client.post("/v1/chat/completions", json=request)
                assert later.content == first.content
                assert len(stages.calls) == 2
                conflict = await client.post(
                    "/v1/chat/completions", json=body(modules, tokens=258)
                )
                assert conflict.status_code == 409
            await app.state.pd_proxy.close()

    asyncio.run(run())


def test_proxy_cache_never_evicts_live_attempt_to_make_room(modules):
    async def run():
        stages = Stages(modules)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(stages.app)
        ) as upstream:
            app = modules.proxy.create_app(
                stages.proxy_config(max_attempts=1), upstream
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://proxy"
            ) as client:
                request = body(modules)
                assert (
                    await client.post("/v1/chat/completions", json=request)
                ).status_code == 200
                assert (
                    await client.post(
                        "/v1/chat/completions", json=body(modules, "other")
                    )
                ).status_code == 503
                assert (
                    await client.post("/v1/chat/completions", json=request)
                ).status_code == 200
                assert len(stages.calls) == 2
                envelope = json.loads(request["messages"][0]["content"])
                cached = next(iter(app.state.pd_proxy.attempts.values()))
                assert cached.expires_at == envelope["issued_at"] + envelope["ttl_s"]
            await app.state.pd_proxy.close()

    asyncio.run(run())


def test_proxy_failure_is_unconfirmed_and_does_not_reselect_or_invent_usage(modules):
    async def run():
        calls = []

        async def unavailable(request):
            calls.append(str(request.url))
            return httpx.Response(503, json={"error": "unavailable"})

        config = Stages(modules).proxy_config()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(unavailable)
        ) as upstream:
            app = modules.proxy.create_app(config, upstream)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://proxy"
            ) as client:
                request = body(modules)
                response = await client.post("/v1/chat/completions", json=request)
                retry = await client.post("/v1/chat/completions", json=request)
                assert response.content == retry.content
                assert '"unconfirmed"' in response.text
                assert '"usage"' not in response.text
                assert calls == ["http://cloud/p"]
            await app.state.pd_proxy.close()

    asyncio.run(run())


def test_expired_envelope_is_rejected_before_any_stage_selection(modules):
    async def run():
        config = Stages(modules).proxy_config()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(500))
        ) as upstream:
            proxy = modules.proxy.LwdPdProxy(config, upstream)
            request = body(modules)
            envelope = json.loads(request["messages"][0]["content"])
            envelope["issued_at"] = time.time() - 1000
            semantic = {key: envelope[key] for key in modules.envelope._SEMANTIC_FIELDS}
            envelope["fingerprint"] = modules.envelope.request_fingerprint(semantic)
            request["messages"][0]["content"] = modules.envelope.envelope_content(
                envelope
            )
            with pytest.raises(Exception, match="expired"):
                await proxy.register(request)
            assert not proxy.attempts

    asyncio.run(run())
