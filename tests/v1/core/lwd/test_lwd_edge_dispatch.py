# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for the LWD edge dispatch-slot workaround."""

from __future__ import annotations

import ast
import logging
import queue
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock


def load_edge_engine_class() -> type[SimpleNamespace]:
    source = (
        Path(__file__).resolve().parents[4]
        / "vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_engine.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LwdEdgeEngineCore"
    )
    names = {
        "_lwd_edge_step",
        "_lwd_can_dispatch_embed",
        "_lwd_edge_consume_c2e",
        "_lwd_queue_mix",
    }
    # Exercise the production step without importing NPU runtime dependencies.
    methods: list[ast.stmt] = [
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    harness = ast.ClassDef(
        name="LwdEdgeEngineUnderTest",
        bases=[ast.Name(id="SimpleNamespace", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {
        "SimpleNamespace": SimpleNamespace,
        "queue": queue,
        "time": time,
        "logger": logging.getLogger(__name__),
        "LWD_EDGE_BATCH_QUEUE_DEPTH": 4,
        "_lwd_disable_down_enabled": lambda: False,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return cast(type[SimpleNamespace], namespace["LwdEdgeEngineUnderTest"])


class LwdEdgeDispatchTests(unittest.TestCase):
    def make_engine(
        self, *, down_notices: int = 0, refills: int = 0
    ) -> SimpleNamespace:
        engine = load_edge_engine_class()()
        engine._lwd_batch_queue = deque()
        engine._lwd_pending_refills = deque(
            SimpleNamespace(num_scheduled_tokens={"request": 1}) for _ in range(refills)
        )
        engine._lwd_embed_futures_inflight = []
        engine.lwd_c2e_meta_queue = queue.Queue()
        for seqno in range(down_notices):
            engine.lwd_c2e_meta_queue.put(
                (SimpleNamespace(down_seqno=seqno), time.monotonic())
            )
        engine.scheduler = Mock()
        engine.scheduler.has_requests.return_value = True
        engine.scheduler.schedule.return_value = SimpleNamespace(
            num_scheduled_tokens={"request": 1}
        )
        engine.model_executor = SimpleNamespace(
            execute_model=Mock(return_value=object())
        )
        engine._lwd_dispatch_embed = Mock(return_value=object())
        engine._lwd_dispatch_unembed = Mock(return_value=object())
        return engine

    def test_no_downlink_uses_all_embed_slots(self) -> None:
        engine = self.make_engine()

        engine._lwd_edge_step()

        self.assertEqual(engine._lwd_queue_mix(), (4, 0))
        self.assertEqual(engine._lwd_dispatch_embed.call_count, 4)
        engine._lwd_dispatch_unembed.assert_not_called()

    def test_queued_downlink_reserves_one_slot(self) -> None:
        engine = self.make_engine(down_notices=1)

        engine._lwd_edge_step()

        self.assertEqual(engine._lwd_queue_mix(), (3, 1))
        self.assertEqual(engine._lwd_dispatch_embed.call_count, 3)
        engine._lwd_dispatch_unembed.assert_called_once()
        self.assertTrue(engine.lwd_c2e_meta_queue.empty())

    def test_downlink_arriving_before_last_slot_is_dispatched(self) -> None:
        engine = self.make_engine()

        def dispatch_embed(_output: SimpleNamespace) -> object:
            if engine._lwd_dispatch_embed.call_count == 3:
                engine.lwd_c2e_meta_queue.put(
                    (SimpleNamespace(down_seqno=0), time.monotonic())
                )
            return object()

        engine._lwd_dispatch_embed.side_effect = dispatch_embed

        engine._lwd_edge_step()

        self.assertEqual(engine._lwd_queue_mix(), (3, 1))
        engine._lwd_dispatch_unembed.assert_called_once()

    def test_refills_share_embed_dispatch_limit(self) -> None:
        engine = self.make_engine(down_notices=1, refills=2)

        engine._lwd_edge_step()

        self.assertEqual(engine._lwd_queue_mix(), (3, 1))
        self.assertEqual(engine.model_executor.execute_model.call_count, 2)
        engine._lwd_dispatch_embed.assert_called_once()
        engine._lwd_dispatch_unembed.assert_called_once()

    def test_refill_backlog_leaves_downlink_slot(self) -> None:
        engine = self.make_engine(down_notices=1, refills=4)

        engine._lwd_edge_step()

        self.assertEqual(engine._lwd_queue_mix(), (3, 1))
        self.assertEqual(len(engine._lwd_pending_refills), 1)
        self.assertEqual(engine.model_executor.execute_model.call_count, 3)
        engine.scheduler.schedule.assert_not_called()
        engine._lwd_dispatch_unembed.assert_called_once()

    def test_downlink_progresses_without_prefill_requests(self) -> None:
        engine = self.make_engine(down_notices=2)
        engine.scheduler.has_requests.return_value = False

        engine._lwd_edge_step()

        self.assertEqual(engine._lwd_queue_mix(), (0, 2))
        self.assertEqual(engine._lwd_dispatch_unembed.call_count, 2)
        engine._lwd_dispatch_embed.assert_not_called()

    def test_zero_scheduled_tokens_still_dispatches_downlink(self) -> None:
        engine = self.make_engine(down_notices=1)
        engine.scheduler.schedule.return_value = SimpleNamespace(
            num_scheduled_tokens={}
        )

        engine._lwd_edge_step()

        self.assertEqual(engine._lwd_queue_mix(), (0, 1))
        engine._lwd_dispatch_embed.assert_not_called()
        engine._lwd_dispatch_unembed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
