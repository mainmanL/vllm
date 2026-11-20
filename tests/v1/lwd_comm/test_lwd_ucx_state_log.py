# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State snapshot of the UCX channel IO loop (runtime diagnostics).

The snapshot used to be logged every 10s per channel whatever the state,
which floods the log of an idle deployment with all-zero lines. What matters
here is the gating: an idle channel stays silent, pending work is reported on
the lwd logger (silent by default), and a dangling pre-post keeps the vllm
error log instead of the snapshot.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from vllm.v1.lwd_comm.data_comm import ucx_channel
from vllm.v1.lwd_comm.data_comm.ucx_channel import LwdUcxChannel

_NOW = 100.0


class _Capture:
    """Logger stand-in recording formatted lines per level."""

    def __init__(self) -> None:
        self.error_lines: list[str] = []
        self.info_lines: list[str] = []

    def error(self, message: str, *args: Any) -> None:
        self.error_lines.append(message % args)

    def info(self, message: str, *args: Any) -> None:
        self.info_lines.append(message % args)


@pytest.fixture()
def logs(monkeypatch) -> tuple[_Capture, _Capture]:
    error_log = _Capture()
    info_log = _Capture()
    monkeypatch.setattr(ucx_channel, "logger", error_log)
    monkeypatch.setattr(ucx_channel, "lwd_logger", info_log)
    return error_log, info_log


def _channel(**state: Any) -> LwdUcxChannel:
    ch = object.__new__(LwdUcxChannel)
    ch._name = "t"
    ch._expects = {}
    ch._expects_t0 = 0.0
    ch._send_reqs = {}
    ch._early_count = 0
    ch._broken = False
    ch._pool_staging = False  # receive channel unless a test says otherwise
    for key, value in state.items():
        setattr(ch, key, value)
    return ch


def test_idle_channel_reports_nothing(logs) -> None:
    error_log, info_log = logs

    _channel()._log_state(_NOW)

    assert error_log.error_lines == []
    assert info_log.info_lines == []


def test_in_flight_send_is_reported(logs) -> None:
    _, info_log = logs
    ch = _channel(_send_reqs={1: (None, 7, 133_120)})

    ch._log_state(_NOW)

    (line,) = info_log.info_lines
    assert line.startswith("[ucx-chan][stall]")
    assert "sends=1" in line
    # both directions of one connection share the channel name
    assert "op=recv" in line


def test_send_channel_is_marked_as_such(logs) -> None:
    _, info_log = logs
    ch = _channel(_pool_staging=True, _send_reqs={1: (None, 7, 133_120)})

    ch._log_state(_NOW)

    (line,) = info_log.info_lines
    assert "op=send" in line


def test_parked_frames_are_reported(logs) -> None:
    _, info_log = logs
    ch = _channel(_early_count=3)

    ch._log_state(_NOW)

    (line,) = info_log.info_lines
    assert "early=3" in line


def test_broken_link_is_reported(logs) -> None:
    _, info_log = logs
    ch = _channel(_broken=True)

    ch._log_state(_NOW)

    (line,) = info_log.info_lines
    assert "broken=True" in line


def test_dangling_pre_post_keeps_the_error_log(logs) -> None:
    error_log, info_log = logs
    ch = _channel(
        _expects={5: deque([object()])},
        _expects_t0=_NOW - ucx_channel._EXPECTS_STUCK_S - 1.0,
    )

    ch._log_state(_NOW)

    (line,) = error_log.error_lines
    assert "recv expects stuck" in line
    # the stuck report replaces the snapshot for this tick
    assert info_log.info_lines == []


def test_fresh_pre_post_is_not_stuck(logs) -> None:
    error_log, info_log = logs
    ch = _channel(_expects={5: deque([object()])}, _expects_t0=_NOW - 1.0)

    ch._log_state(_NOW)

    assert error_log.error_lines == []
    (line,) = info_log.info_lines
    assert "expects=1(tags=1)" in line
