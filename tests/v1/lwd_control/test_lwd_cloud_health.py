# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_health import (
    LwdCloudHealthTracker,
)


def test_failure_threshold_cools_only_one_dp_until_expiry() -> None:
    dp0, dp1 = (0, 0), (0, 1)
    tracker = LwdCloudHealthTracker((dp0, dp1))
    for _ in range(2):
        tracker.record_failure(dp0, now=10.0, threshold=3, cooldown_s=5.0)
        assert tracker.candidates(now=10.0) == (dp0, dp1)

    tracker.record_failure(dp0, now=10.0, threshold=3, cooldown_s=5.0)
    assert tracker.candidates(now=10.0) == (dp1,)
    assert tracker.candidates(now=14.9) == (dp1,)
    assert tracker.candidates(now=15.0) == (dp0, dp1)


def test_all_cooling_dps_fail_open() -> None:
    dp0, dp1 = (0, 0), (1, 0)
    tracker = LwdCloudHealthTracker((dp1, dp0))
    for target in (dp0, dp1):
        tracker.record_failure(target, now=10.0, threshold=1, cooldown_s=5.0)
    assert tracker.candidates(now=10.0) == (dp0, dp1)
    assert tracker.candidates(now=15.0) == (dp0, dp1)


def test_round_robin_staggers_cloud_and_dp_by_edge_id() -> None:
    targets = ((0, 0), (0, 1), (1, 0), (1, 1))
    starts = [
        LwdCloudHealthTracker(targets, edge_id=edge_id).next() for edge_id in range(4)
    ]
    assert starts == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_round_robin_staggers_dps_within_one_cloud() -> None:
    targets = ((0, 0), (0, 1))
    starts = [
        LwdCloudHealthTracker(targets, edge_id=edge_id).next() for edge_id in range(4)
    ]
    assert starts == [(0, 0), (0, 1), (0, 0), (0, 1)]


def test_round_robin_ranks_cloud_ids_by_order_not_value() -> None:
    tracker = LwdCloudHealthTracker(((5, 0), (7, 0), (9, 0)), edge_id=1)
    assert tracker.next() == (7, 0)


def test_round_robin_starts_after_a_cooling_first_cloud() -> None:
    tracker = LwdCloudHealthTracker(((0, 0), (1, 0), (2, 0)), edge_id=1)
    tracker.record_failure((1, 0), now=time.monotonic(), threshold=1, cooldown_s=60.0)
    assert tracker.next() == (2, 0)


def test_success_clears_failures_and_next_respects_logical_edge() -> None:
    dp0, dp1, other_edge = (0, 0), (0, 1), (1, 0)
    tracker = LwdCloudHealthTracker((dp0, dp1, other_edge))
    for _ in range(2):
        tracker.record_failure(dp0, now=10.0, threshold=2, cooldown_s=60.0)
    assert tracker.candidates(now=10.0) == (dp1, other_edge)

    tracker.record_success(dp0)
    assert tracker.candidates(now=10.0) == (dp0, dp1, other_edge)
    assert tracker.next(allowed={dp0, dp1}) in (dp0, dp1)
    tracker.record_failure(dp0, now=11.0, threshold=2, cooldown_s=60.0)
    assert tracker.candidates(now=11.0) == (dp0, dp1, other_edge)
