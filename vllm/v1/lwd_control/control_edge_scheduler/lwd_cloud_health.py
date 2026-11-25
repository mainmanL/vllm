# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Health-aware round-robin selection for direct cloud DP links."""

from __future__ import annotations

import time

LwdCloudDpId = tuple[int, int]


def lwd_rotate_targets(
    targets: tuple[LwdCloudDpId, ...], edge_id: int
) -> tuple[LwdCloudDpId, ...]:
    """Rotate DP targets so the edge's staggered first target leads.

    Edges are staggered over the cloud x DP cross product: the cloud rank
    advances first (``edge_id % cloud_count``), then the DP rank inside that
    cloud (``edge_id // cloud_count`` modulo the cloud's DP count).
    """
    ordered = sorted(set(targets))
    if not ordered:
        return ()
    clouds = sorted({cloud_id for cloud_id, _dp_idx in ordered})
    first_cloud = clouds[edge_id % len(clouds)]
    cloud_dps = [dp_idx for cloud_id, dp_idx in ordered if cloud_id == first_cloud]
    first_dp = cloud_dps[(edge_id // len(clouds)) % len(cloud_dps)]
    start = ordered.index((first_cloud, first_dp))
    return tuple(ordered[start:] + ordered[:start])


class LwdCloudHealthTracker:
    """Track DP health without changing existing request bindings."""

    def __init__(self, targets: tuple[LwdCloudDpId, ...], edge_id: int = 0) -> None:
        self._targets = lwd_rotate_targets(targets, edge_id)
        self._failures = dict.fromkeys(self._targets, 0)
        self._cooldown_until: dict[LwdCloudDpId, float] = {}
        self._cursor = 0

    def candidates(self, now: float | None = None) -> tuple[LwdCloudDpId, ...]:
        now = time.monotonic() if now is None else now
        healthy = tuple(
            target
            for target in self._targets
            if self._cooldown_until.get(target, 0.0) <= now
        )
        return healthy or self._targets

    def next(self, allowed: set[LwdCloudDpId] | None = None) -> LwdCloudDpId | None:
        candidates = self.candidates()
        if allowed is not None:
            candidates = tuple(target for target in candidates if target in allowed)
        if not candidates:
            return None
        target = candidates[self._cursor % len(candidates)]
        self._cursor += 1
        return target

    def record_success(self, target: LwdCloudDpId) -> None:
        self._failures[target] = 0
        self._cooldown_until.pop(target, None)

    def record_failure(
        self,
        target: LwdCloudDpId,
        *,
        now: float | None = None,
        threshold: int = 3,
        cooldown_s: float = 60.0,
    ) -> None:
        now = time.monotonic() if now is None else now
        failures = self._failures.get(target, 0) + 1
        self._failures[target] = failures
        if failures >= threshold:
            self._cooldown_until[target] = now + cooldown_s
