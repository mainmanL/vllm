# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud prefill admission: why a loaded deployment admits prefill
slower than it could.

Throughput loss that lands entirely on TTFT -- unchanged TPOT, unchanged
concurrency, end-to-end latency growing by exactly what the first token
grew by -- means prompts are not reaching the cloud's batches, not that
the batches run slow. This meter answers one question about that: of the
room a step had for prefill, how much did it use? Rows the budget refused
and rows no arrival could fill look the same in the scheduler output and
have opposite fixes.

Where the resulting latency lands, rather than which budget refused it,
is `lwd_ttft_meter` on the cloud and `lwd_edge_ttft_meter` on the edge;
the interval gate and reporting conventions they share are in
`lwd_meter_base`.
"""

from __future__ import annotations

from vllm.v1.lwd_control.lwd_control_diag.lwd_meter_base import (
    LwdIntervalMeter,
    lwd_meter_logger,
)

logger = lwd_meter_logger


class LwdCloudAdmissionMeter(LwdIntervalMeter):
    """Cloud prefill admission: budget offered, budget taken, and what
    stood between them.

    `room` is what the step could still spend on prefill after decode and
    the drafting-slot reservation; `admitted` is what the windows
    actually took. Their ratio is the whole question: a step that admits
    what it offers is limited by the budget (the reservation is charged
    against a queue, so it grows with backlog), while a step that offers
    far more than it admits is limited by arrivals and the budget is
    innocent.

    `blocked_partial` counts requests skipped by the partial-window guard
    -- rows arrived, but fewer than the room left, and the prompt is not
    yet fully received. Those are deferred, not lost, and separating them
    keeps a queue of half-arrived prompts from reading as an arrival
    drought.
    """

    def _reset(self) -> None:
        self._steps = 0
        self._steps_with_room = 0
        self._steps_admitting = 0
        self._room = 0
        self._admitted = 0
        self._windows = 0
        self._blocked_partial = 0
        self._blocked_no_arrival = 0
        self._queue_depth = 0
        self._reserved = 0

    def note_step(
        self,
        room: int,
        admitted: int,
        num_windows: int,
        blocked: tuple[int, int],
        queue_depth: int,
        reserved: int,
    ) -> None:
        """Record one scheduling step's prefill admission.

        `blocked` is (partial_window, no_arrival) -- the two reasons a
        candidate yielded no window. `reserved` is the drafting-slot
        reservation this step withheld from `room`; carrying it here is
        what tells a budget that is merely small from one the backlog
        ate.
        """
        self._steps += 1
        self._room += max(0, room)
        self._admitted += admitted
        self._windows += num_windows
        self._blocked_partial += blocked[0]
        self._blocked_no_arrival += blocked[1]
        self._queue_depth += queue_depth
        self._reserved += reserved
        if room > 0:
            self._steps_with_room += 1
        if admitted:
            self._steps_admitting += 1
        self._roll()

    def _report(self) -> None:
        if not self._steps:
            return
        steps = self._steps
        logger.info(
            "[lwd][cloud-sched] prefill admission over %d steps: "
            "room=%d admitted=%d (%.1f%% of room) windows=%d; "
            "steps with room=%d admitting=%d; "
            "blocked partial=%d no_arrival=%d; "
            "avg queue=%.1f reserved=%.0f tok/step",
            steps,
            self._room,
            self._admitted,
            self._admitted / self._room * 100 if self._room else 0.0,
            self._windows,
            self._steps_with_room,
            self._steps_admitting,
            self._blocked_partial,
            self._blocked_no_arrival,
            self._queue_depth / steps,
            self._reserved / steps,
        )
