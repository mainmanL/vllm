# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Where a first token's latency goes on the edge side of the link.

The cloud meters account for the span between the request notify landing
and the first token going back, and on a loaded run that span is a
minority of the TTFT the benchmark measures. The rest is spent before the
edge ever announces the request -- a stretch no counter times today, and
one that holds the majority of the run-to-run variance.

Three segments, each the wait for one thing, together covering everything
from the client's arrival stamp to the first prompt rows leaving:

  - frontend: the client's arrival stamp to `add_request` running on the
    engine thread. HTTP, tokenization, the ZMQ hop and the engine's input
    queue -- all native vLLM, no LWD code on the path.
  - route: `add_request` entry to the request being admitted and routed.
    Validation and model-family precomputation over the prompt, plus the
    prefix READY round trip when prefix caching is on.
  - send: routed to the notify reaching the cloud's link. A full outbound
    queue is retried here, so this is where that shows up.
  - dispatch: notify sent to the first prompt rows announced for it. The
    edge scheduler choosing this request -- the picker and the gate.

`dispatch` is the same span the cloud already times as `hold`, measured
from the other end. The control plane is about a millisecond one way (the
cloud's own `send` segment), so the two should agree within noise, and a
disagreement means a stamp or a clock is wrong rather than a deployment
being slow. Nothing else here is cross-checkable, so this one is worth
the duplicate.

`queued` counts requests whose notify went out and whose rows have not:
the edge's own backlog, to be read against the cloud's `tracking`. The
cloud knowing about requests the edge is not holding means they never
reached `add_request`, which is the shape that points at the frontend.

Timing the frontend segment means crossing a process boundary, so it is
the one span measured with `time.time()` rather than a monotonic clock;
frontend and engine share a host here, and the segment is seconds long
where the clocks differ by far less.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from vllm.v1.lwd_control.lwd_control_diag.lwd_meter_base import (
    LwdIntervalMeter,
    LwdSpan,
    lwd_meter_logger,
)

logger = lwd_meter_logger

# Worst requests named per window. Enough to see whether one request's
# dispatch covers a whole batch's admit, few enough to stay one line.
_LWD_SLOWEST_KEPT = 3

# Below this the per-request line says nothing the window averages do
# not: a healthy edge spends about 3 ms here, so naming its three worst
# requests every window is noise that hides the window that matters.
_LWD_SLOW_REQUEST_S = 0.010


@dataclass(slots=True)
class _LwdEdgeTimeline:
    """Monotonic stamps of one request's edge hops. Zero means not reached."""

    admit: float = 0.0
    routed: float = 0.0
    notify: float = 0.0
    dispatch: float = 0.0


class LwdEdgeTtftMeter(LwdIntervalMeter):
    """Per-request breakdown of the edge's share of TTFT.

    `admit` is split at the point the request is routed: `route` is the
    scheduler-local admission (validation, model-family precomputation,
    routing) and `send` is the notify reaching the cloud's link. The two
    have nothing in common -- one is CPU work on the prompt, the other
    waits on an outbound queue -- and a measurement that adds them cannot
    say which one grew.

    The slowest requests are also named individually, when any was slow
    enough to be worth a line. The window averages cannot show that one
    request's `dispatch` covers a whole batch's `admit`, because the two
    live in different fields of the same line; the per-request detail
    puts both on one row, for one request id.
    """

    def __init__(self) -> None:
        self._timelines: dict[str, _LwdEdgeTimeline] = {}
        super().__init__()

    def _reset(self) -> None:
        # Timelines survive the window: a request whose rows go out after
        # the report would otherwise lose the stamps that explain it, and
        # those are the slow ones.
        self._arrived = 0
        self._done = 0
        self._frontend = LwdSpan()
        self._route = LwdSpan()
        self._send = LwdSpan()
        self._dispatch = LwdSpan()
        self._slowest: list[tuple[float, str, float, float, float]] = []

    def note_add_request(self, request_id: str, arrival_time: float) -> None:
        """`add_request` reached the engine thread. Closes the frontend
        segment against the client's arrival stamp and opens admit.

        `arrived` counts these, and is the one direct measurement of the
        input stream: every other view of arrivals so far has been
        reconstructed from how a backlog grew.
        """
        self._timelines[request_id] = _LwdEdgeTimeline(admit=time.monotonic())
        self._arrived += 1
        if arrival_time > 0:
            self._frontend.add(max(0.0, time.time() - arrival_time))

    def note_routed(self, request_id: str) -> None:
        """Edge admission finished and picked a cloud DP rank. Splits the
        prompt-side CPU work from the wait on the outbound link."""
        timeline = self._timelines.get(request_id)
        if timeline is not None and not timeline.routed:
            timeline.routed = time.monotonic()

    def note_notify(self, request_id: str) -> None:
        """The request notify went out to the routed cloud DP rank."""
        timeline = self._timelines.get(request_id)
        if timeline is not None and not timeline.notify:
            timeline.notify = time.monotonic()

    def note_dispatch(self, request_id: str) -> None:
        """First prompt rows announced for it. Closes its timeline; later
        chunks of the same request find nothing."""
        timeline = self._timelines.get(request_id)
        if timeline is None or timeline.dispatch:
            return
        now = timeline.dispatch = time.monotonic()
        del self._timelines[request_id]
        self._done += 1
        if not timeline.notify:
            return
        # A request that reached notify without being routed took the
        # prefix path; charge it all to route rather than to both.
        routed = timeline.routed or timeline.notify
        route = routed - timeline.admit
        send = timeline.notify - routed
        dispatch = now - timeline.notify
        self._route.add(route)
        self._send.add(send)
        self._dispatch.add(dispatch)
        self._note_slowest(request_id, route, send, dispatch)

    def _note_slowest(
        self, request_id: str, route: float, send: float, dispatch: float
    ) -> None:
        """Keep the few worst requests of the window, ranked by what they
        spent on the edge in total. Only requests slow enough to explain
        something are kept: on a healthy edge every request would qualify
        and the line would print the same few milliseconds forever."""
        total = route + send + dispatch
        if total < _LWD_SLOW_REQUEST_S:
            return
        self._slowest.append((total, request_id, route, send, dispatch))
        if len(self._slowest) > _LWD_SLOWEST_KEPT:
            self._slowest.sort(reverse=True)
            del self._slowest[_LWD_SLOWEST_KEPT:]

    def forget(self, request_id: str) -> None:
        """A request that will announce no rows (aborted before dispatch)."""
        self._timelines.pop(request_id, None)

    def tick(self) -> None:
        """Close the window if it is due. Driven per step rather than per
        dispatch: an edge that dispatches nothing for ten seconds is the
        case worth a line, and it would have no dispatch to ride on."""
        self._roll()

    def _report(self) -> None:
        # A window with no arrivals and no dispatches still reports the
        # backlog it holds: zero rows announced against a standing queue
        # is the clearest line this meter can print.
        if not self._arrived and not self._done and not self._timelines:
            return
        logger.info(
            "[lwd][edge-ttft] %d arrived %d dispatched: frontend=%s "
            "route=%s send=%s dispatch=%s (avg/max ms); queued=%d",
            self._arrived,
            self._done,
            self._frontend,
            self._route,
            self._send,
            self._dispatch,
            len(self._timelines),
        )
        if not self._slowest:
            return
        self._slowest.sort(reverse=True)
        logger.info(
            "[lwd][edge-slow] worst %d of the window over 10ms: %s",
            len(self._slowest),
            "; ".join(
                f"{request_id} route={route * 1000:.0f} send={send * 1000:.0f} "
                f"dispatch={dispatch * 1000:.0f}"
                for _total, request_id, route, send, dispatch in self._slowest
            ),
        )


class LwdEdgeSendMeter(LwdIntervalMeter):
    """What the send half of admission spends its time on.

    The three things it does are unrelated, and a measurement that adds
    them cannot say which one grew:

      - coord: the FIRST_REQ wake-up. A non-blocking send on a socket no
        other thread touches, so anything above zero here means the edge
        is contending for the coordinator link again.
      - meta: the per-request metadata the notify carries. Builds a tensor
        over the whole prompt and scans it for image runs, per request.
      - wire: encoding the notify and handing it to the link, including
        every retry of a full outbound queue. `retries` counts those, so a
        slow wire can be told from a queue that was merely full.

    Summed per request these are the `send` the TTFT meter reports, so the
    two lines can be read against each other.
    """

    def _reset(self) -> None:
        self._calls = 0
        self._retried = 0
        self._retries = 0
        self._coord = LwdSpan()
        self._meta = LwdSpan()
        self._wire = LwdSpan()

    def note_send(
        self, coord_s: float, meta_s: float, wire_s: float, retries: int
    ) -> None:
        """One request's announcement, split by what it was doing."""
        self._calls += 1
        self._coord.add(coord_s)
        self._meta.add(meta_s)
        self._wire.add(wire_s)
        self._retries += retries
        if retries:
            self._retried += 1
        self._roll()

    def _report(self) -> None:
        if not self._calls:
            return
        logger.info(
            "[lwd][edge-send] %d announced: coord=%s meta=%s wire=%s "
            "(avg/max ms); calls that retried=%d retries=%d",
            self._calls,
            self._coord,
            self._meta,
            self._wire,
            self._retried,
            self._retries,
        )


class LwdEdgeIntakeMeter(LwdIntervalMeter):
    """How long the engine thread spends taking requests in, and how many
    it takes in before it steps again.

    The native busy loop polls the input queue to exhaustion before it
    runs a step, and on this edge each of those requests is admitted and
    announced inline. A burst therefore admits the whole batch back to
    back while no batch of prompt rows goes out, which would make the
    first request's wait for dispatch cover the whole batch's admission.

    That shape is visible today only as an arithmetic coincidence between
    two averages on different lines. This measures it directly: `taken` is
    messages taken in per poll, `held` is how long the poll kept the
    thread, and `gap` is how long the engine went without stepping. A poll
    that takes in tens of messages and holds the thread for minutes is
    the hypothesis confirmed; short holds mean the wait is elsewhere.

    `held` is split, because the native poll blocks on an empty queue when
    the engine has no work: `idle` is that wait, `busy` is time spent
    inside message handlers. Only `busy` is the serial admission this is
    looking for, and between batches the whole hold is `idle`.

    `taken` counts messages, not user requests: aborts, wake-ups and
    prefix replies all arrive through the same queue. The burst it reports
    is therefore an upper bound on how many requests were admitted.
    """

    def _reset(self) -> None:
        self._polls = 0
        self._taken = 0
        self._bursts = 0
        self._max_batch = 0
        self._idle = LwdSpan()
        self._busy = LwdSpan()
        self._gap = LwdSpan()

    def note_poll(
        self, taken: int, held_s: float, busy_s: float, gap_s: float
    ) -> None:
        """One trip through the input-queue poll: how many messages it
        handled, how long it ran, how much of that was inside handlers,
        and how long since the last step."""
        self._polls += 1
        self._taken += taken
        self._max_batch = max(self._max_batch, taken)
        if taken > 1:
            self._bursts += 1
        if busy_s > 0:
            self._busy.add(busy_s)
        idle_s = held_s - busy_s
        if idle_s > 0:
            self._idle.add(idle_s)
        if gap_s > 0:
            self._gap.add(gap_s)
        self._roll()

    def _report(self) -> None:
        if not self._polls:
            return
        logger.info(
            "[lwd][edge-intake] %d polls took in %d messages: bursts=%d "
            "max batch=%d; busy=%s idle=%s step gap=%s (avg/max ms)",
            self._polls,
            self._taken,
            self._bursts,
            self._max_batch,
            self._busy,
            self._idle,
            self._gap,
        )


@dataclass(slots=True)
class _LwdGateSnapshot:
    pending: int
    max_pending: int
    running: int
    max_running: int
    resume: bool
    free_seq: int
    picked: int = -1


class LwdEdgeDispatchMeter(LwdIntervalMeter):
    """Why the edge cannot turn routed prefills into EMBED batches.

    Gate checks and picker calls are counted separately: an all-gated step
    never reaches the picker. The latest per-connection snapshot exposes
    pending and running state even when no HOLD transition is logged.
    """

    def _reset(self) -> None:
        self._checks = 0
        self._seat_gated = 0
        self._pending_gated = 0
        self._free_seq_zero = 0
        self._picks = 0
        self._empty_picks = 0
        self._snapshots: dict[str, _LwdGateSnapshot] = {}

    def note_gate(
        self,
        connection_id: str,
        *,
        pending: int,
        max_pending: int,
        running: int,
        max_running: int,
        resume: bool,
        free_seq: int,
    ) -> None:
        self._checks += 1
        if not resume:
            if running >= max_running:
                self._seat_gated += 1
            if pending >= max_pending:
                self._pending_gated += 1
        if free_seq <= 0:
            self._free_seq_zero += 1
        self._snapshots[connection_id] = _LwdGateSnapshot(
            pending,
            max_pending,
            running,
            max_running,
            resume,
            free_seq,
        )

    def note_pick(self, connection_id: str, picked: int) -> None:
        self._picks += 1
        if picked == 0:
            self._empty_picks += 1
        self._snapshots[connection_id].picked = picked

    def tick(self) -> None:
        self._roll()

    def _report(self) -> None:
        if not self._checks and not self._picks:
            return
        logger.info(
            "[lwd][edge-gate] %d gate checks: seat=%d pending=%d "
            "free_seq0=%d; picks=%d empty=%d",
            self._checks,
            self._seat_gated,
            self._pending_gated,
            self._free_seq_zero,
            self._picks,
            self._empty_picks,
        )
        for connection_id, snapshot in sorted(self._snapshots.items()):
            logger.info(
                "[lwd][edge-gate-detail] %s pending=%d/%d running=%d/%d "
                "resume=%s free_seq=%d picked=%d",
                connection_id,
                snapshot.pending,
                snapshot.max_pending,
                snapshot.running,
                snapshot.max_running,
                snapshot.resume,
                snapshot.free_seq,
                snapshot.picked,
            )
