# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Where a cloud step's wall clock goes, when one takes seconds.

`lwd_ttft_meter`'s step figure is the interval between two commits, so a
step that took eight seconds and a step that waited eight seconds for
work look the same there. Four of its six TTFT segments close on the step
thread as well, so one stalled step widens `flight`, `fill`, `queue` and
`exec` together -- four readings of one event, and none of them says what
the thread was doing.

This meter times the step itself, split at the boundaries the native
engine already has, so each segment names one suspect:

  - pre_gap: the previous step's end to this one's start, less the two
    segments below. The input queue, the request-count publish, and
    waiting for work -- not idle, because the input handlers run here too.
  - dummy: the dummy batch a DP rank runs when it has nothing of its own,
    to keep the MoE ranks in lockstep. Work done for someone else.
  - dp_sync: the all-reduce that decides whether the wave continues. A
    blocking collective, so it costs this rank whatever the slowest rank
    is still doing -- the one segment here that measures another process.
  - sched: the scheduler choosing a batch, LWD admission included.
  - submit: handing the batch to the executor (argument marshalling and
    the worker MQ enqueue), not the execution.
  - grammar: the structured-output bitmask.
  - exec_rpc: waiting for the worker's execute_model. Input preparation,
    the forward pass, collectives and the response trip home, which only
    the worker-side meter can tell apart.
  - sample_rpc / batch_wait: what the step blocked on, named for the path
    (see below).
  - abort: aborts that arrived during execution.
  - update: committing tokens and building the c2e notifies.
  - c2e_send / native_post: the post-step hook, split at the send loop so
    a full outbound queue to the edge is visible as the wait it is.

`exec_rpc` is named for what it measures: the round trip, not the model.
A long one does not say the NPU was busy.

The two waiting segments mean different things on the two native step
paths, so they are labelled differently:

  - `step`: the step resolves the execute future, then calls
    `sample_tokens` and blocks in it. `exec_rpc` is the model round trip
    and `sample_rpc` is sampling plus the DOWN work -- two waits, each
    on its own call.
  - `step_with_batch_queue`: `sample_tokens` is launched non-blocking and
    its future is what the step pops and waits on; the execute future is
    resolved only to re-raise an error. So there is one wait, and it
    covers the whole worker trip -- queueing behind an earlier batch,
    input staging, the forward pass, collectives, sampling, the DOWN
    collect and submit, the output thread, and the response home. It is
    called `batch_wait`, because reading it as sampling would be wrong by
    most of its length, and `exec_rpc` reports `-` rather than `0`: that
    future was never awaited, so nothing was measured there.

A segment the running path never samples prints `-`, never `0`. A zero
clears a suspect; a dash says it was not looked at.

The timings ride on the future rather than on "the last batch
scheduled", because a batch queue deeper than one returns a batch that
was scheduled several steps ago; reading the latest shape would pair
every duration with the wrong batch. The shape of the batch that
produced a slow step is on its own line, because an eight-second step
over 8k prompt rows and one over 30 decode rows are different faults.

One iteration of `step_with_batch_queue` can also schedule a new batch
and wait on an older one, so `sched`/`submit` and `batch_wait`/`update`
on one line need not belong to the same batch. The line is the step
thread's wall clock, which is what a stall is made of; attributing a
single batch end to end takes `down_seqno` across the engine and worker
logs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.lwd_control.lwd_control_diag.lwd_meter_base import (
    LwdIntervalMeter,
    LwdSpan,
    lwd_meter_logger,
)

logger = lwd_meter_logger

# A step worth naming individually. The steady state here is 120-900 ms
# depending on how much prefill the step carried, so a second is clear of
# every healthy shape and still catches anything that stands out.
LWD_SLOW_STEP_S = 1.0

# A gap this long is the engine having nothing to do -- between two
# benchmark batches, or before the first request -- not a step waiting on
# something. Counting those tens of seconds into `pre_gap` would put the
# idle time between batches into the same average as the hundreds of
# milliseconds a step waits mid-load, and the second would vanish. Such a
# gap is reported on its own line and closes the window, so the batch
# that follows is measured from zero.
LWD_IDLE_GAP_S = 5.0


@dataclass(slots=True)
class LwdBatchShape:
    """What one step was asked to run. Snapshotted by the scheduler while
    it still knows, and carried to wherever that batch's timings land."""

    decode_reqs: int = 0
    decode_tokens: int = 0
    prefill_reqs: int = 0
    prefill_windows: int = 0
    prefill_tokens: int = 0
    room: int = 0
    reserved: int = 0
    queue_depth: int = 0
    blocked_partial: int = 0
    blocked_no_arrival: int = 0
    running: int = 0
    waiting: int = 0
    output_tokens: int = 0

    def __str__(self) -> str:
        return (
            f"prefill={self.prefill_tokens}/{self.prefill_reqs}"
            f"of{self.prefill_windows} decode={self.decode_tokens}/"
            f"{self.decode_reqs} room={self.room} reserved={self.reserved} "
            f"queued={self.queue_depth} blocked={self.blocked_partial}/"
            f"{self.blocked_no_arrival} running={self.running} "
            f"waiting={self.waiting}"
        )


@dataclass(slots=True)
class LwdStepTiming:
    """One step's segments, in the order they run. Seconds, monotonic.

    The two waiting segments start as None and are set by whichever
    future the step actually resolved. None means the running path never
    awaited that call, and prints as a dash; a float means it was
    awaited, even if it returned at once.
    """

    pre_gap: float = 0.0
    dummy: float = 0.0
    dp_sync: float = 0.0
    sched: float = 0.0
    submit: float = 0.0
    grammar: float = 0.0
    exec_rpc: float | None = None
    sample_rpc: float | None = None
    abort: float = 0.0
    update: float = 0.0
    c2e_send: float = 0.0
    native_post: float = 0.0
    c2e_retries: int = 0
    down_seqno: int = -1
    shape: LwdBatchShape = field(default_factory=LwdBatchShape)

    def add_wait(self, field_name: str, seconds: float) -> None:
        """Charge a resolved future's wait, starting the segment at zero
        the first time rather than leaving it unmeasured."""
        current = getattr(self, field_name)
        setattr(self, field_name, (current or 0.0) + seconds)

    def total(self) -> float:
        """Everything the step thread spent, waiting for work included."""
        return (
            self.pre_gap
            + self.dummy
            + self.dp_sync
            + self.sched
            + self.submit
            + self.grammar
            + (self.exec_rpc or 0.0)
            + (self.sample_rpc or 0.0)
            + self.abort
            + self.update
            + self.c2e_send
            + self.native_post
        )

    def in_step(self) -> float:
        """Everything but the wait that preceded it. The dummy batch and
        the DP all-reduce count: the thread was running them, and a rank
        that spends its step keeping others in lockstep is the finding,
        not the background."""
        return self.total() - self.pre_gap


class LwdShapeWindow:
    """Aggregates for one non-empty scheduler batch shape."""

    def __init__(self) -> None:
        self.steps = 0
        self.total = LwdSpan()
        self.model_wait = LwdSpan()
        self.model_s = 0.0
        self.prefill_tokens = 0
        self.output_tokens = 0

    def note(self, timing: LwdStepTiming) -> None:
        self.steps += 1
        self.total.add(timing.in_step())
        model_s = timing.sample_rpc or 0.0
        self.model_wait.add(model_s)
        self.model_s += model_s
        self.prefill_tokens += timing.shape.prefill_tokens
        self.output_tokens += timing.shape.output_tokens

    def summary(self, all_model_s: float) -> str:
        ms_per_token = (
            f"{self.model_s * 1000 / self.output_tokens:.2f}"
            if self.output_tokens
            else "-"
        )
        share = self.model_s / all_model_s * 100 if all_model_s else 0.0
        return (
            f"steps={self.steps} prefill={self.prefill_tokens} "
            f"output={self.output_tokens} total={self.total} "
            f"model={self.model_wait} model_ms/token={ms_per_token} "
            f"model_share={share:.1f}%"
        )


class LwdCloudExecMeter(LwdIntervalMeter):
    """Per-step breakdown of the cloud engine's loop.

    Reports two kinds of line. The interval line is every step of the
    window averaged, which says whether a segment is slow in general.
    The slow line is one step named on its own with the batch it ran,
    which says whether a long step was doing something or waiting for
    something -- the averages cannot, because one eight-second step in a
    window of eighty hides inside them.
    """

    def __init__(self) -> None:
        self._last_step_end = 0.0
        # Set by `note_path` at startup, before any step runs. Decides
        # what the waiting segment is called, which changes what it means.
        self._batch_queued = False
        super().__init__()

    @property
    def _wait_label(self) -> str:
        """What the step blocked on, named for the path that ran."""
        return "batch_wait" if self._batch_queued else "sample_rpc"

    def _reset(self) -> None:
        # `_last_step_end` is not cleared: the gap across a report
        # boundary belongs to one of the two windows, and dropping it
        # would drop the long ones first.
        self._steps = 0
        self._empty = 0
        self._shapes = {
            kind: LwdShapeWindow() for kind in ("mixed", "decode_only", "prefill_only")
        }
        self._slow = 0
        self._retries = 0
        self._total = LwdSpan()
        self._pre_gap = LwdSpan()
        self._dummy = LwdSpan()
        self._dp_sync = LwdSpan()
        self._sched = LwdSpan()
        self._submit = LwdSpan()
        self._grammar = LwdSpan()
        self._exec_rpc = LwdSpan()
        self._sample_rpc = LwdSpan()
        self._abort = LwdSpan()
        self._update = LwdSpan()
        self._c2e_send = LwdSpan()
        self._native_post = LwdSpan()

    def note_path(self, batch_queue_size: int, async_scheduling: bool) -> None:
        """Which native step path this deployment runs, said once at
        startup. `step_with_batch_queue` returns a batch scheduled some
        steps earlier, and that changes how every number here reads; the
        choice is derived from config, so printing it beats assuming."""
        self._batch_queued = batch_queue_size > 1
        logger.info(
            "[lwd][cloud-exec] step path: batch_queue_size=%d "
            "async_scheduling=%s step_fn=%s wait_segment=%s",
            batch_queue_size,
            async_scheduling,
            "step_with_batch_queue" if self._batch_queued else "step",
            self._wait_label,
        )

    @staticmethod
    def _kind(timing: LwdStepTiming) -> str:
        shape = timing.shape
        has_prefill = bool(shape.prefill_tokens or shape.prefill_windows)
        has_decode = bool(shape.decode_tokens)
        if has_prefill:
            return "mixed" if has_decode else "prefill_only"
        if has_decode or shape.output_tokens:
            return "decode_only"
        return "empty"

    def note_step(self, timing: LwdStepTiming) -> None:
        """One trip through the engine's step."""
        if timing.pre_gap >= LWD_IDLE_GAP_S:
            self._note_idle(timing.pre_gap)
            timing.pre_gap = 0.0
        kind = self._kind(timing)
        if kind == "empty":
            self._empty += 1
            self._roll()
            return
        self._shapes[kind].note(timing)
        self._steps += 1
        self._retries += timing.c2e_retries
        self._total.add(timing.in_step())
        self._pre_gap.add(timing.pre_gap)
        self._dummy.add(timing.dummy)
        self._dp_sync.add(timing.dp_sync)
        self._sched.add(timing.sched)
        self._submit.add(timing.submit)
        self._grammar.add(timing.grammar)
        # Only if the step awaited it: a path that never resolves a
        # future must report `-`, not an average pulled towards zero.
        if timing.exec_rpc is not None:
            self._exec_rpc.add(timing.exec_rpc)
        if timing.sample_rpc is not None:
            self._sample_rpc.add(timing.sample_rpc)
        self._abort.add(timing.abort)
        self._update.add(timing.update)
        self._c2e_send.add(timing.c2e_send)
        self._native_post.add(timing.native_post)
        if timing.in_step() >= LWD_SLOW_STEP_S:
            self._slow += 1
            self._report_slow(timing)
        self._roll()

    def _note_idle(self, gap_s: float) -> None:
        """An idle stretch: report the window that ended before it and
        start the next one empty.

        Without this the window straddling two benchmark batches carries
        both the idle seconds and the steps on either side, and its
        averages describe neither. The line names the gap so a long one
        is still visible -- it is just not called a slow step.
        """
        if self._steps:
            self._report()
        self._reset()
        self._last_report = time.monotonic()
        logger.info(
            "[lwd][cloud-exec] idle %.1fs, window restarted; ts=%.3f",
            gap_s,
            time.time(),
        )

    def note_step_end(self) -> float:
        """Close the step and open the gap before the next one. Returns
        that gap once the next step starts."""
        self._last_step_end = time.monotonic()
        return self._last_step_end

    def take_pre_gap(self) -> float:
        """How long since the last step ended. Zero on the first step."""
        if not self._last_step_end:
            return 0.0
        return time.monotonic() - self._last_step_end

    @staticmethod
    def _ms(seconds: float | None) -> str:
        """Whole ms, or a dash for a segment this path never measured."""
        return "-" if seconds is None else f"{seconds * 1000:.0f}"

    def _report_slow(self, timing: LwdStepTiming) -> None:
        logger.info(
            "[lwd][cloud-slow-step] %.0fms: pre_gap=%.0f dummy=%.0f "
            "dp_sync=%.0f sched=%.0f "
            "submit=%.0f grammar=%.0f exec_rpc=%s %s=%s "
            "abort=%.0f update=%.0f c2e_send=%.0f native_post=%.0f "
            "(ms); retries=%d down=%d; %s; ts=%.3f",
            timing.in_step() * 1000,
            timing.pre_gap * 1000,
            timing.dummy * 1000,
            timing.dp_sync * 1000,
            timing.sched * 1000,
            timing.submit * 1000,
            timing.grammar * 1000,
            self._ms(timing.exec_rpc),
            self._wait_label,
            self._ms(timing.sample_rpc),
            timing.abort * 1000,
            timing.update * 1000,
            timing.c2e_send * 1000,
            timing.native_post * 1000,
            timing.c2e_retries,
            timing.down_seqno,
            timing.shape,
            # Wall clock, for lining this line up against the edge's log
            # and the other DP rank's. Every duration above is monotonic.
            time.time(),
        )

    def _report_shapes(self) -> None:
        all_model_s = sum(window.model_s for window in self._shapes.values())
        shapes = "; ".join(
            f"{kind}[{window.summary(all_model_s)}]"
            for kind, window in self._shapes.items()
            if window.steps
        )
        if shapes:
            logger.info("[lwd][cloud-shape] %s", shapes)

    def _report(self) -> None:
        if not self._steps:
            return
        logger.info(
            "[lwd][cloud-exec] %d steps (%d slow): step=%s pre_gap=%s; "
            "dummy=%s dp_sync=%s sched=%s submit=%s grammar=%s "
            "exec_rpc=%s %s=%s "
            "abort=%s update=%s c2e_send=%s native_post=%s (avg/max ms); "
            "c2e retries=%d",
            self._steps,
            self._slow,
            self._total,
            self._pre_gap,
            self._dummy,
            self._dp_sync,
            self._sched,
            self._submit,
            self._grammar,
            self._exec_rpc,
            self._wait_label,
            self._sample_rpc,
            self._abort,
            self._update,
            self._c2e_send,
            self._native_post,
            self._retries,
        )
        self._report_shapes()


# A harvest worth naming individually. An order of magnitude under the
# step threshold: this thread is supposed to cost microseconds, and it
# delays the next step by whatever it takes.
LWD_SLOW_OUTPUT_S = 0.1


@dataclass(slots=True)
class LwdOutputRecord:
    """One output crossing from a worker back to the engine.

    `carrier_len` separates the cases the durations alone cannot. A DOWN
    carrier arrives as a five-tuple and leaves as a four-tuple, so a four
    here means this output already passed through -- its harvest belongs
    to an earlier step and must not be reported as a harvest that cost
    nothing. Zero means no carrier at all: a rank that does not own the
    DOWN channel.
    """

    carrier_len: int
    harvested: bool
    harvest_s: float
    output_get_s: float
    enqueue_s: float
    hidden_numel: int
    seqno: int
    output_type: str


class LwdCloudOutputMeter(LwdIntervalMeter):
    """What the executor's output thread spends between a step finishing
    on the device and its result reaching the engine.

    This thread is off the step thread's critical path by design, so its
    cost appears nowhere in the step breakdown -- yet a step cannot be
    collected until it has run, so a slow output thread shows up as the
    *next* step's wait and reads as a slow forward pass.

      - meta_harvest: waiting for the per-step ready event and copying
        the DOWN metadata off the device. Only on outputs that carry a
        five-tuple; `-/-` when none did.
      - output_get: resolving an async runner output.
      - enqueue: handing the result to the engine's response queue.

    The carrier shapes seen in the window are on the line because the
    durations are only readable once you know which outputs they came
    from: a window of harvests that cost nothing and a window of outputs
    that were never harvested print the same three numbers otherwise, and
    the second says the measurement missed, not that the work was free.

    Called from the executor's output thread, which is the only thread
    that touches this meter.
    """

    def __init__(self, rank: int) -> None:
        self._rank = rank
        super().__init__()
        # Said once per worker, so an absent interval line later can be
        # told from a meter that never reached the log.
        logger.info("[lwd][cloud-output] meter up: rank%d", rank)

    def _reset(self) -> None:
        self._outputs = 0
        self._harvests = 0
        self._rows = 0
        self._harvest = LwdSpan()
        self._output_get = LwdSpan()
        self._enqueue = LwdSpan()
        self._total = LwdSpan()
        # How many outputs arrived with each carrier shape, and what the
        # outputs themselves were.
        self._carrier_lens: dict[int, int] = {}
        self._types: dict[str, int] = {}

    def note_output(self, record: LwdOutputRecord) -> None:
        """One output crossing back to the engine."""
        self._outputs += 1
        self._carrier_lens[record.carrier_len] = (
            self._carrier_lens.get(record.carrier_len, 0) + 1
        )
        self._types[record.output_type] = self._types.get(record.output_type, 0) + 1
        self._output_get.add(record.output_get_s)
        self._enqueue.add(record.enqueue_s)
        if record.harvested:
            # Only a real harvest feeds the harvest span and the row
            # count; an output that skipped it has no zero to contribute.
            self._harvests += 1
            self._rows += max(0, record.hidden_numel)
            self._harvest.add(record.harvest_s)
            self._total.add(record.harvest_s + record.output_get_s + record.enqueue_s)
            if record.harvest_s >= LWD_SLOW_OUTPUT_S:
                logger.info(
                    "[lwd][cloud-output-slow] rank%d %.0fms harvest: "
                    "output_get=%.0f enqueue=%.0f (ms); "
                    "down=%d hidden=%d; ts=%.3f",
                    self._rank,
                    record.harvest_s * 1000,
                    record.output_get_s * 1000,
                    record.enqueue_s * 1000,
                    record.seqno,
                    record.hidden_numel,
                    time.time(),
                )
        else:
            self._total.add(record.output_get_s + record.enqueue_s)
        self._roll()

    @staticmethod
    def _histogram(counts: dict[Any, int]) -> str:
        return ",".join(f"{key}:{n}" for key, n in sorted(counts.items(), key=str))

    def _report(self) -> None:
        if not self._outputs:
            return
        logger.info(
            "[lwd][cloud-output] rank%d %d outputs (%d harvested): total=%s "
            "meta_harvest=%s output_get=%s enqueue=%s (avg/max ms); "
            "hidden elems/harvest=%d; carrier_len=%s output=%s",
            self._rank,
            self._outputs,
            self._harvests,
            self._total,
            self._harvest,
            self._output_get,
            self._enqueue,
            self._rows // self._harvests if self._harvests else 0,
            self._histogram(self._carrier_lens),
            self._histogram(self._types),
        )
