# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure functions for edge-side prefill batch assembly: resume-first +
FCFS prefix + tail truncation (PD_mix is the only supported form; the
single-request escape hatch for PD-disaggregated mode was removed along
with the phase machine).

chunk semantics follow lwd_chunked_prefill_design.md §3.1; called by the
LwdEdgeScheduler adapter layer after it assembles candidates. Inputs and
outputs are all plain values with no scheduler state, so the logic is
table-driven testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple


class LwdPrefillCandidate(NamedTuple):
    """Batch assembly candidate: the scheduler distills a Request into
    these three fields at the call site."""

    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int


def lwd_grammar_ready(request: Any, blocked_status: object) -> bool:
    """Picker filter: is the request selectable despite grammar blocking?

    The cloud's native scheduler defers a request whose status is
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR until its grammar finishes
    compiling; the edge picker must not select such a request either, or the
    two sides disagree on the execution set (this repo has no post-step
    reconciliation, the pre-filter is the guarantee). `blocked_status` is
    passed by the caller (the scheduler owns the RequestStatus enum), which
    keeps this module import-free of vllm.

    Reading `grammar` is side-effect free with respect to promotion: it only
    folds a completed future into its result, and a compile failure stays
    visible as an exception (truthy) which the native error path finishes;
    the status promotion itself stays with
    `_try_promote_blocked_waiting_request`.
    """
    if request.status is not blocked_status:
        return True
    structured_request = request.structured_output_request
    return bool(structured_request and structured_request.grammar)


def lwd_pick_prefill_batch(
    running: Sequence[LwdPrefillCandidate],
    waiting: Sequence[LwdPrefillCandidate],
    *,
    cap: int,
    free_seq: int,
    chunk_unit: int,
    single_prefill_min_tokens: int = 0,
) -> tuple[list[str], dict[str, int]]:
    """Pick the next embed batch, returning (req_ids, expected).

    running holds resume candidates (in running order: finish in-flight
    requests before starting new ones); waiting holds new-start candidates
    (the caller pre-joins skipped-first + the FCFS order of waiting).
    chunk_unit is the chunk unit C; each piece self-limits to
    c = min(remaining, C). cap < 1 returns an empty batch; the number of
    new requests is clamped by free_seq; tail truncation: the last entry
    in the batch may be cut short by the remaining budget (cap − used),
    and a truncation closes the batch.

    single_prefill_min_tokens: when > 0, a request whose total prompt
    length reaches this value takes the prefill batch exclusively
    (resumes included; no other entries are packed before or after it in
    the batch); 0 = disabled (DeepSeek V4 passes 8000, other models pass 0).

    """
    picked: list[str] = []
    expected: dict[str, int] = {}
    if cap < 1:
        return picked, expected
    used, single_prefill = _pick_resumes(
        running, picked, expected, cap, chunk_unit, single_prefill_min_tokens
    )
    _pick_new_requests(
        waiting,
        picked,
        expected,
        used,
        cap,
        free_seq,
        chunk_unit,
        single_prefill_min_tokens,
        single_prefill,
    )
    return picked, expected


def _is_single_prefill(single_prefill_min_tokens: int, num_prompt_tokens: int) -> bool:
    """Whether this request triggers the exclusive rule (total prompt
    length reaches the threshold)."""
    return (
        single_prefill_min_tokens > 0 and num_prompt_tokens >= single_prefill_min_tokens
    )


def _append_chunk(
    picked: list[str],
    expected: dict[str, int],
    used: int,
    cap: int,
    request_id: str,
    chunk: int,
) -> tuple[int, bool]:
    """Pack one chunk into the batch; when used + chunk exceeds the
    budget, fall back to tail truncation (remaining budget c = cap − used;
    the entry joins only if c ≥ 1, and the truncated piece still consumes
    budget). A truncation closes the batch. Returns (new used, whether
    the batch is closed)."""
    if used + chunk > cap:
        c = cap - used
        if c >= 1:
            picked.append(request_id)
            expected[request_id] = c
            used += c
        return used, True
    picked.append(request_id)
    expected[request_id] = chunk
    return used + chunk, False


def _pick_resumes(
    running: Sequence[LwdPrefillCandidate],
    picked: list[str],
    expected: dict[str, int],
    cap: int,
    chunk_unit: int,
    single_prefill_min_tokens: int,
) -> tuple[int, bool]:
    """Resume loop: walk requests in running order, each taking
    c = min(remaining, C).

    Exclusive rule: a qualifying request sets the single_prefill flag; if
    the batch already has entries, close it on the spot (long requests
    are never batched with short ones); the flag carries across loops
    (this batch closes, and the new-start loop packs nothing more).
    Returns (used, single_prefill flag)."""
    used = 0
    single_prefill = False
    for cand in running:
        if single_prefill and picked:
            break
        remaining = cand.num_prompt_tokens - cand.num_computed_tokens
        if remaining <= 0:
            continue
        # The exclusive check uses total prompt length, not this chunk's
        # remaining tokens: when a long request arrives later, close the
        # current batch first and leave the next batch to it alone, so
        # waiting new-starts cannot bypass it
        single_prefill = _is_single_prefill(
            single_prefill_min_tokens, cand.num_prompt_tokens
        )
        if single_prefill and picked:
            break
        used, batch_closed = _append_chunk(
            picked,
            expected,
            used,
            cap,
            cand.request_id,
            min(remaining, chunk_unit),
        )
        if batch_closed:
            break
    return used, single_prefill


def _pick_new_requests(
    waiting: Sequence[LwdPrefillCandidate],
    picked: list[str],
    expected: dict[str, int],
    used: int,
    cap: int,
    free_seq: int,
    chunk_unit: int,
    single_prefill_min_tokens: int,
    single_prefill: bool,
) -> int:
    """New-start loop: take the FCFS prefix of the waiting order,
    c = min(prompt, C), new-start count ≤ free_seq; never skip the queue
    head (preserves FCFS and seqno chain order). single_prefill is passed
    in from the resume loop (True means this batch is already closed).
    Returns used."""
    for new_count, cand in enumerate(waiting):
        if single_prefill and picked:
            break
        single_prefill = _is_single_prefill(
            single_prefill_min_tokens, cand.num_prompt_tokens
        )
        if single_prefill and picked:
            break
        remaining = cand.num_prompt_tokens - cand.num_computed_tokens
        if remaining <= 0:
            continue
        chunk = min(remaining, chunk_unit)
        if new_count >= free_seq:
            break
        used, batch_closed = _append_chunk(
            picked, expected, used, cap, cand.request_id, chunk
        )
        if batch_closed:
            break
    return used
