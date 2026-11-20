# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""PD decode admission, independent of the prompt-staging scheduler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
        LwdCloudScheduler,
    )


@dataclass
class LwdPdDecodeAdmission:
    request_id: str
    binding_id: str
    attempt_id: str
    tail_start: int
    tail_end: int
    output_quota: int
    kv_ready: bool = False
    tail_ready: bool = False
    tail_inflight: bool = False
    tail_done: bool = False
    cancelled: bool = False


class LwdPdCloudScheduling:
    """Keep imports progressing while ready tails join existing decodes."""

    def __init__(self, scheduler: LwdCloudScheduler) -> None:
        self.scheduler = scheduler
        self.records: dict[str, LwdPdDecodeAdmission] = {}
        self.fail: Callable[[str, str], None] | None = None

    def register(self, record: LwdPdDecodeAdmission) -> None:
        if record.request_id in self.records:
            raise ValueError(f"duplicate PD decode admission: {record.request_id}")
        self.records[record.request_id] = record

    def note_arrival(self, request_id: str, end: int) -> None:
        record = self.records.get(request_id)
        if record is not None and not record.cancelled and end >= record.tail_end:
            record.tail_ready = True

    def consumed(
        self, request_ids: set[str], identities: dict[str, tuple[str, str]]
    ) -> None:
        for request_id in request_ids:
            record = self.records.get(request_id)
            if (
                record is not None
                and record.tail_inflight
                and not record.cancelled
                and identities.get(request_id) == (record.attempt_id, record.binding_id)
            ):
                record.tail_done = True
                record.tail_inflight = False

    def reconcile_rows(self, scheduled: SchedulerOutput, output: object) -> None:
        actual: dict[tuple[str, str, str], int] = {}
        metadata = getattr(output, "lwd_pd_down_metadata", {})
        for carrier in getattr(output, "lwd_down_carriers", ()) or ():
            if len(carrier) < 5:
                continue
            fields = metadata.get((carrier[4], carrier[3]))
            if fields is None:
                continue
            for request_id, attempt, binding, count in zip(
                carrier[1],
                fields["attempt_ids"],
                fields["binding_ids"],
                fields["row_counts"],
            ):
                actual[request_id, attempt, binding] = int(count)
        for request_id, (attempt, binding, reserved) in getattr(
            scheduled, "lwd_pd_reserved_rows", {}
        ).items():
            record = self.records.get(request_id)
            if record is None or (record.attempt_id, record.binding_id) != (
                attempt,
                binding,
            ):
                continue
            physical = actual.get((request_id, attempt, binding), 0)
            if physical > reserved:
                self._fail(request_id, "PD physical output exceeded its reserved rows")
            elif record.output_quota >= 0:
                # Reserve before async dispatch, then charge actual physical
                # rows. Unused speculative slots were never sent to the edge.
                record.output_quota += reserved - physical

    def _fail(self, request_id: str, reason: str) -> None:
        record = self.records.get(request_id)
        if record is not None:
            record.cancelled = True
        if self.fail is None:
            raise RuntimeError(reason)
        self.fail(request_id, reason)

    def _collect(self) -> tuple[list[str], list[str], list[str]]:
        from vllm.v1.request import RequestStatus

        imports, tails, decodes = [], [], []
        scheduler = self.scheduler
        for request_id, record in list(self.records.items()):
            request = scheduler.requests.get(request_id)
            if request is None or request.is_finished() or record.cancelled:
                continue
            if request.num_preemptions:
                self._fail(request_id, "PD preemption recovery is unsupported")
                continue
            if request_id in scheduler.failed_recving_kv_req_ids:
                self._fail(request_id, "PD remote KV import failed")
                continue
            if not record.kv_ready:
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    if request_id not in scheduler.finished_recving_kv_req_ids:
                        continue
                    scheduler._update_waiting_for_remote_kv(request)
                    request.status = RequestStatus.WAITING
                    if request.num_computed_tokens != record.tail_start:
                        self._fail(request_id, "PD imported KV does not land at R")
                        continue
                    record.kv_ready = True
                else:
                    # Native admission invokes the connector with zero compute
                    # tokens and enters WAITING_FOR_REMOTE_KVS. No tail needed.
                    imports.append(request_id)
                    continue
            if record.output_quota == 0:
                continue
            if record.tail_done:
                physical = 1 + getattr(scheduler, "num_spec_tokens", 0)
                if record.output_quota < 0 or record.output_quota >= physical:
                    decodes.append(request_id)
            elif record.tail_ready and not record.tail_inflight:
                if request.num_computed_tokens != record.tail_start:
                    self._fail(
                        request_id, "PD tail position changed before consumption"
                    )
                else:
                    tails.append(request_id)
        return imports, tails, decodes

    def schedule(self) -> SchedulerOutput:
        scheduler = self.scheduler
        imports, tails, decodes = self._collect()
        room = (
            scheduler.max_num_scheduled_tokens
            - scheduler._lwd_compute_decode_consumption(decodes)
        )
        selected = []
        for request_id in tails:
            record = self.records[request_id]
            size = record.tail_end - record.tail_start
            if size <= room:
                selected.append(request_id)
                room -= size
                scheduler.requests[request_id].lwd_window_cap = size
        for request_id in imports:
            scheduler.requests[request_id].lwd_pd_import_only = True
        try:
            # One native batch: existing MTP decode, ready tails and zero-token
            # connector imports. No global speculative/placeholder overrides.
            out = scheduler._lwd_schedule(decodes + selected + imports)
        finally:
            for request_id in imports:
                request = scheduler.requests.get(request_id)
                if request is not None:
                    request.lwd_pd_import_only = False
            for request_id in selected:
                request = scheduler.requests.get(request_id)
                if request is not None:
                    request.lwd_window_cap = None
        for request_id in imports:
            request = scheduler.requests.get(request_id)
            if request is not None and getattr(request, "lwd_pd_import_failed", False):
                self._fail(
                    request_id, "PD connector did not start an asynchronous KV import"
                )
        for request_id in selected:
            count = out.num_scheduled_tokens.get(request_id, 0)
            if count:
                record = self.records[request_id]
                if count != record.tail_end - record.tail_start:
                    self._fail(
                        request_id, "PD native tail admission exceeded real input"
                    )
                else:
                    record.tail_inflight = True
        reserved_rows = {}
        for request_id, count in out.num_scheduled_tokens.items():
            record = self.records.get(request_id)
            if record is not None and count and record.output_quota > 0:
                # Credit authorizes physical output rows, including speculation.
                charge = max(
                    1 + len(out.scheduled_spec_decode_tokens.get(request_id, ())),
                    1 + getattr(scheduler, "num_spec_tokens", 0)
                    if record.tail_done
                    else 1,
                )
                if charge > record.output_quota:
                    self._fail(request_id, "PD output exceeded its authorized rows")
                else:
                    record.output_quota -= charge
                    reserved_rows[request_id] = (
                        record.attempt_id,
                        record.binding_id,
                        charge,
                    )
        out.lwd_pd_reserved_rows = reserved_rows
        return out
