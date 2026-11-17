# SPDX-License-Identifier: Apache-2.0
"""Binding and credit state retained independently of edge model execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .envelope import validate_handoff_plan
from .output import LwdPdOutputStream


@dataclass
class LwdPdEdgeBinding:
    connection: Any
    notify: Any
    sent_windows: int = 0
    returned_windows: int = 0
    output_watermark: int = 0

    @property
    def available_windows(self) -> int:
        return (
            min(self.notify.recv_windows, self.notify.max_inflight_windows)
            + self.returned_windows
            - self.sent_windows
        )

    def release(self, watermark: int) -> None:
        if type(watermark) is not int or not 0 <= watermark <= self.sent_windows:
            raise ValueError("PD receive credit exceeds sent windows")
        self.returned_windows = max(self.returned_windows, watermark)


@dataclass
class LwdPdEdgeState:
    request_id: str
    attempt_id: str
    fingerprint: str
    n_tokens: int
    deadline: float
    bind_deadline: float
    request: Any
    envelope: dict[str, Any] | None
    bindings: dict[str, LwdPdEdgeBinding] = field(default_factory=dict)
    output: LwdPdOutputStream | None = None
    p_done: str = ""
    cancelled: bool = False
    execution_done: bool = False
    frontend_commit: tuple[int, str, bool] | None = None
    frontend_verified_tokens: int = 0
    receipt: dict[str, Any] | None = None
    http_result: dict[str, Any] | None = None
    error: str = ""
    tail_pending: bool = False

    def bind(self, connection: Any, notify: Any) -> bool:
        if (
            notify.attempt_id != self.attempt_id
            or notify.fingerprint != self.fingerprint
        ):
            raise ValueError("PD BIND changed request semantics")
        if notify.stage_role not in ("prefill", "decode") or not notify.binding_id:
            raise ValueError("invalid PD stage binding")
        if not notify.accepted:
            raise ValueError(notify.error or "PD stage rejected registration")
        validate_handoff_plan(self.n_tokens, notify.p_end, notify.i_end, notify.r_pos)
        if not 0 <= notify.hit_tokens < notify.p_end:
            raise ValueError("PD frozen prefix hit exceeds P plan")
        if (
            notify.max_inflight_windows < 1
            or notify.recv_windows < 1
            or notify.max_window_tokens < 0
        ):
            raise ValueError("invalid PD input credits")
        previous = self.bindings.get(notify.stage_role)
        if previous is not None:
            if previous.connection != connection or previous.notify != notify:
                raise ValueError("conflicting PD stage binding")
            return False
        if notify.stage_role == "decode":
            prefill = self.bindings.get("prefill")
            if prefill is None or notify.source_p_binding != prefill.notify.binding_id:
                raise ValueError("PD decode refers to another prefill binding")
            if (notify.p_end, notify.i_end, notify.r_pos, notify.hit_tokens) != (
                prefill.notify.p_end,
                prefill.notify.i_end,
                prefill.notify.r_pos,
                prefill.notify.hit_tokens,
            ):
                raise ValueError("PD decode changed the frozen handoff plan")
            if (
                not notify.source_export_id
                or self.p_done
                and self.p_done != notify.source_export_id
            ):
                raise ValueError("PD decode refers to another export")
            self.output = LwdPdOutputStream(self.attempt_id, notify.binding_id)
            self.tail_pending = True
        self.bindings[notify.stage_role] = LwdPdEdgeBinding(connection, notify)
        return True

    def match(
        self, connection: Any, attempt_id: str, binding_id: str
    ) -> LwdPdEdgeBinding | None:
        if attempt_id != self.attempt_id:
            return None
        return next(
            (
                binding
                for binding in self.bindings.values()
                if binding.connection == connection
                and binding.notify.binding_id == binding_id
            ),
            None,
        )

    def freeze(self, accepted: int, reason: str, cancel: bool) -> None:
        verified = self.output.verified_tokens if self.output else 0
        if type(accepted) is not int or not 0 <= accepted <= verified:
            raise ValueError("PD frontend K exceeds verified tokens")
        if reason not in (
            "complete",
            "stop_string",
            "cancelled",
            "disconnected",
            "timeout",
            "error",
        ):
            raise ValueError("unknown PD frontend reason")
        commit = accepted, reason, cancel
        if self.frontend_commit is not None and self.frontend_commit != commit:
            raise ValueError("conflicting PD frontend finalization")
        if self.frontend_commit is None:
            self.frontend_verified_tokens = verified
        self.frontend_commit = commit
