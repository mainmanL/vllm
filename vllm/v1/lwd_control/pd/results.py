# SPDX-License-Identifier: Apache-2.0
"""Engine-owned final results. No tensors, execution slots or remote waits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from .envelope import range_digest, request_fingerprint

_EARLY_REASONS = frozenset(
    {"stop_string", "cancelled", "disconnected", "timeout", "error"}
)


@dataclass(frozen=True)
class LwdPdFinalResult:
    result_id: str
    attempt_id: str
    binding_id: str
    prompt_tokens: int
    cached_tokens: int
    generated_tokens: int
    accepted_tokens: int
    frontend_reason: str
    execution_reason: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["usage"] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.accepted_tokens,
            "total_tokens": self.prompt_tokens + self.accepted_tokens,
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
        }
        return result


def validate_result(result: dict[str, Any]) -> None:
    """Check the immutable record and usage as one canonical publication."""
    try:
        record = LwdPdFinalResult(
            **{field.name: result[field.name] for field in fields(LwdPdFinalResult)}
        )
        canonical = record.to_dict()
        material = asdict(record)
        material.pop("result_id")
        counts = (
            record.prompt_tokens,
            record.cached_tokens,
            record.generated_tokens,
            record.accepted_tokens,
        )
        if (
            any(type(value) is not int for value in counts)
            or not 0 <= record.cached_tokens < record.prompt_tokens
            or not 0 <= record.accepted_tokens <= record.generated_tokens
            or record.frontend_reason == "complete"
            and record.accepted_tokens != record.generated_tokens
            or record.result_id != request_fingerprint(material)
            or canonical != result
        ):
            raise ValueError("invalid PD result counts, digest or usage")
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed PD result") from exc


class LwdPdResultLedger:
    """Merge native submitted G and frontend accepted K exactly once per binding."""

    def __init__(
        self, n_tokens: int, hit_tokens: int, attempt_id: str, binding_id: str
    ) -> None:
        if not attempt_id or not binding_id or not 0 <= hit_tokens <= n_tokens:
            raise ValueError("invalid PD result identity or prompt counts")
        self.n_tokens = n_tokens
        self.hit_tokens = hit_tokens
        self.attempt_id = attempt_id
        self.binding_id = binding_id
        self.generated_tokens = 0
        self.execution_reason: str | None = None
        self._commit: tuple[int, str] | None = None
        self._result: LwdPdFinalResult | None = None

    @property
    def result(self) -> dict[str, Any] | None:
        return self._result.to_dict() if self._result is not None else None

    @property
    def final_g(self) -> int | None:
        return self.generated_tokens if self.execution_reason is not None else None

    def append(self, tokens: list[int]) -> tuple[int, int, str]:
        if self.execution_reason is not None:
            raise ValueError("cannot append after PD execution ended")
        start = self.generated_tokens
        end = start + len(tokens)
        digest = range_digest(self.binding_id, start, end, tokens)
        self.generated_tokens = end
        return start, end, digest

    def finish(self, reason: str) -> dict[str, Any] | None:
        if not reason:
            raise ValueError("PD execution reason must be explicit")
        if self.execution_reason is not None and self.execution_reason != reason:
            raise ValueError("conflicting PD execution terminal")
        self.execution_reason = reason
        return self._merge()

    def commit(
        self, accepted_tokens: int, reason: str, verified_tokens: int
    ) -> dict[str, Any] | None:
        if (
            type(accepted_tokens) is not int
            or type(verified_tokens) is not int
            or not 0 <= accepted_tokens <= verified_tokens
        ):
            raise ValueError("PD K exceeds the verified output prefix")
        if reason != "complete" and reason not in _EARLY_REASONS:
            raise ValueError("unknown PD frontend terminal reason")
        commit = (accepted_tokens, reason)
        if self._commit is not None and self._commit != commit:
            raise ValueError("conflicting PD result commit")
        # Validate before freezing; a rejected message cannot poison the ledger.
        if self.execution_reason is not None:
            self._validate_counts(*commit)
        self._commit = commit
        return self._merge()

    def _validate_counts(self, accepted: int, reason: str) -> None:
        if accepted > self.generated_tokens:
            raise ValueError("PD K exceeds G")
        if reason == "complete" and accepted != self.generated_tokens:
            raise ValueError("complete PD result requires K=G")

    def _merge(self) -> dict[str, Any] | None:
        if self._result is not None:
            return self.result
        if self.execution_reason is None or self._commit is None:
            return None
        accepted, reason = self._commit
        self._validate_counts(accepted, reason)
        fields = {
            "attempt_id": self.attempt_id,
            "binding_id": self.binding_id,
            "prompt_tokens": self.n_tokens,
            "cached_tokens": self.hit_tokens,
            "generated_tokens": self.generated_tokens,
            "accepted_tokens": accepted,
            "frontend_reason": reason,
            "execution_reason": self.execution_reason,
            "status": (
                "error"
                if self.execution_reason in ("error", "preempted")
                else "completed"
                if reason in ("complete", "stop_string")
                else reason
            ),
        }
        self._result = LwdPdFinalResult(result_id=request_fingerprint(fields), **fields)
        return self.result
