# SPDX-License-Identifier: Apache-2.0
"""Validate recovered PD output before exposing it to a frontend."""

from __future__ import annotations

from .envelope import range_digest


class LwdPdOutputStream:
    """One attempt/binding's contiguous, verified user-token prefix."""

    def __init__(self, attempt_id: str, binding_id: str) -> None:
        self.attempt_id = attempt_id
        self.binding_id = binding_id
        self.verified_tokens = 0
        self.final_g: int | None = None
        self._intervals: dict[tuple[int, int], tuple[str, int]] = {}

    def accept(
        self,
        interval: tuple[int, int],
        tokens: list[int],
        digest: str,
        physical_rows: int,
        final_g: int = -1,
    ) -> bool:
        """Return False for an exact replay; invalid/overlapping output raises."""
        start, end = interval
        if physical_rows < len(tokens) or end - start != len(tokens):
            raise ValueError("PD output frame geometry mismatch")
        if not digest or digest != range_digest(self.binding_id, start, end, tokens):
            raise ValueError("PD recovered output digest mismatch")
        previous = self._intervals.get(interval)
        if previous is not None:
            # A zero-row live padding record may later close that same endpoint.
            if (
                start == end == self.verified_tokens
                and previous == (digest, -1)
                and final_g == end
                and self.final_g is None
            ):
                self._intervals = {interval: (digest, final_g)}
                self.final_g = final_g
                return True
            if previous != (digest, final_g):
                raise ValueError("conflicting PD output replay")
            return False
        if start != self.verified_tokens:
            raise ValueError("PD output gap or overlap")
        if self.final_g is not None:
            raise ValueError("PD output after stream termination")
        if final_g != -1 and final_g != end:
            raise ValueError("PD final G does not match the output endpoint")
        self._intervals[interval] = (digest, final_g)
        self.verified_tokens = end
        if final_g >= 0:
            self.final_g = final_g
            # Physical frame replay is deduplicated before receiving. Once G
            # is final, result retention needs only this terminal signature.
            self._intervals = {interval: (digest, final_g)}
        return True
