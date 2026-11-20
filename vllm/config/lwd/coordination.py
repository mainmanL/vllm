# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache coordination metadata only; no key reads, clients or server startup."""

from typing import Any

from pydantic.dataclasses import dataclass

from .topology import LwdCoordination, LwdString


@dataclass(frozen=True)
class LwdCoordinationConfig(LwdCoordination):
    instance_id: LwdString = "cloud-0"

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, cloud_id: int | None = None
    ) -> "LwdCoordinationConfig":
        raw = LwdCoordination.reject_legacy_fields(dict(raw))
        defaults = {} if cloud_id is None else {"instance_id": f"cloud-{cloud_id}"}
        return cls(**(defaults | raw))

    def validate_role(self, role: str) -> None:
        if (
            self.enabled
            and role == "edge"
            and not self.tenant_key_file
            and not self.placement_only
        ):
            raise ValueError(
                "[LWD] edge coordination requires tenant_key_file "
                "unless placement_only is true"
            )
