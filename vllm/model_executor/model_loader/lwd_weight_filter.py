# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skip checkpoint tensors of modules this LWD role does not build.

An LWD edge/cloud role only constructs the modules it actually runs and
holds ``PPMissingLayer`` / ``StageMissingLayer`` placeholders for the rest
(see ``LWD_MODULE_OWNERSHIP`` in :mod:`vllm.model_executor.models.utils`).
Pointing such a process at a full checkpoint still loads -- the weights
loader silently drops placeholder tensors -- but every shard is read from
disk first. Mirroring :mod:`vllm.model_executor.model_loader.ep_weight_filter`
(which does the same for experts a rank does not own), this filter drops
the placeholder tensors *before* ``get_tensor`` so a trimmed role reads
only its own slice of the shards.

Skip keys are derived from the same ownership table the build-time guards
use, keeping both layers in agreement. The import is function-local on
purpose: ``models.utils`` imports ``model_loader`` at module scope, so a
module-level import here would be circular.
"""

from __future__ import annotations

from functools import cache

# DeepSeek-V4 Hash MoE routing tables (``model.layers.N.mlp.gate.tid2eid``).
# The edge builds no layers yet consumes these tables straight from the
# checkpoint shards, so they are exempt from the edge's ``model.layers``
# skip rule. Mirrors ``_TID2EID_PATTERN`` in
# scripts/lwd/split_lwd_checkpoint.py, which keeps the same tensors on the
# edge slice.
_TID2EID_SUBSTR = "tid2eid"


@cache
def parse_skip_keys(role: str) -> frozenset[str]:
    """The module-path keys *role* skips, derived from LWD_MODULE_OWNERSHIP."""
    from vllm.model_executor.models.utils import LWD_MODULE_OWNERSHIP

    return frozenset(
        suffix for suffix, owners in LWD_MODULE_OWNERSHIP.items() if role not in owners
    )


def lwd_should_skip_weight(name: str, role: str | None) -> bool:
    """Whether checkpoint tensor *name* belongs to a module *role* skips.

    Checkpoint names differ from module paths in two ways, both covered:
    native DeepSeek checkpoints drop the leading ``model`` component
    (``layers.3.mlp.w1``, ``norm.weight``), so each key matches either its
    dotted form (``model.norm.``) or its last component (``norm.``); and
    matching is anchored at the start of the tensor name, which keeps
    tower-internal names like ``visual.merger.norm.weight`` away from the
    ``norm`` rule. ``model.layers`` matches per-layer prefixes
    (``model.layers.3.``) the way make_layers names them. A None role
    (non-LWD deployment) skips nothing.
    """
    if role is None:
        return False
    if _TID2EID_SUBSTR in name:
        return False
    for key in parse_skip_keys(role):
        stem = key.rsplit(".", 1)[-1]
        if name.startswith(key + ".") or name.startswith(stem + "."):
            return True
    return False
