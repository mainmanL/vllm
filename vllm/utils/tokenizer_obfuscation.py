# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional startup bridge for the independently packaged mixed tokenizer.

The native default configuration never imports the optional package. Deployment
policy and artifact loading live in that package; this module only identifies
configurations that need its startup hooks.
"""

import os
from importlib import import_module
from typing import Any


def _get(value: Any, key: str, default: Any = None) -> Any:
    return (
        value.get(key, default)
        if isinstance(value, dict)
        else getattr(value, key, default)
    )


def needs_obfuscation_hook(args: Any) -> bool:
    """Recognize explicitly configured capabilities, excluding native auto."""
    if getattr(args, "_tokenizer_obfuscation_preparing", False):
        # The trusted builder must first resolve the unmodified native config.
        # This private in-process marker is intentionally absent from the CLI.
        return False
    structured = getattr(args, "structured_outputs_config", None)
    reasoning = getattr(args, "reasoning_config", None)
    return bool(
        getattr(args, "enable_tokenizer_obfuscation", False)
        or getattr(args, "tokenizer_obfuscation_artifacts", None)
        or getattr(args, "tokenizer_mode", None) == "mixed_bytes"
        or getattr(args, "reasoning_parser", None)
        or getattr(args, "reasoning_parser_plugin", None)
        or getattr(args, "enable_auto_tool_choice", False)
        or getattr(args, "tool_call_parser", None)
        or getattr(args, "tool_strict_level", "auto") not in (None, "auto")
        or _get(structured, "backend", "auto") != "auto"
        or _get(structured, "reasoning_parser", "")
        or _get(structured, "reasoning_parser_plugin", "")
        or _get(structured, "enable_in_reasoning", False)
        or _get(structured, "disable_any_whitespace", False)
        or _get(structured, "disable_additional_properties", False)
        or _get(reasoning, "reasoning_parser", "")
        or _get(reasoning, "reasoning_start_str", "")
        or _get(reasoning, "reasoning_end_str", "")
    )


def run_obfuscation_hook(name: str, args: Any, *extra: Any) -> None:
    """Delegate configuration without adding a dependency to native startup."""
    if needs_obfuscation_hook(args):
        module = import_module("mixed_tokenizer.scheme31_startup")
        getattr(module, name)(args, *extra)


def validate_structured_outputs_with_route(
    params: Any,
    model_config: Any,
    structured_config: Any,
    tokenizer: Any,
    validate: Any,
) -> None:
    """Keep original-tokenizer validation native; bind mixed request routing."""
    if getattr(tokenizer, "mixed", None) is not None and os.environ.get(
        "MIXED_TOKENIZER_STARTUP_MANIFEST"
    ):
        module = import_module("mixed_tokenizer.scheme31_backend_route")
        return module.validate_native_request(
            params, model_config, structured_config, tokenizer, validate
        )
    return validate(model_config, structured_config, tokenizer)


def prepare_structured_output_backend_route(
    params: Any, structured_config: Any, tokenizer: Any, backend: Any
) -> str | None:
    if getattr(tokenizer, "mixed", None) is not None and os.environ.get(
        "MIXED_TOKENIZER_STARTUP_MANIFEST"
    ):
        module = import_module("mixed_tokenizer.scheme31_backend_route")
        return module.prepare_engine_backend(
            params, structured_config, tokenizer, backend
        )
    return None


def confirm_structured_output_backend_route(selected: str | None, backend: Any) -> None:
    if selected is not None:
        module = import_module("mixed_tokenizer.scheme31_backend_route")
        module.confirm_engine_backend(selected, backend)
