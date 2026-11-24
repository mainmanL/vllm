# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Discover and reconcile runtime LWD Python sources in both repositories.

The scope is declarative (``lwd_source_scope.json``) and deliberately based on
package-relative path rules rather than a hand-maintained module list.  Both
plaintext ``.py`` files and encrypted ``.py.lwdc`` payloads are discovered, so
running discovery after a full build still reports the complete inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import lwd_source_runtime as runtime

_TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_SCOPE = _TOOL_DIR / "lwd_source_scope.json"
_SKIPPED_DIRECTORIES = frozenset(
    {
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "docs",
        "tests",
        "tools",
    }
)


@dataclass(frozen=True)
class DiscoveredModule:
    """One logical source path and its on-disk plaintext/payload state."""

    repository: str
    package: str
    module: str
    source: str
    payload: str
    origin: Path
    payload_path: Path
    plaintext_present: bool
    payload_present: bool

    @property
    def state(self) -> str:
        if self.plaintext_present and self.payload_present:
            return "both"
        if self.plaintext_present:
            return "plaintext"
        if self.payload_present:
            return "encrypted"
        return "missing"

    @property
    def identity(self) -> tuple[str, str]:
        return self.repository, self.source


def _fail(message: str) -> None:
    raise runtime.LwdSourceProtectionError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot load scope {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"scope {path} must contain a JSON object")
    return value


def load_scope(path: Path = DEFAULT_SCOPE) -> dict[str, Any]:
    """Load and strictly validate the small, reviewable scope file."""
    if path.is_file():
        scope = _load_json(path)
    else:
        scope = globals().get("__bundled_scope__")
        if scope is None:
            _fail(f"scope file does not exist and no embedded scope is present: {path}")
    if scope.get("format_version") != 1:
        _fail(f"{path}: format_version must be 1")
    for field, expected in (
        ("directory_prefix", "lwd"),
        ("file_stem_prefix", "lwd"),
        ("source_suffix", ".py"),
        ("payload_suffix", ".py.lwdc"),
    ):
        if scope.get(field) != expected:
            _fail(f"{path}: {field} must be {expected!r}")
    repositories = scope.get("repositories")
    if not isinstance(repositories, dict) or set(repositories) != {
        "vllm",
        "vllm-ascend",
    }:
        _fail(f"{path}: repositories must declare exactly vllm and vllm-ascend")
    total = 0
    for name, item in repositories.items():
        if not isinstance(item, dict):
            _fail(f"{path}: repository {name} must be an object")
        package = item.get("package")
        expected = item.get("expected_modules")
        required = item.get("required_paths")
        if package not in ("vllm", "vllm_ascend"):
            _fail(f"{path}: invalid package for {name}: {package!r}")
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            _fail(f"{path}: expected_modules for {name} must be a positive integer")
        if not isinstance(required, list) or not required:
            _fail(f"{path}: required_paths for {name} must be a non-empty array")
        for relative in required:
            runtime.validate_relative_path(relative, f"{path}: required_paths")
            if PurePosixPath(relative).parts[0] != package:
                _fail(f"{path}: required path {relative!r} is outside {package}")
        total += expected
    if scope.get("expected_total") != total:
        _fail(f"{path}: expected_total must equal the per-repository sum {total}")
    return scope


def _in_scope(package_root: Path, source: Path, scope: dict[str, Any]) -> bool:
    relative = source.relative_to(package_root)
    if any(part.startswith(scope["directory_prefix"]) for part in relative.parts[:-1]):
        return True
    return source.stem.startswith(scope["file_stem_prefix"])


def _module_name(package: str, package_root: Path, source: Path) -> str:
    relative = source.relative_to(package_root)
    parts = list(relative.parts)
    parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        _fail(f"the top-level package {package} itself cannot be protected")
    return ".".join((package, *parts))


def _check_required_paths(root: Path, item: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for relative in item["required_paths"]:
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.exists():
            continue
        if relative.endswith(".py") and path.with_name(path.name + ".lwdc").is_file():
            continue
        problems.append(f"required scope path is missing: {path}")
    return problems


def discover_modules(
    roots: dict[str, str | os.PathLike[str]],
    scope_path: Path = DEFAULT_SCOPE,
    *,
    require_built: bool = False,
) -> list[DiscoveredModule]:
    """Return the deterministic inventory or raise on any scope inconsistency.

    ``require_built`` is used by post-build/static validation.  During an
    encryption build, plaintext-only and plaintext+payload states are valid so
    existing payloads can be reused when their decrypted bytes are unchanged.
    """
    scope = load_scope(scope_path)
    found: dict[tuple[str, str], dict[str, Any]] = {}
    problems: list[str] = []
    casefold_paths: dict[tuple[str, str], str] = {}
    module_owners: dict[str, str] = {}

    for repository, item in scope["repositories"].items():
        if repository not in roots:
            _fail(f"missing repository root argument for {repository}")
        root = Path(roots[repository]).resolve()
        package = item["package"]
        package_root = root / package
        if not package_root.is_dir():
            problems.append(f"package root does not exist: {package_root}")
            continue
        problems.extend(_check_required_paths(root, item))
        for dirpath, dirnames, filenames in os.walk(package_root):
            dirnames[:] = sorted(
                name for name in dirnames if name not in _SKIPPED_DIRECTORIES
            )
            for filename in sorted(filenames):
                is_payload = filename.endswith(scope["payload_suffix"])
                is_plaintext = filename.endswith(scope["source_suffix"])
                if filename.endswith(".lwdc") and not is_payload:
                    problems.append(
                        f"orphan non-source payload: {Path(dirpath) / filename}"
                    )
                    continue
                if not is_payload and not is_plaintext:
                    continue
                disk_path = Path(dirpath) / filename
                source = (
                    disk_path.with_name(filename[: -len(".lwdc")])
                    if is_payload
                    else disk_path
                )
                source_relative = source.relative_to(root).as_posix()
                if not _in_scope(package_root, source, scope):
                    if is_payload:
                        problems.append(f"orphan out-of-scope payload: {disk_path}")
                    continue
                identity = (repository, source_relative)
                record = found.setdefault(
                    identity,
                    {
                        "package": package,
                        "source_path": source,
                        "plaintext": None,
                        "payload": None,
                    },
                )
                state_field = "payload" if is_payload else "plaintext"
                if record[state_field] is not None:
                    problems.append(f"duplicate {state_field} for {source_relative}")
                record[state_field] = disk_path
                folded = (repository, source_relative.casefold())
                previous = casefold_paths.get(folded)
                if previous is not None and previous != source_relative:
                    problems.append(
                        f"case-colliding source paths: {previous} and {source_relative}"
                    )
                casefold_paths[folded] = source_relative

    discoveries: list[DiscoveredModule] = []
    for (repository, source_relative), record in sorted(found.items()):
        root = Path(roots[repository]).resolve()
        source = record["source_path"]
        package = record["package"]
        module = _module_name(package, root / package, source)
        previous = module_owners.get(module)
        if previous is not None:
            problems.append(
                f"duplicate module {module}: {previous} and {source_relative}"
            )
        module_owners[module] = source_relative
        origin = root.joinpath(*PurePosixPath(source_relative).parts)
        payload_relative = source_relative + ".lwdc"
        payload = root.joinpath(*PurePosixPath(payload_relative).parts)
        discovery = DiscoveredModule(
            repository=repository,
            package=package,
            module=module,
            source=source_relative,
            payload=payload_relative,
            origin=origin,
            payload_path=payload,
            plaintext_present=record["plaintext"] is not None,
            payload_present=record["payload"] is not None,
        )
        discoveries.append(discovery)
        if require_built:
            if discovery.plaintext_present:
                problems.append(f"unexpected plaintext after build: {origin}")
            if not discovery.payload_present:
                problems.append(f"missing payload after build: {payload}")

    expected_total = 0
    for repository, item in scope["repositories"].items():
        actual = sum(item.repository == repository for item in discoveries)
        expected = item["expected_modules"]
        expected_total += expected
        if actual != expected:
            problems.append(
                f"{repository} scope count mismatch: discovered {actual}, "
                f"expected {expected}"
            )
    if len(discoveries) != scope["expected_total"]:
        problems.append(
            f"total scope count mismatch: discovered {len(discoveries)}, "
            f"expected {scope['expected_total']}"
        )
    if expected_total != scope["expected_total"]:
        problems.append("scope expected_total is inconsistent")
    if problems:
        raise runtime.LwdSourceProtectionError(
            "source discovery failed:\n  - " + "\n  - ".join(sorted(problems))
        )
    return discoveries


def reconcile_manifest(
    discoveries: list[DiscoveredModule], manifest: dict[str, Any]
) -> None:
    """Fail on missing, orphan, duplicate, or renamed manifest entries."""
    discovered = {item.identity: item for item in discoveries}
    manifested: dict[tuple[str, str], dict[str, Any]] = {}
    problems: list[str] = []
    for entry in manifest.get("modules", []):
        identity = (entry.get("repository"), entry.get("source"))
        if identity in manifested:
            problems.append(f"duplicate manifest source: {identity}")
            continue
        manifested[identity] = entry
    for identity in sorted(set(discovered) | set(manifested)):
        discovery = discovered.get(identity)
        entry = manifested.get(identity)
        if discovery is None:
            problems.append(f"manifest source was not discovered: {identity}")
            continue
        if entry is None:
            problems.append(f"discovered source is missing from manifest: {identity}")
            continue
        if entry.get("module") != discovery.module:
            problems.append(
                f"module mismatch for {identity}: {entry.get('module')} != "
                f"{discovery.module}"
            )
        if entry.get("payload") != discovery.payload:
            problems.append(f"payload path mismatch for {identity}")
    if problems:
        raise runtime.LwdSourceProtectionError(
            "manifest/discovery reconciliation failed:\n  - " + "\n  - ".join(problems)
        )


def _scope_sha256(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    embedded = globals().get("__bundled_scope_sha256__")
    if not isinstance(embedded, str):
        _fail(f"cannot fingerprint missing scope {path}")
    return embedded


def build_manifest(
    discoveries: list[DiscoveredModule],
    previous: dict[str, Any] | None,
    scope_path: Path,
) -> dict[str, Any]:
    """Create a stable manifest, retaining hashes for already-known sources."""
    scope = load_scope(scope_path)
    previous = previous or {}
    old_entries = {
        (entry["repository"], entry["source"]): entry
        for entry in previous.get("modules", [])
    }
    discovered_identities = {item.identity for item in discoveries}
    missing = sorted(set(old_entries) - discovered_identities)
    if missing:
        raise runtime.LwdSourceProtectionError(
            f"existing manifest entries are no longer discoverable: {missing}"
        )
    modules: list[dict[str, Any]] = []
    for discovery in sorted(discoveries, key=lambda item: item.identity):
        old = old_entries.get(discovery.identity, {})
        if old and old.get("module") != discovery.module:
            raise runtime.LwdSourceProtectionError(
                f"module name changed for {discovery.identity}: "
                f"{old.get('module')} -> {discovery.module}"
            )
        entry: dict[str, Any] = {
            "module": discovery.module,
            "repository": discovery.repository,
            "source": discovery.source,
            "payload": discovery.payload,
        }
        for field in (
            "plaintext_sha256",
            "plaintext_bytes",
            "payload_sha256",
            "payload_bytes",
            "canary_line_sha256",
        ):
            if field in old:
                entry[field] = old[field]
        modules.append(entry)
    return {
        "tool": runtime.TOOL_NAME,
        "description": (
            "Deterministic full-runtime LWD source inventory generated by "
            "discover_lwd_sources.py; do not hand-edit module entries."
        ),
        "format_version": runtime.FORMAT_VERSION,
        "cipher": runtime.CIPHER_NAME,
        "nonce_bytes": runtime.NONCE_BYTES,
        "key_file": runtime.KEY_FILENAME,
        "key_policy": (
            "Runtime accepts only env:LWD_SOURCE_KEY_B64. The ignored local "
            "key file is solely a build/ELF provisioning asset."
        ),
        "scope": {
            "file": scope_path.name,
            "sha256": _scope_sha256(scope_path),
        },
        "expected_total": scope["expected_total"],
        "repositories": {
            name: {
                "package": item["package"],
                "expected_modules": item["expected_modules"],
            }
            for name, item in scope["repositories"].items()
        },
        "modules": modules,
        "verification": previous.get("verification", {}),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Serialize a manifest deterministically with repository-relative paths."""
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def inventory_report(discoveries: list[DiscoveredModule]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    states: dict[str, int] = {}
    for item in discoveries:
        counts[item.repository] = counts.get(item.repository, 0) + 1
        states[item.state] = states.get(item.state, 0) + 1
    return {
        "total": len(discoveries),
        "repositories": dict(sorted(counts.items())),
        "states": dict(sorted(states.items())),
        "modules": [
            {
                "repository": item.repository,
                "module": item.module,
                "source": item.source,
                "payload": item.payload,
                "state": item.state,
            }
            for item in discoveries
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--ascend-root", required=True, type=Path)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--require-built", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        discoveries = discover_modules(
            {"vllm": args.vllm_root, "vllm-ascend": args.ascend_root},
            args.scope.resolve(),
            require_built=args.require_built,
        )
        report = inventory_report(discoveries)
        if args.json_output:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(
                f"discovered {report['total']} modules: "
                f"{report['repositories']}; states={report['states']}"
            )
            for item in report["modules"]:
                print(
                    f"[{item['state']}] {item['repository']}: "
                    f"{item['source']} -> {item['module']}"
                )
        return 0
    except runtime.LwdSourceProtectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
