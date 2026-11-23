# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Command line entry point for the LWD source-encryption PoC.

Subcommands:

``build``      encrypt the manifest modules, update the manifest and delete the
               plaintext sources (local development worktree).
``install``    validate everything, then install the runtime hook, its JSON
               configuration and ``lwd_source_protection.pth`` into purelib.
``launch``     ``install`` followed by ``os.execvp`` of the wrapped command.
``verify``     independent end-to-end check (fresh interpreters, spawn,
               inspect/linecache, optional payload tamper).
``uninstall``  remove only the three files this tool installed.

Run ``python tools/lwd_source_encryption/lwd_poc.py <command> --help`` for the
per-command options.  See ``README.md`` for the full workflow.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import lwd_source_runtime as runtime
import pybase64 as base64

_TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = _TOOL_DIR / runtime.MANIFEST_FILENAME
DEFAULT_KEY = _TOOL_DIR / runtime.KEY_FILENAME
CANARY_COUNT = 8
CANARY_MIN_LINE_LENGTH = 40
MAX_SCANNED_FILE_BYTES = 4 * 1024 * 1024
SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
    }
)
TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".pyx",
        ".txt",
        ".md",
        ".rst",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".sh",
        ".bash",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".cmake",
        ".patch",
        ".diff",
        ".ld",
    }
)
_PTH_TEXT = (
    f"{runtime.PTH_MARKER}\n"
    "import lwd_source_runtime as _lwd_source_runtime; "
    "_lwd_source_runtime.install(suppress_errors=True)\n"
)


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #


def _log(message: str) -> None:
    print(message, flush=True)


def _short_name(module: str) -> str:
    return module.rsplit(".", maxsplit=1)[-1]


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes_atomic(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    """Write ``data`` next to ``path`` and atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".lwd_tmp_")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _write_text_atomic(path: Path, text: str, *, mode: int = 0o644) -> None:
    _write_bytes_atomic(path, text.encode("utf-8"), mode=mode)


def _write_if_changed(path: Path, data: bytes, *, mode: int = 0o644) -> bool:
    """Write ``data`` only when it differs from the current file content."""
    with contextlib.suppress(OSError):
        if path.is_file() and path.read_bytes() == data:
            return False
    _write_bytes_atomic(path, data, mode=mode)
    return True


def _ensure_cryptography() -> None:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
    except ImportError as exc:
        raise runtime.LwdSourceProtectionError(
            "the 'cryptography' package (AESGCM) is required: "
            f"{exc}. Install it with `uv pip install cryptography`."
        ) from exc


def _ensure_writable_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise runtime.LwdSourceProtectionError(f"cannot create {path}: {exc}") from exc
    if not os.access(str(path), os.W_OK):
        raise runtime.LwdSourceProtectionError(f"{path} is not writable")
    handle, probe = tempfile.mkstemp(dir=str(path), prefix=".lwd_probe_")
    os.close(handle)
    os.unlink(probe)


def _resolve_manifest(args: argparse.Namespace) -> Path:
    path = Path(args.manifest).resolve() if args.manifest else DEFAULT_MANIFEST
    if not path.is_file():
        raise runtime.LwdSourceProtectionError(
            f"manifest {path} does not exist; create it or pass --manifest"
        )
    return path


def _resolve_key(
    args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any]
) -> Path:
    if getattr(args, "key", None):
        return Path(args.key).resolve()
    declared = manifest.get("key_file")
    if isinstance(declared, str) and declared:
        runtime.validate_relative_path(declared, "manifest.key_file")
        return (manifest_path.parent / PurePosixPath(declared)).resolve()
    return DEFAULT_KEY


def _load_or_create_key(path: Path) -> tuple[bytes, bool]:
    if path.is_file():
        return runtime.load_key_file(path), False
    key = os.urandom(runtime.AES_KEY_BYTES)
    encoded = base64.b64encode(key).decode("ascii") + "\n"
    _write_bytes_atomic(path, encoded.encode("ascii"), mode=0o600)
    return key, True


def _repository_roots(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> dict[str, str]:
    """Map manifest repository names onto the roots given on the command line."""
    from_flags = {"vllm": args.vllm_root, "vllm-ascend": args.ascend_root}
    roots: dict[str, str] = {}
    for name in manifest["repositories"]:
        if name not in from_flags:
            raise runtime.LwdSourceProtectionError(
                f"repository {name!r} is not covered by --vllm-root/--ascend-root"
            )
        root = Path(from_flags[name]).resolve()
        if not root.is_dir():
            raise runtime.LwdSourceProtectionError(
                f"repository root {root} for {name!r} does not exist"
            )
        roots[name] = str(root)
    return roots


def _entry_paths(roots: dict[str, str], entry: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(roots[entry["repository"]])
    origin = runtime._resolve_inside(
        root, entry["source"], f"{entry['module']}: source"
    )
    payload = runtime._resolve_inside(
        root, entry["payload"], f"{entry['module']}: payload"
    )
    return origin, payload


def _module_pyc_paths(origin: Path) -> list[Path]:
    """Return the bytecode files the regular path finder could pick up."""
    stem = origin.name[: -len(".py")] if origin.name.endswith(".py") else origin.stem
    found: list[Path] = []
    cache_dir = origin.parent / "__pycache__"
    if cache_dir.is_dir():
        found.extend(sorted(cache_dir.glob(f"{stem}.*.pyc")))
    legacy = origin.parent / f"{stem}.pyc"
    if legacy.is_file():
        found.append(legacy)
    return found


def _remove_module_pyc(origin: Path) -> list[Path]:
    removed: list[Path] = []
    for cached in _module_pyc_paths(origin):
        try:
            cached.unlink()
        except OSError as exc:
            raise runtime.LwdSourceProtectionError(
                f"cannot remove stale bytecode {cached}: {exc}"
            ) from exc
        removed.append(cached)
    return removed


# --------------------------------------------------------------------------- #
# Text scan / canary helpers                                                  #
# --------------------------------------------------------------------------- #


def _iter_text_files(roots: dict[str, str], excluded: set[str]) -> list[str]:
    files: list[str] = []
    for root in sorted(roots.values()):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                name for name in dirnames if name not in SKIPPED_DIRECTORIES
            )
            for name in sorted(filenames):
                suffix = os.path.splitext(name)[1].lower()
                if suffix not in TEXT_SUFFIXES:
                    continue
                path = os.path.join(dirpath, name)
                if os.path.normcase(os.path.abspath(path)) in excluded:
                    continue
                try:
                    if os.path.getsize(path) > MAX_SCANNED_FILE_BYTES:
                        continue
                except OSError:
                    continue
                files.append(path)
    return files


def _scan_line_hashes(
    roots: dict[str, str], wanted_by_length: dict[int, set[str]], excluded: set[str]
) -> dict[str, set[str]]:
    """Find candidate canary lines anywhere in the repository trees."""
    hits: dict[str, set[str]] = {}
    lengths = set(wanted_by_length)
    for path in _iter_text_files(roots, excluded=excluded):
        try:
            raw = Path(path).read_bytes()
        except OSError:
            continue
        for line in raw.decode("utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if len(stripped) not in lengths:
                continue
            digest = _sha256_hex(stripped.encode("utf-8"))
            if digest in wanted_by_length[len(stripped)]:
                hits.setdefault(digest, set()).add(path)
    return hits


def _canary_candidates(plaintext: bytes) -> list[tuple[int, str]]:
    """Longest-first distinct lines of one source, as ``(length, digest)``."""
    seen: set[str] = set()
    candidates: list[tuple[int, str]] = []
    for line in plaintext.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if len(stripped) < CANARY_MIN_LINE_LENGTH or stripped in seen:
            continue
        seen.add(stripped)
        candidates.append((len(stripped), _sha256_hex(stripped.encode("utf-8"))))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates


def _select_canaries(
    manifest: dict[str, Any],
    roots: dict[str, str],
    plaintexts: dict[str, bytes | None],
) -> dict[str, list[str]]:
    """Pick canary lines that do not occur anywhere else in the repositories."""
    wanted: dict[int, set[str]] = {}
    candidates: dict[str, list[tuple[int, str]]] = {}
    excluded: set[str] = set()
    for module, plaintext in plaintexts.items():
        if plaintext is None:
            continue
        candidates[module] = _canary_candidates(plaintext)
        for length, digest in candidates[module]:
            wanted.setdefault(length, set()).add(digest)
    if not wanted:
        return {}
    for entry in manifest["modules"]:
        if plaintexts.get(entry["module"]) is None:
            continue
        origin, _payload = _entry_paths(roots, entry)
        excluded.add(os.path.normcase(os.path.abspath(origin)))
    hits = _scan_line_hashes(roots, wanted, excluded)
    selected: dict[str, list[str]] = {}
    for module, items in candidates.items():
        chosen = [digest for _length, digest in items if digest not in hits][
            :CANARY_COUNT
        ]
        selected[module] = chosen
    return selected


def _canary_hits(manifest: dict[str, Any], roots: dict[str, str]) -> list[str]:
    """Report files whose lines match a recorded canary (a plaintext leak)."""
    owners: dict[str, str] = {}
    for entry in manifest["modules"]:
        for digest in entry.get("canary_line_sha256", []):
            owners[digest] = entry["module"]
    if not owners:
        return []
    hits: list[str] = []
    for path in _iter_text_files(roots, excluded=set()):
        try:
            raw = Path(path).read_bytes()
        except OSError:
            continue
        for line in raw.decode("utf-8", errors="replace").splitlines():
            digest = _sha256_hex(line.strip().encode("utf-8"))
            if digest in owners:
                hits.append(f"{path} (matches {owners[digest]})")
                break
    return hits


# --------------------------------------------------------------------------- #
# build                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _BuildContext:
    roots: dict[str, str]
    key: bytes
    force: bool


def _build_module(
    entry: dict[str, Any],
    plaintext: bytes | None,
    canaries: list[str] | None,
    context: _BuildContext,
) -> str:
    module = entry["module"]
    origin, payload = _entry_paths(context.roots, entry)
    if plaintext is None:
        # Plaintext already deleted: only accept a payload that still matches
        # the recorded hash.  Never treat payload bytes as a new source.
        if not payload.is_file():
            raise runtime.LwdSourceProtectionError(
                f"{module}: plaintext {origin} is gone and no payload exists; "
                "restore it from git before building"
            )
        recorded = entry.get("plaintext_sha256")
        if not recorded:
            raise runtime.LwdSourceProtectionError(
                f"{module}: plaintext {origin} is gone and the manifest records no "
                "hash; restore it from git and rebuild"
            )
        try:
            runtime.decrypt_source(
                payload.read_bytes(), context.key, module, expected_sha256=recorded
            )
        except runtime.LwdSourceProtectionError as exc:
            raise runtime.LwdSourceProtectionError(
                f"{module}: plaintext {origin} is gone and the payload does not "
                f"match the manifest ({exc}); restore it from git and rebuild"
            ) from exc
        return "already built (payload verified against the manifest)"
    digest = _sha256_hex(plaintext)
    if payload.is_file():
        try:
            existing = runtime.decrypt_source(
                payload.read_bytes(), context.key, module
            ).encode("utf-8")
        except runtime.LwdSourceProtectionError as exc:
            _log(f"{module}: existing payload is unusable ({exc}); re-encrypting")
        else:
            if existing == plaintext and not context.force:
                entry["plaintext_sha256"] = digest
                entry["plaintext_bytes"] = len(plaintext)
                entry["payload_bytes"] = payload.stat().st_size
                entry["payload_sha256"] = _sha256_hex(payload.read_bytes())
                if canaries is not None:
                    entry["canary_line_sha256"] = canaries
                return "unchanged (payload already matches the source)"
    blob = runtime.encrypt_source(plaintext, context.key, module)
    _write_bytes_atomic(payload, blob)
    verified = runtime.decrypt_source(
        payload.read_bytes(), context.key, module, expected_sha256=digest
    )
    if verified.encode("utf-8") != plaintext:
        raise runtime.LwdSourceProtectionError(
            f"{module}: payload round-trip mismatch after writing {payload}"
        )
    entry["plaintext_sha256"] = digest
    entry["plaintext_bytes"] = len(plaintext)
    entry["payload_bytes"] = len(blob)
    entry["payload_sha256"] = _sha256_hex(blob)
    if canaries is not None:
        entry["canary_line_sha256"] = canaries
    nonce_note = "fresh random nonce"
    return f"encrypted ({len(plaintext)} bytes -> {len(blob)} bytes, {nonce_note})"


def _write_manifest(
    manifest: dict[str, Any], manifest_path: Path, key_path: Path
) -> None:
    if key_path.parent == manifest_path.parent:
        manifest["key_file"] = key_path.name
    else:
        manifest.pop("key_file", None)
    text = json.dumps(manifest, indent=2) + "\n"
    _write_text_atomic(manifest_path, text)


def cmd_build(args: argparse.Namespace) -> int:
    manifest_path = _resolve_manifest(args)
    manifest = runtime.load_manifest(manifest_path, allow_unbuilt=True)
    roots = _repository_roots(args, manifest)
    _ensure_cryptography()
    key_path = _resolve_key(args, manifest_path, manifest)
    key, created = _load_or_create_key(key_path)
    if created:
        _log(f"generated AES-256 key: {key_path}")
    context = _BuildContext(roots=roots, key=key, force=bool(args.force))
    plaintexts: dict[str, bytes | None] = {}
    for entry in manifest["modules"]:
        origin, _payload = _entry_paths(roots, entry)
        if origin.is_file():
            plaintexts[entry["module"]] = origin.read_bytes()
        else:
            plaintexts[entry["module"]] = None
    canaries = _select_canaries(manifest, roots, plaintexts)
    outcomes: list[tuple[str, str]] = []
    for entry in manifest["modules"]:
        module = entry["module"]
        outcome = _build_module(
            entry,
            plaintexts[module],
            canaries.get(module),
            context,
        )
        outcomes.append((module, outcome))
        _log(f"[build] {module}: {outcome}")
    _write_manifest(manifest, manifest_path, key_path)
    for entry in manifest["modules"]:
        origin, _payload = _entry_paths(roots, entry)
        for cached in _remove_module_pyc(origin):
            _log(f"[build] removed stale bytecode {cached}")
        if origin.is_file():
            try:
                origin.unlink()
            except OSError as exc:
                raise runtime.LwdSourceProtectionError(
                    f"cannot delete plaintext {origin}: {exc}"
                ) from exc
            _log(f"[build] deleted plaintext {origin}")
    _log(f"[build] manifest written to {manifest_path}")
    return 0


# --------------------------------------------------------------------------- #
# install / launch                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _InstallPlan:
    purelib: Path
    config: dict[str, Any]
    records: dict[str, runtime.ProtectedModule]
    manifest: dict[str, Any]
    manifest_path: Path
    key_path: Path
    python: str


def _check_package_root(package: str, root: Path) -> None:
    """Fail fast unless ``package`` is installed from ``root`` (editable only)."""
    expected = Path(os.path.normcase(str((root / package).resolve())))
    if not expected.is_dir():
        raise runtime.LwdSourceProtectionError(
            f"repository root {root} does not contain the {package!r} package "
            f"({expected})"
        )
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        raise runtime.LwdSourceProtectionError(
            f"cannot resolve the installed {package!r} package: {exc}"
        ) from exc
    if spec is None or not spec.submodule_search_locations:
        raise runtime.LwdSourceProtectionError(
            f"top-level package {package!r} is not importable from "
            f"{sys.executable}; this PoC supports editable deployments only"
        )
    locations = {
        os.path.normcase(os.path.realpath(str(item)))
        for item in spec.submodule_search_locations
    }
    if str(expected) not in locations:
        raise runtime.LwdSourceProtectionError(
            f"{package!r} is imported from {sorted(locations)}, which differs from "
            f"the repository root {root} given on the command line; refusing to "
            "modify the deployment"
        )


def _preflight(args: argparse.Namespace) -> _InstallPlan:
    manifest_path = _resolve_manifest(args)
    manifest = runtime.load_manifest(manifest_path)
    key_path = _resolve_key(args, manifest_path, manifest)
    key = runtime.load_key_file(key_path)
    _ensure_cryptography()
    roots = _repository_roots(args, manifest)
    for name, repository in manifest["repositories"].items():
        _check_package_root(repository["package"], Path(roots[name]))
    records = runtime.build_module_records(manifest, roots)
    problems: list[str] = []
    for record in records.values():
        try:
            record.validate(key)
        except runtime.LwdSourceProtectionError as exc:
            problems.append(f"{record.name}: {exc}")
    if problems:
        raise runtime.LwdSourceProtectionError(
            "preflight failed; nothing was written: " + "; ".join(problems)
        )
    purelib = (
        Path(args.purelib).resolve()
        if getattr(args, "purelib", None)
        else runtime._default_purelib()
    )
    _ensure_writable_dir(purelib)
    python = getattr(args, "python", None) or sys.executable
    config = {
        "tool": runtime.TOOL_NAME,
        "format_version": runtime.FORMAT_VERSION,
        "manifest_path": str(manifest_path),
        "key_path": str(key_path),
        "repositories": {name: str(Path(root)) for name, root in roots.items()},
        "modules": [entry["module"] for entry in manifest["modules"]],
    }
    return _InstallPlan(
        purelib=purelib,
        config=config,
        records=records,
        manifest=manifest,
        manifest_path=manifest_path,
        key_path=key_path,
        python=python,
    )


def _apply_install(plan: _InstallPlan) -> tuple[list[Path], list[Path]]:
    runtime_source = (_TOOL_DIR / runtime.RUNTIME_FILENAME).read_bytes()
    if runtime.RUNTIME_MARKER.encode() not in runtime_source:
        raise runtime.LwdSourceProtectionError(
            f"{_TOOL_DIR / runtime.RUNTIME_FILENAME} is not the "
            "source-protection runtime"
        )
    removed: list[Path] = []
    for record in plan.records.values():
        removed.extend(_remove_module_pyc(record.origin))
    written: list[Path] = []
    targets = (
        (plan.purelib / runtime.RUNTIME_FILENAME, runtime_source, 0o644),
        (
            plan.purelib / runtime.CONFIG_FILENAME,
            (json.dumps(plan.config, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            0o644,
        ),
        (plan.purelib / runtime.PTH_FILENAME, _PTH_TEXT.encode("utf-8"), 0o644),
    )
    for path, payload, mode in targets:
        if _write_if_changed(path, payload, mode=mode):
            written.append(path)
    return written, removed


def cmd_install(args: argparse.Namespace) -> int:
    plan = _preflight(args)
    written, removed = _apply_install(plan)
    for path in removed:
        _log(f"[install] removed stale bytecode {path}")
    for path in written:
        _log(f"[install] wrote {path}")
    if not written:
        _log("[install] already up to date")
    _log("[install] modules: " + ", ".join(sorted(plan.records)))
    _log(f"[install] purelib: {plan.purelib}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    command = list(args.command)
    while command and command[0] == "--":
        command = command[1:]
    if not command:
        raise runtime.LwdSourceProtectionError(
            "launch requires a command after `--`, e.g. launch ... -- vllm serve ..."
        )
    plan = _preflight(args)
    written, _removed = _apply_install(plan)
    for path in written:
        _log(f"[launch] wrote {path}")
    _log("[launch] execvp: " + " ".join(command))
    os.execvp(command[0], command)
    raise runtime.LwdSourceProtectionError("os.execvp returned unexpectedly")


# --------------------------------------------------------------------------- #
# verify                                                                      #
# --------------------------------------------------------------------------- #


_PROBE_TEMPLATE = r"""# Generated by tools/lwd_source_encryption/lwd_poc.py verify.
# Ephemeral: removed once the probe finishes.
import hashlib
import importlib
import inspect
import json
import linecache
import multiprocessing
import os
import sys
import traceback
import types

DATA = __PROBE_DATA__


def _norm(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check(checks, name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})


def _finder():
    for finder in sys.meta_path:
        cls = type(finder)
        if (
            cls.__name__ == "LwdSourceProtectionFinder"
            and cls.__module__ == "lwd_source_runtime"
        ):
            return finder
    return None


def _pyc_paths(origin):
    directory = os.path.dirname(origin)
    stem = os.path.basename(origin)[:-3]
    found = []
    cache = os.path.join(directory, "__pycache__")
    if os.path.isdir(cache):
        for name in sorted(os.listdir(cache)):
            if name.startswith(stem + ".") and name.endswith(".pyc"):
                found.append(os.path.join(cache, name))
    legacy = os.path.join(directory, stem + ".pyc")
    if os.path.exists(legacy):
        found.append(legacy)
    return found


def _same_path(left, right):
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _check_module(entry):
    module = entry["module"]
    origin = entry["origin"]
    checks = []
    record = {"module": module, "checks": checks, "import_error": None}
    try:
        imported = importlib.import_module(module)
    except BaseException as exc:
        record["import_error"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        return record
    loader = getattr(imported, "__loader__", None)
    spec = getattr(imported, "__spec__", None)
    _check(checks, "import", True, module)
    _check(
        checks,
        "loader",
        type(loader).__name__ == "LwdSourceProtectionLoader"
        and type(loader).__module__ == "lwd_source_runtime",
        type(loader).__name__,
    )
    _check(checks, "file", _same_path(imported.__file__, origin), imported.__file__)
    _check(
        checks,
        "spec_origin",
        spec is not None and _same_path(spec.origin, origin),
        getattr(spec, "origin", None),
    )
    _check(
        checks,
        "filename",
        _same_path(loader.get_filename(module), origin),
        loader.get_filename(module),
    )
    source = loader.get_source(module)
    _check(
        checks,
        "source_sha256",
        _sha(source) == entry["plaintext_sha256"],
        _sha(source),
    )
    _check(
        checks,
        "source_bytes",
        len(source.encode("utf-8")) == entry["plaintext_bytes"],
        len(source.encode("utf-8")),
    )
    module_source = inspect.getsource(imported)
    _check(
        checks,
        "inspect_getsource",
        _norm(module_source).splitlines() == _norm(source).splitlines(),
        "line-normalized comparison",
    )
    lines = linecache.getlines(origin, imported.__dict__)
    _check(
        checks,
        "linecache",
        _norm("".join(lines)).splitlines() == _norm(source).splitlines(),
        str(len(lines)) + " lines",
    )
    code = loader.get_code(module)
    _check(
        checks,
        "get_code",
        isinstance(code, types.CodeType) and code.co_filename == origin,
        getattr(code, "co_filename", None),
    )
    for name in entry["attributes"]:
        target = imported
        for part in name.split("."):
            if not hasattr(target, part):
                target = None
                break
            target = getattr(target, part)
        if target is None:
            _check(checks, "attribute:" + name, False, "missing")
            continue
        if inspect.isfunction(target):
            code_obj = getattr(target, "__code__", None)
            ok = (
                isinstance(code_obj, types.CodeType)
                and code_obj.co_filename == origin
                and bool(inspect.getsource(target))
            )
            _check(
                checks,
                "attribute:" + name,
                ok,
                "co_filename=" + str(getattr(code_obj, "co_filename", None)),
            )
        elif inspect.isclass(target):
            _check(
                checks,
                "attribute:" + name,
                bool(inspect.getsource(target)),
                "class source available",
            )
        else:
            _check(checks, "attribute:" + name, True, type(target).__name__)
    try:
        with open(origin, "rb"):
            pass
    except FileNotFoundError:
        _check(checks, "open_file", True, "FileNotFoundError (expected)")
    except BaseException as exc:
        _check(checks, "open_file", False, repr(exc))
    else:
        _check(checks, "open_file", False, "open() unexpectedly succeeded")
    pycs = _pyc_paths(origin)
    _check(checks, "pyc_absent", not pycs, ", ".join(pycs) or "none")
    cached = getattr(imported, "__cached__", None)
    _check(
        checks,
        "no_cached_pyc",
        cached is None or not os.path.exists(cached),
        str(cached),
    )
    return record


def _spawn_worker(entry):
    imported = importlib.import_module(entry["module"])
    loader = imported.__loader__
    source = loader.get_source(entry["module"])
    return {
        "ok": True,
        "pid": os.getpid(),
        "sha_ok": _sha(source) == entry["plaintext_sha256"],
        "file_ok": _same_path(imported.__file__, entry["origin"]),
        "loader_ok": type(loader).__name__ == "LwdSourceProtectionLoader",
    }


def _spawn_worker_main(entries, queue):
    result = {}
    for entry in entries:
        try:
            result[entry["module"]] = _spawn_worker(entry)
        except BaseException as exc:
            result[entry["module"]] = {
                "ok": False,
                "pid": os.getpid(),
                "error": "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip(),
            }
    queue.put(result)


def _check_spawn(entries, checks):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_worker_main, args=(entries, queue))
    process.start()
    try:
        result = queue.get(timeout=600)
    except BaseException as exc:
        _check(checks, "spawn.result", False, repr(exc))
        result = {}
    finally:
        process.join(60)
    for entry in entries:
        short = entry["module"].rsplit(".", 1)[-1]
        item = result.get(entry["module"], {})
        if not item.get("ok"):
            _check(
                checks,
                "spawn[" + short + "]",
                False,
                item.get("error", "child did not report a result"),
            )
            continue
        checks_ok = (
            item.get("pid") != os.getpid()
            and item.get("sha_ok")
            and item.get("file_ok")
            and item.get("loader_ok")
        )
        _check(
            checks,
            "spawn[" + short + "]",
            checks_ok,
            "child pid=" + str(item.get("pid")) + " source/file/loader checks",
        )


def _check_subclasses(checks):
    for check in DATA["subclass_checks"]:
        label = "subclass[" + check["class_name"] + "]"
        try:
            derived_module = importlib.import_module(check["module"])
            base_module = importlib.import_module(check["base_module"])
            derived = getattr(derived_module, check["class_name"])
            base = getattr(base_module, check["base_class"])
            method = getattr(base, check.get("method", "submit"), None)
            ok = (
                issubclass(derived, base)
                and derived.__module__ == check["module"]
                and bool(inspect.getsource(derived))
                and method is not None
                and isinstance(getattr(method, "__code__", None), types.CodeType)
            )
            _check(
                checks,
                label,
                ok,
                check["module"]
                + "."
                + check["class_name"]
                + " -> "
                + check["base_class"],
            )
        except BaseException as exc:
            _check(
                checks,
                label,
                False,
                "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )


def main():
    checks = []
    finder = _finder()
    _check(
        checks,
        "finder_installed",
        finder is not None,
        "not on sys.meta_path"
        if finder is None
        else "lwd_source_runtime.LwdSourceProtectionFinder",
    )
    if finder is not None:
        names = sorted(getattr(finder, "module_names", lambda: ())())
        expected = sorted(entry["module"] for entry in DATA["modules"])
        _check(checks, "finder_modules", names == expected, "finder=" + repr(names))
    modules = [_check_module(entry) for entry in DATA["modules"]]
    _check_spawn(DATA["modules"], checks)
    _check_subclasses(checks)
    report = {
        "pid": os.getpid(),
        "python": sys.executable,
        "checks": checks,
        "modules": modules,
    }
    with open(sys.argv[1], "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


def _probe_data(plan: _InstallPlan) -> dict[str, Any]:
    verification = plan.manifest.get("verification", {})
    attributes = verification.get("module_attributes", {})
    return {
        "format_version": runtime.FORMAT_VERSION,
        "modules": [
            {
                "module": record.name,
                "origin": str(record.origin),
                "payload": str(record.payload_path),
                "plaintext_sha256": record.plaintext_sha256,
                "plaintext_bytes": record.plaintext_bytes,
                "attributes": list(attributes.get(record.name, [])),
            }
            for record in plan.records.values()
        ],
        "subclass_checks": list(verification.get("subclass_checks", [])),
        "purelib": str(plan.purelib),
    }


def _run_probe(plan: _InstallPlan, label: str) -> dict[str, Any]:
    workdir = Path(tempfile.mkdtemp(prefix=f"lwd_poc_{label}_"))
    try:
        script = workdir / "lwd_source_protection_probe.py"
        output = workdir / "probe_report.json"
        source = _PROBE_TEMPLATE.replace(
            "__PROBE_DATA__", json.dumps(_probe_data(plan))
        )
        script.write_text(source, encoding="utf-8")
        completed = subprocess.run(
            [plan.python, str(script), str(output)],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=900,
        )
        if not output.is_file():
            raise runtime.LwdSourceProtectionError(
                f"fresh-interpreter probe failed (rc={completed.returncode}): "
                f"{(completed.stderr or completed.stdout).strip()[-3000:]}"
            )
        return json.loads(output.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _probe_checks(report: dict[str, Any]) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("probe.python", True, str(report.get("python"))))
    checks.append(("probe.pid", True, str(report.get("pid"))))
    for item in report.get("checks", []):
        checks.append((f"runtime.{item['name']}", bool(item["ok"]), item["detail"]))
    for record in report.get("modules", []):
        short = _short_name(record["module"])
        if record.get("import_error"):
            checks.append((f"runtime[{short}].import", False, record["import_error"]))
            continue
        for item in record.get("checks", []):
            checks.append(
                (f"runtime[{short}].{item['name']}", bool(item["ok"]), item["detail"])
            )
    return checks


def _static_checks(
    manifest: dict[str, Any], roots: dict[str, str]
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    for entry in manifest["modules"]:
        module = entry["module"]
        short = _short_name(module)
        try:
            origin, payload = _entry_paths(roots, entry)
        except runtime.LwdSourceProtectionError as exc:
            checks.append((f"static[{short}].paths", False, str(exc)))
            continue
        checks.append(
            (f"static[{short}].plaintext_absent", not origin.exists(), str(origin))
        )
        checks.append(
            (f"static[{short}].payload_present", payload.is_file(), str(payload))
        )
        if payload.is_file():
            blob = payload.read_bytes()
            recorded = entry.get("payload_sha256")
            checks.append(
                (
                    f"static[{short}].payload_sha256",
                    recorded is None or _sha256_hex(blob) == recorded,
                    _sha256_hex(blob),
                )
            )
            checks.append(
                (
                    f"static[{short}].payload_bytes",
                    entry.get("payload_bytes") in (None, len(blob)),
                    f"{len(blob)} bytes",
                )
            )
        pycs = _module_pyc_paths(origin)
        checks.append(
            (
                f"static[{short}].pyc_absent",
                not pycs,
                ", ".join(str(item) for item in pycs) or "none",
            )
        )
    hits = _canary_hits(manifest, roots)
    checks.append(
        (
            "static.canary_absent",
            not hits,
            "no plaintext line found" if not hits else "; ".join(hits),
        )
    )
    absolute_paths = [
        f"{entry['module']}.{key}"
        for entry in manifest["modules"]
        for key in ("source", "payload")
        if os.path.isabs(entry[key])
    ]
    checks.append(
        (
            "static.manifest_paths_relative",
            not absolute_paths,
            ", ".join(absolute_paths) or "no local absolute paths",
        )
    )
    return checks


def _tamper_checks(
    plan: _InstallPlan, manifest: dict[str, Any]
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    entry = manifest["modules"][0]
    record = plan.records[entry["module"]]
    path = record.payload_path
    original = path.read_bytes()
    original_sha = _sha256_hex(original)
    corrupted = bytearray(original)
    corrupted[-1] ^= 0x01
    _log(f"[verify] tampering with {path} (service must be stopped)")
    try:
        _write_bytes_atomic(path, bytes(corrupted))
        report = _run_probe(plan, "tamper")
    finally:
        _write_bytes_atomic(path, original)
    checks.append(
        (
            "tamper.payload_restored",
            _sha256_hex(path.read_bytes()) == original_sha,
            f"restored {path}",
        )
    )
    record_report = next(
        (
            item
            for item in report.get("modules", [])
            if item["module"] == entry["module"]
        ),
        None,
    )
    if record_report is None:
        checks.append(("tamper.fresh_import_fails", False, "probe reported no module"))
        return checks
    error = record_report.get("import_error")
    if not error:
        checks.append(
            (
                "tamper.fresh_import_fails",
                False,
                "fresh interpreter imported the tampered module (fallback?)",
            )
        )
        return checks
    precise = any(
        marker in error
        for marker in (
            "LwdSourceProtectionError",
            "authentication failed",
            "is not servable",
        )
    )
    checks.append(
        (
            "tamper.fresh_import_fails",
            precise,
            "fresh import failed as expected: " + error,
        )
    )
    return checks


def cmd_verify(args: argparse.Namespace) -> int:
    manifest_path = _resolve_manifest(args)
    manifest = runtime.load_manifest(manifest_path)
    roots = _repository_roots(args, manifest)
    checks = _static_checks(manifest, roots)
    try:
        plan = _preflight(args)
    except runtime.LwdSourceProtectionError as exc:
        checks.append(("install.preflight", False, str(exc)))
        return _report(checks)
    checks.append(
        (
            "install.preflight",
            True,
            "dependencies, manifest, key, payloads and target paths validated",
        )
    )
    written, removed = _apply_install(plan)
    checks.append(
        (
            "install.apply",
            True,
            f"{len(written)} file(s) written, {len(removed)} stale pyc removed",
        )
    )
    try:
        report = _run_probe(plan, "verify")
    except (
        runtime.LwdSourceProtectionError,
        OSError,
        ValueError,
        subprocess.TimeoutExpired,
    ) as exc:
        checks.append(("probe.run", False, str(exc)))
    else:
        checks.extend(_probe_checks(report))
    if args.tamper:
        try:
            checks.extend(_tamper_checks(plan, manifest))
        except (
            runtime.LwdSourceProtectionError,
            OSError,
            ValueError,
            subprocess.TimeoutExpired,
        ) as exc:
            checks.append(("tamper.run", False, str(exc)))
    return _report(checks)


def _report(checks: list[tuple[str, bool, str]]) -> int:
    failures = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        failures += 0 if ok else 1
    total = len(checks)
    verdict = "PASS" if failures == 0 else "FAIL"
    print("")
    print(f"RESULT: {verdict} ({total - failures}/{total} checks passed)")
    return 0 if failures == 0 else 1


# --------------------------------------------------------------------------- #
# uninstall                                                                   #
# --------------------------------------------------------------------------- #


def cmd_uninstall(args: argparse.Namespace) -> int:
    removed = runtime.uninstall(args.purelib)
    if not removed:
        _log("[uninstall] nothing to remove")
    for path in removed:
        _log(f"[uninstall] removed {path}")
    _log(
        "[uninstall] plaintext sources were NOT restored; check out the original "
        "branch of each repository to get them back"
    )
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _add_root_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vllm-root",
        required=True,
        help="path of the vLLM checkout that owns the protected modules",
    )
    parser.add_argument(
        "--ascend-root",
        required=True,
        help="path of the vllm-ascend checkout that owns the protected modules",
    )


def _add_artifact_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest", default=None, help=f"manifest path (default: {DEFAULT_MANIFEST})"
    )
    parser.add_argument(
        "--key", default=None, help=f"key path (default: {DEFAULT_KEY})"
    )


def _add_purelib_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--purelib",
        default=None,
        help="target purelib directory (default: sysconfig purelib; "
        "override only for staging or tests)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lwd_poc.py",
        description="Build, install, launch and verify the LWD source-encryption PoC.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    build = subparsers.add_parser(
        "build", help="encrypt the manifest modules and delete the plaintext"
    )
    _add_root_options(build)
    _add_artifact_options(build)
    build.add_argument(
        "--force",
        action="store_true",
        help="re-encrypt even when the payload already matches the source",
    )
    build.set_defaults(handler=cmd_build)

    install = subparsers.add_parser(
        "install", help="validate everything and install the import hook"
    )
    _add_root_options(install)
    _add_artifact_options(install)
    _add_purelib_option(install)
    install.set_defaults(handler=cmd_install)

    launch = subparsers.add_parser(
        "launch", help="install, then execvp the command given after `--`"
    )
    _add_root_options(launch)
    _add_artifact_options(launch)
    _add_purelib_option(launch)
    launch.add_argument(
        "command", nargs=argparse.REMAINDER, help="command line to exec"
    )
    launch.set_defaults(handler=cmd_launch)

    verify = subparsers.add_parser("verify", help="independent end-to-end verification")
    _add_root_options(verify)
    _add_artifact_options(verify)
    _add_purelib_option(verify)
    verify.add_argument(
        "--python",
        default=None,
        help="interpreter used for the fresh probes (default: sys.executable)",
    )
    verify.add_argument(
        "--tamper",
        action="store_true",
        help="flip a payload byte, require a fresh interpreter to fail, restore",
    )
    verify.set_defaults(handler=cmd_verify)

    uninstall = subparsers.add_parser(
        "uninstall", help="remove the installed runtime, config and .pth"
    )
    _add_purelib_option(uninstall)
    uninstall.set_defaults(handler=cmd_uninstall)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except runtime.LwdSourceProtectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
