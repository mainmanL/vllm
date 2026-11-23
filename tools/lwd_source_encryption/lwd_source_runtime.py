# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime import hook for the LWD source-encryption PoC.

``tools/lwd_source_encryption/lwd_poc.py install`` copies this file into the
environment ``purelib`` directory together with ``lwd_source_protection.json``
(absolute runtime configuration) and ``lwd_source_protection.pth``.  Every
ordinary interpreter of that environment therefore imports this module during
``site`` start-up and registers exactly one
:class:`LwdSourceProtectionFinder` on ``sys.meta_path``, which serves the
modules listed in the manifest from AES-256-GCM payloads (``*.lwdc``).

The plaintext is compiled with the *original* ``.py`` path as ``co_filename``
and is never written back to disk.  Nothing patches ``open``, ``inspect``,
``linecache`` or CPython: the loader implements the standard loader protocol,
so ``inspect.getsource`` and ``linecache`` keep working.

This is a static-protection PoC, not a secret-keeping mechanism: the key is
shipped next to the tool and ``loader.get_source()`` returns plaintext by
design (the accepted boundary, see ``README.md``).
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import struct
import sys
import sysconfig
from pathlib import Path, PurePosixPath
from typing import Any

import pybase64 as base64
import regex as re

__all__ = [
    "FORMAT_VERSION",
    "LwdSourceProtectionError",
    "LwdSourceProtectionFinder",
    "LwdSourceProtectionLoader",
    "build_module_records",
    "build_payload_aad",
    "decrypt_source",
    "encrypt_source",
    "install",
    "load_key_file",
    "load_manifest",
    "load_runtime_config",
    "pack_payload",
    "parse_payload",
    "uninstall",
]

TOOL_NAME = "lwd_source_encryption_poc"
MANIFEST_FILENAME = "manifest.json"
KEY_FILENAME = "runtime_key.b64"
RUNTIME_FILENAME = "lwd_source_runtime.py"
CONFIG_FILENAME = "lwd_source_protection.json"
PTH_FILENAME = "lwd_source_protection.pth"
PTH_MARKER = "# lwd_source_protection: static source-encryption PoC (read-only hook)"
RUNTIME_MARKER = "LWD static source protection runtime"

FORMAT_VERSION = 1
MAGIC = b"LWDCSRC1"
NONCE_BYTES = 12
TAG_BYTES = 16
AES_KEY_BYTES = 32
MAX_MODULE_NAME_BYTES = 255
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
PAYLOAD_SUFFIX = ".lwdc"
CIPHER_NAME = "AES-256-GCM"

# magic | format version | flags | module-name length | nonce | ciphertext length
_HEADER = struct.Struct(">8sHHH12sQ")
_HEADER_BYTES = _HEADER.size
_MODULE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


class LwdSourceProtectionError(RuntimeError):
    """Raised when a protected module cannot be served from its payload."""


# --------------------------------------------------------------------------- #
# Payload format                                                              #
# --------------------------------------------------------------------------- #


def build_payload_aad(version: int, flags: int, module_name: bytes) -> bytes:
    """Bind the format version, flags and the full module name into the AAD."""
    return MAGIC + struct.pack(">HHH", version, flags, len(module_name)) + module_name


def pack_payload(
    module_name: str,
    nonce: bytes,
    ciphertext: bytes,
    *,
    version: int = FORMAT_VERSION,
    flags: int = 0,
) -> bytes:
    """Serialize one payload (``header | module name | ciphertext + tag``)."""
    if version != FORMAT_VERSION:
        raise LwdSourceProtectionError(f"unsupported format version {version}")
    if flags != 0:
        raise LwdSourceProtectionError(f"unsupported flags 0x{flags:04x}")
    if len(nonce) != NONCE_BYTES:
        raise LwdSourceProtectionError(
            f"nonce must be {NONCE_BYTES} bytes, got {len(nonce)}"
        )
    name_bytes = module_name.encode("utf-8")
    if not 0 < len(name_bytes) <= MAX_MODULE_NAME_BYTES:
        raise LwdSourceProtectionError(
            f"module name must be 1..{MAX_MODULE_NAME_BYTES} UTF-8 bytes"
        )
    if not _MODULE_NAME_RE.fullmatch(module_name):
        raise LwdSourceProtectionError(f"invalid module name {module_name!r}")
    if len(ciphertext) <= TAG_BYTES:
        raise LwdSourceProtectionError("ciphertext must contain data and a GCM tag")
    header = _HEADER.pack(
        MAGIC, version, flags, len(name_bytes), nonce, len(ciphertext)
    )
    return header + name_bytes + ciphertext


def parse_payload(
    data: bytes, expected_module: str | None = None
) -> tuple[int, int, str, bytes, bytes]:
    """Strictly parse one payload.

    Args:
        data: Raw payload bytes.
        expected_module: When given, the embedded module name must match it.

    Returns:
        ``(version, flags, module_name, nonce, ciphertext_with_tag)``.

    Raises:
        LwdSourceProtectionError: On truncation, trailing bytes, bad magic,
            unsupported version/flags, invalid name or length mismatch.
    """
    raw = bytes(data)
    if len(raw) < _HEADER_BYTES:
        raise LwdSourceProtectionError(
            f"payload is truncated: {len(raw)} bytes < {_HEADER_BYTES} byte header"
        )
    magic, version, flags, name_len, nonce, ciphertext_len = _HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise LwdSourceProtectionError(f"bad payload magic {magic!r}")
    if version != FORMAT_VERSION:
        raise LwdSourceProtectionError(f"unsupported format version {version}")
    if flags != 0:
        raise LwdSourceProtectionError(f"unsupported flags 0x{flags:04x}")
    if not 0 < name_len <= MAX_MODULE_NAME_BYTES:
        raise LwdSourceProtectionError(f"invalid module-name length {name_len}")
    if len(nonce) != NONCE_BYTES:
        raise LwdSourceProtectionError(
            f"nonce must be {NONCE_BYTES} bytes, got {len(nonce)}"
        )
    if ciphertext_len <= TAG_BYTES:
        raise LwdSourceProtectionError(f"invalid ciphertext length {ciphertext_len}")
    if ciphertext_len > MAX_PAYLOAD_BYTES:
        raise LwdSourceProtectionError(
            f"ciphertext length {ciphertext_len} exceeds the supported maximum"
        )
    expected_total = _HEADER_BYTES + name_len + ciphertext_len
    if len(raw) < expected_total:
        raise LwdSourceProtectionError(
            f"payload is truncated: {expected_total - len(raw)} byte(s) missing"
        )
    if len(raw) > expected_total:
        raise LwdSourceProtectionError(
            f"payload has {len(raw) - expected_total} trailing byte(s)"
        )
    name_bytes = raw[_HEADER_BYTES : _HEADER_BYTES + name_len]
    try:
        module_name = name_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LwdSourceProtectionError(
            "payload module name is not valid UTF-8"
        ) from exc
    if not _MODULE_NAME_RE.fullmatch(module_name):
        raise LwdSourceProtectionError(f"invalid module name {module_name!r}")
    if expected_module is not None and module_name != expected_module:
        raise LwdSourceProtectionError(
            f"payload belongs to module {module_name!r}, expected {expected_module!r}"
        )
    ciphertext = raw[_HEADER_BYTES + name_len :]
    return version, flags, module_name, nonce, ciphertext


def _aes_gcm(key: bytes) -> Any:
    if not isinstance(key, (bytes, bytearray)) or len(key) != AES_KEY_BYTES:
        raise LwdSourceProtectionError(
            f"AES-256 key must be {AES_KEY_BYTES} bytes, got {len(key)}"
        )
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - dependency is validated first
        raise LwdSourceProtectionError(
            "the 'cryptography' package is required for source decryption"
        ) from exc
    return AESGCM(bytes(key))


def encrypt_source(
    plaintext: bytes,
    key: bytes,
    module_name: str,
    *,
    nonce: bytes | None = None,
) -> bytes:
    """Encrypt one module source into a payload blob.

    A fresh 12-byte nonce is generated unless the caller supplies one (only
    tests do); the nonce is stored in the payload header and is never reused
    by the build tool.
    """
    if nonce is None:
        nonce = os.urandom(NONCE_BYTES)
    if len(nonce) != NONCE_BYTES:
        raise LwdSourceProtectionError(
            f"nonce must be {NONCE_BYTES} bytes, got {len(nonce)}"
        )
    if not plaintext:
        raise LwdSourceProtectionError(
            f"refusing to encrypt an empty source for module {module_name!r}"
        )
    name_bytes = module_name.encode("utf-8")
    if not _MODULE_NAME_RE.fullmatch(module_name):
        raise LwdSourceProtectionError(f"invalid module name {module_name!r}")
    aad = build_payload_aad(FORMAT_VERSION, 0, name_bytes)
    ciphertext = _aes_gcm(key).encrypt(nonce, plaintext, aad)
    return pack_payload(module_name, nonce, ciphertext)


def decrypt_source(
    blob: bytes,
    key: bytes,
    module_name: str,
    expected_sha256: str | None = None,
) -> str:
    """Decrypt one payload and return the validated UTF-8 source text."""
    version, flags, name, nonce, ciphertext = parse_payload(
        blob, expected_module=module_name
    )
    aad = build_payload_aad(version, flags, name.encode("utf-8"))
    try:
        plaintext = _aes_gcm(key).decrypt(nonce, ciphertext, aad)
    except Exception as exc:  # InvalidTag, ValueError, ...
        raise LwdSourceProtectionError(
            f"payload authentication failed for module {module_name!r}: "
            "wrong key or modified ciphertext"
        ) from exc
    if expected_sha256 is not None:
        actual = hashlib.sha256(plaintext).hexdigest()
        if actual != expected_sha256.lower():
            raise LwdSourceProtectionError(
                f"plaintext SHA-256 mismatch for module {module_name!r}: "
                f"got {actual}, manifest records {expected_sha256.lower()}"
            )
    try:
        return plaintext.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LwdSourceProtectionError(
            f"payload plaintext of module {module_name!r} is not valid UTF-8"
        ) from exc


# --------------------------------------------------------------------------- #
# Manifest / configuration                                                    #
# --------------------------------------------------------------------------- #


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_relative_path(value: str, field: str) -> str:
    """Reject absolute, drive-qualified or parent-escaping repository paths."""
    if not isinstance(value, str) or not value:
        raise LwdSourceProtectionError(f"{field} must be a non-empty string")
    if "\\" in value:
        raise LwdSourceProtectionError(f"{field} {value!r} must use '/' separators")
    path = PurePosixPath(value)
    if path.is_absolute() or path.drive:
        raise LwdSourceProtectionError(f"{field} {value!r} must be a relative path")
    if ".." in path.parts:
        raise LwdSourceProtectionError(f"{field} {value!r} must not escape the root")
    if str(path) != value:
        raise LwdSourceProtectionError(f"{field} {value!r} is not normalized")
    return value


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LwdSourceProtectionError(f"{field} must be a JSON object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise LwdSourceProtectionError(f"{field} must be a non-empty JSON array")
    return value


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LwdSourceProtectionError(f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= minimum:
        raise LwdSourceProtectionError(f"{field} must be an integer > {minimum}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    text = _require_str(value, field).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise LwdSourceProtectionError(
            f"{field} must be a lowercase SHA-256 hex digest"
        )
    return text


def _require_module_name(value: Any, field: str) -> str:
    text = _require_str(value, field)
    if not _MODULE_NAME_RE.fullmatch(text):
        raise LwdSourceProtectionError(f"{field} {text!r} is not a dotted module name")
    return text


def validate_manifest(
    manifest: Any, *, source: str = "manifest", allow_unbuilt: bool = False
) -> None:
    """Validate the manifest schema the runtime and the CLI both rely on.

    Args:
        manifest: Parsed manifest JSON.
        source: Label used in error messages.
        allow_unbuilt: Accept ``None`` hashes so that ``build`` can fill them in
            for a manifest that has never been built.

    Raises:
        LwdSourceProtectionError: On any schema violation.
    """
    data = _require_mapping(manifest, source)
    if data.get("tool") != TOOL_NAME:
        raise LwdSourceProtectionError(
            f"{source}: tool must be {TOOL_NAME!r}, got {data.get('tool')!r}"
        )
    if data.get("format_version") != FORMAT_VERSION:
        raise LwdSourceProtectionError(
            f"{source}: unsupported format_version {data.get('format_version')!r}"
        )
    if data.get("cipher") != CIPHER_NAME:
        raise LwdSourceProtectionError(f"{source}: cipher must be {CIPHER_NAME!r}")
    if data.get("nonce_bytes") != NONCE_BYTES:
        raise LwdSourceProtectionError(f"{source}: nonce_bytes must be {NONCE_BYTES}")
    key_file = data.get("key_file")
    if key_file is not None:
        validate_relative_path(key_file, f"{source}: key_file")
    repositories = _require_mapping(data.get("repositories"), f"{source}: repositories")
    for name, entry in repositories.items():
        item = _require_mapping(entry, f"{source}: repositories[{name!r}]")
        package = _require_module_name(
            item.get("package"), f"{source}: repositories[{name!r}].package"
        )
        _ = package
    modules = _require_list(data.get("modules"), f"{source}: modules")
    seen: set[str] = set()
    for index, entry in enumerate(modules):
        field = f"{source}: modules[{index}]"
        item = _require_mapping(entry, field)
        name = _require_module_name(item.get("module"), f"{field}.module")
        if name in seen:
            raise LwdSourceProtectionError(f"{field}: duplicate module {name!r}")
        seen.add(name)
        repository = _require_str(item.get("repository"), f"{field}.repository")
        if repository not in repositories:
            raise LwdSourceProtectionError(
                f"{field}: repository {repository!r} is not declared"
            )
        validate_relative_path(item.get("source"), f"{field}.source")
        validate_relative_path(item.get("payload"), f"{field}.payload")
        payload = _require_str(item.get("payload"), f"{field}.payload")
        if not payload.endswith(PAYLOAD_SUFFIX):
            raise LwdSourceProtectionError(
                f"{field}.payload must end with {PAYLOAD_SUFFIX!r}"
            )
        if payload == item.get("source"):
            raise LwdSourceProtectionError(f"{field}.payload must differ from source")
        sha_value = item.get("plaintext_sha256")
        if not (allow_unbuilt and sha_value is None):
            _require_sha256(sha_value, f"{field}.plaintext_sha256")
        size_value = item.get("plaintext_bytes")
        if not (allow_unbuilt and size_value is None):
            _require_int(size_value, f"{field}.plaintext_bytes")
        if "payload_bytes" in item:
            _require_int(item.get("payload_bytes"), f"{field}.payload_bytes")
        if "payload_sha256" in item:
            _require_sha256(item.get("payload_sha256"), f"{field}.payload_sha256")
        canaries = item.get("canary_line_sha256", [])
        if not isinstance(canaries, list):
            raise LwdSourceProtectionError(
                f"{field}.canary_line_sha256 must be a JSON array"
            )
        for canary in canaries:
            _require_sha256(canary, f"{field}.canary_line_sha256[]")
    verification = data.get("verification", {})
    if verification:
        _validate_verification(verification, seen, f"{source}: verification")


def _validate_verification(
    verification: Any, module_names: set[str], field: str
) -> None:
    item = _require_mapping(verification, field)
    attributes = item.get("module_attributes", {})
    for name, values in _require_mapping(
        attributes, f"{field}.module_attributes"
    ).items():
        if name not in module_names:
            raise LwdSourceProtectionError(
                f"{field}.module_attributes: unknown module {name!r}"
            )
        for value in _require_list(values, f"{field}.module_attributes[{name!r}]"):
            _require_str(value, f"{field}.module_attributes[{name!r}][]")
    checks = item.get("subclass_checks", [])
    if not isinstance(checks, list):
        raise LwdSourceProtectionError(f"{field}.subclass_checks must be a JSON array")
    for index, check in enumerate(checks):
        entry = _require_mapping(check, f"{field}.subclass_checks[{index}]")
        for key in ("module", "class_name", "base_module", "base_class"):
            _require_str(entry.get(key), f"{field}.subclass_checks[{index}].{key}")


def load_manifest(path: Path, *, allow_unbuilt: bool = False) -> dict[str, Any]:
    """Load and validate a manifest file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise LwdSourceProtectionError(f"cannot read manifest {path}: {exc}") from exc
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LwdSourceProtectionError(
            f"manifest {path} is not valid JSON: {exc}"
        ) from exc
    validate_manifest(manifest, source=str(path), allow_unbuilt=allow_unbuilt)
    return manifest


def load_key_file(path: Path) -> bytes:
    """Load a base64-encoded 32-byte AES key."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise LwdSourceProtectionError(f"cannot read key file {path}: {exc}") from exc
    compact = "".join(text.split())
    try:
        key = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise LwdSourceProtectionError(
            f"key file {path} is not valid base64: {exc}"
        ) from exc
    if len(key) != AES_KEY_BYTES:
        raise LwdSourceProtectionError(
            f"key file {path} must decode to {AES_KEY_BYTES} bytes, got {len(key)}"
        )
    return key


def load_runtime_config(path: Path) -> dict[str, Any]:
    """Load and validate the installed runtime configuration."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise LwdSourceProtectionError(
            f"cannot read runtime configuration {path}: {exc}"
        ) from exc
    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LwdSourceProtectionError(
            f"runtime configuration {path} is not valid JSON: {exc}"
        ) from exc
    data = _require_mapping(config, str(path))
    if data.get("tool") != TOOL_NAME:
        raise LwdSourceProtectionError(
            f"{path}: tool must be {TOOL_NAME!r}, got {data.get('tool')!r}"
        )
    if data.get("format_version") != FORMAT_VERSION:
        raise LwdSourceProtectionError(
            f"{path}: unsupported format_version {data.get('format_version')!r}"
        )
    for key in ("manifest_path", "key_path"):
        value = _require_str(data.get(key), f"{path}: {key}")
        if not os.path.isabs(value):
            raise LwdSourceProtectionError(f"{path}: {key} must be absolute")
    repositories = _require_mapping(data.get("repositories"), f"{path}: repositories")
    for name, value in repositories.items():
        text_root = _require_str(value, f"{path}: repositories[{name!r}]")
        if not os.path.isabs(text_root):
            raise LwdSourceProtectionError(
                f"{path}: repositories[{name!r}] must be absolute"
            )
    modules = _require_list(data.get("modules"), f"{path}: modules")
    for index, name in enumerate(modules):
        _require_module_name(name, f"{path}: modules[{index}]")
    return data


def _validate_config_against_manifest(
    config: dict[str, Any], manifest: dict[str, Any]
) -> None:
    configured = list(config["modules"])
    manifested = [entry["module"] for entry in manifest["modules"]]
    if configured != manifested:
        raise LwdSourceProtectionError(
            "runtime configuration module list does not match the manifest; "
            f"configured={configured} manifest={manifested}"
        )
    if set(config["repositories"]) != set(manifest["repositories"]):
        raise LwdSourceProtectionError(
            "runtime configuration repositories do not match the manifest"
        )


# --------------------------------------------------------------------------- #
# Module records                                                              #
# --------------------------------------------------------------------------- #


def _resolve_inside(root: Path, relative: str, field: str) -> Path:
    validate_relative_path(relative, field)
    root_abs = os.path.abspath(str(root))
    parts = PurePosixPath(relative).parts
    candidate = os.path.abspath(os.path.join(root_abs, *parts))
    try:
        common = os.path.commonpath(
            [os.path.normcase(root_abs), os.path.normcase(candidate)]
        )
    except ValueError as exc:
        raise LwdSourceProtectionError(
            f"{field} {relative!r} does not resolve inside {root}"
        ) from exc
    if common != os.path.normcase(root_abs):
        raise LwdSourceProtectionError(
            f"{field} {relative!r} escapes repository root {root}"
        )
    return Path(candidate)


class ProtectedModule:
    """One module name resolved onto a payload inside a repository checkout."""

    def __init__(
        self,
        *,
        name: str,
        repository: str,
        origin: Path,
        payload_path: Path,
        plaintext_sha256: str,
        plaintext_bytes: int,
        payload_sha256: str | None = None,
        payload_bytes: int | None = None,
    ) -> None:
        self.name = name
        self.repository = repository
        self.origin = Path(origin)
        self.payload_path = Path(payload_path)
        self.plaintext_sha256 = plaintext_sha256.lower()
        self.plaintext_bytes = plaintext_bytes
        self.payload_sha256 = payload_sha256.lower() if payload_sha256 else None
        self.payload_bytes = payload_bytes
        self.error: LwdSourceProtectionError | None = None

    def read_payload(self) -> bytes:
        try:
            size = self.payload_path.stat().st_size
        except OSError as exc:
            raise LwdSourceProtectionError(
                f"cannot read payload {self.payload_path} "
                f"for module {self.name!r}: {exc}"
            ) from exc
        if size <= _HEADER_BYTES + TAG_BYTES:
            raise LwdSourceProtectionError(
                f"payload {self.payload_path} is too small ({size} bytes)"
            )
        if size > MAX_PAYLOAD_BYTES + _HEADER_BYTES + MAX_MODULE_NAME_BYTES:
            raise LwdSourceProtectionError(
                f"payload {self.payload_path} is unreasonably large ({size} bytes)"
            )
        try:
            return self.payload_path.read_bytes()
        except OSError as exc:
            raise LwdSourceProtectionError(
                f"cannot read payload {self.payload_path}: {exc}"
            ) from exc

    def validate(self, key: bytes) -> None:
        """Fail fast when the deployment is not in a servable state."""
        if not self.origin.is_absolute() or not self.payload_path.is_absolute():
            raise LwdSourceProtectionError(
                f"module {self.name!r} must resolve to absolute paths"
            )
        if self.origin.is_file():
            raise LwdSourceProtectionError(
                f"plaintext source {self.origin} is still present; the regular "
                "path finder would win over the encrypted payload"
            )
        blob = self.read_payload()
        if self.payload_bytes is not None and len(blob) != self.payload_bytes:
            raise LwdSourceProtectionError(
                f"payload {self.payload_path} size {len(blob)} does not match the "
                f"manifest ({self.payload_bytes} bytes)"
            )
        if self.payload_sha256 is not None:
            actual = _sha256_hex(blob)
            if actual != self.payload_sha256:
                raise LwdSourceProtectionError(
                    f"payload SHA-256 mismatch for {self.payload_path}: "
                    f"got {actual}, manifest records {self.payload_sha256}"
                )
        text = decrypt_source(
            blob, key, self.name, expected_sha256=self.plaintext_sha256
        )
        encoded = text.encode("utf-8")
        if len(encoded) != self.plaintext_bytes:
            raise LwdSourceProtectionError(
                f"plaintext size mismatch for module {self.name!r}: "
                f"got {len(encoded)} bytes, manifest records {self.plaintext_bytes}"
            )

    def source(self, key: bytes) -> str:
        """Decrypt and hash-check the source text of this module."""
        return decrypt_source(
            self.read_payload(), key, self.name, expected_sha256=self.plaintext_sha256
        )


def build_module_records(
    manifest: dict[str, Any], repositories: dict[str, str]
) -> dict[str, ProtectedModule]:
    """Resolve every manifest module onto absolute paths under its repository."""
    records: dict[str, ProtectedModule] = {}
    for entry in manifest["modules"]:
        name = entry["module"]
        repository = entry["repository"]
        if repository not in repositories:
            raise LwdSourceProtectionError(
                f"module {name!r} needs repository {repository!r}, which is not "
                f"configured ({sorted(repositories)})"
            )
        root = Path(repositories[repository])
        if not root.is_absolute():
            raise LwdSourceProtectionError(
                f"repository root for {repository!r} must be absolute"
            )
        if not root.is_dir():
            raise LwdSourceProtectionError(
                f"repository root {root} for {repository!r} does not exist"
            )
        origin = _resolve_inside(root, entry["source"], f"{name}: source")
        payload = _resolve_inside(root, entry["payload"], f"{name}: payload")
        if origin.suffix != ".py":
            raise LwdSourceProtectionError(
                f"module {name!r}: source {origin} must be a .py file"
            )
        records[name] = ProtectedModule(
            name=name,
            repository=repository,
            origin=origin,
            payload_path=payload,
            plaintext_sha256=entry["plaintext_sha256"],
            plaintext_bytes=entry["plaintext_bytes"],
            payload_sha256=entry.get("payload_sha256"),
            payload_bytes=entry.get("payload_bytes"),
        )
    return records


# --------------------------------------------------------------------------- #
# Loader / finder                                                             #
# --------------------------------------------------------------------------- #


class LwdSourceProtectionLoader(importlib.abc.Loader):
    """Decrypt, compile and execute one protected module from its payload."""

    def __init__(self, record: ProtectedModule, key: bytes) -> None:
        self._record = record
        self._key = key

    def _check_name(self, fullname: str) -> None:
        if fullname != self._record.name:
            raise ImportError(
                f"loader for {self._record.name!r} cannot serve {fullname!r}"
            )

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        origin = str(self._record.origin)
        source = self.get_source(module.__name__)
        module.__file__ = origin
        code = compile(source, origin, "exec", dont_inherit=True)
        exec(code, module.__dict__)

    def get_filename(self, fullname: str) -> str:
        self._check_name(fullname)
        return str(self._record.origin)

    def get_source(self, fullname: str) -> str:
        self._check_name(fullname)
        return self._record.source(self._key)

    def get_code(self, fullname: str) -> Any:
        source = self.get_source(fullname)
        return compile(source, str(self._record.origin), "exec", dont_inherit=True)

    def get_data(self, path: str) -> bytes:
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
            str(self._record.origin)
        ):
            return self._record.source(self._key).encode("utf-8")
        raise OSError(
            f"{path!r} is not available: protected module "
            f"{self._record.name!r} has no on-disk source"
        )

    def is_package(self, fullname: str) -> bool:
        self._check_name(fullname)
        return False


class LwdSourceProtectionFinder(importlib.abc.MetaPathFinder):
    """Serve exactly the manifest modules and nothing else."""

    def __init__(self, modules: dict[str, ProtectedModule], key: bytes) -> None:
        self._modules = dict(modules)
        self._key = key

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        _ = path, target
        record = self._modules.get(fullname)
        if record is None:
            return None
        if record.error is not None:
            raise ImportError(
                f"protected module {fullname!r} is not servable: {record.error}"
            ) from record.error
        loader = LwdSourceProtectionLoader(record, self._key)
        spec = importlib.machinery.ModuleSpec(
            fullname, loader, origin=str(record.origin)
        )
        spec.has_location = True
        return spec

    def update(self, modules: dict[str, ProtectedModule], key: bytes) -> None:
        self._modules = dict(modules)
        self._key = key

    def module_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._modules))


# --------------------------------------------------------------------------- #
# Installation                                                                #
# --------------------------------------------------------------------------- #


def _warn(message: str) -> None:
    sys.stderr.write(f"[LWD] source protection: {message}\n")


def _default_config_path() -> Path:
    return Path(__file__).resolve().with_name(CONFIG_FILENAME)


def _default_purelib() -> Path:
    return Path(sysconfig.get_path("purelib"))


def _register_finder(
    finder_type: type[LwdSourceProtectionFinder],
    modules: dict[str, ProtectedModule],
    key: bytes,
) -> None:
    for existing in sys.meta_path:
        if isinstance(existing, finder_type):
            existing.update(modules, key)
            return
    sys.meta_path.insert(0, finder_type(modules, key))


def _unregister_finder(finder_type: type[LwdSourceProtectionFinder]) -> None:
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not isinstance(finder, finder_type)
    ]


def install(
    config_path: str | os.PathLike[str] | None = None, *, suppress_errors: bool = False
) -> bool:
    """Register the protected-module finder from the installed configuration.

    Idempotent: a second call updates the existing finder in place.  With
    ``suppress_errors=True`` (used by the ``.pth`` hook) a broken deployment
    never kills interpreter start-up: a warning is printed instead and imports
    of the affected modules raise a precise error (never a plaintext fallback).
    """
    path = Path(config_path) if config_path is not None else _default_config_path()
    if not path.is_file():
        if suppress_errors:
            return False
        raise LwdSourceProtectionError(
            f"runtime configuration {path} does not exist; run the install command"
        )
    try:
        config = load_runtime_config(path)
        manifest = load_manifest(Path(config["manifest_path"]))
        _validate_config_against_manifest(config, manifest)
        key = load_key_file(Path(config["key_path"]))
        records = build_module_records(manifest, config["repositories"])
        errors: list[str] = []
        for record in records.values():
            try:
                record.validate(key)
            except LwdSourceProtectionError as exc:
                record.error = exc
                errors.append(f"{record.name}: {exc}")
        _register_finder(LwdSourceProtectionFinder, records, key)
        if errors:
            message = "unusable payload(s): " + "; ".join(errors)
            if suppress_errors:
                _warn(message)
                return True
            raise LwdSourceProtectionError(message)
        return True
    except LwdSourceProtectionError as exc:
        if not suppress_errors:
            raise
        _warn(f"disabled ({exc})")
        return False


def _looks_like_runtime_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return RUNTIME_MARKER in text or "LwdSourceProtectionFinder" in text


def _looks_like_config_file(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and data.get("tool") == TOOL_NAME


def _looks_like_pth_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "lwd_source_runtime" in text


def uninstall(purelib: str | os.PathLike[str] | None = None) -> list[Path]:
    """Remove the runtime, runtime configuration and ``.pth`` this tool added.

    Only files carrying our markers are removed, so a foreign ``.pth`` or an
    unrelated ``lwd_source_runtime.py`` is never touched.
    """
    directory = Path(purelib) if purelib is not None else _default_purelib()
    targets = (
        (directory / RUNTIME_FILENAME, _looks_like_runtime_file),
        (directory / CONFIG_FILENAME, _looks_like_config_file),
        (directory / PTH_FILENAME, _looks_like_pth_file),
    )
    removed: list[Path] = []
    for path, predicate in targets:
        if not path.is_file() or not predicate(path):
            continue
        try:
            path.unlink()
        except OSError as exc:
            raise LwdSourceProtectionError(f"cannot remove {path}: {exc}") from exc
        removed.append(path)
    cache_dir = directory / "__pycache__"
    if cache_dir.is_dir():
        for cached in sorted(cache_dir.glob(f"{Path(RUNTIME_FILENAME).stem}.*.pyc")):
            try:
                cached.unlink()
            except OSError as exc:
                raise LwdSourceProtectionError(
                    f"cannot remove {cached}: {exc}"
                ) from exc
            removed.append(cached)
    _unregister_finder(LwdSourceProtectionFinder)
    return removed


def status(purelib: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Describe the installed hook (used by the CLI and for troubleshooting)."""
    directory = Path(purelib) if purelib is not None else _default_purelib()
    config_path = directory / CONFIG_FILENAME
    finder = None
    for candidate in sys.meta_path:
        if isinstance(candidate, LwdSourceProtectionFinder):
            finder = candidate
            break
    result: dict[str, Any] = {
        "purelib": str(directory),
        "runtime_present": (directory / RUNTIME_FILENAME).is_file(),
        "config_present": config_path.is_file(),
        "pth_present": (directory / PTH_FILENAME).is_file(),
        "finder_installed": finder is not None,
        "finder_modules": list(finder.module_names()) if finder else [],
    }
    if config_path.is_file():
        try:
            config = load_runtime_config(config_path)
            result["manifest_path"] = config["manifest_path"]
            result["key_path"] = config["key_path"]
            result["repositories"] = dict(config["repositories"])
            result["modules"] = list(config["modules"])
        except LwdSourceProtectionError as exc:
            result["error"] = str(exc)
    return result


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m lwd_source_runtime``."""
    parser = argparse.ArgumentParser(
        prog="python -m lwd_source_runtime",
        description="Inspect or remove the installed LWD source-protection hook.",
    )
    parser.add_argument(
        "--purelib",
        default=None,
        help="purelib directory that holds the installed files "
        "(default: sysconfig purelib)",
    )
    parser.add_argument("--config", default=None, help="runtime configuration path")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--install", action="store_true", help="install the finder from --config"
    )
    action.add_argument(
        "--uninstall", action="store_true", help="remove the installed files"
    )
    action.add_argument(
        "--check", action="store_true", help="print the installed state (default)"
    )
    args = parser.parse_args(argv)
    if args.uninstall:
        removed = uninstall(args.purelib)
        if not removed:
            print("lwd_source_runtime: nothing to uninstall")
        for path in removed:
            print(f"removed {path}")
        return 0
    if args.install:
        install(args.config, suppress_errors=False)
        print(f"lwd_source_runtime: installed hook from {args.config or 'default'}")
        return 0
    print(json.dumps(status(args.purelib), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
