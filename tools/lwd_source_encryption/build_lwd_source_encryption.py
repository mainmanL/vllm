#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-command full build: discover -> manifest -> encrypt -> static [-> ELF].

The runtime key is provisioned only through ``LWD_SOURCE_KEY_B64`` in child
processes.  Its value is never included in command arguments or log output.
On Windows, ``--elf`` delegates the cross build to WSL Ubuntu-22.04.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import lwd_source_runtime as runtime
import pybase64 as base64

_TOOL_DIR = Path(__file__).resolve().parent
_DEFAULT_MANIFEST = _TOOL_DIR / runtime.MANIFEST_FILENAME
_DEFAULT_KEY = _TOOL_DIR / runtime.KEY_FILENAME
_DEFAULT_SCOPE = _TOOL_DIR / "lwd_source_scope.json"


def _run(label: str, command: Sequence[str], env: Mapping[str, str]) -> None:
    print(f"[wrapper] {label}: {shlex.join(command)}", flush=True)
    subprocess.run(list(command), check=True, env=dict(env))


def _write_key_asset(path: Path, key: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64encode(key) + b"\n")
    os.chmod(path, 0o600)


def _provision_key(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    key_path = Path(args.key).resolve()
    env_key = runtime.load_key_from_env()
    if env_key is None:
        if not key_path.is_file():
            _run(
                "keygen",
                [
                    sys.executable,
                    str(_TOOL_DIR / "lwd_poc.py"),
                    "keygen",
                    "--key",
                    str(key_path),
                ],
                env,
            )
        key = runtime.load_key_file(key_path)
        env[runtime.KEY_ENV_VAR] = base64.b64encode(key).decode("ascii")
    else:
        key = env_key
        if key_path.is_file() and runtime.load_key_file(key_path) != key:
            raise runtime.LwdSourceProtectionError(
                f"local build key {key_path} differs from {runtime.KEY_ENV_VAR}; "
                "refusing to mix payload/ELF keys"
            )
        if args.elf and not key_path.is_file():
            _write_key_asset(key_path, key)
            print(f"[wrapper] provisioned ignored ELF key asset: {key_path}")
    print("[wrapper] runtime key provisioned through environment (value hidden)")
    return env


def _wslpath(path: Path, distro: str) -> str:
    command = "wslpath -a " + shlex.quote(str(path.resolve()))
    completed = subprocess.run(
        ["wsl", "-d", distro, "--", "bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _elf_command(args: argparse.Namespace) -> list[str]:
    script = _TOOL_DIR / "build_lwd_source_poc.py"
    output = (
        Path(args.elf_output).resolve()
        if args.elf_output
        else _TOOL_DIR / "lwd_source_poc"
    )
    compiler_args = ["--output", str(output), "--cc", args.cc]
    if args.qemu:
        compiler_args.append("--qemu")
    if os.name != "nt":
        return [sys.executable, str(script), *compiler_args]
    linux_args = [
        "--output",
        _wslpath(output, args.wsl_distro),
        "--cc",
        args.cc,
    ]
    if args.qemu:
        linux_args.append("--qemu")
    command = ["python3", _wslpath(script, args.wsl_distro), *linux_args]
    return [
        "wsl",
        "-d",
        args.wsl_distro,
        "--",
        "bash",
        "-lc",
        shlex.join(command),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--ascend-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--scope", type=Path, default=_DEFAULT_SCOPE)
    parser.add_argument("--key", type=Path, default=_DEFAULT_KEY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--elf", action="store_true", help="also build the aarch64 ELF")
    parser.add_argument("--elf-output", type=Path)
    parser.add_argument("--cc", default="aarch64-linux-gnu-gcc")
    parser.add_argument("--qemu", action="store_true", help="run ELF/QEMU smoke checks")
    parser.add_argument("--wsl-distro", default="Ubuntu-22.04")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        env = _provision_key(args)
        common = [
            "--vllm-root",
            str(args.vllm_root.resolve()),
            "--ascend-root",
            str(args.ascend_root.resolve()),
            "--scope",
            str(args.scope.resolve()),
        ]
        tool = str(_TOOL_DIR / "lwd_poc.py")
        _run("1/4 discover", [sys.executable, tool, "discover", *common], env)
        _run(
            "2/4 manifest",
            [
                sys.executable,
                tool,
                "manifest",
                *common,
                "--manifest",
                str(args.manifest.resolve()),
            ],
            env,
        )
        encrypt = [
            sys.executable,
            tool,
            "build",
            *common,
            "--manifest",
            str(args.manifest.resolve()),
        ]
        if args.force:
            encrypt.append("--force")
        _run("3/4 encrypt", encrypt, env)
        _run(
            "4/4 static",
            [
                sys.executable,
                tool,
                "static",
                *common,
                "--manifest",
                str(args.manifest.resolve()),
            ],
            env,
        )
        if args.elf:
            _run("optional ELF", _elf_command(args), env)
        print("[wrapper] FULL BUILD PASS (static validation; no NPU imports)")
        return 0
    except runtime.LwdSourceProtectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(
            f"ERROR: build step failed with exit code {exc.returncode}", file=sys.stderr
        )
        return exc.returncode or 1


if __name__ == "__main__":
    sys.exit(main())
