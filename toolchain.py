"""Toolchain provisioning: ensure BridgeStan + CmdStan are present, fail loud."""

import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from errors import ToolchainError


def _provision(script: str) -> str:
    # Installers print progress to stdout. A child process owns that stream so
    # provisioning cannot redirect or corrupt concurrent MCP protocol writes.
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    lines = result.stdout.splitlines()
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode:
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="")
        raise RuntimeError(
            f"toolchain provisioning exited {result.returncode}: {result.stderr.strip()}"
        )
    if not lines:
        raise RuntimeError("toolchain provisioning returned no path")
    if len(lines) > 1:
        print("\n".join(lines[:-1]), file=sys.stderr)
    return str(json.loads(lines[-1]))


def _bridgestan_path() -> str:
    # Downloads the pinned BridgeStan source to ~/.bridgestan on first call.
    path = Path(
        os.environ.get("BRIDGESTAN")
        or Path.home() / ".bridgestan" / f"bridgestan-{version('bridgestan')}"
    )
    if (path / "Makefile").is_file():
        return str(path)
    return _provision(
        "import json; from bridgestan.compile import get_bridgestan_path; "
        "print(json.dumps(str(get_bridgestan_path())))"
    )


def _cmdstan_path() -> str:
    import cmdstanpy

    return str(cmdstanpy.cmdstan_path())


def _install_cmdstan() -> None:
    path = _provision(
        "import json, cmdstanpy; "
        "ok = cmdstanpy.install_cmdstan(); "
        "assert ok, 'install_cmdstan failed'; "
        "print(json.dumps(str(cmdstanpy.cmdstan_path())))"
    )
    os.environ["CMDSTAN"] = path


def ensure_bridgestan() -> str:
    try:
        return _bridgestan_path()
    except Exception as exc:  # noqa: BLE001 - surfaced as a remediation message
        raise ToolchainError(
            "BridgeStan source tree unavailable and could not be downloaded "
            f"({exc}). Check network access or run "
            '`python -c "from bridgestan.compile import get_bridgestan_path; '
            'get_bridgestan_path()"` manually.'
        ) from exc


def ensure_cmdstan() -> str:
    try:
        return _cmdstan_path()
    except Exception:
        try:
            _install_cmdstan()
            return _cmdstan_path()
        except Exception as exc:  # noqa: BLE001
            raise ToolchainError(
                "CmdStan not installed and install_cmdstan() failed "
                f'({exc}). Run `python -c "import cmdstanpy; '
                'cmdstanpy.install_cmdstan()"` manually.'
            ) from exc
