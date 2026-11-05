import json
import os
import pty
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[1] / "install.py"


@pytest.fixture
def installation(tmp_path):
    source = tmp_path / "source ' tree"
    (source / "stan_mcp").mkdir(parents=True)
    (source / "stan_mcp/__main__.py").write_text("")
    (source / "pixi.toml").write_text('[workspace]\nname = "stan-mcp"\n')
    (source / "pixi.lock").write_text("version: 6\n")
    binary = tmp_path / "bin with spaces"
    binary.mkdir()
    fake = (
        f"#!{sys.executable}\n"
        + """
import json, os, shutil, sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ['INSTALL_LOG'], 'a') as log:
    log.write(json.dumps([name, *sys.argv[1:]]) + '\\n')
if name == 'git':
    shutil.copytree(os.environ['INSTALL_SOURCE'], sys.argv[-1])
if name == 'gh':
    sys.exit('GitHub CLI is not authenticated')
if name == 'pixi' and sys.argv[1] == 'install':
    sys.exit(int(os.environ.get('INSTALL_FAIL', '0')))
if name == 'claude':
    path = Path(os.environ['CLAUDE_CONFIG_DIR']) / '.claude.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    separator = sys.argv.index('--')
    config.setdefault('mcpServers', {})[sys.argv[separator - 1]] = {
        'command': sys.argv[separator + 1], 'args': sys.argv[separator + 2:],
    }
    path.write_text(json.dumps(config))
if name == 'codex':
    sys.exit(int(os.environ.get('INSTALL_CODEX_FAIL', '0')))
"""
    )
    for name in ("pixi", "codex", "claude", "git", "gh"):
        path = binary / name
        path.write_text(fake)
        path.chmod(0o755)
    codex = tmp_path / "codex"
    claude = tmp_path / "claude"
    codex.mkdir()
    claude.mkdir()
    env = {
        **os.environ,
        "PATH": str(binary),
        "CODEX_HOME": str(codex),
        "CLAUDE_CONFIG_DIR": str(claude),
        "INSTALL_LOG": str(tmp_path / "commands.jsonl"),
        "INSTALL_SOURCE": str(source),
    }
    return source, env


def _invoke(source: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(INSTALLER), "--source", str(source), *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _commands(env: dict[str, str]) -> list[list[str]]:
    path = Path(env["INSTALL_LOG"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_selected_client_and_rerun_preserve_settings(installation):
    source, env = installation
    codex = Path(env["CODEX_HOME"]) / "config.toml"
    codex.write_text('[mcp_servers.stan]\ncommand = "existing-command"\n')
    untouched = codex.read_bytes()
    result = _invoke(source, env, "--claude")
    assert result.returncode == 0, result.stderr
    calls = _commands(env)
    launch = [
        str(Path(env["PATH"]) / "pixi"),
        "run",
        "--locked",
        "--manifest-path",
        str(source.resolve() / "pixi.toml"),
        "serve",
    ]
    assert calls[-1] == [
        "claude",
        "mcp",
        "add",
        "--scope",
        "user",
        "--transport",
        "stdio",
        "stan",
        "--",
        *launch,
    ]
    assert not any(call[0] == "codex" for call in calls)
    assert codex.read_bytes() == untouched

    claude = Path(env["CLAUDE_CONFIG_DIR"]) / ".claude.json"
    claude.write_text(
        json.dumps(
            {
                "theme": "dark",
                "mcpServers": {
                    "stan": {"command": launch[0], "args": launch[1:], "env": {"KEEP": "yes"}}
                },
            }
        )
    )
    saved = claude.read_bytes()
    result = _invoke(source, env, "--claude")
    assert result.returncode == 0, result.stderr
    assert not any(call[0] in {"codex", "claude"} for call in _commands(env)[len(calls) :])
    assert claude.read_bytes() == saved and codex.read_bytes() == untouched


def test_menu_with_consumed_stdin_toggles_each_client(installation):
    source, env = installation
    pid, terminal = pty.fork()
    if pid == 0:
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0)
        os.close(null)
        # Pixi's stdin script mode executes with -c and no __file__.
        os.execve(
            sys.executable,
            [sys.executable, "-c", INSTALLER.read_text(), "--source", str(source)],
            env,
        )
    output = b""
    finished = False
    try:
        deadline = time.monotonic() + 15
        answered = False
        while time.monotonic() < deadline:
            ready, _, _ = select.select([terminal], [], [], 0.1)
            if ready:
                try:
                    output += os.read(terminal, 65536)
                except OSError:
                    pass
            if b"q: quit:" in output and not answered:
                os.write(terminal, b"1\n2\n2\n\n")
                answered = True
            child, status = os.waitpid(pid, os.WNOHANG)
            if child:
                finished = True
                assert os.waitstatus_to_exitcode(status) == 0, output.decode()
                break
        assert finished, output.decode()
    finally:
        if not finished:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        os.close(terminal)
    registrations = [call for call in _commands(env) if call[0] in {"codex", "claude"}]
    assert len(registrations) == 1
    assert registrations[0][:4] == ["codex", "mcp", "add", "stan"]


def test_failed_environment_install_does_not_register_clients(installation):
    source, env = installation
    result = _invoke(source, {**env, "INSTALL_FAIL": "7"}, "--claude", "--codex")
    assert result.returncode != 0
    assert [call[:2] for call in _commands(env)] == [["pixi", "install"]]


def test_conflicting_entry_prevents_all_client_writes(installation):
    source, env = installation
    config = Path(env["CODEX_HOME"]) / "config.toml"
    config.write_text('[mcp_servers.stan]\ncommand = "keep-me"\n')
    saved = config.read_bytes()
    result = _invoke(source, env, "--claude", "--codex")
    assert result.returncode != 0
    assert _commands(env) == []
    assert config.read_bytes() == saved


def test_partial_registration_can_resume(installation):
    source, env = installation
    result = _invoke(source, {**env, "INSTALL_CODEX_FAIL": "9"}, "--claude", "--codex")
    assert result.returncode != 0
    config = Path(env["CLAUDE_CONFIG_DIR"]) / ".claude.json"
    saved = config.read_bytes()
    previous_calls = len(_commands(env))
    result = _invoke(source, env, "--claude", "--codex")
    assert result.returncode == 0, result.stderr
    assert config.read_bytes() == saved
    calls = _commands(env)[previous_calls:]
    assert [call[0] for call in calls if call[0] in {"codex", "claude"}] == ["codex"]


def test_public_download_reuses_checkout_without_authentication(installation, tmp_path):
    source, env = installation
    checkout = tmp_path / "installed ' source"
    command = [
        sys.executable,
        str(INSTALLER),
        "--directory",
        str(checkout),
        "--no-clients",
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert (checkout / "pixi.toml").read_bytes() == (source / "pixi.toml").read_bytes()
    marker = checkout / "local-edit.txt"
    marker.write_text("keep")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "keep"
    assert [call[1] for call in _commands(env) if call[0] == "git"] == ["clone"]
