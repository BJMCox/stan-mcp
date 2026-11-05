# /// script
# requires-python = ">=3.15.2,<3.16"
# dependencies = []
# [tool.pixi.dependencies]
# git = "*"
# ///
"""Install the Pixi environment and register selected MCP clients."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

REPOSITORY = "https://github.com/BJMCox/stan-mcp.git"
CLIENTS = {"claude": "Claude Code", "codex": "Codex"}


def _command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Install {name} and put it on PATH before continuing.")
    return str(Path(path).absolute())


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, check=True, stdout=sys.stderr, cwd=cwd)


def _choose_clients() -> list[str] | None:
    selected = {name: shutil.which(name) is not None for name in CLIENTS}
    try:
        # A downloaded script consumes stdin. Keep the menu on the controlling terminal.
        terminal = nullcontext(sys.stdin) if sys.stdin.isatty() else Path("/dev/tty").open()
    except OSError as exc:
        raise RuntimeError("No terminal. Pass --claude, --codex, or --no-clients.") from exc
    with terminal as reader:
        while True:
            print("\nInstall stan-mcp for:", file=sys.stderr)
            for number, (name, label) in enumerate(CLIENTS.items(), 1):
                mark = "x" if selected[name] else " "
                print(f"  {number}. [{mark}] {label}", file=sys.stderr)
            print("1/2: toggle, Enter: install, q: quit: ", file=sys.stderr, end="", flush=True)
            answer = reader.readline()
            if not answer or answer.strip().lower() == "q":
                return None
            choice = answer.strip()
            if not choice:
                return [name for name, enabled in selected.items() if enabled]
            if choice in {"1", "2"}:
                name = list(CLIENTS)[int(choice) - 1]
                selected[name] = not selected[name]


def _existing_server(client: str, name: str) -> dict[str, Any] | None:
    if client == "codex":
        path = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"
        if not path.exists():
            return None
        config = tomllib.loads(path.read_text())
        return config.get("mcp_servers", {}).get(name)
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    path = (Path(config_dir) if config_dir else Path.home()) / ".claude.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("mcpServers", {}).get(name)


def _registrations(clients: list[str], name: str, launch: list[str]) -> list[list[str]]:
    commands = []
    for client in clients:
        executable = _command(client)
        current = _existing_server(client, name)
        if current is not None:
            if current.get("command") == launch[0] and current.get("args", []) == launch[1:]:
                print(f"{CLIENTS[client]}: {name} already uses this checkout.", file=sys.stderr)
                continue
            raise RuntimeError(
                f"{CLIENTS[client]} already has a different '{name}' server. "
                "Use --name with a new name, or remove the old entry in that client first."
            )
        options = ["--scope", "user", "--transport", "stdio"] if client == "claude" else []
        commands.append([executable, "mcp", "add", *options, name, "--", *launch])
    return commands


def _checkout(source: Path | None, directory: Path, repository: str) -> Path:
    checkout = (source if source is not None else directory).expanduser().resolve()
    if source is None and not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run([_command("git"), "clone", "--depth", "1", "--", repository, str(checkout)])
    manifest = checkout / "pixi.toml"
    if not manifest.is_file() or not (checkout / "stan_mcp/__main__.py").is_file():
        raise RuntimeError(f"{checkout} is not a stan-mcp checkout.")
    if tomllib.loads(manifest.read_text()).get("workspace", {}).get("name") != "stan-mcp":
        raise RuntimeError(f"{manifest} does not describe stan-mcp.")
    if not (checkout / "pixi.lock").is_file():
        raise RuntimeError(f"{checkout} lacks pixi.lock. Use a revision with the repo installer.")
    return checkout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claude", action="store_true", help="register Claude Code without the menu"
    )
    parser.add_argument("--codex", action="store_true", help="register Codex without the menu")
    parser.add_argument(
        "--no-clients", action="store_true", help="install only the Pixi environment"
    )
    parser.add_argument("--name", default="stan", help="MCP server name (default: stan)")
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--source", type=Path, help="use an existing checkout")
    location.add_argument("--directory", type=Path, help="destination for a downloaded checkout")
    parser.add_argument("--repo", default=REPOSITORY, help="Git URL used for a new checkout")
    args = parser.parse_args(argv)
    if args.no_clients and (args.claude or args.codex):
        parser.error("--no-clients cannot be combined with --claude or --codex")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        parser.error("--name must start with a letter or digit and contain letters, digits, - or _")
    try:
        pixi = _command("pixi")
        clients = [name for name in CLIENTS if getattr(args, name)]
        if not clients and not args.no_clients:
            clients = _choose_clients()
            if clients is None:
                return 0
        for client in clients:
            _command(client)

        source = args.source
        # URL and stdin execution have no adjacent checkout. A local script can reuse its own.
        script = globals().get("__file__")
        if source is None and args.directory is None and script:
            adjacent = Path(script).resolve().parent
            if (adjacent / "stan_mcp/__main__.py").is_file():
                source = adjacent
        data_root = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
        directory = args.directory or data_root / "stan-mcp"
        checkout = _checkout(source, directory, args.repo)
        run = [pixi, "run", "--locked", "--manifest-path", str(checkout / "pixi.toml")]
        registrations = _registrations(clients, args.name, [*run, "serve"])
        print(f"Installing from {checkout}", file=sys.stderr)
        _run([pixi, "install", "--locked", "--manifest-path", str(checkout / "pixi.toml")])
        _run(
            [
                *run,
                "python",
                "-c",
                "import asyncio; from stan_mcp.server import mcp; "
                "print(f'{len(asyncio.run(mcp.list_tools()))} MCP tools loaded.')",
            ],
            cwd=checkout,
        )
        for command in registrations:
            _run(command)
        print(
            "Installation complete. Restart the selected clients to load stan-mcp.", file=sys.stderr
        )
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Installation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
