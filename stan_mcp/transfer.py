"""Stage Stan inputs and fetch exports with rsync over the existing SSH alias."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

from stan_mcp import config

_SSH = ("ssh", "-T", "-o", "BatchMode=yes")
_PARTIAL = ".rsync-partial"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_FORMAT = "stan-mcp.stage.v1"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"expected an absolute path: {value!r}")
    return path


def _relative_file(root: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        path = path.relative_to(root)
    if not path.parts or any(part in {"..", _PARTIAL} for part in path.parts):
        raise ValueError(f"invalid file in staging inventory: {value!r}")
    candidate = root
    for part in path.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"symbolic links are not allowed in staging inputs: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"staging input is not a regular file: {candidate}")
    return path.as_posix()


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    stat = path.stat()
    if path.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise ValueError(f"transfer directory must be private and user-owned: {path}")


@contextmanager
def _lease(path: Path) -> Iterator[BinaryIO]:
    with path.open("ab") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another transfer owns this staging path") from exc
        yield stream


def _rsync(*, local: bool) -> str:
    candidates = dict.fromkeys(
        (shutil.which("rsync"), "/opt/homebrew/bin/rsync" if local else None)
    )
    for binary in candidates:
        if not binary or not Path(binary).is_file():
            continue
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, check=False)
        match = re.search(r"^rsync\s+version\s+(\d+)\.(\d+)", result.stdout, re.MULTILINE)
        if result.returncode == 0 and match and tuple(map(int, match.groups())) >= (3, 2):
            return binary
    where = "the client" if local else "the remote host"
    raise ValueError(f"rsync 3.2 or newer is required on {where}")


def _copy(binary: str, arguments: list[str], *, inventory: bytes | None = None) -> None:
    argv = [
        binary,
        "-ts",
        "--checksum",
        "--no-whole-file",
        f"--partial-dir={_PARTIAL}",
        "--info=progress2",
        "-e",
        shlex.join(_SSH),
        *arguments,
    ]
    subprocess.run(argv, input=inventory, stdout=sys.stderr, check=True)


def _remote_command(args: argparse.Namespace, *command: str) -> str:
    checkout = _absolute(args.remote_checkout)
    argv = [args.remote_pixi, "run", "--locked", "python", "-m", "stan_mcp.transfer", *command]
    return f"cd {shlex.quote(str(checkout))} && exec {shlex.join(argv)}"


def _remote_request(args: argparse.Namespace, request: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [*_SSH, "--", args.host, _remote_command(args, "_prepare")],
        input=json.dumps(request),
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    return dict(json.loads(result.stdout))


def _stage_root(value: str | None) -> Path:
    root = _absolute(value) if value is not None else _absolute(str(config.home())) / "staging"
    _private_dir(root)
    return root.resolve()


def _file_seal(bundle: Path, files: dict[str, str], *, publish: bool) -> dict[str, list[int]]:
    """Check the exact inventory and record, or compare, its file identities."""
    directory = bundle / "files"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"invalid staging files directory: {directory}")
    parents = {parent.as_posix() for name in files for parent in Path(name).parents} - {"."}
    seal = {}
    for path in directory.rglob("*"):
        name = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise ValueError(f"staging contains a symbolic link: {name}")
        if path.is_dir() and name in parents:
            continue
        if name not in files or not path.is_file():
            raise ValueError(f"unexpected file or directory in staging: {name}")
        if publish:
            if _digest(path) != files[name]:
                raise ValueError(f"staged content changed during transfer: {name}")
            path.chmod(0o400)
        stat = path.stat()
        seal[name] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    if seal.keys() != files.keys():
        raise ValueError("received files do not match staging inventory")
    return seal


def _prepare(request: dict[str, Any]) -> dict[str, Any]:
    record = request["record"]
    key = _identity(record)
    root = _stage_root(request["remote_root"])
    locks = root / ".locks"
    _private_dir(locks)
    final = root / key
    with _lease(locks / key):
        if final.is_symlink():
            raise ValueError("published staging path must not be a symbolic link")
        if final.is_dir():
            if json.loads((final / "manifest.json").read_text()) != record:
                raise ValueError("published staging manifest does not match its identity")
            if _file_seal(final, record["files"], publish=False) != json.loads(
                (final / "seal.json").read_text()
            ):
                raise ValueError(
                    "published staging files changed; copy to a new stage after inspection"
                )
            return {"ready": True, "path": str(final / "files"), "id": key}

        if request.get("check"):
            return {"ready": False}
        binary = _rsync(local=False)
        incoming = root / ".incoming" / key
        _private_dir(root / ".incoming")
        _private_dir(incoming)
        _private_dir(incoming / "files")
        manifest = incoming / "manifest.json"
        if manifest.is_file():
            if json.loads(manifest.read_text()) != record:
                raise ValueError("incoming staging manifest does not match its identity")
        else:
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(record))
            temporary.replace(manifest)
        return {
            "ready": False,
            "path": str(incoming / "files"),
            "id": key,
            "root": str(root),
            "rsync": binary,
        }


def _receive(root: Path, key: str, binary: str, flags: list[str]) -> None:
    if not _DIGEST.fullmatch(key) or flags[:1] != ["--server"] or "--sender" in flags:
        raise ValueError("invalid rsync staging receiver")
    with _lease(root / ".locks" / key) as lease:
        incoming = root / ".incoming" / key
        if (root / key).exists() or not incoming.is_dir():
            raise ValueError("staging destination changed; repeat the stage command")
        record = json.loads((incoming / "manifest.json").read_text())
        if _identity(record) != key:
            raise ValueError("staging manifest does not match its identity")
        subprocess.run([binary, *flags], pass_fds=(lease.fileno(),), check=True)
        seal = _file_seal(incoming, record["files"], publish=True)
        (incoming / "seal.json").write_text(json.dumps(seal))
        incoming.rename(root / key)


def _stage_bundle(
    args: argparse.Namespace, binary: str, root: Path, names: list[str], kind: str
) -> dict[str, Any]:
    files = {name: _digest(root / name) for name in sorted(set(names))}
    record = {"schema": _FORMAT, "kind": kind, "files": files}
    request = {"record": record, "remote_root": args.remote_root}
    target = _remote_request(args, request)
    reused = target["ready"]
    if not reused:
        receiver = _remote_command(
            args, "_receive", target["root"], target["id"], target["rsync"], "--"
        )
        inventory = b"".join(os.fsencode(name) + b"\0" for name in files)
        _copy(
            binary,
            [
                "--from0",
                "--files-from=-",
                f"--rsync-path={receiver}",
                "--",
                str(root) + "/",
                f"{args.host}:{target['path']}/",
            ],
            inventory=inventory,
        )
        target = _remote_request(args, {**request, "check": True})
        if not target["ready"]:
            raise ValueError("rsync exited without publishing a verified bundle")
    return {"id": target["id"], "root": target["path"], "reused": reused}


def _stage(args: argparse.Namespace) -> dict[str, Any]:
    root = _absolute(args.root)
    model = _relative_file(root, args.model)
    includes = [_relative_file(root, name) for name in args.include]
    data = [_absolute(name) for name in args.data]
    if not model.endswith(".stan") or any(path.suffix != ".json" for path in data):
        raise ValueError("stage expects a .stan model and .json data files")
    for path in data:
        _relative_file(path.parent, path.name)
    binary = _rsync(local=True)
    staged = _stage_bundle(args, binary, root, [model, *includes], "model")
    staged["path"] = str(Path(staged["root"]) / model)
    datasets = []
    for path in data:
        item = _stage_bundle(args, binary, path.parent, [path.name], "data")
        item.update(source=str(path), path=str(Path(item["root"]) / path.name))
        datasets.append(item)
    return {"host": args.host, "model": staged, "data": datasets}


def _fetch(args: argparse.Namespace) -> dict[str, Any]:
    source, output = _absolute(args.remote_file), _absolute(args.output)
    expected = args.sha256.lower()
    if not _DIGEST.fullmatch(expected):
        raise ValueError("expected a SHA-256 hex digest")
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output directory does not exist: {output.parent}")
    binary = _rsync(local=True)
    key = _identity(
        {"host": args.host, "source": str(source), "output": str(output), "sha256": expected}
    )
    spool = output.parent / f".stan-fetch-{key}"
    _private_dir(spool)
    with _lease(spool / "lock"):
        if output.exists() or output.is_symlink():
            raise ValueError(f"output already exists: {output}")
        payload = spool / "payload"
        # With -s the remote rsync sender still interprets wildcard syntax.
        escaped = re.sub(r"([\\*?\[])", r"\\\1", str(source))
        _copy(binary, ["--", f"{args.host}:{escaped}", str(payload)])
        if payload.is_symlink() or not payload.is_file():
            raise ValueError("remote export is not a regular file")
        if _digest(payload) != expected:
            payload.unlink()
            raise ValueError("downloaded export does not match its SHA-256")
        payload.chmod(0o600)
        os.link(payload, output)
        shutil.rmtree(spool)
    return {"path": str(output), "sha256": expected}


def main() -> None:
    try:
        # Internal actions run on the remote SSH endpoint.
        if sys.argv[1:] == ["_prepare"]:
            print(json.dumps(_prepare(json.load(sys.stdin))))
            return
        if len(sys.argv) >= 7 and sys.argv[1] == "_receive" and sys.argv[5] == "--":
            _receive(Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[6:])
            return

        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        stage = commands.add_parser(
            "stage", help="Upload a model bundle and independent data files"
        )
        stage.add_argument("--host", required=True)
        stage.add_argument("--remote-checkout", required=True)
        stage.add_argument("--remote-pixi", default="pixi")
        stage.add_argument(
            "--remote-root", help="Absolute remote root, default STAN_MCP_HOME/staging"
        )
        stage.add_argument("--root", required=True, help="Absolute local model bundle root")
        stage.add_argument("--model", required=True, help="Model path within root")
        stage.add_argument(
            "--include", action="append", default=[], help="Include path within root"
        )
        stage.add_argument(
            "--data", action="append", default=[], help="Absolute local .json data path"
        )
        fetch = commands.add_parser("fetch", help="Fetch an export and verify its checksum")
        fetch.add_argument("--host", required=True)
        fetch.add_argument("--remote-file", required=True)
        fetch.add_argument("--output", required=True)
        fetch.add_argument("--sha256", required=True)
        args = parser.parse_args()
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.host):
            raise ValueError("host must be an SSH alias or user@host")
        result = _stage(args) if args.command == "stage" else _fetch(args)
        print(json.dumps(result))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"stan-mcp transfer: {exc}") from exc


if __name__ == "__main__":
    main()
