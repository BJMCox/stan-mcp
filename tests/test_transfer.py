"""Exercise actual rsync on both ends of the SSH command boundary."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from stan_mcp import compilation, transfer


@pytest.fixture
def peer(tmp_path):
    try:
        binary = transfer._rsync(local=True)
    except ValueError as exc:
        pytest.skip(str(exc))
    checkout = Path(__file__).resolve().parents[1]
    remote = tmp_path / "remote ' checkout"
    remote.symlink_to(checkout, target_is_directory=True)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "rsync").symlink_to(binary)
    ssh = binaries / "ssh"
    ssh.write_text("""#!/bin/sh
set -eu
test "$1" = -T && test "$2" = -o && test "$3" = BatchMode=yes
shift 3
if [ "$1" = -- ]; then shift; fi
test "$1" = compute-host
shift
case "$*" in
  *"_receive"*)
    if [ -n "${MUTATE_STAGE:-}" ]; then
      printf 'model {} // bad\\n' > "$MUTATE_STAGE"
    fi
    printf 'send\\n' >> "$TRANSFER_LOG"
    ;;
esac
exec sh -c "$*"
""")
    ssh.chmod(0o700)
    pixi = binaries / "remote ' pixi"
    pixi.write_text(
        '#!/bin/sh\n[ "$1" = run ] && [ "$2" = --locked ] && [ "$3" = python ] || exit 2\n'
        f'shift 3\nexec {shlex.quote(sys.executable)} "$@"\n'
    )
    pixi.chmod(0o700)
    log = tmp_path / "transfers.log"
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "TRANSFER_LOG": str(log),
    }
    base = [sys.executable, "-m", "stan_mcp.transfer"]
    remote_root = tmp_path / "staging ' root"

    def call(command, *args, mutate=None):
        remote_args = (
            [
                "--remote-checkout",
                str(remote),
                "--remote-pixi",
                str(pixi),
                "--remote-root",
                str(remote_root),
            ]
            if command == "stage"
            else []
        )
        return subprocess.run(
            [*base, command, "--host", "compute-host", *remote_args, *map(str, args)],
            cwd=checkout,
            env={**env, **({"MUTATE_STAGE": str(mutate)} if mutate else {})},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    return call, remote_root, log


def _json(result):
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_upload_preserves_include_layout_and_reuses_data_after_a_model_edit(tmp_path, peer):
    call, _, log = peer
    root = tmp_path / "local models"
    root.mkdir()
    model = root / "root [model].stan"
    model.write_text('parameters { real x; }\n#include "inc dir/model.stan"\n')
    includes = root / "inc dir"
    includes.mkdir()
    block = includes / "model.stan"
    block.write_text("model { x ~ std_normal(); }\n")
    (root / "unlisted.txt").write_text("leave me local")
    data = tmp_path / "data.json"
    data.write_text('{"N":5}\n')
    args = (
        "--root",
        root,
        "--model",
        model.name,
        "--include",
        "inc dir/model.stan",
        "--data",
        data,
    )
    first = _json(call("stage", *args))
    staged = Path(first["model"]["path"])
    assert staged.read_bytes() == model.read_bytes()
    assert (staged.parent / "inc dir/model.stan").read_bytes() == block.read_bytes()
    assert "x ~ std_normal()" in compilation.resolve_source(str(staged))
    assert not (staged.parent / "unlisted.txt").exists()
    assert Path(first["data"][0]["path"]).read_bytes() == data.read_bytes()
    repeated = _json(call("stage", *args))
    assert repeated["model"]["path"] == staged.as_posix() and repeated["model"]["reused"]
    assert repeated["data"][0]["reused"]
    assert log.read_text().splitlines() == ["send", "send"]
    block.write_text("model { x ~ normal(1, 1); }\n")
    changed = _json(call("stage", *args))
    assert changed["model"]["id"] != first["model"]["id"]
    assert changed["data"][0]["id"] == first["data"][0]["id"]
    assert changed["data"][0]["reused"]
    assert log.read_text().splitlines() == ["send", "send", "send"]

    staged.chmod(0o600)
    staged.write_text("tampered")
    block.write_text("model { x ~ std_normal(); }\n")
    failed = call("stage", *args)
    assert failed.returncode != 0 and not failed.stdout
    assert staged.read_text() == "tampered"


def test_failed_verification_leaves_only_owned_partial_and_retry_repairs_it(tmp_path, peer):
    call, remote_root, _ = peer
    root = tmp_path / "inputs"
    root.mkdir()
    model = root / "model.stan"
    original = "model {} // yes\n"
    assert len(original) == len("model {} // bad\n")
    model.write_text(original)
    args = ("--root", root, "--model", model.name)
    failed = call("stage", *args, mutate=model)
    assert failed.returncode != 0 and not failed.stdout
    assert "staged content changed during transfer" in failed.stderr
    assert not [p for p in remote_root.iterdir() if not p.name.startswith(".")]
    (partial,) = (remote_root / ".incoming").iterdir()
    (partial_file,) = (partial / "files").iterdir()
    stat = partial_file.stat()
    model.write_text(original)
    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    result = _json(call("stage", *args))
    assert result["model"]["id"] == partial.name
    assert Path(result["model"]["path"]).read_text() == original
    assert not partial.exists()


def test_fetch_treats_source_literally_and_never_publishes_unverified_bytes(tmp_path, peer):
    call, _, _ = peer
    source = tmp_path / "remote [one] ' results.safetensors"
    source.write_bytes(b"test binary data")
    (tmp_path / "remote o ' results.safetensors").write_bytes(b"glob decoy")
    output = tmp_path / "local result.safetensors"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    args = ("--remote-file", source, "--output", output)
    failed = call("fetch", *args, "--sha256", "0" * 64)
    assert failed.returncode != 0 and not output.exists()
    assert "does not match its SHA-256" in failed.stderr
    result = _json(call("fetch", *args, "--sha256", digest))
    assert result == {"path": str(output), "sha256": digest}
    assert output.read_bytes() == source.read_bytes()
    source.write_bytes(b"changed")
    failed = call("fetch", *args, "--sha256", hashlib.sha256(source.read_bytes()).hexdigest())
    assert failed.returncode != 0
    assert output.read_bytes() == b"test binary data"


def test_outside_or_symlink_inputs_are_rejected_before_remote_staging(tmp_path, peer):
    call, remote_root, _ = peer
    root = tmp_path / "source"
    root.mkdir()
    model = root / "inside.stan"
    model.write_text("model {}")
    outside = tmp_path / "outside.stan"
    outside.write_text("model {}")
    (root / "link.stan").symlink_to(outside)
    for choice in ("../outside.stan", "link.stan"):
        failed = call("stage", "--root", root, "--model", model, "--include", choice)
        assert failed.returncode != 0 and not remote_root.exists()
