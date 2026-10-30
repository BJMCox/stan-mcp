import json
import os
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_ssh_launcher_preserves_protocol_and_quoted_paths(tmp_path):
    project = Path(__file__).resolve().parents[1]
    remote = tmp_path / "remote ' checkout"
    remote.symlink_to(project, target_is_directory=True)
    binary = tmp_path / "bin"
    binary.mkdir()
    # Execute the remote command locally after validating transport options.
    ssh = binary / "ssh"
    ssh.write_text("""#!/bin/sh
set -eu
[ "$1" = -T ] && [ "$2" = -o ] && [ "$3" = BatchMode=yes ]
[ "$4" = -- ] && [ "$5" = test-host ]
exec sh -c "$6"
""")
    ssh.chmod(0o755)
    pixi = binary / "remote ' pixi"
    pixi.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m stan_mcp\n')
    pixi.chmod(0o755)
    data = tmp_path / "data.json"
    data.write_text('{"n": 2}')
    params = StdioServerParameters(
        command=str(project / "scripts/serve-ssh"),
        args=["test-host", str(remote), str(pixi)],
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "STAN_MCP_HOME": str(tmp_path / "home"),
        },
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            assert len((await session.list_tools()).tools) == 32
            result = await session.call_tool("register_data", {"data": str(data)})
            handle = json.loads(next(item.text for item in result.content if item.type == "text"))[
                "data_id"
            ]
            assert len(handle) == 16
            result = await session.call_tool("register_data", {"data": {"n": 2}})
            assert (
                json.loads(next(item.text for item in result.content if item.type == "text"))[
                    "data_id"
                ]
                == handle
            )


def test_ssh_launcher_rejects_relative_remote_checkout(tmp_path):
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [str(project / "scripts/serve-ssh"), "host", "relative"], capture_output=True, text=True
    )
    assert result.returncode == 2 and not result.stdout
