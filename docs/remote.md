# Run over SSH

SSH carries MCP stdio without changing the tools. Compile and inference run on
the remote host. Files and returned paths refer to that host.

Install the checkout and Pixi on the remote host. Run `pixi install` there.
Provision CmdStan and BridgeStan there, or let the first compile install them.
Use SSH keys and verify the host through your usual SSH configuration first.

Configure the client to launch:

```toml
[mcp_servers.stan]
command = "/absolute/local/stan-mcp/scripts/serve-ssh"
args = ["compute-host", "/absolute/remote/stan-mcp", "/absolute/remote/bin/pixi"]
```

The third argument is optional when Pixi is on the remote noninteractive PATH.
The launcher disables the remote TTY and password prompts. SSH diagnostics go
to stderr. Shell startup files must also keep stdout clean.

Use remote absolute `.stan` and `.json` paths in MCP calls. The transfer commands
below produce those paths. Reuse Stan's model and data handles across calls.

## Stage inputs and fetch results

Install rsync 3.2 or newer on both hosts. On macOS, use a current rsync such as
Homebrew's. Use the same SSH alias as the launcher. Install this checkout and its
locked Pixi environment on the remote host before staging.

Run the transfer task on your local machine:

```sh
pixi run --locked transfer stage \
  --host compute-host --remote-checkout /absolute/remote/stan-mcp \
  --root /absolute/local/project --model models/fit.stan \
  --include models/includes/common.stan --data /absolute/local/project/data.json
```

List each include with a separate `--include`. Paths under `--root` keep their
relative layout. Include paths outside the root and symbolic links are rejected.
The JSON reply gives the remote `model.path` and `data[].path`. Use those paths
with `compile_model` and `register_data`, then use the returned Stan handles.
For a separate include search root, pass its staged absolute path to stanc.

Stage keeps verified bundles under the remote `STAN_MCP_HOME/staging` by
default. Set `--remote-root` to use another private directory. Model and data
bundles have separate hashes, so model edits leave unchanged data reusable.
Repeat the same command after an interruption to reuse partial files. Use
`--remote-pixi` when the remote Pixi executable is outside the SSH PATH.

After `export_draws` returns a remote path and SHA-256, fetch the file:

```sh
pixi run --locked transfer fetch \
  --host compute-host --remote-file /absolute/remote/fit.safetensors \
  --output /absolute/local/fit.safetensors --sha256 '<hash from export_draws>'
```

Fetch verifies the hash before publishing a new local file. Both commands put
progress on stderr and return JSON on stdout. If the host lacks rsync, use
OpenSSH SFTP to copy the files yourself and pass the resulting remote paths
to the same MCP tools.

## Run lifetime

Run IDs and caches persist remotely across client disconnects. A server restart
marks its abandoned runs failed. It does not adopt orphaned computations.

The manifest supports `osx-arm64`, `linux-64` and `linux-aarch64`. Linux targets
need their own Pixi environment and native C++ toolchain. Native
libraries and executables are platform-specific. Copy source files and data,
then compile on the destination. Do not copy the local `.pixi` environment.

A deployment check requires a configured SSH host. Local transport tests verify
argument quoting and a complete MCP exchange through a mock SSH transport.
