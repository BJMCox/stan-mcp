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

Use remote absolute `.stan` and `.json` paths. Copy inputs with `scp` or `rsync`
before the tool call. Reuse model and data handles to avoid repeated transfers.
The launcher does not translate local paths or sync files automatically.
Run IDs and caches persist remotely across client disconnects. A server restart
marks its abandoned runs failed. It does not adopt orphaned computations.

The manifest supports `osx-arm64`, `linux-64` and `linux-aarch64`. Linux targets
need their own Pixi environment and native C++ toolchain. Native
libraries and executables are platform-specific. Copy source files and data,
then compile on the destination. Do not copy the local `.pixi` environment.

A deployment check requires a configured SSH host. Local transport tests verify
argument quoting and a complete MCP exchange through a mock SSH transport.
