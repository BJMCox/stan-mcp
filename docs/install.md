# Install with Pixi

Install [Pixi](https://pixi.sh) and the clients you want to use.
The standalone installer requires Pixi 0.83 or later.
Supported hosts are Apple Silicon macOS, Linux x86-64, and Linux ARM64.

## Without a checkout

```sh
pixi run --script https://raw.githubusercontent.com/BJMCox/stan-mcp/main/install.py
```

Press `1` or `2` to toggle Claude Code or Codex. Press Enter to install.
Installed clients start selected. Clear both to install only the environment.
Press `q` to quit.

For an unattended install, select clients explicitly:

```sh
pixi run --script https://raw.githubusercontent.com/BJMCox/stan-mcp/main/install.py -- --claude --codex
```

Use either flag alone for one client, or `--no-clients` for the environment alone.
Restart each selected client after installation.

Pixi supplies Python and Git. The installer clones the public repository over HTTPS.
No GitHub account, GitHub CLI, or manual download is required.
Downloaded installs use `~/.local/share/stan-mcp`, or `$XDG_DATA_HOME/stan-mcp`.
Pass `--directory /absolute/path` to choose another location.
Pass `--source /absolute/checkout` to use an existing checkout.
Keep that directory: the client registrations point to its Pixi environment.

For a downloaded script on stdin, use `pixi run --script -`.
The menu still reads your terminal after Pixi consumes the script.

## From a checkout

```sh
git clone https://github.com/BJMCox/stan-mcp.git
cd stan-mcp
pixi run setup
```

Use `pixi run setup --claude --codex` to select both clients without the menu.

## Registration and updates

The installer uses the clients' `mcp add` commands. Claude Code uses user scope.
Both clients receive absolute Pixi and manifest paths, so launching works from any directory.
An identical registration remains unchanged. A different existing `stan` entry stops installation.
Pass `--name stan-other` to keep both entries, or remove the old entry in the client first.
Unselected clients remain unchanged.

The checked-in lockfile fixes Python, Python packages, and the C++ toolchain.
BridgeStan and CmdStan source trees install on first use, as before.
Stored runs remain under `~/stan-mcp`, independently of the checkout.

To update, enter the installed checkout and run:

```sh
git pull --ff-only
pixi run --locked setup
```

The installer never pulls or resets an existing checkout itself.
Restart the clients after an update.

To remove registrations:

```sh
codex mcp remove stan
claude mcp remove --scope user stan
```

The checkout and saved runs remain on disk.

See [Pixi script mode](https://pixi.prefix.dev/latest/python/scripts/),
[Codex MCP setup](https://developers.openai.com/codex/mcp/), and
[Claude Code MCP setup](https://code.claude.com/docs/en/mcp) for client details.
