# stan-mcp

Use [Stan](https://mc-stan.org) from an MCP client. Compile models, evaluate log
densities and derivatives, run inference, and inspect convergence. Batch jobs
evaluate many points or compare Stan against results from another implementation.

## Install

Install [Pixi](https://pixi.sh) 0.83 or later, then run:

```sh
pixi run --script https://raw.githubusercontent.com/BJMCox/stan-mcp/main/install.py
```

Toggle **Claude Code** and **Codex** independently, then press Enter.
The installer downloads the repo and installs its Pixi environment.
No prior checkout or GitHub account is needed. Restart the selected clients.
See [installation](docs/install.md) for unattended setup, updates, and removal.
From an existing checkout, run `pixi run setup` or `pixi run serve`.

BridgeStan and CmdStan download their source trees on first use. Pixi supplies
Python and the C++ toolchain. Compiled models and run results live in
`~/stan-mcp`; set `STAN_MCP_HOME` to change that location.

## Use

Give your client absolute paths to a `.stan` model and a data `.json` file.
For example:

> Sample `/absolute/model.stan` with `/absolute/data.json`, then report the
> parameter estimates and any convergence problems.

Files must exist on the machine running the server. Runs retain input snapshots.
Batch inputs and outputs use Safetensors shards with JSON manifests.

- [Tools and examples](docs/usage.md)
- [Batch evaluation and comparisons](docs/batches.md)
- [Predictive checks, studies and calibration](docs/validation.md)
- [Remote execution over SSH](docs/remote.md)
- [Configuration and storage](docs/configuration.md)
