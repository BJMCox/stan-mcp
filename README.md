# stan-mcp

Use [Stan](https://mc-stan.org) from an MCP client. Compile models, evaluate log
densities and derivatives, run inference, and inspect convergence. Batch jobs
evaluate many points or compare Stan against results from another implementation.

## Run

Install [Pixi](https://pixi.sh), then run:

```sh
pixi install
pixi run serve
```

The server uses stdio. Example MCP client configuration:

```json
{
  "mcpServers": {
    "stan": {
      "command": "pixi",
      "args": ["run", "--manifest-path", "/absolute/path/stan-mcp/pixi.toml", "serve"]
    }
  }
}
```

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
