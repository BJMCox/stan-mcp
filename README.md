# stan-mcp

An MCP server for compiling, inspecting, and running [Stan](https://mc-stan.org)
models. The primary use is fast in-memory log-density and gradient interaction
through [BridgeStan](https://github.com/roualdes/bridgestan); full inference runs
(all six CmdStanPy methods) are available through
[CmdStanPy](https://github.com/stan-dev/cmdstanpy).

It is a general Stan harness for any `.stan` model, and doubles as the runner for
the FlatPPL validation loop (compile emitted Stan, check its log density, sample
from it).

## Setup

```sh
pixi install          # Python deps + C++ toolchain (clang, make)
pixi run serve        # start the MCP server
```

The C++ toolchain is provided by pixi. The BridgeStan and CmdStan source trees
are downloaded automatically to `STAN_MCP_HOME` on first compile (large — the
first compile takes a while).

## Tasks

| Task | Command |
|------|---------|
| Serve | `pixi run serve` |
| Test | `pixi run test` |
| Live smoke (needs toolchain) | `pixi run test-live` |
| Lint | `pixi run lint` |
| Format | `pixi run format` |
| Typecheck | `pixi run typecheck` |

## Tools

- **Compile / metadata:** `compile_model`, `param_info`
- **BridgeStan density (primary):** `log_density`, `log_density_gradient`,
  `log_density_hessian`, `param_constrain`, `param_unconstrain`, `check_gradient`
- **Runs (all methods):** `start_run` (`method` ∈ `sample`, `optimize`,
  `variational`, `pathfinder`, `laplace`, `generate_quantities`),
  `get_run_status`, `await_run`, `list_runs`, `get_run_summary`,
  `get_run_diagnostics`, `extract_draws`

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `STAN_MCP_HOME` | `~/stan-mcp` | Root for source trees, compiled artifacts, and runs |
| `STAN_MCP_LRU_SIZE` | `8` | Max loaded BridgeStan models kept in memory |

## Design

Full design: `docs/2026-07-14-stan-mcp-design.md` (local-only).
