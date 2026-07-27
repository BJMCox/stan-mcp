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
first compile takes a while). Pixi-managed, single platform (`osx-arm64`),
Python 3.11/3.12.

## Tasks

| Task | Command |
|------|---------|
| Serve | `pixi run serve` |
| Test | `pixi run test` |
| Live smoke (needs toolchain) | `pixi run test-live` |
| Lint | `pixi run lint` |
| Format | `pixi run format` |
| Typecheck | `pixi run typecheck` |

Single test: `pixi run pytest tests/test_density_eval.py::test_log_density_shapes -q`
(swap in any test file/node id; `pixi run` prefixes any command with the
project's pixi env, not just the declared tasks). `pytest.ini` sets
`asyncio_mode = auto`, so async test functions need no `@pytest.mark.asyncio`
marker. Live tests (`tests/test_live_smoke.py`) are `skipif`-gated on
`STAN_MCP_LIVE=1` since they compile and sample a real Bernoulli model — every
other test in `tests/` runs against fakes (`tests/fakes.py`) and needs no
toolchain.

## Quickstart

Compile a model, then evaluate its log density and gradient at a point in
unconstrained space:

```python
compile_model(stan="parameters { real x; } model { x ~ normal(0, 1); }")
# -> {"model_id": "…", "so_path": "…", "exe_path": "…"}

log_density_gradient(stan="<model_id from above>", data=None, theta_unc=[0.5])
# -> {"lp": -0.375, "gradient": [-0.5]}
```

Or run full inference and pull a summary once it finishes:

```python
start_run(stan="<model_id>", data={"N": 10, ...}, method="sample")
# -> {"run_id": "sample-0001", "method": "sample", "status": "running"}

await_run(run_id="sample-0001")
get_run_diagnostics(run_id="sample-0001")
get_run_summary(run_id="sample-0001")
```

A `model_id` (from `compile_model`) or `data_id` (from `register_data`) can be
passed anywhere a tool expects `stan`/`data`, so a full model source or data
payload only needs to be sent once.

## Documentation

| Page | Contents |
|------|----------|
| [docs/usage.md](docs/usage.md) | Every MCP tool: parameters, return shape, behavior |
| [docs/configuration.md](docs/configuration.md) | Env vars, defaults, on-disk layout |
| [docs/architecture.md](docs/architecture.md) | Module map, data flow, design decisions |
| [docs/internals.md](docs/internals.md) | Developer reference: internal functions and types |

Full design rationale: `docs/2026-07-14-stan-mcp-design.md` (local-only, not
pushed to remotes).
