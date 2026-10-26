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
Python 3.15.

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

Prefer file inputs. Pass an absolute `.stan` path as `stan` and an absolute
`.json` path as `data`. Both files must exist on the MCP server's filesystem.
Relative paths resolve from the server's working directory. Inline Stan and
JSON strings remain available as fallbacks when files are unavailable.

Compile an existing model file, then evaluate its log density and gradient at
a point in unconstrained space:

```python
compile_model(stan="/absolute/path/model.stan")
# -> {"model_id": "…", "so_path": "…", "exe_path": "…"}

log_density_gradient(
    stan="<model_id from above>",
    data="/absolute/path/data.json",
    theta_unc="/absolute/path/theta.json",
)
# theta.json contains a numeric array, e.g. [0.5]. Its length must match the model.
```

Or run full inference and pull a summary once it finishes:

```python
start_run(stan="/absolute/path/model.stan", data="/absolute/path/data.json", method="sample")
# -> {"run_id": "sample-0001", "method": "sample", "status": "running"}

await_run(run_id="sample-0001")
get_run_diagnostics(run_id="sample-0001")
get_run_summary(run_id="sample-0001")
```

A `model_id` (from `compile_model`) or `data_id` (from
`register_data(data="/absolute/path/data.json")`) can be passed anywhere a tool
expects `stan`/`data`. Reuse these IDs for repeated calls. Registered data is a
snapshot of the file contents. Runs save their model source and data contents,
so later file changes do not alter their stored inputs.

The density, gradient, Hessian, constrain, and gradient-check tools accept a
JSON array file as `theta_unc`. `param_unconstrain` accepts a JSON object file
as `constrained`, for example `{"theta": 0.5}`. Prefer these files over inline
lists and dictionaries, which remain supported.

For generated quantities from an external fit, pass CmdStan CSV paths:

```python
start_run(
    stan="/absolute/path/model_with_gq.stan",
    data="/absolute/path/data.json",
    method="generate_quantities",
    method_args={"csv_files": ["/absolute/path/chain-1.csv", "/absolute/path/chain-2.csv"]},
)
```

`csv_files` accepts one path or a list in chain order. Files must retain the
CmdStan headers from a sample, optimize, or variational fit. Optimize and
variational accept one CSV each. Each run saves copies, so `rerun` works after
the original files change or disappear. Draw
summaries exclude saved warmup, intermediate optimization iterations, and the
variational mean row. Use either `csv_files` or `from_run_id`, never both.

## Documentation

| Page | Contents |
|------|----------|
| [docs/usage.md](docs/usage.md) | Every MCP tool: parameters, return shape, behavior |
| [docs/configuration.md](docs/configuration.md) | Env vars, defaults, on-disk layout |
| [docs/architecture.md](docs/architecture.md) | Module map, data flow, design decisions |
| [docs/internals.md](docs/internals.md) | Developer reference: internal functions and types |

Full design rationale: `docs/2026-07-14-stan-mcp-design.md` (local-only, not
pushed to remotes).
