# Working on stan-mcp

## Environment and checks

Use Pixi for Python and dependencies. The supported environment uses Python 3.15.
Use Python MCP for interactive Python work when available.

```sh
pixi install
pixi run serve
pixi run test
pixi run test-live
pixi run format
pixi run lint
pixi run typecheck
```

Run focused tests with `pixi run pytest tests/<file>.py -q`.
Native tests require the C++ toolchain and run through `test-live`.
Pytest, Ruff and Pyright settings live in `pyproject.toml`.

Keep the README short and written for users. Put agent instructions here.
Keep plans, review ledgers and experiment reports outside Git.

## Module ownership

- `stan_mcp/server.py` registers MCP tools and offloads blocking work.
- `stan_mcp/compilation.py` owns source/data resolution, snapshots and build identity.
- `stan_mcp/density.py` owns native models, the model cache and process-wide native locking.
- `stan_mcp/inference.py` owns background inference and warm starts.
- `stan_mcp/runstore.py` owns persisted runs and process ownership.
- `stan_mcp/fitstorage.py` owns fit persistence and typed CSV sidecars.
- `stan_mcp/draws.py` owns typed draw loading and bounded extraction.
- `stan_mcp/diagnostics.py` owns summaries, convergence, pathologies and LOO.
- `stan_mcp/geometry.py` owns posterior conditioning checks.
- `stan_mcp/predictive.py` owns checks of explicit replicated and observed quantities.
- `stan_mcp/studies.py` owns multi-run studies, simulator mappings and resume checkpoints.
- `stan_mcp/calibration.py` owns descriptive SBC reports and artifacts.
- `stan_mcp/validation_stats.py` owns shared uncertainty calculations.
- `stan_mcp/batch/` separates artifact storage, evaluation, comparison and job coordination.
- `stan_mcp/serialization.py` normalizes numerical values for JSON tool responses.

Keep package imports light. Launch with `python -m stan_mcp`.
Keep root `server.py` as a guarded compatibility launcher.
Keep multiprocessing targets at module scope so spawned workers can import them.
Import each module through its canonical package path. Avoid `sys.path` and `sys.modules` aliases.

## Contracts to preserve

- Prefer absolute server-local `.stan`, data `.json` and parameter-file paths in tool descriptions.
- Keep inline model/data values as fallbacks. Preserve MCP tool names and argument defaults.
- Preserve model/data/build identities, include snapshots and native build flags.
- Keep native calls, construction and destruction behind the same process-wide lock.
- Give each batch worker its own native model. Never transfer native pointers between processes.
- Preserve stable row IDs, tensor axes/dtypes, schema versions and content hashes.
- Keep completed shards immutable. Verify inputs, settings, seed and build identity before resume.
- Keep job completion distinct from numerical agreement. Missing checks must remain visible.
- Compare native derivatives only when coordinates, ordering and target conventions agree.
- Require autodiff builds for batch Hessians/HVPs. Never add a silent finite-difference fallback.
- Keep finite differences explicit and label empirical uncertainty honestly.
- Keep raw CmdStan CSVs as the durable draw values. Preserve typed sidecar column ordering.
- Preserve input CSV and initialization snapshots for reproducible reruns.
- Use MCMC diagnostics only for MCMC draws. Approximate fits do not acquire R-hat/ESS claims.
- Treat fixed-parameter simulation separately from MCMC. Require an explicit declaration for iid uncertainty.
- Preserve every planned study case. Never replace a failed simulated dataset with a new successful case.
- Keep SBC ranks descriptive when retained posterior independence is unverified.
- Keep numerical payloads and error summaries bounded. Send arrays through files for batches.
- Keep installer/build progress on stderr. Reserve stdout for the MCP protocol.

## Refactoring and tests

Use type hints at module interfaces. Preserve numeric precision and array semantics.
Prefer domain modules and small functions over broad class hierarchies.
Keep numerical algorithms separate from file and process coordination.
Keep shared test fakes in `tests/fakes.py`. Do not import collected test modules.
Record each new test's purpose and nearest existing coverage outside source files.
Use analytic controls for numerical changes. Retain spawn, cancellation, resume and disk-failure coverage.
Run the ordinary and native suites after changes to imports or process ownership.
Verify all MCP tool schemas and SSH stdio behavior after entry-point changes.

Sign commits. Keep each commit focused. Get explicit approval before pushing or opening a PR.
