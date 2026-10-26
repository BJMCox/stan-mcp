# Configuration

All configuration is environment-variable based, resolved in `config.py`.
There is no config file.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `STAN_MCP_HOME` | `~/stan-mcp` | Root directory for compiled artifacts, registered data, and runs (`config.home()`) |
| `STAN_MCP_LRU_SIZE` | `8` | Max BridgeStan `StanModel` objects kept loaded in memory at once (`config.MODEL_LRU_SIZE`, read once at import) |
| `STAN_MCP_MAX_WORKERS` | `2` | Size of the background run thread pool (`config.max_workers()`, read at `RunManager` construction) |
| `STAN_MCP_MAX_PARALLEL_CHAINS` | `cpu_count // max_workers` | Per-run cap on concurrently executing CmdStan chain subprocesses (`config.max_parallel_chains()`, applied as the `parallel_chains` default for `sample`) |
| `STAN_MCP_LIVE` | unset | Not read by application code — gates `tests/test_live_smoke.py` via `pytest.mark.skipif`; set to `1` to run the live-toolchain smoke test |

`STAN_MCP_LRU_SIZE` is read once at module import (`int(os.environ.get(...) or
8)`), so it must be set before the process starts. `STAN_MCP_MAX_WORKERS` is
read via a function (`config.max_workers()`) called when `RunManager` is
constructed (module-level `MANAGER = RunManager()` in `sample.py`), so it's
also effectively fixed at process start.

`STAN_MCP_MAX_PARALLEL_CHAINS` is read per run. Without it CmdStanPy defaults
`parallel_chains` to `min(cpu_count(), chains)` *per run*, so `max_workers`
concurrent runs would oversubscribe the machine by that factor; the default
divides the CPUs between the pool slots instead. A caller can still override it
for a single run by passing `method_args={"parallel_chains": N}` (or
`threads_per_chain`) to `start_run`.

## On-disk layout (`STAN_MCP_HOME`)

```
compiled/       compiled artifacts, keyed by model_id (see below)
  <model_id>.stan            persisted source
  <model_id>_model.so        BridgeStan shared object
  <model_id>*                CmdStan executable and any other stanc outputs
  data/<data_id>.json        registered data payloads, keyed by content hash
runs/<run_id>/  one directory per persisted inference run (see below)
```

`compiled/` and `runs/` are created on demand. BridgeStan downloads its source
into `~/.bridgestan/bridgestan-<version>` and CmdStanPy installs CmdStan into
`~/.cmdstan/cmdstan-<version>`. Set `BRIDGESTAN` and `CMDSTAN` to existing source
trees. These locations are independent of `STAN_MCP_HOME`.

### `compiled/` — keyed by `model_id`

Model handles identify a persisted source snapshot, compiler arguments, included
source contents, and build identity. A changed include or toolchain requires a new
artifact. Manifests retain arguments and source origin when a handle is reused.
Data handles identify one canonical JSON snapshot. Build locks serialize artifact
creation. Loaded native calls are serialized across server threads.

Everything under `compiled/` is recomputable, so `prune_compiled` allows
`keep=0`.

### `runs/<run_id>/` — one directory per run

`run_id` is `<method>-NNNN`. A process lock and exclusive directory creation
reserve each ID. Run metadata records the owning PID and process creation time.
A new server marks abandoned runs failed and preserves runs owned by live servers.

```
model.stan     exact Stan source used (persist_inputs, for reproduction)
data.json      exact data payload used (omitted if data was None)
live/          CmdStan's live streamed CSV output during the run (per-run,
               so cancel_run can target this run's subprocess and
               get_run_progress can count draws written); deleted once the
               CSVs are safely copied into csv/, so a finished run never
               keeps two copies of its draws
csv/           the same CSVs copied here once the run finishes (source of
               truth for generate_quantities reuse, and the fallback data
               source for summary/diagnostics/extract_draws on methods
               arviz can't convert)
draws_schema.json  typed shapes, ordered files, and exact CSV header
inits.json     resolved initialization snapshot
source_manifest.json  model arguments and source snapshot identity
input_csv/     owned fitted-parameter and Laplace mode snapshots
estimate.json  optimized_params_dict (optimize only)
meta.json      run_id, method, status, model_id, seq, method_args, seed,
               created_at/finished_at/duration_s, stderr (on failure),
               method_diagnostics (VI/pathfinder/laplace scalars),
               persist_warning / csv_warning (fatal persistence failures)
```

`meta.json` is written atomically (`runstore.write_meta`: write to a `.tmp`
file, then `Path.replace`) so a concurrent reader (`list_runs`,
`get_run_status`, ...) never observes a truncated file; readers that hit a
`JSONDecodeError` mid-write simply skip that run for the current call.

`status` is one of `running`, `cancelling`, `done`, `failed`, `cancelled`.
Runs still `running` or `cancelling` are never deleted by `prune_runs` (would race the writer).

Everything under `runs/` is **not** recomputable in general (a run's exact
random seed/draws), so `prune_runs` refuses `keep=0`.
