# Architecture

`stan_mcp/server.py` registers 37 MCP tools with `MCPServer`. Blocking work runs in
`asyncio.to_thread`. Tool errors become structured error dictionaries. Results
normalize nonfinite numbers recursively before MCP serialization.

| Module | Responsibility |
|---|---|
| `stan_mcp/compilation.py` | Resolve files and handles, snapshot includes, retain compiler settings, identify and build artifacts |
| `stan_mcp/density.py` | Bound model cache, serialize native calls, density derivatives and parameter transforms |
| `stan_mcp/batch/store.py` | Validate, snapshot and publish bounded Safetensors shards with content hashes |
| `stan_mcp/batch/evaluation.py` | Reuse process-owned native contexts across point shards, select autodiff curvature builds |
| `stan_mcp/batch/comparison.py` | Compare declared reference outputs and explicit finite-difference stencils |
| `stan_mcp/batch/jobs.py` | Persist batch jobs, cap worker processes, checkpoint, cancel and resume |
| `stan_mcp/toolchain.py` | Provision BridgeStan and CmdStan without contaminating protocol stdout |
| `stan_mcp/inference.py` | Submit background inference, cancel owned processes, await completion, rerun saved inputs |
| `stan_mcp/runstore.py` | Reserve run IDs, recover abandoned owners, store inputs, outputs and metadata, prune runs |
| `stan_mcp/draws.py` | Validate typed CSV sidecars and headers, stream numeric data, preserve component shapes |
| `stan_mcp/fitstorage.py` | Persist fits, raw CSVs and typed sidecars |
| `stan_mcp/diagnostics.py` | Summarize fits, report sampler health, localize divergences and compute LOO |
| `stan_mcp/geometry.py` | Inspect posterior conditioning in verified parameter coordinates |
| `stan_mcp/predictive.py` | Compare explicit observed and replicated statistics, write plots and reports |
| `stan_mcp/studies.py` | Snapshot study inputs, coordinate shared run workers, preserve and resume cases |
| `stan_mcp/calibration.py` | Report descriptive SBC ranks, coverage and missing-case bounds |
| `stan_mcp/validation_stats.py` | Calculate Wilson intervals and draw-law-aware Monte Carlo uncertainty |
| `stan_mcp/serialization.py` | Normalize numerical tool responses for JSON |
| `stan_mcp/stansummary.py` | Invoke and parse CmdStan's MCMC summaries |
| `stan_mcp/methods.py` | Validate method-specific arguments |
| `stan_mcp/config.py` | Resolve storage roots and concurrency limits |

`python -m stan_mcp` starts the server. Package imports do not start it.
Root `server.py` remains a compatibility launcher. `pyproject.toml` holds
pytest, Ruff and Pyright settings; Pixi owns the environment and dependencies.

## Source and data

Prefer absolute server-side files. Model snapshots contain expanded source,
include contents, source origin, compiler arguments and build identity. Handles
retain that snapshot after the original files disappear. Runs copy their
source manifests independently of the recompilable artifact cache.

Data files resolve once per operation. The consumed JSON snapshot also defines
its cache identity. Registered data snapshots remain stable after file changes.

BridgeStan's nonthreaded native autodiff state is shared across loaded models.
A process-wide lock covers native calls, model construction and destruction.
CmdStan runs use separate child processes and a bounded run pool.

## Runs and durable outputs

A process lock and exclusive directory creation reserve run IDs. Metadata
records the owning process identity. New servers fail abandoned runs and retain
records owned by other live servers. Cancellation remains set once successful.
Timed-out or cancelled awaiters release their registrations.

Each run owns resolved initialization files, generated-quantities input CSVs and
Laplace mode CSVs. Pruning source runs cannot break those snapshots or reruns.
All methods write enough CSV precision for constrained values to roundtrip.

CSV persistence must finish before a run succeeds or deletes its live output.
A failed copy retains every original. Optimize stores raw CSVs, a flattened
estimate and shaped initialization. Draw-producing methods also store a sidecar.

Sidecar v3 records exact CSV headers, ordered chain files, draw-row offsets,
variable types, shapes and real component leaves for complex/tuple variables.
Legacy v2 files remain readable with an explicit header warning. No NetCDF file
is written. Raw CSVs remain the durable draw values.

## Statistics

MCMC summaries use CmdStan's `stansummary`. Approximate fits use NumPy moments
and quantiles without MCMC convergence claims. Summaries bound displayed rows.
Extraction uses a common retained row stride and a strict total scalar cap.

Undefined diagnostics remain visible. A measured failure yields `warn`, missing
required diagnostic coverage yields `unknown`, and complete clean checks yield
`pass`. Geometry uses declared model parameters, excludes derived quantities,
and checks requested dimensions before transforming each draw.

Ordinary PSIS-LOO accepts MCMC posterior draws or GQ snapshots with verified MCMC
provenance. Approximate fits need target/proposal correction and are rejected.
Warm starts retain parameter shapes and check constrained support. An invalid
posterior mean falls back to a valid stored draw.

## Remote execution

`scripts/serve-ssh` carries stdio over SSH. Computation, paths and stored state
remain on the remote host. See [remote.md](remote.md). Linux environments resolve
separately and rebuild their own native artifacts.
