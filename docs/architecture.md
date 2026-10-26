# Architecture

`server.py` registers 28 MCP tools with `MCPServer`. Blocking work runs in
`asyncio.to_thread`. Tool errors become structured error dictionaries. Results
normalize nonfinite numbers recursively before MCP serialization.

| Module | Responsibility |
|---|---|
| `compile.py` | Resolve files and handles, snapshot includes, retain compiler settings, identify and build artifacts |
| `density.py` | Bound model cache, serialize native calls, density derivatives and parameter transforms |
| `toolchain.py` | Provision BridgeStan and CmdStan without contaminating protocol stdout |
| `sample.py` | Submit background inference, cancel owned processes, await completion, rerun saved inputs |
| `runstore.py` | Reserve run IDs, recover abandoned owners, store inputs, outputs and metadata, prune runs |
| `draws.py` | Validate typed CSV sidecars and headers, stream numeric data, preserve component shapes |
| `diagnostics.py` | Persist fits, summarize, extract, diagnose, inspect geometry, compute LOO, form warm starts |
| `stansummary.py` | Invoke and parse CmdStan's MCMC summaries |
| `methods.py` | Validate method-specific arguments |
| `config.py` | Resolve storage roots and concurrency limits |

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
