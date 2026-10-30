# Internal interfaces

See [architecture.md](architecture.md) for module ownership and
[usage.md](usage.md) for the public tools.

All modules live under `stan_mcp`. Batch modules live under `stan_mcp.batch`.
Import canonical package paths to retain one cache, native lock and manager.

## Compilation

`resolve_source(stan, stanc_args=None)` accepts source, a `.stan` path or a
model handle. A `SourceSnapshot` behaves as source text and carries its manifest.
Includes resolve using Stan's configured search roots. Snapshots preserve the
expanded source and each dependency after the original files disappear.

`model_manifest(source)` returns an owned JSON manifest. `restore_source(source,
manifest)` verifies and restores a saved snapshot. `stan_file_for`, `compiled_so`
and `cmdstan_model` own cache materialization and builds. Build identity records
toolchain versions, relevant files, compiler settings and platform.

`canonical_data` accepts a mapping, JSON string, JSON file, registered handle or
null. `register_data` writes the exact canonical payload under its content hash.
A density request resolves its data once for cache identity and native loading.

`toolchain.ensure_bridgestan` and `ensure_cmdstan` provision source trees through
the libraries. Installer child-process output is captured and sent to stderr,
leaving MCP stdout available to the protocol.

## Native models

`density.get_model` returns a bounded cached model facade. The facade serializes
native methods across all model objects because nonthreaded artifacts share
native autodiff state. Retained bound methods keep their serialized owner alive.
Construction and destruction use the same lock.

`constrained_values(names, values)` reconstructs Stan JSON values from indexed
native names. Stanio handles real arrays, complex components and nested tuples.
Parameter transform tools use this representation in both directions.

## Run ownership

`runstore.reserve_run(method)` reserves an ID and directory exclusively across
processes. Metadata records PID and process creation time. `reconcile_abandoned`
marks unverifiable/dead owners failed while retaining other live owners.

`RunManager.start_run` owns resolved model, data, initialization and CSV inputs
before worker submission. Submission errors remove incomplete reservations.
`rerun` consumes owned snapshots. `_execute` records the effective build identity
after cold provisioning, persists outputs, then writes a terminal status.

A run handle owns a completion event, asynchronous waiter registrations and a
cancellation lock. Successful cancellation remains monotonic through worker
completion. Await timeouts and task cancellation remove waiter registrations.

`persist_inputs` stores source, data, resolved initialization and source manifest.
`persist_input_csvs` preserves caller chain order with numbered file names.
`persist_initialization` stores shaped optimize values. `csv_paths` exposes the
raw durable outputs, including optimization output for generated quantities.

## Draws

`draws.schema` validates sidecar structure and reports legacy header uncertainty.
`csv_files` validates complete headers before returning chain-ordered paths.
`load` streams numeric CSV rows through NumPy, skips recorded nondraw rows and
assembles arrays shaped `(chain, draw, *declared_dims)`.

Schema v3 retains each original variable's recursive type tree. `params` contains
real component leaves for complex and tuple variables. `spec_element_labels`
returns display labels and exact CSV names for a leaf. `matrix` preserves these
labels. `parameter_matrix` selects exact CSV names in native parameter order.

Extraction chooses one stride for all selected leaves. A positive value budget
must hold one complete retained row, including every chain in by-chain mode.
Scalar, length-one vector, matrix and array axes remain distinct.

## Diagnostics and persistence

`fitstorage.persist_run` copies every raw CSV before writing a success sidecar
or deleting live output. Copy failure records a warning and raises. Optimize
stores both the flattened estimate and shaped JSON initialization.

`_sidecar_from_fit` uses CmdStanPy metadata and raw headers. `_draw_row_offset`
excludes variational means, saved warmup and intermediate optimization rows.
Generated-quantities runs retain the source sampling law in run metadata.

`run_summary` uses CmdStan summaries for MCMC and NumPy summaries otherwise.
`run_diagnostics` tracks undefined metrics and incomplete chain coverage.
`run_loo` and `compare_runs` verify posterior provenance before ordinary ArviZ LOO.

`inference.posterior_mean_inits` returns a shaped estimate or verified constrained mean,
then a valid stored draw if the mean leaves the support. Historical runs without
source retain their means with an explicit initialization warning in metadata.

`geometry._unconstrained_draws` restores compiler settings, checks dimensions, selects
model parameters and transforms them. If inverse transforms reject rounded
CSV values, the constrained fallback still excludes derived quantities. If the
model cannot be rebuilt, geometry fails explicitly rather than guessing its
coordinate set.

`serialization.json_safe` handles NumPy arrays/scalars and replaces nonfinite
numbers with JSON nulls. Tool response normalization uses this shared function.
