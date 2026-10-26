# Usage: MCP tools

All tools are registered in `server.py` via `@mcp.tool()` and are thin wrappers:
they validate/resolve arguments, offload the actual work to a module function
via `asyncio.to_thread` (so a slow compile or density call doesn't block the
event loop), and catch `StanMcpError` (and generic exceptions) into a
structured `{"error": ..., "error_type": ...}` dict rather than raising —
MCPServer therefore never returns a raw traceback.

Two handles thread through nearly every tool:

- **`model_id`** — returned by `compile_model`. Pass it as `stan` on any other
  tool instead of resending the full source.
- **`data_id`** — returned by `register_data`. Pass it as `data` on any other
  tool instead of resending the full payload.

`stan` also accepts a path ending in `.stan` (read from disk) or raw Stan
source. `data` also accepts a dict, a path ending in `.json`, or a raw JSON
string. See `compile.resolve_source` / `compile.canonical_data` in
[docs/internals.md](internals.md).

## Compile / metadata

### `compile_model(stan, stanc_args=None, warn_pedantic=False) -> dict`
Compiles a Stan model to both a BridgeStan `.so` and a CmdStan executable from
one source (either backend may be asked for later on the same model).
Compilation is disk-cached by a hash of `(source, stanc_args)`; a repeat call
with the same source and args returns the cached artifact paths without
recompiling.

- `stanc_args` — extra `stanc` compiler flags (e.g. `["--O1"]`); part of the
  cache key.
- `warn_pedantic` — if true, additionally runs a standalone `stanc
  --warn-pedantic` pass and returns any findings under `warnings`. Kept out of
  band deliberately: pedantic mode doesn't change generated code, so it must
  not perturb `model_id`, and it should surface even when the binary is
  already cached.

Returns `{"model_id", "so_path", "exe_path"}`, plus `"warnings"` if
`warn_pedantic` found any. On compile failure returns
`{"error", "error_type": "CompileError", "compile_log"}` instead of raising.

### `register_data(data) -> dict`
Caches a data payload (dict, `.json` path, or JSON string) to
`compiled/data/<data_id>.json` keyed by its content hash. Returns
`{"data_id"}`. The id can then be passed as `data` anywhere.

### `validate_data(stan, data=None) -> dict`
Builds the BridgeStan model against `data` to confirm it satisfies the data
block. Building fails loudly (a missing/malformed variable raises inside
BridgeStan) if invalid; on success returns
`{"ok": true, "param_num", "param_unc_num"}`.

### `list_compiled(limit=50) -> dict`
Enumerates the compiled-model and registered-data caches, most recently used
first: `{"models": [{"model_id", "last_used", "so_built", "source_bytes"}, ...],
"data": [{"data_id", "bytes"}, ...]}`. Lets a caller recover a lost handle
without re-sending the model source or data payload.

### `prune_compiled(keep=None, older_than_days=None) -> dict`
Prunes the recompilable `compiled/` cache and clears the in-process BridgeStan
LRU (which would otherwise keep serving models whose artifacts were just
deleted). `keep` retains the newest N models by mtime — mtime is refreshed on
every *use*, not just on first compile, so this is genuine LRU eviction
(across all `.stan`/`.so`/exe files sharing a `model_id` stem);
`older_than_days` additionally drops models and cached data older than the
cutoff. At least one of `keep`/`older_than_days` is required. Unlike
`prune_runs`, `keep=0` is allowed — everything recompiles on demand. Returns
`{"removed_models": [...], "removed_data": [...]}`.

### `param_info(stan, data=None) -> dict`
Returns `{"param_names", "param_unc_names", "param_num", "param_unc_num"}` for
the model (builds/loads it via the BridgeStan LRU cache; see
[docs/architecture.md](architecture.md)).

## BridgeStan density (primary path)

All five density tools resolve `stan`/`data`, fetch (or build) the BridgeStan
model from the in-process LRU cache, and validate `theta_unc` has exactly
`param_unc_num()` entries (raises `ValidationError` otherwise).

Prefer an absolute JSON file path for `theta_unc`. The file must contain a
numeric array. Inline numeric lists remain supported. `param_unconstrain`
likewise accepts a JSON object file path for `constrained`, with an inline
named parameter dict as a fallback.

### `log_density(stan, data, theta_unc, propto=True, jacobian=True) -> dict`
Returns `{"lp": float}` — log density at the unconstrained parameter vector.
`propto` drops additive constants; `jacobian` includes the constrained-space
Jacobian adjustment.

### `log_density_gradient(stan, data, theta_unc, propto=True, jacobian=True) -> dict`
Returns `{"lp", "gradient": [float, ...]}`.

### `log_density_hessian(stan, data, theta_unc, propto=True, jacobian=True) -> dict`
Returns `{"lp", "gradient", "hessian": [[float, ...], ...]}`.

### `param_constrain(stan, data, theta_unc, include_tp=False, include_gq=False) -> dict`
Maps an unconstrained vector to named constrained values:
`{"constrained": {name: value, ...}}`. Requesting `include_gq` allocates a
BridgeStan RNG (seed 1234) since generated quantities may sample.

### `param_unconstrain(stan, data, constrained) -> dict`
Inverse of `param_constrain`: maps a `{name: value}` dict to the unconstrained
vector. Returns `{"theta_unc": [float, ...]}`.

### `check_gradient(stan, data, theta_unc, eps=1e-6, tol=1e-4) -> dict`
Compares the autodiff gradient against central finite differences component by
component. Returns `{"max_abs_err", "max_rel_err", "mismatches": [{"index",
"autodiff", "finite_diff"}, ...]}` where `mismatches` lists only components
whose absolute error exceeds `tol`.

## Runs (all six CmdStanPy methods)

`method` is one of `sample`, `optimize`, `variational`, `pathfinder`,
`laplace`, `generate_quantities` (see `methods.METHODS` in
[docs/internals.md](internals.md) for each method's allowed `method_args` and
capabilities).

### `start_run(stan, data=None, method="sample", seed=None, inits=None, method_args=None, inits_from=None) -> dict`
Resolves and persists the model source and data, assigns a sequential
`run_id` (`<method>-NNNN`), and submits execution to a background thread pool.
Returns immediately: `{"run_id", "method", "status": "running"}`.

- `method_args` — dict of method-specific kwargs (validated against
  `methods.METHODS[method]["args"]`; unknown keys raise `ValidationError`).
  For `generate_quantities`, use either `method_args.from_run_id` to reuse a
  managed run or `method_args.csv_files` for external CmdStan CSVs. Prefer
  absolute paths. `csv_files` accepts a single path or a list in chain order,
  with original CmdStan headers from sample, optimize, or variational. Optimize
  and variational require one CSV each. The run saves copies for reproduction,
  and summaries exclude saved warmup, intermediate optimization iterations,
  and the variational mean row. The two source options are mutually exclusive.
- `inits_from` — a prior `run_id` to warm-start at (posterior mean for
  draws-producing runs when valid, a stored draw otherwise; shaped estimate for `optimize`); ignored if `inits` is
  explicitly given. Not accepted by `laplace` or `generate_quantities`
  (`methods.METHODS[...]["inits"]` is `False` for those). `posterior_mean_inits`
  now covers every parameter rank: with dims coming from the run's
  `draws_schema.json` sidecar rather than being reconstructed from flat CSV
  column names, a matrix or higher-rank parameter gets a real nested-list mean
  instead of being silently left for Stan to default-init.

  **It does not check that the two models match**, deliberately: warm-starting a
  *related* model is the point (same source with more data, a centered→
  non-centered rewrite, one more parameter than last time), and a `model_id`
  equality check would reject exactly those. What CmdStan does with a partial or
  mismatched init is what makes this safe — an init name the target model doesn't
  have is ignored, a parameter absent from the init is default-inited by Stan, a
  dimension clash errors with the variable name and both shapes, and a
  constraint violation errors naming the bound. Inits cannot bias a posterior,
  so the worst outcome of a bad warm start is wasted warmup, not a wrong answer.
- `variational` runs are started with `require_converged=False` by default
  (ADVI is approximate; draws are still returned even if the ELBO gate isn't
  met).
- `sample` runs default `parallel_chains` to `STAN_MCP_MAX_PARALLEL_CHAINS` (so
  concurrent runs don't oversubscribe the box) and `sig_figs` to 17. The latter
  is what makes constrained draws round-trip through BridgeStan, which
  `get_run_geometry` needs; CmdStan's default 6 is lossy enough that a simplex
  read back fails `simplex_free`'s 1e-8 check. It costs ~1.9x the CSV bytes —
  pass `method_args={"sig_figs": 8}` to trade that back on a very large run,
  accepting that its geometry will only be readable on the constrained scale.

### `get_run_status(run_id) -> dict`
Non-blocking. Returns `{"run_id", "method", "status"}` plus `"error"` (first
line of stderr) if the run failed, and `"persist_warning"`/`"csv_warning"` if
present — a `status="done"` run whose draws failed to persist is still
flagged.

### `get_run_progress(run_id) -> dict`
Status plus `elapsed_s` (wall-clock so far, or the final `duration_s` if
finished). For a running `sample`, also parses the live CmdStan CSVs to report
`draws_written`, `expected_draws` (`chains × iter_sampling`), and `progress`
(0–1, clamped).

### `await_run(run_id, timeout=None) -> dict`
Blocks server-side until the run finishes or `timeout` elapses. Returns
`{"status": "running", "run_id"}` on timeout, otherwise
`{"status", "run_dir", "meta"}`. Works across process restarts: if the run's
in-memory handle is gone (finished in a prior process), reads status straight
from `meta.json`.

### `cancel_run(run_id) -> dict`
Only a still-queued run (thread-pool future not yet started) can be truly
stopped — it's finalized as `status="cancelled"` immediately. Once CmdStan is
executing, this instead SIGKILLs the run's subprocess(es) (identified by the
run's `live/` output path appearing in their command line) and returns
`cancelled` true/false depending on whether a process was found; the worker
thread then relabels the run `"cancelled"` once it sees the nonzero exit. A
run already finished returns `{"cancelled": false, "reason": "run already
finished"}`.

### `rerun(run_id) -> dict`
Starts a fresh run from a prior run's persisted source, data, method, seed,
and `method_args` (inits are not persisted and are not carried over). Returns
a new `run_id` in the same shape as `start_run`.

### `list_runs(model_id=None, limit=50, status=None, method=None) -> dict`
Returns `{"runs": [...]}`, newest first (by sequence number), reading every
`meta.json` under `runs/` and filtering by the given criteria. Each row's
`stderr` is truncated to its first line; use `get_run_status` for a bounded
excerpt of one run's failure, or `await_run` for the full traceback.

### `get_run_inputs(run_id) -> dict`
Returns the persisted `{"source", "data"}` for a run (for manual reproduction).
An unknown `run_id` raises `RunError` rather than returning two nulls.

### `prune_runs(keep=None, older_than_days=None) -> dict`
Deletes stored run directories. At least one of `keep`/`older_than_days` is
required; `keep=0` is refused (runs aren't recomputable — refuses to delete
everything). Never deletes a run with `status="running"` (would race its
writes). Returns `{"removed": [run_id, ...]}`.

## Diagnostics / summary

These are method-aware: `sample` runs read their statistics from CmdStan's own
`stansummary` binary; the other draws-producing methods (`variational`,
`pathfinder`, `laplace`, `generate_quantities`) compute mean/sd/quantiles
directly from the typed draws — no R-hat/ESS, since they aren't MCMC chains;
`optimize` reads a stored point estimate. All draws-producing methods share one
reader (`draws.load`, keyed off a per-run `draws_schema.json` sidecar) instead
of arviz's `inference.nc`, which never existed for `variational`/`pathfinder`/
`laplace` in the first place (`arviz.from_cmdstanpy` cannot convert those
fits).

**Runs persisted before the typed draws layer cannot be read.** They have an
`inference.nc` and no `draws_schema.json`, and nothing migrates one to the
other — the sidecar is written from the live fit's own column names, which a
finished run no longer has. So on an older run `get_run_summary`,
`extract_draws`, `get_run_pathologies`, `get_run_geometry`, `run_loo`,
`compare_runs` and `posterior_mean_inits` all raise `RunError` ("has no
draws_schema.json … re-run the model"). Re-running the model is the remedy;
the persisted `model.stan`/`data.json` make that a `rerun` call.
(`docs/internals.md` covers the neighbouring case of a sidecar whose *version*
does not match.)

**`get_run_diagnostics` is the one exception, deliberately.** On a `sample` run
with no readable sidecar it degrades to `convergence_warning` +
`sampler_stats_warning` + `verdict: "unknown"` rather than raising, while
`get_run_summary` on the same run raises. The two answer different questions:
a summary with no statistics in it is not a summary, so there is nothing to
return; whereas "I could not check this run" is itself the useful answer to
"is this run healthy?", and is exactly what `"unknown"` exists to say. The
same split applies when `stansummary` cannot run at all (no CmdStan install):
diagnostics degrade, the summary raises.

**`ess_tail` changed value, and it's a correction.** arviz's Dataset-path
`method="tail"` ESS — what this tool reported before `stansummary` became the
source — disagrees with both `stansummary` and arviz's own bare-array path at
the documented `prob=(0.05, 0.95)`; the latter two agree with each other.
Measured on `sample-0013`'s `theta`: 506.52 (arviz Dataset path) against
403.87 (`stansummary`, matching arviz's bare-array path). Across all 151
elements the disagreement is bidirectional and median 3% (reaching 1.76×),
not a systematic shift in either direction — the below-400 counts are nearly
identical, 69 against 68, even though 27 elements cross that threshold one
way or the other. The one figure that does move in a single, clear
direction is `min_ess_tail` — the number `get_run_diagnostics`' verdict
gates on — which falls from 199.3 to 148.5 on that run. arviz's Dataset path
is the odd one out here (its own bare-array path sides with CmdStan), so
sourcing from `stansummary` settles the disagreement by construction rather
than by this tool's own adjudication.

### `get_run_summary(run_id, params=None, max_rows=100) -> dict`
The primary way to read a run: statistics are what you can act on, whereas raw
draws (`extract_draws`) mostly just consume context.
- `optimize` → `{"estimate": {...}}` (the stored `optimized_params_dict`).
- `sample` → `{"summary": [...]}`, one record per parameter **element**, from
  CmdStan's own `stansummary`: `param`, `mean`, `sd`, `mad`, `q5`, `q50`,
  `q95`, `ess_bulk`, `ess_tail`, `r_hat`, `mcse`. `param` is 1-based Stan
  syntax (`theta[1]`, `L[2,1]`), not arviz's 0-based `z[0]` or a flat index.
  `q5`/`q95` is a 90% equal-tailed interval — replacing the 89% one arviz
  produced — still an ETI and deliberately not configurable: transformation
  invariant, stable from few draws, and well defined for multimodal
  posteriors, none of which holds for an HDI. `mad` is new (`stansummary`
  reports it directly); `mcse_mean`/`mcse_sd` are gone, replaced by a single
  `mcse` (`stansummary` reports one MCSE, not separate mean/sd ones). Every
  statistic is reported to six **significant figures** — not six decimal
  places, which would round a well-mixed parameter's MCSE, or any statistic of
  a parameter whose sd is ~1e-5, down to a flat `0.0`. A statistic
  `stansummary` reports as `nan` — the MCSE, ESS and R-hat of a constant
  column — comes back as `null`, since NaN is not valid JSON.
- The other draws-producing methods (`variational`, `pathfinder`, `laplace`,
  `generate_quantities`) → `{"summary": [...]}` with `mean`/`sd`/`q5`/`q50`/
  `q95`/`n` computed directly from the typed draws (no R-hat/ESS — these
  aren't MCMC chains). There is no `source` key here either: this path used to
  report `source: "csv"` to distinguish itself from an arviz-backed one, and
  with every method now reading the same typed layer there is no longer a
  fallback to flag.
- If a run's chains come back **ragged** — unequal row counts across its draws
  CSVs — every chain is truncated to the shortest before statistics are
  computed and the result carries `"chains_truncated_to": <n>`, so a caller can
  tell the summary covers fewer draws than the run nominally has. This means a
  **malformed or partially-written CSV set**, not a cancelled run: a run is only
  persisted once its runner returns, so a cancelled or killed one has no
  `draws_schema.json` at all and fails earlier and differently. Truncating is
  there so such a set yields a reported number rather than a raw numpy broadcast
  error.
- **`sample` runs fail loudly on ragged chains instead of truncating**, and that
  asymmetry is deliberate. Their statistics come from `stansummary`, which
  rejects the input itself (`Error during processing. Error: chain 2, missing or
  extra rows.`, exit 1) before any truncation could apply, so `get_run_summary`
  raises rather than quietly reporting over a shortened array. Truncation is the
  weaker guarantee, and it applies only where nothing stronger is available.
- `params` restricts to a subset by parameter **base name** (e.g. `beta`
  returns every element of `beta`) — not per-element; requesting `beta[1]`
  alone is not supported, since the sidecar and the typed arrays are both
  keyed by variable name.
- Rows are per parameter **element**, so one `vector[500]` is 500 rows. The
  result is capped at `max_rows` (0 = unlimited) and carries
  `{"truncated": {"shown", "total", "selected_by", "hint"}}` when it bites —
  name `params` for the ones you care about, or read `get_run_diagnostics`,
  which reports worst-case R-hat/ESS across every parameter in one line.
- When the cap bites, the rows kept are the **least converged** (worst `r_hat`,
  then lowest `ess_bulk`), not the first N — otherwise the tail of a big vector
  disappears unseen and truncation hides exactly what it should surface. The
  approximate-method summary has no convergence columns, so there it stays
  first-N (`selected_by: "order"`).

### `get_run_diagnostics(run_id) -> dict`
- `optimize` → `{"method": "optimize", "lp__": float}`.
- `sample` → `{"method", "max_rhat", "max_rhat_param", "min_ess_bulk",
  "min_ess_tail", "divergences", "max_treedepth", "treedepth_saturations",
  "mean_accept_stat", "min_ebfmi", "verdict": "pass"|"warn"|"unknown",
  "issues": [...]}`. `max_rhat`/`min_ess_bulk`/`min_ess_tail` come from
  CmdStan's own `stansummary` (the same output `get_run_summary` reads, so
  the two cannot disagree); `divergences`/`treedepth_saturations`/
  `mean_accept_stat` come from the typed sampler stats; `min_ebfmi` from
  `arviz.bfmi` — the one sampler diagnostic `stansummary` doesn't report
  (CmdStan has it only in `diagnose`, as prose). `verdict` is `"warn"` if
  R-hat > 1.01, bulk/tail ESS < 400, any divergences, any tree-depth
  saturations, or E-BFMI < 0.3 (see `diagnostics._RHAT_MAX` / `_ESS_MIN` /
  `_EBFMI_MIN`). Undefined or missing convergence metrics and undefined BFMI
  chains remain visible. Incomplete coverage yields `"unknown"` unless a
  measured failure already yields `"warn"`. A pass requires clean available
  checks without missing required diagnostic coverage.
- `variational`/`pathfinder`/`laplace` → method-specific scalars captured at
  persist time (`eta`, `is_resampled`, or `mode_lp` respectively), merged in
  under the same keys.
- `generate_quantities` → `{"method": "generate_quantities"}` and nothing else.
  It is the one draws-producing method with no diagnostic of its own: the HMC
  statistics need chains it does not have, and unlike the other three it has no
  method-specific scalar to report either. Its draws come from re-evaluating a
  generated-quantities block against another run, so that run's diagnostics are
  the ones to read.

### `extract_draws(run_id, param=None, thin=1, by_chain=False, params=None, max_draws=100) -> dict`
A spot-check of raw draws, not the main way to read a run — for statistics use
`get_run_summary` and for convergence `get_run_diagnostics`; there is little to
be learned from reading thousands of raw numbers that those two don't say
better. Raise `max_draws` when feeding the draws somewhere else.

Not available for `optimize` (raises `RunError`). Single-param form returns
`{"param", "draws": [...], "thin"}` (or `{"param", "by_chain": true, "draws":
[[... per chain], ...], "thin"}` with `by_chain=True`). `params` requests
several at once: `{"draws": {param: [...], ...}, "by_chain", "thin"}`. `thin`
subsamples every Nth draw (must be ≥ 1). `max_draws` caps the draws the **whole
call** returns: the ceiling is divided across the requested parameters and (with
`by_chain`) across chains, and scaled by each draw's width, so a length-3 vector
costs three times a scalar. The effective thinning step is raised as needed and
reported back in `thin`; pass `max_draws=0` for everything. Capping per chain
per parameter instead would admit `max_draws × chains × params × vector-length`
numbers — enough for a nominally capped call to overrun a tool-result limit on
its own (5000 scalar draws alone serialize to ~64 KB). One path for all six
draws-producing methods now: reads the typed draws layer (`draws.load`, the
same `draws_schema.json`-backed reader every other tool uses), so `by_chain`
works identically regardless of method. The draws are loaded once per call,
not once per requested parameter.

### `get_run_pathologies(run_id, top_k=5, permutations=2000) -> dict`
Localizes a `sample` run's divergences: `get_run_diagnostics` says a run is
pathological, this says *where*. Returns `n_draws`, `n_divergent`, and a
`ranking` of up to `top_k` parameter elements by how far the divergent
transitions sit from the rest of the draws, in units of that parameter's sd
(`shift`, plus `direction` above/below). Constant parameters are skipped.

Read `separation`, not `shift`. The raw shift has no fixed interpretation: its
noise floor moves ~10× with the divergence count (on Neal's funnel, permuted
labels give a p95 of 2.25 at 3 divergences but 0.21 at 200), so no single cutoff
works. `separation.margin` is the top parameter's shift over the runner-up's —
about 1.3 under permuted labels, about 10 when one parameter genuinely carries
the pathology — and `separation.p_value` permutation-tests exactly that,
self-calibrating to the divergence count. Deterministic: the same run always
gives the same p-value.

Below five divergences the result carries a `warning`; the ranking is often
still right (verified on the funnel at six) but has outrun its evidence. Errors
for non-`sample` methods, which have no `diverging` sampler stat. Detects
*marginal* displacement only — a pathology living purely in a pairwise
interaction, with no marginal signature, will not show up.

### `get_run_geometry(run_id, params=None, max_dims=500, min_abs_r=0.5, min_z=4.0, top_k=10) -> dict`
Posterior conditioning for a `sample` run. Catches **linear degeneracy** —
collinear predictors burning gradient evaluations along a ridge — which can
leave R-hat, ESS, divergences and E-BFMI all looking healthy.

It does **not** catch funnels. That dependence is non-linear, so a funnel's max
`|r|` is indistinguishable from noise (0.262 measured on Neal's funnel against a
`1/√ESS` noise scale of 0.236). Use `get_run_diagnostics` and
`get_run_pathologies` for those; between them ESS, R-hat, divergences and E-BFMI
flag a funnel loudly.

Computed on the **unconstrained** scale, since that is the space NUTS explores
and whose conditioning costs step size; correlating constrained draws would
report the mechanical dependence of simplex components and Cholesky factors as
findings. Each chain is **centred first**: pooling unmixed chains manufactures
correlation out of chain offsets, and you reach for this tool exactly when
mixing is doubtful.

Names in `pairs`/`eigen.direction` depend on `scale`. On `"unconstrained"` they
are BridgeStan's own dotted, 1-based unconstrained names (`z.1`, not Stan's
`z[1]`) — deliberately not relabelled into Stan syntax, since the unconstrained
space is genuinely different (a `simplex[3]` has only two unconstrained
elements) and constrained-style labels would misrepresent it. On a
`"constrained"` fallback (see below) they use the same 1-based Stan-syntax
labels (`z[1]`, `L[2,1]`) as `get_run_summary`/`extract_draws`/
`run_pathologies`.

If `scale` comes back `"constrained"`, read `scale_reason`. Runs recorded before
the `sig_figs` default (see below) cannot be unconstrained at all: Stan's
`simplex_free` validates that a simplex sums to 1 to within 1e-8, and a draw
written at CmdStan's default 6 significant figures only manages about that, so
BridgeStan rejects it. Those runs still get the full analysis, but on a scale
where some dependence is mechanical rather than a finding — re-run the model to
get the unconstrained geometry. New runs are sampled at `sig_figs=17`, where a
double round-trips exactly, so this does not arise.

Returns `scale`, `n_dims`, `min_ess`, `pairs`, `max_abs_r_cross_block`,
`within_block`, `eigen`, and a `pass`/`warn` `verdict` with `issues`.

- **Pairs** are calibrated by Fisher z against each pair's effective sample size
  (`n = min(ess_i, ess_j)`), because raw `|r|` has no fixed meaning across runs —
  its noise floor scales as `1/√ESS`, which varied 5× between two runs of the
  same model. Only pairs clearing both `min_abs_r` and `min_z` are itemized;
  `max_abs_r_cross_block` is always reported so you can see the ceiling that was
  checked. Direction lives in `r`; `z` is a magnitude.
- A pair whose ESS is too low to calibrate is reported on `|r|` alone with
  `z: null` and a `z_note`. Being unable to certify a correlation is not
  evidence against it — and a degeneracy often destroys the very ESS that would
  certify it.
- **Within-block pairs are collapsed** to one row per block (`n_pairs`,
  `max_abs_r`, `median_abs_r`). For `theta[1..200]` they are 19,900 of 20,000
  pairs, all reporting one fact, and would crowd out the cross-block pair that
  names the actual problem.
- **`eigen`** carries `condition_number`, `min_eigenvalue`,
  `noise_condition_number`, and `direction` (top loadings of the smallest
  eigenvector) — the multi-way degeneracy pairwise cannot see: if
  `x3 ≈ x1 + x2`, every pair can sit near 0.5 while the posterior is
  near-singular along `x1 + x2 − x3`. Loadings are withheld when the smallest
  eigenvalues are not separated, since the eigenvector is then arbitrary within
  its subspace.
- Read `condition_number` against `noise_condition_number`, which is what
  *independent* parameters alone would produce at this `p/n_eff`
  (Marchenko–Pastur). Some conditioning is structural, not a finding: measured on
  independent data, the condition number is 3.7 at `p/n_eff = 0.10`, 8.0 at 0.25,
  19.5 at 0.40 and 219.6 at 0.80.
- The eigen block is suppressed entirely, with a reason, when
  `p × 4 > min_ess` — i.e. `p/n_eff > 0.25`, past which noise-driven conditioning
  starts to approach the level at which a condition number is worth reporting.
  (A stricter `× 10` was tried first and suppressed the block on every real run,
  including a 151-dimension model at ESS 673, which made the only statistic that
  catches multi-way degeneracy effectively unreachable.)
- `provisional` appears when max R-hat exceeds 1.01: with chains disagreeing
  there is not yet one posterior whose geometry this describes.
- The **first call on a freshly sampled model returns `status: "preparing"`**.
  A run compiles only the CmdStan executable, so the BridgeStan artifact this
  needs does not exist yet; it builds on a background thread (never a
  `RunManager` slot) and the next call answers. Running `compile_model` before
  `start_run` avoids the wait entirely.
- `max_dims`, `min_abs_r`, `min_z` and `top_k` all accept `0` for unlimited.
  Above `max_dims` it refuses rather than silently spending minutes, since the
  matrix is `O(p²)` and its eigendecomposition `O(p³)`; scope with `params`,
  which matches either element names or a block name.

### `run_loo(run_id, var_name="log_lik") -> dict`
PSIS-LOO cross-validation via `arviz.loo`. The model must emit a generated
quantity named `var_name` (default `log_lik`). Only MCMC posterior draws and
GQ snapshots with verified MCMC provenance are accepted. Variational, Pathfinder,
and Laplace fits need target/proposal importance correction and are rejected.
Returns `{"elpd_loo", "se", "p_loo", "n_high_pareto_k", "warning"}`.

### `compare_runs(run_ids, var_name="log_lik") -> dict`
Ranks 2+ runs by PSIS-LOO via `arviz.compare` (best first). Returns
`{"comparison": [...]}`, one record per run with non-finite floats replaced by
`null` for valid JSON.

Optimize defaults to `jacobian=False`, while Laplace defaults to `jacobian=True`.
Set `method_args.jacobian` on both calls when reusing an optimized mode.
Extraction uses one common retained row stride and preserves declared axes.
A positive `max_draws` below one complete row is rejected.
Sidecar v3 verifies the complete CSV header. Legacy v2 reports `header_warning`.
Complex and tuple variables expose real component leaves and retain their type tree.
Undefined convergence metrics produce an unknown verdict unless a measured failure exists.
