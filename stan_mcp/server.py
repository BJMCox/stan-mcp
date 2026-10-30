"""MCP tool registration and asynchronous dispatch to the Stan modules."""

import asyncio
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer

from stan_mcp import compilation as _compile
from stan_mcp import density, diagnostics, draws, geometry, runstore, serialization
from stan_mcp.batch import jobs as batchjobs
from stan_mcp.errors import StanMcpError, as_error_dict
from stan_mcp.inference import MANAGER

# Keep the JSON text response format independent of Python return annotations.
mcp = MCPServer("stan-mcp")
BATCH_MANAGER = batchjobs.BatchManager()


def _safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return serialization.json_safe(fn(*args, **kwargs))
    except Exception as exc:
        return as_error_dict(exc)


@mcp.tool(structured_output=False)
async def compile_model(
    stan: str, stanc_args: list[str] | None = None, warn_pedantic: bool = False
) -> dict[str, Any]:
    """Prefer a file input: set `stan="/absolute/path/model.stan"`.

    Use an absolute path to a file on the MCP server. Inline Stan source is
    a fallback when no file is available.

    Compile a Stan model (BridgeStan .so + CmdStan exe). Return artifact paths.

    The returned `model_id` can be passed as `stan` to any other tool instead of
    re-sending the full source (saves tokens on repeated calls). Set
    `warn_pedantic` to run stanc's pedantic checks; any findings come back under
    a `warnings` key."""
    return await asyncio.to_thread(_compile.compile_model, stan, stanc_args, warn_pedantic)


@mcp.tool(structured_output=False)
async def register_data(data: dict[str, Any] | str | None) -> dict[str, Any]:
    """Prefer a file input: set `data="/absolute/path/data.json"`.

    Use an absolute path to a file on the MCP server. Inline JSON is a fallback
    when no file is available. Cache the file contents and return a `data_id`.
    Reuse that ID as `data` on later calls to avoid resending the payload.
    The cached snapshot does not change when the original file changes."""
    return await asyncio.to_thread(_safe, lambda: {"data_id": _compile.register_data(data)})


@mcp.tool(structured_output=False)
async def validate_data(stan: str, data: dict[str, Any] | str | None = None) -> dict[str, Any]:
    """Prefer file inputs: `stan="/absolute/path/model.stan"`, `data="/absolute/path/data.json"`.

    Use absolute paths to files on the MCP server. Reuse `model_id` and `data_id`
    handles for repeated calls. Inline Stan and JSON strings are fallbacks
    when files are unavailable.

    Check that `data` satisfies the model's data block (returns param counts,
    or an error naming the missing/invalid variable)."""
    return await asyncio.to_thread(_safe, density.validate_data, stan, data)


@mcp.tool(structured_output=False)
async def list_compiled(limit: int = 50) -> dict[str, Any]:
    """List cached `model_id`s and `data_id`s (most recently used first), so a
    handle can be recovered without re-sending the model source or data."""
    return await asyncio.to_thread(_safe, _compile.list_compiled, limit)


def _prune_compiled(keep: int | None, older_than_days: float | None) -> dict[str, Any]:
    result = _compile.prune_compiled(keep, older_than_days)
    # The in-process BridgeStan LRU would otherwise keep serving models whose
    # on-disk artifacts have just been deleted.
    density.clear_cache()
    return result


@mcp.tool(structured_output=False)
async def prune_compiled(
    keep: int | None = None, older_than_days: float | None = None
) -> dict[str, Any]:
    """Prune the recompilable compiled/ cache: keep the newest `keep` models
    and/or drop artifacts older than `older_than_days`. Returns what was removed."""
    return await asyncio.to_thread(_safe, _prune_compiled, keep, older_than_days)


@mcp.tool(structured_output=False)
async def param_info(stan: str, data: dict[str, Any] | str | None = None) -> dict[str, Any]:
    """Prefer file inputs: `stan="/absolute/path/model.stan"`, `data="/absolute/path/data.json"`.

    Use absolute paths to files on the MCP server. Reuse `model_id` and `data_id`
    handles for repeated calls. Inline Stan and JSON strings are fallbacks
    when files are unavailable.

    Return parameter names and constrained/unconstrained counts for the model."""
    return await asyncio.to_thread(_safe, density.param_info, stan, data)


@mcp.tool(structured_output=False)
async def log_density(
    stan: str,
    data: dict[str, Any] | str | None,
    theta_unc: list[float] | str,
    propto: bool = True,
    jacobian: bool = True,
) -> dict[str, Any]:
    """Prefer file inputs: `stan="/absolute/path/model.stan"`, `data="/absolute/path/data.json"`.

    Use absolute paths to files on the MCP server. Reuse `model_id` and `data_id`
    handles for repeated calls. Inline Stan and JSON strings are fallbacks
    when files are unavailable. Prefer `theta_unc="/absolute/path/theta.json"`,
    containing a numeric array such as `[0.5]`. An inline numeric list is a fallback.

    Return the log density at an unconstrained parameter vector."""
    return await asyncio.to_thread(
        _safe, density.log_density, stan, data, theta_unc, propto, jacobian
    )


@mcp.tool(structured_output=False)
async def log_density_gradient(
    stan: str,
    data: dict[str, Any] | str | None,
    theta_unc: list[float] | str,
    propto: bool = True,
    jacobian: bool = True,
) -> dict[str, Any]:
    """Prefer file inputs: `stan="/absolute/path/model.stan"`, `data="/absolute/path/data.json"`.

    Use absolute paths to files on the MCP server. Reuse `model_id` and `data_id`
    handles for repeated calls. Inline Stan and JSON strings are fallbacks
    when files are unavailable. Prefer `theta_unc="/absolute/path/theta.json"`,
    containing a numeric array such as `[0.5]`. An inline numeric list is a fallback.

    Return the log density and gradient at an unconstrained parameter vector."""
    return await asyncio.to_thread(
        _safe, density.log_density_gradient, stan, data, theta_unc, propto, jacobian
    )


@mcp.tool(structured_output=False)
async def log_density_hessian(
    stan: str,
    data: dict[str, Any] | str | None,
    theta_unc: list[float] | str,
    propto: bool = True,
    jacobian: bool = True,
) -> dict[str, Any]:
    """Prefer file inputs: `stan="/absolute/path/model.stan"`, `data="/absolute/path/data.json"`.

    Use absolute paths to files on the MCP server. Reuse `model_id` and `data_id`
    handles for repeated calls. Inline Stan and JSON strings are fallbacks
    when files are unavailable. Prefer `theta_unc="/absolute/path/theta.json"`,
    containing a numeric array such as `[0.5]`. An inline numeric list is a fallback.

    Return the log density, gradient, and Hessian at an unconstrained parameter vector."""
    return await asyncio.to_thread(
        _safe, density.log_density_hessian, stan, data, theta_unc, propto, jacobian
    )


@mcp.tool(structured_output=False)
async def param_constrain(
    stan: str,
    data: dict[str, Any] | str | None,
    theta_unc: list[float] | str,
    include_tp: bool = False,
    include_gq: bool = False,
) -> dict[str, Any]:
    """Prefer file inputs: `stan="/absolute/path/model.stan"`, `data="/absolute/path/data.json"`.

    Use absolute paths to files on the MCP server. Reuse `model_id` and `data_id`
    handles for repeated calls. Inline Stan and JSON strings are fallbacks
    when files are unavailable. Prefer `theta_unc="/absolute/path/theta.json"`,
    containing a numeric array such as `[0.5]`. An inline numeric list is a fallback.

    Map unconstrained params to the constrained space (named)."""
    return await asyncio.to_thread(
        _safe, density.param_constrain, stan, data, theta_unc, include_tp, include_gq
    )


@mcp.tool(structured_output=False)
async def param_unconstrain(
    stan: str, data: dict[str, Any] | str | None, constrained: dict[str, Any] | str
) -> dict[str, Any]:
    """Prefer file inputs: `stan="/absolute/path/model.stan"`, `data="/absolute/path/data.json"`.

    Use absolute paths to files on the MCP server. Reuse `model_id` and `data_id`
    handles for repeated calls. Inline Stan and JSON strings are fallbacks
    when files are unavailable. Prefer `constrained="/absolute/path/params.json"`,
    containing a named parameter object such as `{"x": 0.5}`. An inline dict is a fallback.

    Map named constrained params to the unconstrained vector."""
    return await asyncio.to_thread(_safe, density.param_unconstrain, stan, data, constrained)


@mcp.tool(structured_output=False)
async def check_gradient(
    stan: str,
    data: dict[str, Any] | str | None,
    theta_unc: list[float] | str,
    eps: float = 1e-6,
    tol: float = 1e-4,
) -> dict[str, Any]:
    """Prefer file inputs: `stan="/absolute/path/model.stan"`, `data="/absolute/path/data.json"`.

    Use absolute paths to files on the MCP server. Reuse `model_id` and `data_id`
    handles for repeated calls. Inline Stan and JSON strings are fallbacks
    when files are unavailable. Prefer `theta_unc="/absolute/path/theta.json"`,
    containing a numeric array such as `[0.5]`. An inline numeric list is a fallback.

    Compare autodiff gradient against central finite differences (mismatch
    threshold `tol`)."""
    return await asyncio.to_thread(_safe, density.check_gradient, stan, data, theta_unc, eps, tol)


@mcp.tool(structured_output=False)
async def start_run(
    stan: str,
    data: dict[str, Any] | str | None = None,
    method: str = "sample",
    seed: int | None = None,
    inits: dict[str, Any] | list[dict[str, Any] | str] | str | float | None = None,
    method_args: dict[str, Any] | None = None,
    inits_from: str | None = None,
) -> dict[str, Any]:
    """Prefer file inputs: `stan="/absolute/path/model.stan"`, `data="/absolute/path/data.json"`.

    Use absolute paths to files on the MCP server. Reuse `model_id` and `data_id`
    handles for repeated calls. Inline Stan and JSON strings are fallbacks
    when files are unavailable. Model source and data contents are saved with
    the run, so later file changes do not alter its stored inputs.

    Prefer `inits="/absolute/path/inits.json"` when providing initial values.
    Per-chain JSON paths are supported. Inline objects and numeric initialization
    radii remain available. Initializer file contents are saved with the run.

    For `generate_quantities`, prefer `method_args={"csv_files": ["/absolute/path/chain.csv"]}`
    for external CmdStan fitted parameters. A single CSV path also works.
    Supply sample, optimize, or variational CSVs in chain order, with their
    original CmdStan comment headers. Optimize and variational accept one CSV
    each. Input CSVs are copied into the run for reproduction.
    Warmup rows, saved optimization iterations, and a variational
    mean row are excluded from stored draw summaries. Alternatively, use
    `method_args={"from_run_id": "sample-0001"}` for a managed run. Choose one source.

    Start a background inference run (sample/optimize/variational/pathfinder/
    laplace/generate_quantities). Returns a run_id immediately. `inits_from` warm-
    starts at a shaped estimate or verified posterior mean, using a stored draw
    when the mean violates constraints (ignored if `inits` is set).
    Optimize defaults to `jacobian=False` and Laplace to `jacobian=True`.
    Set `method_args.jacobian` explicitly to align a separately optimized mode."""
    return await asyncio.to_thread(
        _safe, MANAGER.start_run, stan, data, method, seed, inits, method_args, inits_from
    )


@mcp.tool(structured_output=False)
async def evaluate_batch(request_file: str) -> dict[str, Any]:
    """Use an absolute server-local JSON request path; keep numerical arrays in Safetensors files.

    The request names `stan` (.stan path preferred), `data` (.json path preferred),
    a `points` manifest path, and optional `quantities`, target flags, seed and workers.
    Evaluate many points in one model/data context. Outputs include log_density,
    gradient, Hessian and HVP. Native derivatives use Stan unconstrained coordinates.
    Curvature uses an autodiff BridgeStan build, never implicit finite differences.
    Returns a background job_id/run_id. Use get_run_progress, await_run, cancel_run,
    get_batch_result and resume_batch. See docs/batches.md for the file schema."""
    return await asyncio.to_thread(_safe, BATCH_MANAGER.start, "evaluate", request_file)


@mcp.tool(structured_output=False)
async def compare_batches(request_file: str) -> dict[str, Any]:
    """Use an absolute server-local JSON request path with `left` and `right` result manifests.

    Compare supplied Safetensors files; this tool does not execute external runtimes.
    Match constrained points and explicit target flags. Compare native derivatives
    only with matching declared coordinates, names/order and HVP directions.
    Explicit per-quantity atol/rtol enables pass/fail; otherwise report errors only.
    Optional constant-offset and caller-supplied finite-difference stencils require
    explicit settings. Unstable finite differences are inconclusive.
    Returns a background job_id/run_id. Use get_batch_result for a bounded summary
    and discrepancy manifest. See docs/batches.md for request and artifact schemas."""
    return await asyncio.to_thread(_safe, BATCH_MANAGER.start, "compare", request_file)


@mcp.tool(structured_output=False)
async def get_batch_result(job_id: str) -> dict[str, Any]:
    """Return a bounded batch summary and absolute output manifest path.

    Numerical arrays stay in Safetensors shards. A completed comparison job can
    report failed or inconclusive checks; job completion is not numerical agreement."""
    return await asyncio.to_thread(_safe, BATCH_MANAGER.result, job_id)


@mcp.tool(structured_output=False)
async def resume_batch(job_id: str) -> dict[str, Any]:
    """Resume unfinished shards of an interrupted batch using its owned file snapshots.

    Verify inputs, settings, seed, build identity and completed output hashes.
    Preserve completed shard bytes. Changed computation requires a new job.
    A job interrupted before preparation finished cannot resume."""
    return await asyncio.to_thread(_safe, BATCH_MANAGER.resume, job_id)


def _batch_job(run_id: str) -> bool:
    return runstore.read_meta(run_id).get("method") in {"evaluate_batch", "compare_batches"}


def _run_status(run_id: str) -> dict[str, Any]:
    return BATCH_MANAGER.status(run_id) if _batch_job(run_id) else MANAGER.get_run_status(run_id)


def _run_progress(run_id: str) -> dict[str, Any]:
    return (
        BATCH_MANAGER.progress(run_id) if _batch_job(run_id) else MANAGER.get_run_progress(run_id)
    )


def _cancel_run(run_id: str) -> dict[str, Any]:
    return BATCH_MANAGER.cancel(run_id) if _batch_job(run_id) else MANAGER.cancel_run(run_id)


@mcp.tool(structured_output=False)
async def get_run_status(run_id: str) -> dict[str, Any]:
    """Non-blocking status of an inference run or batch job."""
    return await asyncio.to_thread(_safe, _run_status, run_id)


@mcp.tool(structured_output=False)
async def get_run_progress(run_id: str) -> dict[str, Any]:
    """Inference status and elapsed time, or batch status and completed row/shard counts."""
    return await asyncio.to_thread(_safe, _run_progress, run_id)


@mcp.tool(structured_output=False)
async def cancel_run(run_id: str) -> dict[str, Any]:
    """Cancel owned inference processes or interrupt a batch, preserving completed batch shards.

    Batch compilation finishes before cancellation takes effect; active numerical
    workers stop immediately."""
    return await asyncio.to_thread(_safe, _cancel_run, run_id)


@mcp.tool(structured_output=False)
async def rerun(run_id: str) -> dict[str, Any]:
    """Start a fresh run reusing a prior run's persisted source, data, method,
    seed, and method_args. Returns the new run_id."""
    return await asyncio.to_thread(_safe, MANAGER.rerun, run_id)


@mcp.tool(structured_output=False)
async def await_run(run_id: str, timeout: float | None = None) -> dict[str, Any]:
    """Block server-side until the run finishes (or timeout), then return its status."""
    try:
        if await asyncio.to_thread(_batch_job, run_id):
            return await BATCH_MANAGER.await_job(run_id, timeout=timeout)
        return await MANAGER.await_run(run_id, timeout=timeout)
    except StanMcpError as exc:
        return as_error_dict(exc)


@mcp.tool(structured_output=False)
async def list_runs(
    model_id: str | None = None,
    limit: int = 50,
    status: str | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    """List persisted runs, newest first, optionally filtered by model_id,
    status (running/done/failed/cancelled), and/or method."""
    return {"runs": await asyncio.to_thread(runstore.list_runs, model_id, limit, status, method)}


@mcp.tool(structured_output=False)
async def get_run_inputs(run_id: str) -> dict[str, Any]:
    """Return the persisted Stan source and data for a run (for reproduction)."""
    return await asyncio.to_thread(_safe, runstore.load_inputs, run_id)


@mcp.tool(structured_output=False)
async def prune_runs(
    keep: int | None = None, older_than_days: float | None = None
) -> dict[str, Any]:
    """Delete stored runs: keep only the newest `keep`, and/or drop runs older
    than `older_than_days`. Returns the run_ids removed."""
    result = await asyncio.to_thread(_safe, runstore.prune_runs, keep, older_than_days)
    return result if isinstance(result, dict) else {"removed": result}


@mcp.tool(structured_output=False)
async def get_run_summary(
    run_id: str,
    params: list[str] | None = None,
    max_rows: int = diagnostics.DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """Method-aware summary of a run. `sample` gets mean/sd/mad/quantiles plus
    bulk-tail ESS, R-hat and MCSE from CmdStan's own stansummary; the other
    draws-producing methods get mean/sd/quantiles computed from the draws, with
    no R-hat/ESS (they are not MCMC chains); optimize gets its point estimate.
    One row per parameter ELEMENT, so a big vector is many rows: capped at
    `max_rows` (0 = unlimited), with a `truncated` block reporting the true
    total. Name `params` to summarize specific parameters."""
    return await asyncio.to_thread(_safe, diagnostics.run_summary, run_id, params, max_rows)


@mcp.tool(structured_output=False)
async def get_run_diagnostics(run_id: str) -> dict[str, Any]:
    """Method-aware diagnostics (divergences/R-hat for sample, etc.)."""
    return await asyncio.to_thread(_safe, diagnostics.run_diagnostics, run_id)


@mcp.tool(structured_output=False)
async def extract_draws(
    run_id: str,
    param: str | None = None,
    thin: int = 1,
    by_chain: bool = False,
    params: list[str] | None = None,
    max_draws: int = draws.DEFAULT_MAX_DRAWS,
) -> dict[str, Any]:
    """A spot-check sample of raw draws (draws-producing methods only). For
    statistics use `get_run_summary` and for convergence `get_run_diagnostics` —
    reading thousands of raw numbers achieves nothing that those don't. Pass
    `params` for several at once; `by_chain=True` keeps chain structure.
    `max_draws` bounds what the whole call returns (0 = unlimited), split across
    the requested parameters and chains and scaled by each draw's width; `thin`
    reports the step actually applied."""
    return await asyncio.to_thread(
        _safe, draws.extract_draws, run_id, param, thin, by_chain, params, max_draws
    )


@mcp.tool(structured_output=False)
async def get_run_pathologies(
    run_id: str, top_k: int = 5, permutations: int = diagnostics.DEFAULT_PERMUTATIONS
) -> dict[str, Any]:
    """Localize a sample run's divergences: which parameters do the divergent
    transitions sit away from the rest of the draws on, and in which direction.
    `get_run_diagnostics` says a run is pathological; this says where. Ranking is
    calibrated by permuting the divergence labels — see `separation.margin` and
    its `p_value`, not the raw `shift`, for how much to trust the top hit."""
    return await asyncio.to_thread(_safe, diagnostics.run_pathologies, run_id, top_k, permutations)


@mcp.tool(structured_output=False)
async def get_run_geometry(
    run_id: str,
    params: list[str] | None = None,
    max_dims: int = geometry.DEFAULT_MAX_DIMS,
    min_abs_r: float = 0.5,
    min_z: float = 4.0,
    top_k: int = 10,
) -> dict[str, Any]:
    """Posterior conditioning for a sample run: correlated cross-block parameter
    pairs and the worst-conditioned direction, computed on the unconstrained
    scale the sampler actually explores, with each chain centred first.

    Catches linear degeneracy — collinear predictors wasting gradient
    evaluations along a ridge — which can leave R-hat, ESS, divergences and
    E-BFMI all looking healthy. It does NOT catch funnels: that dependence is
    non-linear, so use `get_run_pathologies` and `get_run_diagnostics` for those.

    Pairs are calibrated by Fisher z against each pair's effective sample size,
    since raw |r| has no fixed meaning across runs (its noise floor scales as
    1/sqrt(ESS)). Only pairs clearing both `min_abs_r` and `min_z` are itemized;
    thresholds and `max_dims` accept 0 for unlimited. The first call on a freshly
    sampled model returns `status: "preparing"` while the BridgeStan artifact
    builds in the background — retry shortly."""
    return await asyncio.to_thread(
        _safe, geometry.run_geometry, run_id, params, max_dims, min_abs_r, min_z, top_k
    )


@mcp.tool(structured_output=False)
async def run_loo(run_id: str, var_name: str = "log_lik") -> dict[str, Any]:
    """PSIS-LOO for MCMC draws, including GQ snapshots of MCMC draws.

    The model must emit a `log_lik` GQ. Approximate fits require target/proposal
    correction and are rejected. External CSVs must retain sampling metadata."""
    return await asyncio.to_thread(_safe, diagnostics.run_loo, run_id, var_name)


@mcp.tool(structured_output=False)
async def compare_runs(run_ids: list[str], var_name: str = "log_lik") -> dict[str, Any]:
    """Rank MCMC posterior runs by PSIS-LOO, best first.

    Each model must emit a `log_lik` GQ. GQ runs must retain verified MCMC
    provenance. Approximate fits require an unsupported importance correction."""
    return await asyncio.to_thread(_safe, diagnostics.compare_runs, run_ids, var_name)


def main() -> None:
    """Serve MCP over stdio and stop owned batch workers on exit."""
    try:
        mcp.run()
    finally:
        BATCH_MANAGER.shutdown()
