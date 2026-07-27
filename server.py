"""stan-mcp — MCP server for compiling, inspecting, and running Stan models.

Primary use: in-memory log-density / gradient interaction via BridgeStan.
Secondary: full inference runs (all six CmdStanPy methods) via CmdStanPy.

See docs/2026-07-14-stan-mcp-design.md for the full design. Tools are registered
here per the design's tool surface:

- A. compile / metadata      -> compile.py
- B. BridgeStan density      -> density.py   (primary)
- C. runs (all methods)      -> sample.py + runstore.py + diagnostics.py

Modules keep the logic; this file stays thin (registration + arg validation +
offload of blocking work via asyncio.to_thread).
"""

import asyncio

from mcp.server.fastmcp import FastMCP

import compile as _compile
import density
import diagnostics
import runstore
from errors import StanMcpError, as_error_dict
from sample import MANAGER

mcp = FastMCP("stan-mcp")


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return as_error_dict(exc)


@mcp.tool()
async def compile_model(
    stan: str, stanc_args: list[str] | None = None, warn_pedantic: bool = False
) -> dict:
    """Compile a Stan model (BridgeStan .so + CmdStan exe). Returns artifact paths.

    The returned `model_id` can be passed as `stan` to any other tool instead of
    re-sending the full source (saves tokens on repeated calls). Set
    `warn_pedantic` to run stanc's pedantic checks; any findings come back under
    a `warnings` key."""
    return await asyncio.to_thread(_compile.compile_model, stan, stanc_args, warn_pedantic)


@mcp.tool()
async def register_data(data) -> dict:
    """Cache a data payload and return a `data_id` usable as `data` on any tool
    instead of re-sending the full data (saves tokens on repeated calls)."""
    return await asyncio.to_thread(_safe, lambda: {"data_id": _compile.register_data(data)})


@mcp.tool()
async def validate_data(stan: str, data=None) -> dict:
    """Check that `data` satisfies the model's data block (returns param counts,
    or an error naming the missing/invalid variable)."""
    return await asyncio.to_thread(_safe, density.validate_data, stan, data)


@mcp.tool()
async def list_compiled(limit: int = 50) -> dict:
    """List cached `model_id`s and `data_id`s (most recently used first), so a
    handle can be recovered without re-sending the model source or data."""
    return await asyncio.to_thread(_safe, _compile.list_compiled, limit)


def _prune_compiled(keep, older_than_days) -> dict:
    result = _compile.prune_compiled(keep, older_than_days)
    # The in-process BridgeStan LRU would otherwise keep serving models whose
    # on-disk artifacts have just been deleted.
    density.clear_cache()
    return result


@mcp.tool()
async def prune_compiled(keep: int | None = None, older_than_days: float | None = None) -> dict:
    """Prune the recompilable compiled/ cache: keep the newest `keep` models
    and/or drop artifacts older than `older_than_days`. Returns what was removed."""
    return await asyncio.to_thread(_safe, _prune_compiled, keep, older_than_days)


@mcp.tool()
async def param_info(stan: str, data=None) -> dict:
    """Return parameter names and constrained/unconstrained counts for the model."""
    return await asyncio.to_thread(_safe, density.param_info, stan, data)


@mcp.tool()
async def log_density(
    stan: str, data, theta_unc: list[float], propto: bool = True, jacobian: bool = True
) -> dict:
    """Log density at an unconstrained parameter vector."""
    return await asyncio.to_thread(
        _safe, density.log_density, stan, data, theta_unc, propto, jacobian
    )


@mcp.tool()
async def log_density_gradient(
    stan: str, data, theta_unc: list[float], propto: bool = True, jacobian: bool = True
) -> dict:
    """Log density and gradient at an unconstrained parameter vector."""
    return await asyncio.to_thread(
        _safe, density.log_density_gradient, stan, data, theta_unc, propto, jacobian
    )


@mcp.tool()
async def log_density_hessian(
    stan: str, data, theta_unc: list[float], propto: bool = True, jacobian: bool = True
) -> dict:
    """Log density, gradient, and Hessian at an unconstrained parameter vector."""
    return await asyncio.to_thread(
        _safe, density.log_density_hessian, stan, data, theta_unc, propto, jacobian
    )


@mcp.tool()
async def param_constrain(
    stan: str, data, theta_unc: list[float], include_tp: bool = False, include_gq: bool = False
) -> dict:
    """Map unconstrained params to the constrained space (named)."""
    return await asyncio.to_thread(
        _safe, density.param_constrain, stan, data, theta_unc, include_tp, include_gq
    )


@mcp.tool()
async def param_unconstrain(stan: str, data, constrained: dict) -> dict:
    """Map named constrained params to the unconstrained vector."""
    return await asyncio.to_thread(_safe, density.param_unconstrain, stan, data, constrained)


@mcp.tool()
async def check_gradient(
    stan: str, data, theta_unc: list[float], eps: float = 1e-6, tol: float = 1e-4
) -> dict:
    """Compare autodiff gradient against central finite differences (mismatch
    threshold `tol`)."""
    return await asyncio.to_thread(_safe, density.check_gradient, stan, data, theta_unc, eps, tol)


@mcp.tool()
async def start_run(
    stan: str,
    data=None,
    method: str = "sample",
    seed: int | None = None,
    inits=None,
    method_args: dict | None = None,
    inits_from: str | None = None,
) -> dict:
    """Start a background inference run (sample/optimize/variational/pathfinder/
    laplace/generate_quantities). Returns a run_id immediately. `inits_from` warm-
    starts at another run's posterior/estimate mean (ignored if `inits` is set)."""
    return await asyncio.to_thread(
        _safe, MANAGER.start_run, stan, data, method, seed, inits, method_args, inits_from
    )


@mcp.tool()
async def get_run_status(run_id: str) -> dict:
    """Non-blocking status of a run."""
    return await asyncio.to_thread(_safe, MANAGER.get_run_status, run_id)


@mcp.tool()
async def get_run_progress(run_id: str) -> dict:
    """Run status plus elapsed wall-clock seconds."""
    return await asyncio.to_thread(_safe, MANAGER.get_run_progress, run_id)


@mcp.tool()
async def cancel_run(run_id: str) -> dict:
    """Cancel a run. Only queued (not-yet-started) runs can be stopped."""
    return await asyncio.to_thread(_safe, MANAGER.cancel_run, run_id)


@mcp.tool()
async def rerun(run_id: str) -> dict:
    """Start a fresh run reusing a prior run's persisted source, data, method,
    seed, and method_args. Returns the new run_id."""
    return await asyncio.to_thread(_safe, MANAGER.rerun, run_id)


@mcp.tool()
async def await_run(run_id: str, timeout: float | None = None) -> dict:
    """Block server-side until the run finishes (or timeout), then return its status."""
    try:
        return await MANAGER.await_run(run_id, timeout=timeout)
    except StanMcpError as exc:
        return as_error_dict(exc)


@mcp.tool()
async def list_runs(
    model_id: str | None = None,
    limit: int = 50,
    status: str | None = None,
    method: str | None = None,
) -> dict:
    """List persisted runs, newest first, optionally filtered by model_id,
    status (running/done/failed/cancelled), and/or method."""
    return {"runs": await asyncio.to_thread(runstore.list_runs, model_id, limit, status, method)}


@mcp.tool()
async def get_run_inputs(run_id: str) -> dict:
    """Return the persisted Stan source and data for a run (for reproduction)."""
    return await asyncio.to_thread(_safe, runstore.load_inputs, run_id)


@mcp.tool()
async def prune_runs(keep: int | None = None, older_than_days: float | None = None) -> dict:
    """Delete stored runs: keep only the newest `keep`, and/or drop runs older
    than `older_than_days`. Returns the run_ids removed."""
    result = await asyncio.to_thread(_safe, runstore.prune_runs, keep, older_than_days)
    return result if isinstance(result, dict) else {"removed": result}


@mcp.tool()
async def get_run_summary(
    run_id: str,
    params: list[str] | None = None,
    max_rows: int = diagnostics.DEFAULT_MAX_ROWS,
) -> dict:
    """Method-aware summary of a run. `sample` gets mean/sd/mad/quantiles plus
    bulk-tail ESS, R-hat and MCSE from CmdStan's own stansummary; the other
    draws-producing methods get mean/sd/quantiles computed from the draws, with
    no R-hat/ESS (they are not MCMC chains); optimize gets its point estimate.
    One row per parameter ELEMENT, so a big vector is many rows: capped at
    `max_rows` (0 = unlimited), with a `truncated` block reporting the true
    total. Name `params` to summarize specific parameters."""
    return await asyncio.to_thread(_safe, diagnostics.run_summary, run_id, params, max_rows)


@mcp.tool()
async def get_run_diagnostics(run_id: str) -> dict:
    """Method-aware diagnostics (divergences/R-hat for sample, etc.)."""
    return await asyncio.to_thread(_safe, diagnostics.run_diagnostics, run_id)


@mcp.tool()
async def extract_draws(
    run_id: str,
    param: str | None = None,
    thin: int = 1,
    by_chain: bool = False,
    params: list[str] | None = None,
    max_draws: int = diagnostics.DEFAULT_MAX_DRAWS,
) -> dict:
    """A spot-check sample of raw draws (draws-producing methods only). For
    statistics use `get_run_summary` and for convergence `get_run_diagnostics` —
    reading thousands of raw numbers achieves nothing that those don't. Pass
    `params` for several at once; `by_chain=True` keeps chain structure.
    `max_draws` bounds what the whole call returns (0 = unlimited), split across
    the requested parameters and chains and scaled by each draw's width; `thin`
    reports the step actually applied."""
    return await asyncio.to_thread(
        _safe, diagnostics.extract_draws, run_id, param, thin, by_chain, params, max_draws
    )


@mcp.tool()
async def get_run_pathologies(
    run_id: str, top_k: int = 5, permutations: int = diagnostics.DEFAULT_PERMUTATIONS
) -> dict:
    """Localize a sample run's divergences: which parameters do the divergent
    transitions sit away from the rest of the draws on, and in which direction.
    `get_run_diagnostics` says a run is pathological; this says where. Ranking is
    calibrated by permuting the divergence labels — see `separation.margin` and
    its `p_value`, not the raw `shift`, for how much to trust the top hit."""
    return await asyncio.to_thread(_safe, diagnostics.run_pathologies, run_id, top_k, permutations)


@mcp.tool()
async def get_run_geometry(
    run_id: str,
    params: list[str] | None = None,
    max_dims: int = diagnostics.DEFAULT_MAX_DIMS,
    min_abs_r: float = 0.5,
    min_z: float = 4.0,
    top_k: int = 10,
) -> dict:
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
        _safe, diagnostics.run_geometry, run_id, params, max_dims, min_abs_r, min_z, top_k
    )


@mcp.tool()
async def run_loo(run_id: str, var_name: str = "log_lik") -> dict:
    """PSIS-LOO cross-validation for a run (model must emit a `log_lik` GQ)."""
    return await asyncio.to_thread(_safe, diagnostics.run_loo, run_id, var_name)


@mcp.tool()
async def compare_runs(run_ids: list[str], var_name: str = "log_lik") -> dict:
    """Rank runs by PSIS-LOO (best first). Model must emit a `log_lik` GQ."""
    return await asyncio.to_thread(_safe, diagnostics.compare_runs, run_ids, var_name)


if __name__ == "__main__":
    mcp.run()
