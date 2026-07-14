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
    except StanMcpError as exc:
        return as_error_dict(exc)


@mcp.tool()
async def compile_model(stan: str, stanc_args: list[str] | None = None) -> dict:
    """Compile a Stan model (BridgeStan .so + CmdStan exe). Returns artifact paths."""
    return await asyncio.to_thread(_compile.compile_model, stan, stanc_args)


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
async def check_gradient(stan: str, data, theta_unc: list[float], eps: float = 1e-6) -> dict:
    """Compare autodiff gradient against central finite differences."""
    return await asyncio.to_thread(_safe, density.check_gradient, stan, data, theta_unc, eps)


@mcp.tool()
async def start_run(
    stan: str,
    data=None,
    method: str = "sample",
    seed: int | None = None,
    inits=None,
    method_args: dict | None = None,
) -> dict:
    """Start a background inference run (sample/optimize/variational/pathfinder/
    laplace/generate_quantities). Returns a run_id immediately."""
    return await asyncio.to_thread(
        _safe, MANAGER.start_run, stan, data, method, seed, inits, method_args
    )


@mcp.tool()
async def get_run_status(run_id: str) -> dict:
    """Non-blocking status of a run."""
    return await asyncio.to_thread(_safe, MANAGER.get_run_status, run_id)


@mcp.tool()
async def await_run(run_id: str, timeout: float | None = None) -> dict:
    """Block server-side until the run finishes (or timeout), then return its status."""
    try:
        return await MANAGER.await_run(run_id, timeout=timeout)
    except StanMcpError as exc:
        return as_error_dict(exc)


@mcp.tool()
async def list_runs(model_id: str | None = None, limit: int = 50) -> dict:
    """List persisted runs, newest first, optionally filtered by model_id."""
    return {"runs": await asyncio.to_thread(runstore.list_runs, model_id, limit)}


@mcp.tool()
async def get_run_summary(run_id: str, params: list[str] | None = None) -> dict:
    """Method-aware summary of a run (arviz summary for draws; estimate for optimize)."""
    return await asyncio.to_thread(_safe, diagnostics.run_summary, run_id, params)


@mcp.tool()
async def get_run_diagnostics(run_id: str) -> dict:
    """Method-aware diagnostics (divergences/R-hat for sample, etc.)."""
    return await asyncio.to_thread(_safe, diagnostics.run_diagnostics, run_id)


@mcp.tool()
async def extract_draws(run_id: str, param: str, thin: int = 1) -> dict:
    """Draws for one parameter (draws-producing methods only)."""
    return await asyncio.to_thread(_safe, diagnostics.extract_draws, run_id, param, thin)


if __name__ == "__main__":
    mcp.run()
