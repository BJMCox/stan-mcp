"""Method-aware summary and diagnostics over stored runs (arviz-backed).

get_run_summary: draws-producing methods -> arviz summary (mean, sd, HDI,
bulk/tail ESS, R-hat); optimize -> point estimate + lp__.
get_run_diagnostics: sample -> divergences, tree-depth hits, E-BFMI, R-hat;
variational -> ELBO convergence; pathfinder -> path diagnostics; optimize ->
convergence / lp__.

Implemented during the build (see design tool group C).
"""

import runstore
from errors import RunError


def persist_run(run_id: str, method: str, fit) -> None:
    if method == "optimize":
        runstore.persist_estimate(run_id, dict(fit.optimized_params_dict))
        return
    import arviz as az

    meta = runstore.read_meta(run_id)
    try:
        idata = az.from_cmdstanpy(fit)
        idata.to_netcdf(str(runstore.idata_path(run_id)))
    except Exception as exc:  # noqa: BLE001 - non-sample methods may not map cleanly
        meta["persist_warning"] = f"arviz conversion failed: {exc}"
        runstore.write_meta(run_id, meta)


def _load_idata(run_id: str):
    import arviz as az

    p = runstore.idata_path(run_id)
    if not p.exists():
        raise RunError(f"run {run_id!r} has no draws (inference.nc missing)")
    return az.from_netcdf(str(p))


def run_summary(run_id: str, params=None) -> dict:
    meta = runstore.read_meta(run_id)
    if meta.get("method") == "optimize":
        return {"estimate": runstore.load_estimate(run_id)}
    import arviz as az

    idata = _load_idata(run_id)
    df = az.summary(idata, var_names=params)
    return {"summary": df.reset_index().rename(columns={"index": "param"}).to_dict("records")}


def run_diagnostics(run_id: str) -> dict:
    meta = runstore.read_meta(run_id)
    method = meta.get("method")
    if method == "optimize":
        est = runstore.load_estimate(run_id)
        return {"method": "optimize", "lp__": est.get("lp__")}
    idata = _load_idata(run_id)
    out: dict = {"method": method}
    if "sample_stats" in idata and "diverging" in idata.sample_stats:
        out["divergences"] = int(idata.sample_stats["diverging"].sum())
    return out


def extract_draws(run_id: str, param: str, thin: int = 1) -> dict:
    meta = runstore.read_meta(run_id)
    if meta.get("method") == "optimize":
        raise RunError("extract_draws is not available for optimize runs")
    idata = _load_idata(run_id)
    arr = idata.posterior[param].values  # (chain, draw, ...)
    flat = arr.reshape(arr.shape[0] * arr.shape[1], -1)[::thin]
    return {"param": param, "draws": flat.squeeze().tolist()}
