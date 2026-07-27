"""Method-aware summary and diagnostics over stored runs (arviz-backed).

persist_run: after a run finishes, convert its fit to persistent storage —
optimize writes an estimate.json (optimized_params_dict); draws-producing
methods first persist the raw CmdStan CSVs (source of truth; enables
generate_quantities reuse via runstore.csv_paths), best-effort and
independent of arviz, then convert via arviz.from_cmdstanpy to inference.nc,
recording a persist_warning in meta.json if the conversion fails (a
csv_warning is recorded separately if CSV persistence itself fails).
run_summary: draws-producing methods -> arviz summary (mean, sd, HDI,
bulk/tail ESS, R-hat) over inference.nc; optimize -> the stored estimate.
run_diagnostics: optimize -> lp__ from the estimate; draws-producing methods ->
worst R-hat, min bulk/tail ESS, divergences, tree-depth saturations, mean
accept stat, E-BFMI, plus a pass/warn verdict with an issues list.
run_loo / compare_runs: PSIS-LOO cross-validation and multi-run ranking (models
must emit a `log_lik` generated quantity).
extract_draws: flattened draws for one parameter from inference.nc (draws-
producing methods only; not available for optimize).
"""

from typing import Any

import numpy as np

import runstore
from errors import RunError, ValidationError


def _method_diagnostics(method: str, fit) -> dict:
    """Method-specific health scalars pulled from the fit (arviz can't convert
    VI/pathfinder/laplace fits, so these are the only diagnostics for them)."""
    try:
        if method == "variational":
            return {"eta": float(fit.eta)}
        if method == "pathfinder":
            return {"is_resampled": bool(fit.is_resampled)}
        if method == "laplace":
            return {"mode_lp": float(fit.mode.optimized_params_dict.get("lp__"))}
    except Exception:
        return {}
    return {}


def persist_run(run_id: str, method: str, fit) -> None:
    if method == "optimize":
        runstore.persist_estimate(run_id, dict(fit.optimized_params_dict))
        return

    # persist raw cmdstan CSVs (source of truth; enables generate_quantities reuse).
    # Prefer the live output dir we route every method into — robust across fit
    # types (CmdStanPathfinder / CmdStanLaplace expose no .runset).
    try:
        live = runstore.live_dir(run_id)
        csvs = sorted(str(p) for p in live.glob("*.csv")) if live.is_dir() else []
        if not csvs and hasattr(fit, "runset"):
            csvs = list(fit.runset.csv_files)
        runstore.persist_csvs(run_id, csvs)
    except Exception as exc:  # best-effort
        meta = runstore.read_meta(run_id)
        meta["csv_warning"] = f"csv persist failed: {exc}"
        runstore.write_meta(run_id, meta)

    md = _method_diagnostics(method, fit)
    if md:
        meta = runstore.read_meta(run_id)
        meta["method_diagnostics"] = md
        runstore.write_meta(run_id, meta)

    import arviz as az

    try:
        idata = az.from_cmdstanpy(fit)
        idata.to_netcdf(str(runstore.idata_path(run_id)))
    except Exception as exc:  # non-sample methods may not map cleanly
        meta = runstore.read_meta(run_id)
        meta["persist_warning"] = f"arviz conversion failed: {exc}"
        runstore.write_meta(run_id, meta)

    # Last, once nothing else needs to read the fit's CSVs: drop live/. csv/ is
    # the durable copy, so keeping both doubles every run's disk footprint.
    if runstore.csv_paths(run_id):
        runstore.clear_live(run_id)


def _load_idata(run_id: str):
    import arviz as az

    p = runstore.idata_path(run_id)
    if not p.exists():
        warning = runstore.read_meta(run_id).get("persist_warning")
        detail = f" ({warning})" if warning else ""
        raise RunError(f"run {run_id!r} has no draws (inference.nc missing){detail}")
    return az.from_netcdf(str(p))


def _coerce_numbers(rec: dict) -> dict:
    """arviz.summary formats numbers as strings; return machine-usable floats."""
    out: dict = {}
    for k, v in rec.items():
        if k == "param":
            out[k] = v
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def _param_columns(columns, params):
    """CSV columns that are model params (not `__` diagnostics), optionally
    filtered to a requested set (matching base name for indexed params)."""
    cols = [c for c in columns if not c.endswith("__")]
    if params:
        want = set(params)
        cols = [c for c in cols if c in want or c.split(".")[0] in want]
    return cols


# Default ceiling on summary rows. arviz emits one row per parameter ELEMENT, so
# a single vector[500] is 500 rows x 9 fields. The summary is the tool you
# actually reach for — raw draws are near-useless to read — so it is the one that
# most needs to stay small. Name `params` for specifics, or pass max_rows=0.
DEFAULT_MAX_ROWS = 100


def _worst_first(records: list[dict]):
    """Sort key putting the least-converged rows first, or None if the run has
    no convergence columns (the CSV fallback for VI/pathfinder/laplace)."""
    if not records or not ({"r_hat", "ess_bulk"} & set(records[0])):
        return None

    def key(r):
        rhat = _finite(r.get("r_hat"))
        ess = _finite(r.get("ess_bulk"))
        return (-(rhat if rhat is not None else 0.0), ess if ess is not None else float("inf"))

    return key


def _cap_rows(records: list[dict], max_rows: int) -> dict:
    if max_rows <= 0 or len(records) <= max_rows:
        return {"summary": records}
    # Keeping the FIRST n rows is arbitrary — it drops the tail of a big vector
    # unseen. The rows worth keeping are the ones you would act on, so rank by
    # worst convergence where those columns exist.
    key = _worst_first(records)
    return {
        "summary": sorted(records, key=key)[:max_rows] if key else records[:max_rows],
        "truncated": {
            "shown": max_rows,
            "total": len(records),
            "selected_by": "worst r_hat, then lowest ess_bulk" if key else "order",
            "hint": "name parameters via `params`, or pass max_rows=0; "
            "get_run_diagnostics reports worst-case R-hat/ESS over all of them",
        },
    }


def _summary_from_csv(run_id: str, params, max_rows: int) -> dict:
    df = _draws_df(run_id)
    cols = _param_columns(df.columns, params)
    recs = []
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        recs.append(
            {
                "param": c,
                "mean": round(float(v.mean()), 6),
                "sd": round(float(v.std(ddof=1)), 6) if v.size > 1 else 0.0,
                "q5": round(float(np.quantile(v, 0.05)), 6),
                "q50": round(float(np.quantile(v, 0.5)), 6),
                "q95": round(float(np.quantile(v, 0.95)), 6),
                "n": int(v.size),
            }
        )
    # No R-hat/ESS: approximate methods are not MCMC chains.
    return {**_cap_rows(recs, max_rows), "source": "csv"}


def run_summary(run_id: str, params=None, max_rows: int = DEFAULT_MAX_ROWS) -> dict:
    meta = runstore.read_meta(run_id)
    if meta.get("method") == "optimize":
        return {"estimate": runstore.load_estimate(run_id)}
    import arviz as az

    try:
        idata = _load_idata(run_id)
    except RunError:
        # variational / pathfinder / laplace have no inference.nc; summarize the CSVs.
        return _summary_from_csv(run_id, params, max_rows)
    # Intervals stay arviz's default 89% ETI (eti89_lb/eti89_ub), deliberately:
    # quantiles are transformation invariant (exp(ETI(log x)) == ETI(x)), stable
    # from few draws, and well defined when the posterior is multimodal. An HDI
    # is none of those, and its parameterization dependence is disqualifying for
    # a harness that sees the same parameter constrained and unconstrained.
    # Do NOT reach for az.summary(ci_kind="hdi") if this is ever revisited: as of
    # arviz 1.2 it renames the columns to hdi89_* but returns the ETI quantiles
    # unchanged (verified against az.hdi on Exponential(1): summary gives
    # [0.0557, 2.855], the true HDI is [0.0002, 2.215]). az.hdi is the real one.
    df = az.summary(idata, var_names=params)
    records = df.reset_index().rename(columns={"index": "param"}).to_dict("records")
    return _cap_rows([_coerce_numbers(r) for r in records], max_rows)


# Standard Stan-workflow thresholds for the pass/warn verdict.
_RHAT_MAX = 1.01
_ESS_MIN = 400
_EBFMI_MIN = 0.3


def _finite(x):
    import math

    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _convergence(idata) -> dict:
    """Worst R-hat and smallest bulk/tail ESS across all posterior params."""
    import arviz as az

    try:
        summ: Any = az.summary(idata, kind="diagnostics")
    except Exception:  # single chain / degenerate posterior
        return {}
    out: dict = {}
    if "r_hat" in summ.columns:
        s = summ["r_hat"].dropna()
        if len(s):
            out["max_rhat"] = round(float(s.max()), 4)
            out["max_rhat_param"] = str(s.idxmax())
    for col, key in (("ess_bulk", "min_ess_bulk"), ("ess_tail", "min_ess_tail")):
        if col in summ.columns:
            s = summ[col].dropna()
            if len(s):
                out[key] = round(float(s.min()), 1)
    return out


def _sampler_stats(idata, meta) -> dict:
    import arviz as az

    ss = idata.sample_stats
    out: dict = {}

    def _pick(*names):
        for n in names:
            if n in ss:
                return ss[n]
        return None

    div = _pick("diverging", "divergent")
    if div is not None:
        out["divergences"] = int(div.values.sum())

    td = _pick("tree_depth", "treedepth")
    if td is not None:
        max_td = int(meta.get("method_args", {}).get("max_treedepth", 10))
        out["max_treedepth"] = max_td
        out["treedepth_saturations"] = int((td.values >= max_td).sum())

    accept = _pick("acceptance_rate", "accept_stat", "mean_accept_stat")
    if accept is not None:
        out["mean_accept_stat"] = round(float(accept.values.mean()), 4)

    energy = _pick("energy")
    if energy is not None:
        # Pass the energy DataArray, NOT the whole idata: under arviz 1.2 that is
        # a DataTree, and az.bfmi(tree) raises TypeError every time — which a bare
        # `except: pass` here turned into E-BFMI silently missing from every run,
        # for the diagnostic most specific to funnel geometry (measured on Neal's
        # funnel: min E-BFMI 0.06 against 1.01 for its non-centered twin).
        try:
            bfmi = [v for v in (_finite(b) for b in np.asarray(az.bfmi(energy))) if v is not None]
            if bfmi:
                out["min_ebfmi"] = round(min(bfmi), 4)
        except Exception as exc:  # noqa: BLE001
            # Surfaced, never swallowed: a diagnostic that quietly returns nothing
            # reads as "checked and fine".
            out["ebfmi_warning"] = f"E-BFMI unavailable: {exc}"
    return out


def _verdict(d: dict) -> dict:
    issues: list[str] = []
    if (r := d.get("max_rhat")) is not None and r > _RHAT_MAX:
        issues.append(f"R-hat {r} > {_RHAT_MAX} (worst: {d.get('max_rhat_param')})")
    for key, label in (("min_ess_bulk", "bulk"), ("min_ess_tail", "tail")):
        if (e := d.get(key)) is not None and e < _ESS_MIN:
            issues.append(f"low {label} ESS {e} < {_ESS_MIN}")
    if d.get("divergences", 0) > 0:
        issues.append(f"{d['divergences']} divergent transition(s)")
    if d.get("treedepth_saturations", 0) > 0:
        issues.append(f"{d['treedepth_saturations']} iteration(s) hit max_treedepth")
    if (b := d.get("min_ebfmi")) is not None and b < _EBFMI_MIN:
        issues.append(f"E-BFMI {b} < {_EBFMI_MIN}")
    return {"verdict": "pass" if not issues else "warn", "issues": issues}


def run_diagnostics(run_id: str) -> dict:
    meta = runstore.read_meta(run_id)
    method = meta.get("method")
    if method == "optimize":
        est = runstore.load_estimate(run_id)
        return {"method": "optimize", "lp__": est.get("lp__")}
    out: dict = {"method": method}
    # HMC-style convergence/sampler diagnostics require inference.nc, which arviz
    # only writes for sample runs; VI/pathfinder/laplace fall back to the
    # method-specific scalars captured at persist time.
    try:
        idata = _load_idata(run_id)
    except RunError:
        idata = None
    if idata is not None:
        out.update(_convergence(idata))
        if "sample_stats" in idata:
            out.update(_sampler_stats(idata, meta))
        out.update(_verdict(out))
    md = meta.get("method_diagnostics")
    if md:
        out.update(md)
    return out


def _idata_with_loglik(run_id: str, var_name: str):
    """Load a run's idata, ensuring a `log_likelihood` group exists.

    Stan models expose the pointwise log-likelihood as a generated quantity
    (conventionally `log_lik`); from_cmdstanpy files it under `posterior`, so
    lift it into the `log_likelihood` group that arviz.loo/compare expect.
    """
    import xarray as xr

    idata = _load_idata(run_id)
    if "log_likelihood" in idata.children:
        return idata
    if var_name not in idata.posterior:
        raise RunError(
            f"run {run_id!r} has no log-likelihood: model must emit a generated "
            f"quantity named {var_name!r}"
        )
    idata["log_likelihood"] = xr.Dataset({var_name: idata.posterior[var_name]})
    return idata


def run_loo(run_id: str, var_name: str = "log_lik") -> dict:
    """PSIS-LOO cross-validation for a run (needs a `log_lik` generated quantity)."""
    import arviz as az

    meta = runstore.read_meta(run_id)
    if meta.get("method") == "optimize":
        raise RunError("run_loo is not available for optimize runs")
    idata = _idata_with_loglik(run_id, var_name)
    loo: Any = az.loo(idata, var_name=var_name, pointwise=True)
    # arviz >=1.2 renamed elpd_loo/p_loo -> elpd/p; support both.
    elpd = getattr(loo, "elpd", None)
    elpd = loo.elpd_loo if elpd is None else elpd
    p = getattr(loo, "p", None)
    p = getattr(loo, "p_loo", None) if p is None else p
    khat = np.asarray(getattr(loo, "pareto_k", []), dtype=float)
    good_k = float(getattr(loo, "good_k", 0.7) or 0.7)
    return {
        "elpd_loo": round(float(elpd), 3),
        "se": round(float(loo.se), 3),
        "p_loo": round(float(p), 3) if p is not None else None,
        "n_high_pareto_k": int((khat > good_k).sum()) if khat.size else 0,
        "warning": bool(getattr(loo, "warning", False)),
    }


def compare_runs(run_ids: list[str], var_name: str = "log_lik") -> dict:
    """Rank runs by PSIS-LOO (arviz.compare). Best model first."""
    import arviz as az

    if len(run_ids) < 2:
        raise ValidationError("compare_runs needs at least two run_ids")
    dataset = {rid: _idata_with_loglik(rid, var_name) for rid in run_ids}
    df: Any = az.compare(dataset)  # PSIS-LOO by default
    records = df.reset_index().rename(columns={"index": "run_id"}).to_dict("records")
    return {"comparison": [_json_safe(r) for r in records]}


def _json_safe(record: dict) -> dict:
    """Replace non-finite floats (NaN/inf) with None so the result is valid JSON."""
    import math

    out = {}
    for k, v in record.items():
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


# Default ceiling on the draws one extract_draws call returns. Deliberately
# small: draws come back through the MCP transport into a model's context, and
# there is nothing useful to do by reading thousands of raw numbers — that is
# what run_summary (statistics) and run_diagnostics (convergence) are for. The
# default is a spot-check of the posterior's location and spread; callers
# feeding the draws somewhere else raise it or pass max_draws=0.
#
# The ceiling bounds the WHOLE call: split across requested parameters and
# chains and scaled by each draw's width, since capping per chain per parameter
# really admits max_draws x chains x params x vector-length numbers.
DEFAULT_MAX_DRAWS = 100


def _chain_frames(run_id: str):
    """Per-chain draw frames from a run's raw CmdStan CSVs (one file per chain,
    excluding the laplace optimization CSV)."""
    import pandas as pd

    paths = [p for p in runstore.csv_paths(run_id) if "-opt-" not in p]
    if not paths:
        raise RunError(f"run {run_id!r} has no persisted draws (no CSVs)")
    return [pd.read_csv(p, comment="#") for p in paths]


def _draws_df(run_id: str):
    """Concatenated draws from a run's raw CmdStan CSVs (excludes the laplace
    optimization CSV). Fallback source for methods with no inference.nc."""
    import pandas as pd

    return pd.concat(_chain_frames(run_id), ignore_index=True)


def _thin_step(n: int, thin: int, budget: int) -> int:
    """Thinning step that also honours a ceiling of `budget` draws (0 = none)."""
    if budget > 0 and len(range(0, n, thin)) > budget:
        return max(thin, -(-n // budget))  # ceil division
    return thin


def _draw_budget(max_draws: int, width: int, n_chains: int = 1) -> int:
    """Draws to return per chain, so the VALUES the call returns stay near the
    ceiling. `width` is the number of scalar components in one draw of the
    parameter (1 for a scalar, N for a length-N vector)."""
    if max_draws <= 0:
        return 0  # unlimited
    return max(1, max_draws // max(1, width) // max(1, n_chains))


def _shape_draws(arr: np.ndarray) -> list:
    """Stable shape regardless of draw count: scalar param -> list of floats,
    multidimensional param -> list of per-draw lists. Never collapse the draw
    axis (squeeze() would turn a single remaining draw into a scalar)."""
    return arr[:, 0].tolist() if arr.shape[1] == 1 else arr.tolist()


def _param_cols(columns, param: str) -> list[str]:
    return [c for c in columns if c == param or c.startswith(param + ".")]


def _from_csv_source(frames, run_id: str, param: str, thin: int, by_chain: bool, max_draws: int):
    cols = _param_cols(frames[0].columns, param)
    if not cols:
        raise RunError(f"parameter {param!r} not found in run {run_id!r}")
    if by_chain:
        # One CmdStan CSV per chain, so the chain structure is recoverable here
        # too — it must not be silently dropped just because there's no idata.
        per_chain = [f[cols].to_numpy(dtype=float) for f in frames]
        budget = _draw_budget(max_draws, len(cols), len(per_chain))
        step = _thin_step(max(len(a) for a in per_chain), thin, budget)
        return {
            "param": param,
            "by_chain": True,
            "draws": [_shape_draws(a[::step]) for a in per_chain],
            "source": "csv",
            "thin": step,
        }
    import pandas as pd

    stacked = pd.concat(frames, ignore_index=True)[cols].to_numpy(dtype=float)
    step = _thin_step(len(stacked), thin, _draw_budget(max_draws, len(cols)))
    return {
        "param": param,
        "draws": _shape_draws(stacked[::step]),
        "source": "csv",
        "thin": step,
    }


def _from_idata_source(idata, param: str, thin: int, by_chain: bool, max_draws: int):
    arr = idata.posterior[param].values  # (chain, draw, ...)
    width = int(np.prod(arr.shape[2:])) if arr.ndim > 2 else 1
    if by_chain:
        budget = _draw_budget(max_draws, width, arr.shape[0])
        step = _thin_step(arr.shape[1], thin, budget)
        per_chain = [_shape_draws(c.reshape(c.shape[0], -1)[::step]) for c in arr]
        return {"param": param, "by_chain": True, "draws": per_chain, "thin": step}
    flat = arr.reshape(arr.shape[0] * arr.shape[1], -1)
    step = _thin_step(flat.shape[0], thin, _draw_budget(max_draws, width))
    return {"param": param, "draws": _shape_draws(flat[::step]), "thin": step}


def extract_draws(
    run_id: str,
    param: str | None = None,
    thin: int = 1,
    by_chain: bool = False,
    params: list[str] | None = None,
    max_draws: int = DEFAULT_MAX_DRAWS,
) -> dict:
    if thin < 1:
        raise ValidationError(f"thin must be >= 1, got {thin}")
    if max_draws < 0:
        raise ValidationError(f"max_draws must be >= 0 (0 = unlimited), got {max_draws}")
    meta = runstore.read_meta(run_id)
    if meta.get("method") == "optimize":
        raise RunError("extract_draws is not available for optimize runs")
    wanted = list(params) if params else ([param] if param else [])
    if not wanted:
        raise ValidationError("extract_draws requires `param` or `params`")

    # Load the draws source ONCE: doing it per parameter re-opened inference.nc
    # (or re-parsed every CSV) for each name in `params`.
    from functools import partial

    # Split the ceiling across the requested parameters: it bounds the whole
    # call, so asking for N names must not return N times as much.
    per_param = max(1, max_draws // len(wanted)) if max_draws > 0 else 0
    shared = {"thin": thin, "by_chain": by_chain, "max_draws": per_param}
    try:
        extract = partial(_from_idata_source, _load_idata(run_id), **shared)
    except RunError:
        extract = partial(_from_csv_source, _chain_frames(run_id), run_id, **shared)

    if params:
        results = {p: extract(p) for p in wanted}
        return {
            "draws": {p: r["draws"] for p, r in results.items()},
            "by_chain": by_chain,
            "thin": max(r["thin"] for r in results.values()),
        }
    return extract(wanted[0])


# --- divergence localization ---------------------------------------------
#
# run_diagnostics reports the divergence COUNT, which already tells you a run is
# pathological. This tells you WHICH parameter, by how far the divergent
# transitions sit from the rest of the draws in units of that parameter's sd.
#
# Calibration is by permutation of the divergence labels rather than a fixed
# threshold, because a fixed threshold cannot work: measured on Neal's funnel,
# the noise floor of the raw shift moves ~10x with the divergence count (null
# p95 = 2.25 at 3 divergences, 0.21 at 200), so any cutoff both over-fires when
# divergences are few and goes blind when they are many. Normalizing by the
# standard error narrows that spread but leaves a heavy tail at small counts.
#
# What is stable is the RANKING and the top/second margin: ~1.3 under permuted
# labels, ~10 when one parameter genuinely carries the pathology, and correct on
# the funnel even at 6 divergences where the absolute shift sits below its own
# noise floor. So the margin is what gets a permutation p-value, which
# self-calibrates to the divergence count for free.
DEFAULT_PERMUTATIONS = 2000
_MIN_DIVERGENCES = 5
# Ceiling on permutations x divergences x parameter elements, so a wide model
# does not turn a diagnostic into a minute of resampling.
_PERMUTATION_BUDGET = 50_000_000


def _draw_matrix(idata) -> tuple[np.ndarray, list[str]]:
    """(draws, elements) matrix over every posterior parameter element, and the
    matching element names."""
    cols, names = [], []
    for name in idata.posterior.data_vars:
        arr = np.asarray(idata.posterior[name].values)
        flat = arr.reshape(arr.shape[0] * arr.shape[1], -1)
        for j in range(flat.shape[1]):
            cols.append(flat[:, j])
            names.append(str(name) if flat.shape[1] == 1 else f"{name}[{j}]")
    return np.column_stack(cols), names


def _shift_vector(Xz: np.ndarray, idx: np.ndarray, n_div: int, n: int) -> np.ndarray:
    """mean(divergent) - mean(rest), per column, in sd units.

    Xz columns are mean-centred, so the two group sums are negatives of each
    other and only the divergent rows need touching — the difference is
    sum_div * (1/n_div + 1/n_rest). That keeps a permutation pass O(n_div) per
    column instead of O(n_draws)."""
    return Xz[idx].sum(0) * (1.0 / n_div + 1.0 / (n - n_div))


def _margin_p_value(Xz: np.ndarray, n_div: int, observed: float, permutations: int) -> dict:
    """Fraction of random divergence labellings whose top/second margin is at
    least as extreme as the observed one."""
    n, n_params = Xz.shape
    budget = max(200, _PERMUTATION_BUDGET // max(1, n_div * n_params))
    perms = int(min(permutations, budget))
    rng = np.random.default_rng(0)  # deterministic: same run -> same p-value
    hits = 0
    for _ in range(perms):
        idx = rng.choice(n, n_div, replace=False)
        s = np.sort(np.abs(_shift_vector(Xz, idx, n_div, n)))[::-1]
        m = s[0] / s[1] if s[1] > 0 else np.inf
        if m >= observed:
            hits += 1
    # Add-one estimator: a permutation test can bound a p-value, never show it
    # to be zero, and reporting 0.0 would overstate what 2000 draws can support.
    return {"p_value": round((hits + 1) / (perms + 1), 5), "permutations": perms}


def run_pathologies(run_id: str, top_k: int = 5, permutations: int = DEFAULT_PERMUTATIONS) -> dict:
    """Rank parameters by how far a run's divergent transitions sit from the
    rest of its draws — i.e. where in parameter space the geometry breaks."""
    if top_k < 1:
        raise ValidationError(f"top_k must be >= 1, got {top_k}")
    meta = runstore.read_meta(run_id)
    method = meta.get("method")
    if method != "sample":
        raise RunError(
            f"divergence localization needs HMC sampler stats; run {run_id!r} used {method!r}"
        )
    idata = _load_idata(run_id)
    if "sample_stats" not in idata or "diverging" not in idata.sample_stats:
        raise RunError(f"run {run_id!r} has no `diverging` sampler stat to localize")

    div = np.asarray(idata.sample_stats["diverging"].values).reshape(-1).astype(bool)
    n, n_div = int(div.size), int(div.sum())
    out: dict = {"run_id": run_id, "n_draws": n, "n_divergent": n_div, "ranking": []}
    if n_div == 0:
        return {**out, "note": "no divergent transitions to localize"}
    if n_div == n:
        return {**out, "note": "every draw diverged; no non-divergent draws to compare against"}

    X, names = _draw_matrix(idata)
    sd = X.std(0)
    keep = np.isfinite(sd) & (sd > 0)  # constants carry no location information
    if not keep.any():
        return {**out, "note": "no varying parameters to compare"}
    kept = [nm for nm, k in zip(names, keep, strict=True) if k]
    Xz = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]

    shift = _shift_vector(Xz, np.flatnonzero(div), n_div, n)
    order = np.argsort(np.abs(shift))[::-1]
    out["ranking"] = [
        {
            "param": kept[i],
            "shift": round(float(shift[i]), 4),
            "direction": "below" if shift[i] < 0 else "above",
        }
        for i in order[: min(top_k, len(kept))]
    ]

    ranked = np.sort(np.abs(shift))[::-1]
    if ranked.size > 1 and ranked[1] > 0:
        margin = float(ranked[0] / ranked[1])
        out["separation"] = {
            "margin": round(margin, 2),
            **_margin_p_value(Xz, n_div, margin, permutations),
            "reading": "ratio of the top parameter's shift to the runner-up; "
            "~1.3 under permuted divergence labels, ~10 when one parameter "
            "genuinely carries the pathology",
        }
    if n_div < _MIN_DIVERGENCES:
        out["warning"] = (
            f"only {n_div} divergent transition(s): the ranking is indicative at best. "
            "Re-run with a lower adapt_delta to surface more of them."
        )
    return out


def posterior_mean_inits(run_id: str) -> dict:
    """Posterior (or estimate) mean per parameter, shaped as CmdStan inits, for
    warm-starting another run. Scalars and 1-D vectors are reconstructed;
    higher-dimensional params are omitted (Stan default-inits those)."""
    meta = runstore.read_meta(run_id)
    if meta.get("method") == "optimize":
        return {k: v for k, v in runstore.load_estimate(run_id).items() if k != "lp__"}
    df = _draws_df(run_id)
    scalars: dict = {}
    vectors: dict[str, dict[int, float]] = {}
    for c in df.columns:
        if c.endswith("__"):
            continue
        parts = c.split(".")
        mean = float(df[c].to_numpy(dtype=float).mean())
        if len(parts) == 1:
            scalars[parts[0]] = mean
        elif len(parts) == 2:  # 1-D vector element base.i
            vectors.setdefault(parts[0], {})[int(parts[1])] = mean
        # higher-dimensional params (base.i.j...) are left to Stan's default init
    inits = dict(scalars)
    for base, idx_to_mean in vectors.items():
        inits[base] = [idx_to_mean[i] for i in sorted(idx_to_mean)]
    return inits
