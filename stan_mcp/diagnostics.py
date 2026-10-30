"""Method-aware summaries, sampler diagnostics, divergence localization and LOO."""

from collections.abc import Callable
from typing import Any

import numpy as np

from stan_mcp import draws, runstore, stansummary
from stan_mcp.errors import RunError, ValidationError
from stan_mcp.serialization import json_safe

# Default ceiling on summary rows. The summary emits one row per parameter
# ELEMENT, so a single vector[500] is 500 rows x 9 fields. It is the tool you
# actually reach for — raw draws are near-useless to read — so it is the one that
# most needs to stay small. Name `params` for specifics, or pass max_rows=0.
DEFAULT_MAX_ROWS = 100


def _worst_first(
    records: list[dict[str, Any]],
) -> Callable[[dict[str, Any]], tuple[float, float]] | None:
    """Sort key putting the least-converged rows first, or None if the run has no
    convergence columns — the approximate methods' numpy summary, which reports
    no R-hat or ESS because their draws are not MCMC chains."""
    if not records or not ({"r_hat", "ess_bulk"} & set(records[0])):
        return None

    def key(r: dict[str, Any]) -> tuple[float, float]:
        rhat = _finite(r.get("r_hat"))
        ess = _finite(r.get("ess_bulk"))
        return (-(rhat if rhat is not None else 0.0), ess if ess is not None else float("inf"))

    return key


def _cap_rows(records: list[dict[str, Any]], max_rows: int) -> dict[str, Any]:
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


# stansummary column -> our output key. `MAD` is new; `mcse_sd` is gone because
# stansummary reports a single MCSE. `q5`/`q95` is a 90% equal-tailed interval
# replacing the 89% one arviz produced -- still an ETI, and it matches the naming
# the CSV path already used.
_SUMMARY_COLUMNS = {
    "Mean": "mean",
    "StdDev": "sd",
    "MAD": "mad",
    "5%": "q5",
    "50%": "q50",
    "95%": "q95",
    "ESS_bulk": "ess_bulk",
    "ESS_tail": "ess_tail",
    "R_hat": "r_hat",
    "MCSE": "mcse",
}


def _wanted(name: str, params: list[str] | None) -> bool:
    return not params or any(
        name == root or name.startswith((root + ".", root + ":")) for root in params
    )


def _sig6(x: float) -> float:
    """Six SIGNIFICANT figures, for a statistic of unknown magnitude.

    NOT `round(x, 6)`, which rounds decimal PLACES and so partly undoes
    `stansummary.SIG_FIGS = 18`: it collapses any small-magnitude statistic to
    0.0 outright — a well-mixed parameter's MCSE (`sd/sqrt(ESS)`), or the
    mean/sd/quantiles of a parameter whose sd is ~1e-5. `%g` keeps a row the same
    size on screen without that cliff.
    """
    return float(f"{float(x):.6g}")


def _no_match(
    run_id: str, n_elements: int, looked_up: list[str], stats: dict[str, Any]
) -> RunError:
    """The error for a join that matched nothing at all.

    Both consumers of stansummary's rows `continue` past an element with no row,
    which is right for a PARTIAL match. Wholesale it is not: if the join ever
    missed everything — a naming change on either side of it — `run_summary` would
    return `{"summary": []}` as a SUCCESS, and `run_diagnostics` would report no
    R-hat with no reason given. Same class of guard `stansummary.parse` already
    applies one layer up to a header with no data rows.
    """
    return RunError(
        f"run {run_id!r}: none of its {n_elements} parameter element(s) matched a "
        f"stansummary row. Looked up {', '.join(looked_up[:5])}; stansummary "
        f"reported {', '.join(sorted(stats)[:5])}. The sidecar and stansummary "
        "disagree on element naming, so the result would be silently empty."
    )


def _summary_from_stansummary(
    run_id: str, params: list[str] | None, max_rows: int
) -> dict[str, Any]:
    """Statistics from CmdStan's own implementation, relabelled into Stan syntax.

    Only the sidecar is read here, not the draws: stansummary has already parsed
    the CSVs, so loading them again to learn names and dims would be wasted work.
    """
    sc = draws.schema(run_id)
    stats = stansummary.summarize(draws.csv_files(run_id))
    records = []
    looked_up: list[str] = []
    for name, spec in sc["params"].items():
        if not _wanted(name, params):
            continue
        for stan_label, cmdstan_name in draws.spec_element_labels(name, spec):
            looked_up.append(cmdstan_name)
            row = stats.get(cmdstan_name)
            if row is None:
                continue
            rec: dict[str, Any] = {"param": stan_label}
            for src, dst in _SUMMARY_COLUMNS.items():
                if src in row:
                    rec[dst] = _sig6(row[src])
            # stansummary reports `nan` for the MCSE/ESS/R_hat of a CONSTANT
            # column (verified: `"fixed",2.5,nan,0,0,...,nan,nan,nan,nan`), and
            # NaN is not valid JSON. A constant parameter is common enough that
            # this is a normal path, not an edge case.
            records.append(json_safe(rec))
    if looked_up and not records:
        raise _no_match(run_id, len(looked_up), looked_up, stats)
    out = _cap_rows(records, max_rows)
    if sc.get("header_warning"):
        out["header_warning"] = sc["header_warning"]
    return out


def _summary_from_draws(run_id: str, params: list[str] | None, max_rows: int) -> dict[str, Any]:
    """Moments and quantiles for the approximate methods.

    No R-hat or ESS: variational/pathfinder/laplace draws are not MCMC chains, and
    stansummary would happily report both anyway.
    """
    d = draws.load(run_id)
    sc = draws.schema(run_id)
    selected = [(name, arr) for name, arr in d.params.items() if _wanted(name, params)]
    total = sum(int(np.prod(arr.shape[2:])) for _, arr in selected)
    records = []
    for name, arr in selected:
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        for pos, (label, _) in enumerate(draws.spec_element_labels(name, sc["params"][name])):
            if max_rows > 0 and len(records) >= max_rows:
                break
            v = flat[:, :, pos].reshape(-1)
            q5, q50, q95 = np.quantile(v, [0.05, 0.5, 0.95])
            records.append(
                json_safe(
                    {
                        "param": label,
                        "mean": _sig6(v.mean()),
                        "sd": _sig6(v.std(ddof=1)) if v.size > 1 else 0.0,
                        "q5": _sig6(q5),
                        "q50": _sig6(q50),
                        "q95": _sig6(q95),
                        "n": int(v.size),
                    }
                )
            )
        if max_rows > 0 and len(records) >= max_rows:
            break
    out: dict[str, Any] = {"summary": records}
    if total > len(records):
        out["truncated"] = {
            "shown": len(records),
            "total": total,
            "selected_by": "order",
            "hint": "name parameters via `params`, or pass max_rows=0; "
            "get_run_diagnostics reports worst-case R-hat/ESS over all of them",
        }
    if d.truncated_to is not None:
        out["chains_truncated_to"] = d.truncated_to
    if d.header_warning:
        out["header_warning"] = d.header_warning
    return out


def run_summary(
    run_id: str, params: list[str] | None = None, max_rows: int = DEFAULT_MAX_ROWS
) -> dict[str, Any]:
    meta = runstore.read_meta(run_id)
    method = meta.get("method")
    if method == "optimize":
        return {"estimate": runstore.load_estimate(run_id)}
    if method == "sample":
        return _summary_from_stansummary(run_id, params, max_rows)
    return _summary_from_draws(run_id, params, max_rows)


# Standard Stan-workflow thresholds for the pass/warn verdict.
RHAT_MAX = 1.01
_ESS_MIN = 400
_EBFMI_MIN = 0.3


def _finite(x: Any) -> float | None:
    import math

    try:
        f = float(x)
    except TypeError, ValueError:
        return None
    return f if math.isfinite(f) else None


def convergence(run_id: str) -> dict[str, Any]:
    """`_convergence_stats`, degraded to a warning on any failure.

    Surfaced, never swallowed: silently dropping R-hat and ESS, the two headline
    diagnostics, is exactly how E-BFMI went missing. And never fatal: a run with
    no sidecar (predates the typed draws layer, or the sidecar write failed)
    lands here too, and must degrade to the method-diagnostics fallback rather
    than raise out of `run_diagnostics`.

    The clause is deliberately broad rather than `except RunError`. This is a
    degrade-to-warning site by design, so what matters is that NO failure to read
    one diagnostic becomes fatal for the whole call — independent of which layer
    below failed and which exception type it chose. Narrowing it to `RunError`
    is what let `cmdstanpy.cmdstan_path()`'s `ValueError` escape both
    `run_diagnostics` and `run_geometry` on a box with no CmdStan install.
    """
    try:
        return _convergence_stats(run_id)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        return {"convergence_warning": f"R-hat/ESS unavailable: {exc}"}


def _convergence_stats(run_id: str) -> dict[str, Any]:
    """Worst R-hat and smallest bulk/tail ESS over all parameter elements.

    Same stansummary output the summary uses, so the two cannot disagree. Sampler
    stats (`lp__` and friends) are excluded: they are not model parameters and their
    R-hat is not what the verdict is about.

    Free to raise: `convergence` is the only caller and turns any failure into a
    `convergence_warning`.
    """
    sc = draws.schema(run_id)
    stats = stansummary.summarize(draws.csv_files(run_id))

    worst_rhat: tuple[float, str] | None = None
    min_bulk: float | None = None
    min_tail: float | None = None
    looked_up: list[str] = []
    matched = 0
    undefined: dict[str, list[str]] = {}
    for name, spec in sc["params"].items():
        for stan_label, cmdstan_name in draws.spec_element_labels(name, spec):
            looked_up.append(cmdstan_name)
            row = stats.get(cmdstan_name)
            if row is None:
                undefined.setdefault("missing", []).append(stan_label)
                continue
            matched += 1
            for metric in ("R_hat", "ESS_bulk", "ESS_tail"):
                if _finite(row.get(metric)) is None:
                    undefined.setdefault(metric, []).append(stan_label)
            if (r := _finite(row.get("R_hat"))) is not None:
                if worst_rhat is None or r > worst_rhat[0]:
                    worst_rhat = (r, stan_label)
            if (b := _finite(row.get("ESS_bulk"))) is not None:
                min_bulk = b if min_bulk is None else min(min_bulk, b)
            if (t := _finite(row.get("ESS_tail"))) is not None:
                min_tail = t if min_tail is None else min(min_tail, t)

    if looked_up and not matched:
        # `convergence` turns this into a `convergence_warning` naming the
        # mismatch, so the verdict reports "unknown" WITH a reason rather than
        # silently omitting R-hat and ESS.
        raise _no_match(run_id, len(looked_up), looked_up, stats)

    out: dict[str, Any] = {}
    if undefined:
        out["undefined_diagnostics"] = {
            key: {"count": len(values), "parameters": values[:10]}
            for key, values in undefined.items()
        }
    if sc.get("header_warning"):
        out["header_warning"] = sc["header_warning"]
    if worst_rhat is not None:
        out["max_rhat"] = round(worst_rhat[0], 4)
        out["max_rhat_param"] = worst_rhat[1]
    if min_bulk is not None:
        out["min_ess_bulk"] = round(min_bulk, 1)
    if min_tail is not None:
        out["min_ess_tail"] = round(min_tail, 1)
    return out


def _sampler_stats(d: draws.Draws, meta: dict[str, Any]) -> dict[str, Any]:
    ss = d.sampler_stats
    out: dict[str, Any] = {}

    if (div := ss.get("divergent__")) is not None:
        out["divergences"] = int(np.asarray(div).sum())

    if (td := ss.get("treedepth__")) is not None:
        max_td = int(meta.get("method_args", {}).get("max_treedepth", 10))
        out["max_treedepth"] = max_td
        out["treedepth_saturations"] = int((np.asarray(td) >= max_td).sum())

    if (accept := ss.get("accept_stat__")) is not None:
        out["mean_accept_stat"] = round(float(np.asarray(accept).mean()), 4)

    if (energy := ss.get("energy__")) is not None:
        import arviz as az

        # E-BFMI is not in stansummary (only in `diagnose`, as prose), so it stays
        # an arviz call -- now on a clean (chain, draw) array rather than a
        # DataTree. Passing the whole idata (arviz 1.2: a DataTree) raised
        # TypeError every time, which a bare `except: pass` turned into E-BFMI
        # silently missing from every run (measured on Neal's funnel: min E-BFMI
        # 0.06 against 1.01 for its non-centered twin).
        try:
            values = [_finite(b) for b in np.asarray(az.bfmi(energy))]
            bfmi = [v for v in values if v is not None]
            if missing := [i + 1 for i, value in enumerate(values) if value is None]:
                out["ebfmi_warning"] = f"E-BFMI undefined for chain(s) {missing}"
            if bfmi:
                out["min_ebfmi"] = round(min(bfmi), 4)
        except Exception as exc:  # noqa: BLE001
            # Surfaced, never swallowed: a diagnostic that quietly returns nothing
            # reads as "checked and fine".
            out["ebfmi_warning"] = f"E-BFMI unavailable: {exc}"
    return out


# Each entry pairs a diagnostic key with the check that turns it into an issue
# string (or None). This is the ONE place the six key names are listed:
# `_verdict` also uses it to tell "nothing was measured" from "measured and
# clean", rather than keeping a second, separately-maintained key list.
_VERDICT_CHECKS: tuple[tuple[str, Callable[[dict[str, Any]], str | None]], ...] = (
    (
        "max_rhat",
        lambda d: (
            f"R-hat {d['max_rhat']} > {RHAT_MAX} (worst: {d.get('max_rhat_param')})"
            if d["max_rhat"] > RHAT_MAX
            else None
        ),
    ),
    (
        "min_ess_bulk",
        lambda d: (
            f"low bulk ESS {d['min_ess_bulk']} < {_ESS_MIN}"
            if d["min_ess_bulk"] < _ESS_MIN
            else None
        ),
    ),
    (
        "min_ess_tail",
        lambda d: (
            f"low tail ESS {d['min_ess_tail']} < {_ESS_MIN}"
            if d["min_ess_tail"] < _ESS_MIN
            else None
        ),
    ),
    (
        "divergences",
        lambda d: f"{d['divergences']} divergent transition(s)" if d["divergences"] > 0 else None,
    ),
    (
        "treedepth_saturations",
        lambda d: (
            f"{d['treedepth_saturations']} iteration(s) hit max_treedepth"
            if d["treedepth_saturations"] > 0
            else None
        ),
    ),
    (
        "min_ebfmi",
        lambda d: (
            f"E-BFMI {d['min_ebfmi']} < {_EBFMI_MIN}" if d["min_ebfmi"] < _EBFMI_MIN else None
        ),
    ),
)


def _verdict(d: dict[str, Any]) -> dict[str, Any]:
    """Pass/warn/unknown verdict from whichever of `_VERDICT_CHECKS`'s keys
    landed in `d`.

    `unknown`, not `pass`, when NONE of them are present: an empty diagnostics
    dict here means every reader failed (both `convergence` and
    `_sampler_stats` recorded a `*_warning` instead of a statistic, or
    `convergence` ran but matched nothing), and reporting "pass" on the
    headline field for a run nothing was actually checked on is a false-clean
    -- the exact failure mode the comments on those two readers argue
    against. A `*_warning` alongside SOME measured keys is left alone: it
    already says what was missed, and folding it into `issues` would flip a
    partially-checked run to "warn" and read as "the model is pathological"
    when the truth is "one thing could not be checked".
    """
    if not any(key in d for key, _ in _VERDICT_CHECKS):
        return {"verdict": "unknown", "issues": []}
    issues = [msg for key, check in _VERDICT_CHECKS if key in d and (msg := check(d))]
    if not issues and (
        d.get("undefined_diagnostics")
        or d.get("convergence_warning")
        or d.get("ebfmi_warning")
        or d.get("sampler_stats_warning")
    ):
        return {"verdict": "unknown", "issues": []}
    return {"verdict": "pass" if not issues else "warn", "issues": issues}


def run_diagnostics(run_id: str) -> dict[str, Any]:
    meta = runstore.read_meta(run_id)
    method = meta.get("method")
    if method == "optimize":
        est = runstore.load_estimate(run_id)
        return {"method": "optimize", "lp__": est.get("lp__")}
    out: dict[str, Any] = {"method": method}
    # HMC-style convergence/sampler diagnostics are only meaningful for sample
    # runs (stansummary's R-hat/ESS assume MCMC chains); VI/pathfinder/laplace
    # fall back to the method-specific scalars captured at persist time.
    if method == "sample":
        out.update(convergence(run_id))
        try:
            out.update(_sampler_stats(draws.load(run_id), meta))
        except RunError as exc:
            out["sampler_stats_warning"] = f"sampler statistics unavailable: {exc}"
        out.update(_verdict(out))
    md = meta.get("method_diagnostics")
    if md:
        out.update(md)
    return out


def _idata_with_loglik(run_id: str, var_name: str) -> Any:
    """An arviz DataTree with a `log_likelihood` group, built from typed draws.

    arviz is needed only for PSIS-LOO, which CmdStan has no equivalent for, so the
    object is constructed here rather than persisted. Stan models expose the
    pointwise log-likelihood as a generated quantity (conventionally `log_lik`),
    which arviz wants in its own group rather than under `posterior`.
    """
    import arviz as az

    _require_posterior_draws(run_id)
    d = draws.load(run_id)
    if var_name not in d.params:
        # A run with zero parameters would otherwise end the message at
        # "(have: )", with a trailing space and nothing after it.
        have = ", ".join(sorted(d.params)) or "no parameters at all"
        raise RunError(
            f"run {run_id!r} has no log-likelihood: model must emit a generated "
            f"quantity named {var_name!r} (have: {have})"
        )
    posterior = {k: v for k, v in d.params.items() if k != var_name}
    # arviz 1.2's from_dict takes ONE nested {group: {var: array}} dict, not the
    # per-group keyword arguments (posterior=..., log_likelihood=...) the brief
    # was written against. This says nothing about az.loo/az.compare below: those
    # take an idata or dataset plus ordinary keyword arguments (`var_name=`,
    # `pointwise=`), and never took per-group ones.
    return az.from_dict(
        {
            "posterior": posterior or {var_name: d.params[var_name]},
            "log_likelihood": {var_name: d.params[var_name]},
        }
    )


def _require_posterior_draws(run_id: str) -> None:
    meta = runstore.read_meta(run_id)
    method = meta.get("method")
    if method == "sample":
        return
    if method == "generate_quantities":
        # A CSV snapshot retains the source sampling law after pruning.
        source_method = meta.get("input_method")
        source = meta.get("method_args", {}).get("from_run_id")
        if source_method is None and source:
            _require_posterior_draws(source)
            return
        if source_method == "sample":
            return
    raise RunError(
        f"run {run_id!r} has no verified posterior sampling law ({method!r}); "
        "ordinary PSIS-LOO requires MCMC posterior draws. Approximate fits need "
        "target/proposal importance correction, which this tool does not implement."
    )


def run_loo(run_id: str, var_name: str = "log_lik") -> dict[str, Any]:
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
    out = {
        "elpd_loo": round(float(elpd), 3),
        "se": round(float(loo.se), 3),
        "p_loo": round(float(p), 3) if p is not None else None,
        "n_high_pareto_k": int((khat > good_k).sum()) if khat.size else 0,
        "warning": bool(getattr(loo, "warning", False)),
    }
    if warning := draws.schema(run_id).get("header_warning"):
        out["header_warning"] = warning
    return out


def compare_runs(run_ids: list[str], var_name: str = "log_lik") -> dict[str, Any]:
    """Rank runs by PSIS-LOO (arviz.compare). Best model first."""
    import arviz as az

    if len(run_ids) < 2:
        raise ValidationError("compare_runs needs at least two run_ids")
    dataset = {rid: _idata_with_loglik(rid, var_name) for rid in run_ids}
    df: Any = az.compare(dataset)  # PSIS-LOO by default
    records = df.reset_index().rename(columns={"index": "run_id"}).to_dict("records")
    out: dict[str, Any] = {"comparison": [json_safe(r) for r in records]}
    warnings = {
        rid: warning for rid in run_ids if (warning := draws.schema(rid).get("header_warning"))
    }
    if warnings:
        out["header_warnings"] = warnings
    return out


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


def _shift_vector(Xz: np.ndarray, idx: np.ndarray, n_div: int, n: int) -> np.ndarray:
    """mean(divergent) - mean(rest), per column, in sd units.

    Xz columns are mean-centred, so the two group sums are negatives of each
    other and only the divergent rows need touching — the difference is
    sum_div * (1/n_div + 1/n_rest). That keeps a permutation pass O(n_div) per
    column instead of O(n_draws)."""
    return Xz[idx].sum(0) * (1.0 / n_div + 1.0 / (n - n_div))


def _margin_p_value(
    Xz: np.ndarray, n_div: int, observed: float, permutations: int
) -> dict[str, Any]:
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


def run_pathologies(
    run_id: str, top_k: int = 5, permutations: int = DEFAULT_PERMUTATIONS
) -> dict[str, Any]:
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
    d = draws.load(run_id)
    if "divergent__" not in d.sampler_stats:
        raise RunError(f"run {run_id!r} has no `divergent__` sampler stat to localize")

    # CmdStan's own column name; arviz renamed it to `diverging`.
    div = np.asarray(d.sampler_stats["divergent__"]).reshape(-1).astype(bool)
    n, n_div = int(div.size), int(div.sum())
    out: dict[str, Any] = {"run_id": run_id, "n_draws": n, "n_divergent": n_div, "ranking": []}
    if d.header_warning:
        out["header_warning"] = d.header_warning
    if n_div == 0:
        return {**out, "note": "no divergent transitions to localize"}
    if n_div == n:
        return {**out, "note": "every draw diverged; no non-divergent draws to compare against"}

    X, names = draws.matrix(d)
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
    sep = out.get("separation")
    if n_div >= _MIN_DIVERGENCES and (sep is None or sep["p_value"] > 0.05):
        # Divergences with no parameter standing out is the signature of a
        # pathology that is not marginal — which is what run_geometry looks for.
        out["hint"] = (
            "divergences are present but no single parameter separates: the cause "
            "may be a degenerate direction rather than one location. Try run_geometry."
        )
    return out
