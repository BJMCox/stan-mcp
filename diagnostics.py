"""Method-aware summary and diagnostics over stored runs (arviz-backed).

persist_run: after a run finishes, convert its fit to persistent storage —
optimize writes an estimate.json (optimized_params_dict); draws-producing
methods first persist the raw CmdStan CSVs (source of truth; enables
generate_quantities reuse via runstore.csv_paths), best-effort and
independent of the sidecar, then write a draws_schema.json sidecar (dims and
row-major CSV column indices per variable, from the fit's own column names).
The sidecar is the only record of a run's dims/columns, so unlike the CSV
step it is NOT best-effort: a failure records a persist_warning in meta.json
(a csv_warning is recorded separately if CSV persistence itself fails) and
then raises, which `sample.py`'s caller turns into status="failed" rather
than letting the run rest at "done" with unreadable draws.
run_summary: sample -> CmdStan's own `stansummary` (mean, sd, mad, 90% ETI,
bulk/tail ESS, R-hat, MCSE) over the run's draws CSVs, relabelled into Stan
syntax; the other draws-producing methods -> numpy moments/quantiles over the
typed draws (no R-hat/ESS -- they are not MCMC chains); optimize -> the stored
estimate.
run_diagnostics: sample -> worst R-hat/min bulk-tail ESS (stansummary) plus
divergences, tree-depth saturations, mean accept stat, E-BFMI (typed sampler
stats), with a pass/warn verdict; the other draws-producing methods -> just
the method-specific scalars captured at persist time; optimize -> lp__ from
the estimate.
run_loo / compare_runs: PSIS-LOO cross-validation and multi-run ranking (models
must emit a `log_lik` generated quantity).
extract_draws: flattened draws for one parameter from the typed draws layer
(draws.load) -- one path for all six draws-producing methods (not available
for optimize).
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import draws
import runstore
import stansummary
from errors import RunError, ValidationError


def _method_diagnostics(method: str, fit) -> dict:
    """Method-specific health scalars pulled from the fit. arviz cannot convert
    VI/pathfinder/laplace fits, so for those three methods these are the ONLY
    diagnostics there are — which is why a failure here is recorded rather than
    swallowed."""
    getters = {
        "variational": lambda: {"eta": float(fit.eta)},
        "pathfinder": lambda: {"is_resampled": bool(fit.is_resampled)},
        "laplace": lambda: {"mode_lp": float(fit.mode.optimized_params_dict.get("lp__"))},
    }
    get = getters.get(method)
    if get is None:
        return {}
    try:
        return get()
    except Exception as exc:  # noqa: BLE001
        # These read cmdstanpy fit attributes, which do get renamed across
        # versions. Returning {} on failure left the method with no diagnostics
        # at all and no sign that anything was meant to be there.
        return {"method_diagnostics_warning": f"{method} diagnostics unavailable: {exc}"}


def _csv_columns(fit) -> tuple[list[str], str]:
    """The fit's CSV column names in CmdStan's own dot notation (`m.1.1`), plus
    which header source they came from.

    NOT `fit.column_names`: cmdstanpy documents that property as "names munged
    to array notation, e.g. `beta[1]` not `beta.1`" (confirmed live: a matrix
    parameter comes back as `m[1,1]`, `m[2,1]`, ...), which cannot be looked up
    against the `name + "." + ".".join(idx)` strings `draws.element_labels`
    builds — every non-scalar variable would KeyError and lose its sidecar
    entirely. `metadata.cmdstan_config["raw_header"]` is cmdstanpy's own
    unmunged copy of the CSV header row it parsed, so it always matches.
    Falls back to `column_names` for fakes/mocks that only set that attribute.

    The source name is returned so `_sidecar_from_fit` can report it: a lookup
    miss against the munged fallback means something quite different from a miss
    against the unmunged `raw_header`, and the fix differs accordingly.
    """
    try:
        return fit.metadata.cmdstan_config["raw_header"].split(","), "raw_header"
    except (AttributeError, KeyError, TypeError):
        return list(fit.column_names), "column_names"


def _draw_row_offset(method: str, run_id: str) -> int:
    """How many leading data rows in each of this run's draws CSVs are NOT
    draws. Checked directly against cmdstanpy's own row assembly for every
    method that reaches here (cmdstanpy 1.3.0):

    `variational` (`stanfit/vb.py`) splits `draws_np[0]` off as the
    variational mean and only treats `draws_np[1:]` as the sample. `sample`
    (`mcmc.py`), `pathfinder` and `laplace` (`pathfinder.py` / `laplace.py`,
    both via `parse_comments_header_and_draws`) use every data row as a draw,
    no leading row skipped.

    `generate_quantities` (`gq.py`) usually does too -- EXCEPT when its
    source run (`method_args.from_run_id`) was `variational`: cmdstanpy's
    `generate_quantities` re-evaluates the generated-quantities block once
    per row of the source fit, including a variational source's mean row, so
    that extra row lands in the GQ run's OWN output CSV too. `gq.py`'s
    `_draws_start` computes `draw1 = 1` for a `CmdStanVB` previous_fit and
    applies it to `self._draws[start_idx:, :, :]` -- the GQ run's own
    assembled draws, not merely a slice of the source's. Resolved via the
    source run's persisted `method` (`runstore.read_meta`), not
    `isinstance(fit.previous_fit, CmdStanVB)`, so this does not depend on a
    cmdstanpy internal that could be renamed.
    """
    if method == "variational":
        return 1
    if method == "generate_quantities":
        from_run_id = runstore.read_meta(run_id).get("method_args", {}).get("from_run_id")
        if from_run_id and runstore.read_meta(from_run_id).get("method") == "variational":
            return 1
    return 0


def _sidecar_from_fit(fit, method: str, run_id: str) -> dict:
    """The draws sidecar body for a fit: dims and CSV column indices per variable.

    `cols` is written in the order that fills the array ROW-MAJOR, because
    numpy.reshape is row-major while CmdStan's CSV header is column-major
    (`m.1.1, m.2.1, m.1.2, ...`). Recording the contiguous column run instead and
    reshaping plainly puts `m.2.1`'s values into `m[1,2]` — a silent transpose that
    mislabels rather than raises. Doing the permutation once, here, is what lets
    every reader use a plain reshape with no order= argument.

    See `_draw_row_offset` for `draw_row_offset`.

    A column that cannot be placed raises a RunError naming the variable, the
    element, and the header source. Sampling itself has already SUCCEEDED by the
    time this runs, so a bare `KeyError: 'm[1,1]'` propagating up through
    `persist_run` landed such a run at status="failed" with a cryptic message and
    no hint that the header source was the problem.
    """
    cols, header_source = _csv_columns(fit)
    index = {name: i for i, name in enumerate(cols)}
    dims = {name: tuple(v.dimensions) for name, v in fit.metadata.stan_vars.items()}

    def column_of(var: str, cmdstan_name: str) -> int:
        try:
            return index[cmdstan_name]
        except KeyError as exc:
            hint = (
                "cmdstanpy munges `column_names` to array notation (`m[1,1]`, not "
                "CmdStan's own `m.1.1`), so no non-scalar variable can be placed "
                "from it; the fit exposed no cmdstan_config['raw_header'] to use "
                "instead"
                if header_source == "column_names"
                else "the header does not carry every element of the fit's own declared variables"
            )
            raise RunError(
                f"run {run_id!r}: could not place element {cmdstan_name!r} of "
                f"parameter {var!r} (dims {list(dims[var])}) against the fit's "
                f"{header_source} header — {hint}. Sampling succeeded and the "
                "run's CSVs are persisted, but its draws cannot be typed."
            ) from exc

    params: dict[str, dict] = {}
    for name, shape in dims.items():
        if not shape:
            params[name] = {"dims": [], "cols": [column_of(name, name)]}
            continue
        ordered = [column_of(name, cn) for _, cn in draws.element_labels(name, shape)]
        params[name] = {"dims": list(shape), "cols": ordered}

    stats = {name: {"dims": [], "cols": [i]} for i, name in enumerate(cols) if name.endswith("__")}
    return {
        "schema": draws.SCHEMA_VERSION,
        "params": params,
        "sampler_stats": stats,
        "draw_row_offset": _draw_row_offset(method, run_id),
    }


def _is_draws_csv(name: str) -> bool:
    """CmdStan draws CSVs, not `laplace`'s internal-optimize mode CSV.

    When `laplace_sample` gets no explicit `mode=`, cmdstanpy runs its own
    optimize step into the SAME output directory, dropping an `-opt-...` CSV
    (fewer columns, one row) beside the draws CSV. `_ordered_draws_files`
    below excludes it before recording the sidecar's `files` list, or
    `draws.load` stacks arrays of two different shapes and dies with a bare
    numpy `ValueError` instead of a `RunError` naming the run.
    """
    return "-opt-" not in name


def _ordered_draws_files(fit, run_id: str) -> list[str]:
    """Persisted draws CSV filenames, in true per-chain run order.

    NOT a filename sort of `csv/`: past 9 chains that puts "_10" before "_2",
    silently permuting chains relative to the run (chain-symmetric statistics
    are unaffected, but a per-chain reader like `extract_draws(by_chain=True)`
    would mislabel them). `fit.runset.csv_files` is cmdstanpy's own
    per-chain-ordered list, so it is used verbatim where present.
    `CmdStanPathfinder`/`CmdStanLaplace` expose no `.runset`, so the fallback
    sorts by the filename's trailing `_<chain>` id numerically — falling back
    to a plain name sort where that suffix is absent, e.g. the single
    unsuffixed CSV variational/pathfinder/laplace write.
    """
    persisted = [Path(p).name for p in runstore.csv_paths(run_id)]
    want = set(persisted)

    if hasattr(fit, "runset"):
        try:
            ordered = [Path(p).name for p in fit.runset.csv_files]
        except Exception:  # noqa: BLE001
            ordered = []
        ordered = [n for n in ordered if _is_draws_csv(n) and n in want]
        if ordered:
            return ordered

    def chain_key(name: str):
        stem = name.rsplit(".", 1)[0]
        tail = stem.rsplit("_", 1)[-1]
        return (0, int(tail)) if tail.isdigit() else (1, stem)

    return sorted((n for n in persisted if _is_draws_csv(n)), key=chain_key)


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

    # The draws sidecar replaces inference.nc: it covers all six methods (arviz's
    # from_cmdstanpy raises on variational/pathfinder/laplace), it is written from
    # the live fit's own column names, and it is the only record of chain order and
    # of which CSVs hold draws.
    try:
        sidecar = _sidecar_from_fit(fit, method, run_id)
        files = _ordered_draws_files(fit, run_id)
        if not files:
            # Can happen if CSV persistence above failed (already recorded as
            # csv_warning) or found nothing to copy. A sidecar with files: []
            # would still write "successfully" and leave status="done" for a
            # run draws.load can never open (min([]) on an empty file list)
            # -- exactly the unqualified-success failure mode this whole
            # try/except exists to prevent.
            raise RunError(f"run {run_id!r} has no persisted draws CSVs to record in the sidecar")
        sidecar["files"] = files
        draws.schema_path(run_id).write_text(json.dumps(sidecar, indent=2))
    except Exception as exc:  # noqa: BLE001
        meta = runstore.read_meta(run_id)
        meta["persist_warning"] = f"draws schema not written: {exc}"
        runstore.write_meta(run_id, meta)
        # Unlike the arviz conversion this replaces (whose failure still left
        # CSV-derived fallbacks readable for variational/pathfinder/laplace),
        # the sidecar is now the ONLY record of a run's dims/column layout —
        # without it the run's draws cannot be read at all. Such a run must
        # not rest at status="done" with only this field to notice by, so
        # re-raise into `_execute`'s existing failure path (status="failed",
        # the CSVs already persisted above are kept, not lost).
        raise RunError(f"draws schema not written for run {run_id!r}: {exc}") from exc

    # Last, once nothing else needs to read the fit's CSVs: drop live/. csv/ is
    # the durable copy, so keeping both doubles every run's disk footprint.
    if runstore.csv_paths(run_id):
        runstore.clear_live(run_id)


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


def _wanted(name: str, params) -> bool:
    return not params or name in set(params)


def _summary_from_stansummary(run_id: str, params, max_rows: int) -> dict:
    """Statistics from CmdStan's own implementation, relabelled into Stan syntax.

    Only the sidecar is read here, not the draws: stansummary has already parsed
    the CSVs, so loading them again to learn names and dims would be wasted work.
    """
    sc = draws.schema(run_id)
    stats = stansummary.summarize(draws.csv_files(run_id))
    records = []
    for name, spec in sc["params"].items():
        if not _wanted(name, params):
            continue
        for stan_label, cmdstan_name in draws.element_labels(name, tuple(spec["dims"])):
            row = stats.get(cmdstan_name)
            if row is None:
                continue
            rec: dict[str, Any] = {"param": stan_label}
            for src, dst in _SUMMARY_COLUMNS.items():
                if src in row:
                    rec[dst] = round(row[src], 6)
            records.append(rec)
    return _cap_rows(records, max_rows)


def _summary_from_draws(run_id: str, params, max_rows: int) -> dict:
    """Moments and quantiles for the approximate methods.

    No R-hat or ESS: variational/pathfinder/laplace draws are not MCMC chains, and
    stansummary would happily report both anyway.
    """
    d = draws.load(run_id)
    records = []
    for name, arr in d.params.items():
        if not _wanted(name, params):
            continue
        for label, values in draws.elements(name, arr):
            v = values.reshape(-1)
            records.append(
                {
                    "param": label,
                    "mean": round(float(v.mean()), 6),
                    "sd": round(float(v.std(ddof=1)), 6) if v.size > 1 else 0.0,
                    "q5": round(float(np.quantile(v, 0.05)), 6),
                    "q50": round(float(np.quantile(v, 0.5)), 6),
                    "q95": round(float(np.quantile(v, 0.95)), 6),
                    "n": int(v.size),
                }
            )
    out = _cap_rows(records, max_rows)
    if d.truncated_to is not None:
        out["chains_truncated_to"] = d.truncated_to
    return out


def run_summary(run_id: str, params=None, max_rows: int = DEFAULT_MAX_ROWS) -> dict:
    meta = runstore.read_meta(run_id)
    method = meta.get("method")
    if method == "optimize":
        return {"estimate": runstore.load_estimate(run_id)}
    if method == "sample":
        return _summary_from_stansummary(run_id, params, max_rows)
    return _summary_from_draws(run_id, params, max_rows)


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


def _convergence(run_id: str) -> dict:
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


def _convergence_stats(run_id: str) -> dict:
    """Worst R-hat and smallest bulk/tail ESS over all parameter elements.

    Same stansummary output the summary uses, so the two cannot disagree. Sampler
    stats (`lp__` and friends) are excluded: they are not model parameters and their
    R-hat is not what the verdict is about.

    Free to raise: `_convergence` is the only caller and turns any failure into a
    `convergence_warning`.
    """
    sc = draws.schema(run_id)
    stats = stansummary.summarize(draws.csv_files(run_id))

    worst_rhat: tuple[float, str] | None = None
    min_bulk: float | None = None
    min_tail: float | None = None
    for name, spec in sc["params"].items():
        for stan_label, cmdstan_name in draws.element_labels(name, tuple(spec["dims"])):
            row = stats.get(cmdstan_name)
            if row is None:
                continue
            if (r := _finite(row.get("R_hat"))) is not None:
                if worst_rhat is None or r > worst_rhat[0]:
                    worst_rhat = (r, stan_label)
            if (b := _finite(row.get("ESS_bulk"))) is not None:
                min_bulk = b if min_bulk is None else min(min_bulk, b)
            if (t := _finite(row.get("ESS_tail"))) is not None:
                min_tail = t if min_tail is None else min(min_tail, t)

    out: dict = {}
    if worst_rhat is not None:
        out["max_rhat"] = round(worst_rhat[0], 4)
        out["max_rhat_param"] = worst_rhat[1]
    if min_bulk is not None:
        out["min_ess_bulk"] = round(min_bulk, 1)
    if min_tail is not None:
        out["min_ess_tail"] = round(min_tail, 1)
    return out


def _sampler_stats(d: draws.Draws, meta: dict) -> dict:
    ss = d.sampler_stats
    out: dict = {}

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
            bfmi = [v for v in (_finite(b) for b in np.asarray(az.bfmi(energy))) if v is not None]
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
_VERDICT_CHECKS: tuple[tuple[str, Callable[[dict], str | None]], ...] = (
    (
        "max_rhat",
        lambda d: (
            f"R-hat {d['max_rhat']} > {_RHAT_MAX} (worst: {d.get('max_rhat_param')})"
            if d["max_rhat"] > _RHAT_MAX
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


def _verdict(d: dict) -> dict:
    """Pass/warn/unknown verdict from whichever of `_VERDICT_CHECKS`'s keys
    landed in `d`.

    `unknown`, not `pass`, when NONE of them are present: an empty diagnostics
    dict here means every reader failed (both `_convergence` and
    `_sampler_stats` recorded a `*_warning` instead of a statistic, or
    `_convergence` ran but matched nothing), and reporting "pass" on the
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
    return {"verdict": "pass" if not issues else "warn", "issues": issues}


def run_diagnostics(run_id: str) -> dict:
    meta = runstore.read_meta(run_id)
    method = meta.get("method")
    if method == "optimize":
        est = runstore.load_estimate(run_id)
        return {"method": "optimize", "lp__": est.get("lp__")}
    out: dict = {"method": method}
    # HMC-style convergence/sampler diagnostics are only meaningful for sample
    # runs (stansummary's R-hat/ESS assume MCMC chains); VI/pathfinder/laplace
    # fall back to the method-specific scalars captured at persist time.
    if method == "sample":
        out.update(_convergence(run_id))
        try:
            out.update(_sampler_stats(draws.load(run_id), meta))
        except RunError as exc:
            out["sampler_stats_warning"] = f"sampler statistics unavailable: {exc}"
        out.update(_verdict(out))
    md = meta.get("method_diagnostics")
    if md:
        out.update(md)
    return out


def _idata_with_loglik(run_id: str, var_name: str):
    """An arviz DataTree with a `log_likelihood` group, built from typed draws.

    arviz is needed only for PSIS-LOO, which CmdStan has no equivalent for, so the
    object is constructed here rather than persisted. Stan models expose the
    pointwise log-likelihood as a generated quantity (conventionally `log_lik`),
    which arviz wants in its own group rather than under `posterior`.
    """
    import arviz as az

    d = draws.load(run_id)
    if var_name not in d.params:
        raise RunError(
            f"run {run_id!r} has no log-likelihood: model must emit a generated "
            f"quantity named {var_name!r} (have: {', '.join(sorted(d.params))})"
        )
    posterior = {k: v for k, v in d.params.items() if k != var_name}
    # arviz 1.2's from_dict takes one nested {group: {var: array}} dict, not
    # per-group keyword arguments (the signature the brief was written
    # against, and still in use by az.compare/az.loo elsewhere in this file).
    return az.from_dict(
        {
            "posterior": posterior or {var_name: d.params[var_name]},
            "log_likelihood": {var_name: d.params[var_name]},
        }
    )


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


def _from_typed_source(
    d: draws.Draws, run_id: str, param: str, thin: int, by_chain: bool, max_draws: int
) -> dict:
    arr = d.params.get(param)
    if arr is None:
        raise RunError(
            f"parameter {param!r} not found in run {run_id!r} (have: {', '.join(sorted(d.params))})"
        )
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

    from functools import partial

    # Split the ceiling across the requested parameters: it bounds the whole
    # call, so asking for N names must not return N times as much.
    per_param = max(1, max_draws // len(wanted)) if max_draws > 0 else 0
    shared = {"thin": thin, "by_chain": by_chain, "max_draws": per_param}
    # Load the draws ONCE: doing it per parameter re-parsed every CSV for each
    # name in `params`.
    extract = partial(_from_typed_source, draws.load(run_id), run_id, **shared)

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
    d = draws.load(run_id)
    if "divergent__" not in d.sampler_stats:
        raise RunError(f"run {run_id!r} has no `divergent__` sampler stat to localize")

    # CmdStan's own column name; arviz renamed it to `diverging`.
    div = np.asarray(d.sampler_stats["divergent__"]).reshape(-1).astype(bool)
    n, n_div = int(div.size), int(div.sum())
    out: dict = {"run_id": run_id, "n_draws": n, "n_divergent": n_div, "ranking": []}
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


# --- posterior geometry (linear degeneracy) ------------------------------
#
# Complements run_pathologies rather than overlapping it. Divergences, ESS, R-hat
# and E-BFMI between them catch non-linear pathology loudly (measured on Neal's
# funnel: ESS 18 of 4000, E-BFMI 0.06). What none of them catch is a posterior
# that is merely very ill-conditioned — collinear predictors burning gradient
# evaluations along a ridge while every one of those four looks healthy. That is
# what this is for, and it is why a funnel does NOT show up here: its dependence
# is not linear, so its max |r| is indistinguishable from noise (0.262 measured,
# against a 1/sqrt(ESS) noise scale of 0.236).
DEFAULT_MAX_DIMS = 500
_MIN_ABS_R = 0.5  # large enough to hurt sampling...
_MIN_FISHER_Z = 4.0  # ...and firmly established given the effective sample size
# The sample correlation matrix over p dims from n_eff effective draws is
# ill-conditioned BY CONSTRUCTION once p is a meaningful fraction of n_eff: the
# smallest eigenvalue collapses toward zero from noise alone. Eigen output is
# suppressed unless p is below n_eff / this.
#
# 4 (i.e. p/n_eff <= 0.25) rather than a stricter guess, from measuring what
# independent data actually yields: condition number 3.7 at p/n_eff = 0.10, 8.0
# at 0.25, 19.5 at 0.40, 219.6 at 0.80. Only 0.8 breaches the threshold at which
# a condition number is reported as an issue, so 0.25 keeps noise a comfortable
# 12x clear of it. A stricter ratio of 10 suppressed the eigen block on every
# real run tried — including a 151-dimension model at ESS 673, which would have
# needed near-perfect mixing to qualify — making the one statistic that catches
# multi-way degeneracy effectively unreachable.
_EIGEN_ESS_RATIO = 4
_EIGEN_GAP = 1.5  # min relative gap to the next eigenvalue before naming loadings


def _constrained_matrix(d: draws.Draws, bs_names: list[str]) -> np.ndarray:
    """(draws, n_params) in BridgeStan's constrained parameter order.

    BridgeStan spells elements `z.3` / `m.2.1` with 1-based indices; the sidecar
    records each variable's dims, so an index tuple resolves directly against the
    typed array. This replaces resolving them against arviz's dim ordering with
    ravel_multi_index."""
    cols = []
    for nm in bs_names:
        base, *idx = nm.split(".")
        arr = d.params.get(base)
        if arr is None:
            raise RunError(f"BridgeStan names parameter {base!r}, absent from the run's draws")
        flat = arr.reshape(arr.shape[0] * arr.shape[1], -1)
        if idx:
            try:
                pos = int(np.ravel_multi_index(tuple(int(i) - 1 for i in idx), arr.shape[2:]))
            except ValueError as exc:
                raise RunError(
                    f"BridgeStan name {nm!r} does not fit parameter {base!r}'s "
                    f"recorded dims {arr.shape[2:]!r}: {exc}"
                ) from exc
        else:
            pos = 0
        cols.append(flat[:, pos])
    return np.column_stack(cols)


def _unconstrained_draws(run_id: str, d: draws.Draws) -> tuple[np.ndarray, list[str], dict]:
    """(chain, draw, dim) unconstrained draws and their names.

    The sampler explores the UNCONSTRAINED space, so that is the geometry whose
    conditioning costs step size. Correlating constrained draws would report the
    mechanical dependence of simplex components and Cholesky factors as findings.
    Falls back to constrained draws, saying so, only if the model cannot be
    rebuilt at all."""
    import compile as _compile
    import density

    inputs = runstore.load_inputs(run_id)
    source = inputs["source"]
    n_chain, n_draw = d.n_chain, d.n_draw
    if source is None:
        raise RunError(f"run {run_id!r} has no persisted model.stan to unconstrain with")

    state = _compile.request_so(source)
    if state["status"] == "building":
        raise _GeometryPending(
            "building the BridgeStan artifact for this model (a run compiles only "
            "the CmdStan executable); retry shortly"
        )
    if state["status"] == "failed":
        X, names = draws.matrix(d)
        u = X.reshape(n_chain, n_draw, -1)
        return u, names, {"scale": "constrained", "scale_reason": state["error"]}

    m = density.get_model(source, inputs["data"])
    X = _constrained_matrix(d, m.param_names())
    unc = np.empty((X.shape[0], m.param_unc_num()))
    try:
        for i in range(X.shape[0]):
            unc[i] = m.param_unconstrain(X[i])
    except Exception as exc:  # noqa: BLE001
        # Stan's *_free functions VALIDATE the constraint they invert, and a draw
        # read back from a CSV only holds it to the written precision: at CmdStan's
        # default 6 sig figs a simplex sums to 1 to about 1e-8, which IS
        # simplex_free's tolerance, so ~1.5% of draws are rejected (measured:
        # 15/1000) — and one rejection aborts the whole loop. Runs sampled since
        # sample._OUTPUT_SIG_FIGS avoid it; a run recorded before that cannot be
        # unconstrained at all, so report constrained draws and say why instead of
        # failing the call. Same for any other constraint whose inverse validates
        # (unit_vector, corr_matrix, cholesky_factor_corr).
        Xc, cnames = draws.matrix(d)
        return (
            Xc.reshape(n_chain, n_draw, -1),
            cnames,
            {
                "scale": "constrained",
                "scale_reason": (
                    f"could not unconstrain this run's draws ({exc}); reporting the "
                    "constrained scale, on which some dependence is mechanical. "
                    "Re-run the model to get the unconstrained geometry."
                ),
            },
        )
    return unc.reshape(n_chain, n_draw, -1), list(m.param_unc_names()), {"scale": "unconstrained"}


class _GeometryPending(RunError):
    """The BridgeStan artifact is still building; the caller should retry."""


def _fisher_z(r: float, n_eff: float) -> float:
    """atanh(r) has standard error 1/sqrt(n-3), which is how a correlation gets
    calibrated against the effective sample size rather than the draw count."""
    if n_eff <= 4 or abs(r) >= 1.0:
        return float("inf") if abs(r) >= 1.0 else 0.0
    return float(np.arctanh(r) * np.sqrt(n_eff - 3))


def _block_of(name: str) -> str:
    return name.split(".")[0].split("[")[0]


def _mp_condition(p: int, n_eff: float) -> float:
    """Condition number independent parameters alone would give at this p/n_eff.

    Marchenko-Pastur: the sample eigenvalues of independent data spread to
    (1 -/+ sqrt(p/n))^2, so some conditioning is structural rather than a finding.
    Reported alongside the observed value so a moderate one is not over-read."""
    ratio = p / n_eff if n_eff > 0 else 1.0
    if ratio >= 1.0:
        return float("inf")
    s = np.sqrt(ratio)
    return float(((1 + s) / (1 - s)) ** 2)


def run_geometry(
    run_id: str,
    params: list[str] | None = None,
    max_dims: int = DEFAULT_MAX_DIMS,
    min_abs_r: float = _MIN_ABS_R,
    min_z: float = _MIN_FISHER_Z,
    top_k: int = 10,
) -> dict:
    """Posterior conditioning: correlated parameter pairs and the worst-
    conditioned direction, on the unconstrained scale the sampler works in."""
    import arviz as az

    meta = runstore.read_meta(run_id)
    if meta.get("method") != "sample":
        raise RunError(
            f"geometry needs MCMC draws with chain structure; run {run_id!r} "
            f"used {meta.get('method')!r}"
        )
    d = draws.load(run_id)
    try:
        u3, names, scale = _unconstrained_draws(run_id, d)
    except _GeometryPending as pending:
        # Not an error: the artifact is being built off the caller's thread.
        return {"run_id": run_id, "status": "preparing", "note": str(pending)}

    if params:
        want = set(params)
        sel = [i for i, nm in enumerate(names) if nm in want or _block_of(nm) in want]
        if not sel:
            raise ValidationError(f"no parameters matched {params!r} (have {len(names)} elements)")
        u3, names = u3[:, :, sel], [names[i] for i in sel]

    p = len(names)
    if max_dims > 0 and p > max_dims:
        raise ValidationError(
            f"{p} unconstrained dimensions exceeds max_dims={max_dims}; the "
            "correlation matrix is O(p^2) and its eigendecomposition O(p^3). "
            "Scope with `params`, or raise max_dims to accept the cost."
        )

    # Centre each chain before correlating. Pooling unmixed chains manufactures
    # correlation out of chain offsets: two chains sitting at different points
    # look collinear even when the within-chain geometry is spherical.
    resid = (u3 - u3.mean(axis=1, keepdims=True)).reshape(-1, p)
    sd = resid.std(0)
    keep = np.isfinite(sd) & (sd > 0)
    if keep.sum() < 2:
        return {
            "run_id": run_id,
            **scale,
            "note": "fewer than two varying parameters; no geometry to describe",
        }
    resid, names = resid[:, keep], [nm for nm, k in zip(names, keep, strict=True) if k]
    p = len(names)

    ess = np.asarray(
        az.ess(az.from_dict({"posterior": {"u": u3[:, :, keep]}}))["u"].values, dtype=float
    )
    # Left as NaN where arviz could not compute it, rather than clamped to a
    # fake value that would silently read as "calibrated against 1 draw".
    ess = np.where(np.isfinite(ess) & (ess > 0), ess, np.nan)
    finite_ess = ess[np.isfinite(ess)]

    # asarray: np.corrcoef is typed as possibly-scalar, which blocks 2-D indexing.
    R: np.ndarray = np.asarray(np.corrcoef(resid, rowvar=False), dtype=float)
    np.fill_diagonal(R, 0.0)
    out: dict = {
        "run_id": run_id,
        **scale,
        "n_dims": p,
        "min_ess": round(float(finite_ess.min()), 1) if finite_ess.size else None,
    }

    # Within-block pairs are collapsed: for theta[1..200] they are 19,900 of the
    # 20,000 pairs, all saying the same one thing, and would crowd out the single
    # cross-block pair that names the actual problem.
    blocks = np.array([_block_of(nm) for nm in names])
    iu, ju = np.triu_indices(p, k=1)
    same = blocks[iu] == blocks[ju]
    absr = np.abs(R[iu, ju])
    # abs: Fisher z carries the SIGN of r, and it is used here purely as strength
    # of evidence. Comparing the signed value against a positive threshold drops
    # every negatively correlated pair — which is the commonest form collinearity
    # takes, two coefficients trading off (measured: b1~b2 at r=-0.99995, z=-133,
    # silently discarded). Direction is already reported in `r`.
    # np.minimum, not min(): it propagates NaN, where Python's min() would pick
    # whichever operand came first and quietly launder a missing ESS into a real one.
    n_eff = np.minimum(ess[iu], ess[ju])
    # Below ~4 effective draws Fisher z is undefined, and arviz can hand back a
    # NaN ESS outright. Such a pair is NOT dropped: an inability to calibrate is
    # not evidence of absence, and a strong |r| whose ESS is too low to certify is
    # still the finding — often *because* the degeneracy destroyed the ESS.
    calibratable = np.isfinite(n_eff) & (n_eff > 4)
    raw_z = [
        _fisher_z(R[i, j], nz if np.isfinite(nz) else 0.0)
        for (i, j), nz in zip(zip(iu, ju, strict=True), n_eff, strict=True)
    ]
    zs = np.where(calibratable, np.abs(raw_z), np.nan)

    issues: list[str] = []
    pairs = []
    cross = ~same
    if cross.any():
        sel = np.flatnonzero(cross)
        # Both thresholds at 0 admits everything, per the 0 = unlimited convention.
        # An uncalibratable pair qualifies on |r| alone rather than vanishing.
        ok_z = (~calibratable[sel]) | (zs[sel] >= min_z)
        keepsel = sel[(absr[sel] >= min_abs_r) & ok_z]
        for k in keepsel[np.argsort(absr[keepsel])[::-1]][: max(top_k, 0) or None]:
            pair = {
                "a": names[iu[k]],
                "b": names[ju[k]],
                "r": round(float(R[iu[k], ju[k]]), 4),
                "z": round(float(zs[k]), 1) if calibratable[k] else None,
            }
            if not calibratable[k]:
                pair["z_note"] = f"effective sample size {n_eff[k]:.0f} too low to calibrate"
            pairs.append(pair)
    out["pairs"] = pairs
    out["max_abs_r_cross_block"] = round(float(absr[cross].max()), 4) if cross.any() else None
    if pairs:
        issues.append(
            f"{len(pairs)} correlated cross-block pair(s), worst |r|={abs(pairs[0]['r'])}"
        )

    within = []
    for b in dict.fromkeys(blocks):
        m = same & (blocks[iu] == b)
        if m.any():
            within.append(
                {
                    "block": b,
                    "n_pairs": int(m.sum()),
                    "max_abs_r": round(float(absr[m].max()), 4),
                    "median_abs_r": round(float(np.median(absr[m])), 4),
                }
            )
    if within:
        out["within_block"] = within
        for w in within:
            if w["max_abs_r"] >= min_abs_r:
                issues.append(
                    f"block {w['block']} internally correlated (max |r|={w['max_abs_r']})"
                )

    # Pairwise is structurally blind to multi-way degeneracy: if x3 ~ x1 + x2,
    # every pair can sit near 0.5 while the posterior is near-singular along
    # x1 + x2 - x3. The eigenvalues of the same matrix see it.
    np.fill_diagonal(R, 1.0)
    # A missing ESS cannot license the eigen block: treat it as the worst case,
    # since the guard exists precisely to stop reporting artifact.
    ess_min = float(finite_ess.min()) if finite_ess.size == p else 0.0
    if p * _EIGEN_ESS_RATIO > ess_min:
        out["eigen_note"] = (
            f"suppressed: {p} dimensions against min ESS {ess_min:.0f}. A sample "
            "correlation matrix is ill-conditioned by construction when p is not "
            f"small relative to n_eff, so the condition number would be artifact."
        )
    else:
        ev = np.linalg.eigvalsh(R)
        lo = max(float(ev[0]), 1e-12)
        eig: dict = {
            "condition_number": round(float(ev[-1]) / lo, 1),
            "min_eigenvalue": round(float(ev[0]), 5),
            # What independent parameters alone would produce at this p/n_eff
            # (Marchenko-Pastur; matches measurement to ~0.4 and is conservative
            # beyond). Without it a condition number of 8 looks like mild
            # degeneracy when at p/n_eff = 0.25 it is exactly what noise gives.
            "noise_condition_number": round(_mp_condition(p, ess_min), 1),
        }
        if ev.size > 1 and ev[1] > 0 and ev[1] / lo >= _EIGEN_GAP:
            w = np.linalg.eigh(R)[1][:, 0]
            top = np.argsort(np.abs(w))[::-1][:5]
            eig["direction"] = [{"param": names[i], "loading": round(float(w[i]), 3)} for i in top]
        else:
            eig["direction_note"] = (
                "smallest eigenvalues are not separated, so the eigenvector is "
                "arbitrary within its subspace and loadings would be misleading"
            )
        out["eigen"] = eig
        if eig["condition_number"] > 100:
            issues.append(f"condition number {eig['condition_number']}")

    # `_convergence`, not `run_diagnostics`: the same number, without spawning a
    # second stansummary subprocess and re-parsing every CSV for sampler stats to
    # reach one field. It also keeps this the only public entry point that is not
    # re-entered from inside another — which is what dragged the stansummary
    # binary into the offline geometry tests.
    rhat = _convergence(run_id).get("max_rhat")
    if rhat is not None and rhat > _RHAT_MAX:
        out["provisional"] = (
            f"max R-hat {rhat} > {_RHAT_MAX}: the chains disagree, so there is not "
            "yet one posterior whose geometry this describes. Fix mixing first."
        )
    out.update({"verdict": "pass" if not issues else "warn", "issues": issues})
    return out


def posterior_mean_inits(run_id: str) -> dict:
    """Posterior (or estimate) mean per parameter, shaped as CmdStan inits, for
    warm-starting another run.

    Every rank is supported: with dims recorded in the sidecar this is just the
    mean over the chain and draw axes. The previous version parsed flat CSV names
    and silently skipped anything with two or more dots -- so warm-starting a model
    with a matrix parameter looked like it worked while Stan default-inited it.

    No check that the source run's model matches the target's, deliberately:
    warm-starting a RELATED model is the point (same source with more data, a
    centered->non-centered rewrite), and CmdStan ignores an init name the model
    does not have while default-initing one the init omits. An init cannot bias a
    posterior, so a mismatch costs warmup, not correctness.
    """
    meta = runstore.read_meta(run_id)
    if meta.get("method") == "optimize":
        return {k: v for k, v in runstore.load_estimate(run_id).items() if k != "lp__"}
    d = draws.load(run_id)
    inits: dict = {}
    for name, arr in d.params.items():
        mean = np.asarray(arr).mean(axis=(0, 1))
        inits[name] = float(mean) if mean.ndim == 0 else mean.tolist()
    return inits
