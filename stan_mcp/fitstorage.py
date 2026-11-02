"""Persist completed CmdStan fits as durable CSVs, estimates and typed sidecars."""

import json
from pathlib import Path
from typing import Any

from stan_mcp import draws, runstore
from stan_mcp.errors import RunError


def read_fitted_csvs(csvs: list[str]) -> Any:
    """Load fitted parameters, retaining the fixed-parameter sampling law.

    CmdStanPy 1.3's from_csv reconstructs fixed_param CSVs with adaptation
    enabled. Its zero-warmup validation then fails. Keep that compatibility
    adapter here, and let CmdStanMCMC validate every supplied chain normally.
    """
    import cmdstanpy
    from cmdstanpy.cmdstan_args import CmdStanArgs, SamplerArgs
    from cmdstanpy.stanfit.runset import RunSet
    from cmdstanpy.utils.stancsv import parse_config

    comments = []
    with Path(csvs[0]).open("rb") as stream:
        for line in stream:
            if line.strip() and not line.startswith(b"#"):
                break
            comments.append(line)
    config = parse_config(comments)
    if config.get("method") != "sample" or config.get("algorithm") != "fixed_param":
        return cmdstanpy.from_csv(csvs)
    sampler = SamplerArgs(
        iter_sampling=int(config["num_samples"]),
        iter_warmup=int(config["num_warmup"]),
        thin=int(config["thin"]),
        save_warmup=bool(config["save_warmup"]),
        fixed_param=True,
        adapt_engaged=False,
    )
    args = CmdStanArgs(
        model_name=str(config["model"]),
        model_exe=str(config["model"]),
        chain_ids=list(range(1, len(csvs) + 1)),
        method_args=sampler,
    )
    runset = RunSet(args=args, chains=len(csvs))
    # The same RunSet attachment used by CmdStanPy's from_csv implementation.
    runset._csv_files = list(csvs)
    for index in range(len(csvs)):
        runset._set_retcode(index, 0)
    fit = cmdstanpy.CmdStanMCMC(runset)
    fit.draws()
    return fit


def _method_diagnostics(method: str, fit: Any) -> dict[str, Any]:
    """Method-specific health scalars pulled from the fit.

    For variational/pathfinder/laplace these are the ONLY diagnostics there are.
    Not because arviz cannot convert those fits — nothing here goes through arviz
    any more — but because R-hat, ESS, divergences and E-BFMI all describe HMC
    chains, and these three methods produce neither chains nor the sampler stats
    those statistics are computed from. That is why a failure here is recorded
    rather than swallowed: there is no other diagnostic to fall back on.
    """
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


def _csv_columns(fit: Any) -> tuple[list[str], str]:
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
    except AttributeError, KeyError, TypeError:
        return list(fit.column_names), "column_names"


def _draw_row_offset(method: str, run_id: str, fit: Any = None) -> int:
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

    External CSV inputs have no source run. Use the parsed previous fit's
    metadata and public draw counts instead. Its saved warmup or intermediate
    optimization rows also appear in the generated CSV and must be skipped.
    """
    if method == "variational":
        return 1
    if method == "generate_quantities":
        meta = runstore.read_meta(run_id)
        args = meta.get("method_args", {})
        if meta.get("input_method") == "variational":
            return 1
        if "csv_files" in args:
            previous = fit.previous_fit
            config = previous.metadata.cmdstan_config
            source_method = config["method"]
            if source_method == "variational":
                return 1
            if source_method == "sample" and config.get("save_warmup"):
                return previous.num_draws_warmup
            if source_method == "optimize" and config.get("save_iterations"):
                return len(previous.optimized_iterations_np) - 1
            return 0
        from_run_id = args.get("from_run_id")
        if from_run_id and runstore.read_meta(from_run_id).get("method") == "variational":
            return 1
    return 0


def _sidecar_from_fit(fit: Any, method: str, run_id: str) -> dict[str, Any]:
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

    params: dict[str, dict[str, Any]] = {}
    variables: dict[str, dict[str, Any]] = {}

    def describe(variable: Any) -> dict[str, Any]:
        return {
            "type": getattr(getattr(variable, "type", None), "name", "SCALAR").lower(),
            "dims": list(variable.dimensions),
            "contents": [describe(child) for child in getattr(variable, "contents", [])],
        }

    def leaves(
        variable: Any,
        key: str,
        prefixes: list[tuple[str, str]],
        outer_dims: tuple[int, ...] = (),
    ) -> None:
        shape = tuple(variable.dimensions)
        # Array indices belong before a tuple field or complex component.
        expanded = [
            (stan_label, column + cmdstan_name[len(label) :])
            for label, column in prefixes
            for stan_label, cmdstan_name in draws.element_labels(label, shape)
        ]
        kind = getattr(getattr(variable, "type", None), "name", "SCALAR")
        if kind == "TUPLE":
            for i, child in enumerate(variable.contents, 1):
                leaves(
                    child,
                    f"{key}:{i}",
                    [(f"{label}:{i}", f"{column}:{i}") for label, column in expanded],
                    outer_dims + shape,
                )
            return
        for component in ("real", "imag") if kind == "COMPLEX" else (None,):
            suffix = f".{component}" if component else ""
            labels = [label + suffix for label, _ in expanded]
            columns = [column + suffix for _, column in expanded]
            spec: dict[str, Any] = {
                "dims": list(outer_dims + shape),
                "cols": [column_of(key.split(":")[0], column) for column in columns],
            }
            if suffix or ":" in key:
                spec.update(labels=labels, column_names=columns)
            params[key + suffix] = spec

    for name, variable in fit.metadata.stan_vars.items():
        variables[name] = describe(variable)
        leaves(variable, name, [(name, name)])

    stats = {name: {"dims": [], "cols": [i]} for i, name in enumerate(cols) if name.endswith("__")}
    return {
        "schema": draws.SCHEMA_VERSION,
        "header": cols,
        "variables": variables,
        "params": params,
        "sampler_stats": stats,
        "draw_row_offset": _draw_row_offset(method, run_id, fit),
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


def _ordered_draws_files(fit: Any, run_id: str) -> list[str]:
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

    def chain_key(name: str) -> tuple[int, int | str]:
        stem = name.rsplit(".", 1)[0]
        tail = stem.rsplit("_", 1)[-1]
        return (0, int(tail)) if tail.isdigit() else (1, stem)

    return sorted((n for n in persisted if _is_draws_csv(n)), key=chain_key)


def persist_run(run_id: str, method: str, fit: Any) -> None:
    # persist raw cmdstan CSVs (source of truth; enables generate_quantities reuse).
    # Prefer the live output dir we route every method into — robust across fit
    # types (CmdStanPathfinder / CmdStanLaplace expose no .runset).
    try:
        live = runstore.live_dir(run_id)
        csvs = sorted(str(p) for p in live.glob("*.csv")) if live.is_dir() else []
        if not csvs and hasattr(fit, "runset"):
            csvs = list(fit.runset.csv_files)
        runstore.persist_csvs(run_id, csvs)
    except Exception as exc:
        meta = runstore.read_meta(run_id)
        meta["csv_warning"] = f"csv persist failed: {exc}"
        runstore.write_meta(run_id, meta)
        # Keep every original, including chains not yet copied. A partial
        # durable copy cannot support a successful run or justify cleanup.
        raise RunError(f"CSV persistence failed for run {run_id!r}: {exc}") from exc

    if method == "optimize":
        from stan_mcp import density

        estimate = dict(fit.optimized_params_dict)
        runstore.persist_estimate(run_id, estimate)
        values = {k: v for k, v in estimate.items() if not k.endswith("__")}
        runstore.persist_initialization(
            run_id, density.constrained_values(list(values), values.values())
        )
        if runstore.csv_paths(run_id):
            runstore.clear_live(run_id)
        return

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
