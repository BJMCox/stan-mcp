"""Typed access to a run's persisted draws.

Reads the per-run sidecar written by `diagnostics.persist_run` and returns draws as
correctly-shaped arrays keyed by Stan variable name.

Deliberately depends on numpy, stdlib and `runstore` only:

- no pandas, because `pd.read_csv` hands back flat, string-keyed `z.1 ... z.150`
  columns with the declared shape discarded, leaving every caller to reconstruct it;
- no cmdstanpy, because `from_csv` parses CmdStan's config/adaptation comment block
  and rejects a bare header, which would force every offline fixture to mimic
  CmdStan's file format;
- no arviz, because `from_cmdstanpy` cannot represent variational/pathfinder/laplace
  fits at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import runstore
from errors import RunError

SCHEMA_VERSION = 3


@dataclass(frozen=True)
class Draws:
    """Draws as `(chain, draw, *var_dims)` arrays keyed by Stan variable name.

    The approximate methods have no chain structure and come back with one chain,
    rather than a special case for every caller to remember.

    Carries its own `run_id` so a helper handed only a `Draws` can still name the
    run in an error. Without it `matrix` was the one failure in this module whose
    message identified nothing.
    """

    run_id: str
    params: dict[str, np.ndarray]
    sampler_stats: dict[str, np.ndarray]
    n_chain: int
    n_draw: int
    truncated_to: int | None = None
    header_warning: str | None = None


def schema_path(run_id: str) -> Path:
    return runstore.run_dir(run_id) / "draws_schema.json"


def schema(run_id: str) -> dict[str, Any]:
    """The validated sidecar, without reading any draws.

    Separate from `load` so a summary sourced from stansummary can get names and
    dims without parsing megabytes of CSV that stansummary has already read.

    Validated means: present, JSON-parseable, the version this build reads, and
    carrying every key its readers then index WITHOUT a `.get` — `files`, `params`
    and `sampler_stats`, each of the latter two's specs with `dims` and `cols`.
    Checking only the version left a malformed sidecar to raise a bare `KeyError`
    from inside `load` or `csv_files`, naming a dict key rather than the run.
    """
    p = schema_path(run_id)
    if not p.exists():
        raise RunError(
            f"run {run_id!r} has no draws_schema.json, so its draws cannot be "
            "typed; it predates the typed draws layer — re-run the model"
        )
    try:
        sc = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise RunError(f"run {run_id!r} has an unreadable draws_schema.json: {exc}") from exc
    if not isinstance(sc, dict):
        raise RunError(
            f"run {run_id!r} has a draws_schema.json holding {type(sc).__name__}, "
            "not an object — re-run the model"
        )
    got = sc.get("schema")
    if got not in (2, SCHEMA_VERSION):
        raise RunError(
            f"run {run_id!r} has draws_schema version {got!r}, but this build reads "
            f"version {SCHEMA_VERSION} — re-run the model"
        )
    for key, kind in (("files", list), ("params", dict), ("sampler_stats", dict)):
        if key not in sc:
            raise RunError(
                f"run {run_id!r} has a malformed draws_schema.json: no {key!r} — re-run the model"
            )
        if not isinstance(sc[key], kind):
            raise RunError(
                f"run {run_id!r} has a malformed draws_schema.json: {key!r} is a "
                f"{type(sc[key]).__name__}, expected a {kind.__name__} — re-run the model"
            )
    if got == SCHEMA_VERSION and not (
        isinstance(sc.get("header"), list)
        and sc["header"]
        and all(isinstance(name, str) for name in sc["header"])
    ):
        raise RunError(f"run {run_id!r} has a malformed draws_schema.json: no valid header")
    if got == 2:
        sc["header_warning"] = (
            "legacy draws schema has no recorded CSV header identity; column labels are unverified"
        )
    for group in ("params", "sampler_stats"):
        for name, spec in sc[group].items():
            missing = [k for k in ("dims", "cols") if not isinstance(spec, dict) or k not in spec]
            if missing:
                raise RunError(
                    f"run {run_id!r} has a malformed draws_schema.json: entry "
                    f"{group}.{name} is missing {', '.join(missing)} — re-run the model"
                )
    return sc


def csv_files(run_id: str) -> list[str]:
    """The run's draws CSVs, in the order the run produced them.

    By record, not by glob: `laplace` writes a mode CSV beside its draws CSV, and
    chain order must not depend on a filename sort (cmdstanpy's `from_csv` sorts by
    filename and thereby permutes chains relative to the original run).
    """
    d = runstore.run_dir(run_id) / "csv"
    sc = schema(run_id)
    paths = [d / name for name in sc["files"]]
    if "header" in sc:
        for path in paths:
            if not path.exists():
                raise RunError(f"run {run_id!r} is missing its draws CSV {path.name!r}")
            with path.open() as stream:
                _validate_header(path, _csv_header(path, stream), sc["header"])
    return [str(path) for path in paths]


def _csv_header(path: Path, stream: Iterable[str]) -> list[str]:
    for line in stream:
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip().split(",")
    raise RunError(f"draws CSV {path.name!r} has no header or draws")


def _validate_header(path: Path, header: list[str], expected: list[str] | None) -> None:
    if expected is not None and header != expected:
        raise RunError(
            f"draws CSV {path.name!r} header differs from draws_schema; "
            "the sidecar is stale or its CSV was replaced"
        )


def _read_csv(path: Path, expected_header: list[str] | None = None) -> np.ndarray:
    """`(draw, column)` floats from one CmdStan CSV."""
    with path.open() as stream:
        header = _csv_header(path, stream)
        _validate_header(path, header, expected_header)
        try:
            rows = np.loadtxt(stream, delimiter=",", comments="#", ndmin=2)
        except ValueError as exc:
            raise RunError(f"draws CSV {path.name!r} has invalid numeric rows: {exc}") from exc
    if rows.size == 0:
        raise RunError(f"draws CSV {path.name!r} has no draws")
    if rows.shape[1] != len(header):
        raise RunError(f"draws CSV {path.name!r} has numeric rows inconsistent with its header")
    return rows


def load(run_id: str) -> Draws:
    sc = schema(run_id)
    # variational's first data row is cmdstanpy's own mean-field mean, not a
    # draw (cmdstanpy stanfit/vb.py: draws_np[0] is the mean, draws_np[1:] the
    # sample) -- every other method's rows are all draws, so this is 0 for
    # them. Recorded at persist time (diagnostics._sidecar_from_fit) since
    # this reader must stay method-agnostic.
    offset = sc.get("draw_row_offset", 0)
    csvdir = runstore.run_dir(run_id) / "csv"
    per_chain = []
    for name in sc["files"]:
        p = csvdir / name
        if not p.exists():
            raise RunError(f"run {run_id!r} is missing its draws CSV {name!r}")
        rows = _read_csv(p, sc.get("header"))
        if offset:
            rows = rows[offset:]
            if rows.shape[0] == 0:
                raise RunError(
                    f"run {run_id!r}'s draws CSV {name!r} has no draws left after "
                    f"skipping its leading {offset} summary row(s)"
                )
        per_chain.append(rows)

    lengths = [a.shape[0] for a in per_chain]
    n_draw = min(lengths)
    # A (chain, draw, ...) array cannot represent ragged chains. Truncate to the
    # shortest and RECORD it: silently dropping the excess is the failure mode
    # this layer exists to remove.
    #
    # NOT a cancelled run, despite how this once read. `persist_run` only runs
    # when `_runner` RETURNS (sample.py), so a cancelled or killed run never gets
    # a sidecar at all and can never reach this code — `schema()` fails first with
    # "has no draws_schema.json". What this actually defends against is a CSV set
    # that is malformed or was only partly written: a truncated copy, a
    # half-populated `csv/`, a hand-built fixture. Worth keeping for that alone,
    # since the alternative is a raw numpy broadcast error from `np.stack`.
    #
    # Only `_summary_from_draws` surfaces this as `chains_truncated_to`; the
    # `sample` path never sees it, because `stansummary` itself rejects ragged
    # input ("Error during processing. Error: chain 2, missing or extra rows.",
    # exit 1) and so fails loudly instead. See docs/usage.md.
    truncated_to = n_draw if len(set(lengths)) > 1 else None
    arr = np.stack([a[:n_draw] for a in per_chain])  # (chain, draw, column)

    highest = max(
        (
            max(spec["cols"])
            for group in ("params", "sampler_stats")
            for spec in sc[group].values()
            if spec["cols"]
        ),
        default=-1,
    )
    if highest >= arr.shape[2]:
        raise RunError(
            f"run {run_id!r}: draws_schema references column {highest} but its CSVs "
            f"have {arr.shape[2]} columns — the sidecar is stale, re-run the model"
        )

    def build(group: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, spec in sc[group].items():
            block = arr[:, :, spec["cols"]]
            # cols are recorded in row-major fill order, so numpy's default
            # reshape is correct and no order= argument is needed anywhere.
            out[name] = block.reshape(arr.shape[0], n_draw, *tuple(spec["dims"]))
        return out

    return Draws(
        run_id=run_id,
        params=build("params"),
        sampler_stats=build("sampler_stats"),
        n_chain=arr.shape[0],
        n_draw=n_draw,
        truncated_to=truncated_to,
        header_warning=sc.get("header_warning"),
    )


def element_labels(name: str, dims: tuple[int, ...]) -> Iterator[tuple[str, str]]:
    """`(stan_label, cmdstan_name)` per scalar element, row-major.

    Both are built from the same 1-based index tuple, so a statistic looked up by
    CmdStan's name (`L.2.1`) can be reported under Stan's (`L[2,1]`) without the
    two drifting. Stan is 1-based, so arviz's `z[0]` would name an element that
    does not exist in the user's model.
    """
    if not dims:
        yield name, name
        return
    for pos in range(int(np.prod(dims))):
        one = [int(i) + 1 for i in np.unravel_index(pos, dims)]
        yield f"{name}[{','.join(map(str, one))}]", name + "." + ".".join(map(str, one))


def elements(name: str, arr: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
    """`(stan_label, (chain, draw) values)` per scalar element, row-major so it
    agrees with `element_labels`."""
    if arr.ndim == 2:
        yield name, arr
        return
    flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
    for pos, (label, _) in enumerate(element_labels(name, arr.shape[2:])):
        yield label, flat[:, :, pos]


def spec_element_labels(name: str, spec: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Scalar labels and exact header identities, including tuple/complex leaves."""
    if "labels" in spec:
        yield from zip(spec["labels"], spec["column_names"], strict=True)
    else:
        yield from element_labels(name, tuple(spec["dims"]))


def matrix(d: Draws) -> tuple[np.ndarray, list[str]]:
    """`(draws, elements)` design matrix over every parameter element, plus labels.

    Parameters only — never sampler stats.
    """
    sc = schema(d.run_id)
    cols, names = [], []
    for name, arr in d.params.items():
        flat = arr.reshape(d.n_chain * d.n_draw, -1)
        for pos, (label, _) in enumerate(spec_element_labels(name, sc["params"][name])):
            cols.append(flat[:, pos])
            names.append(label)
    if not cols:
        raise RunError(f"run {d.run_id!r} has no parameters to build a draw matrix from")
    return np.column_stack(cols), names


def parameter_matrix(d: Draws, names: list[str]) -> np.ndarray:
    """Constrained parameter columns in an external model's exact CSV order.

    Resolving against recorded leaf identities also covers complex components
    and nested tuple fields without inferring structure from dotted names.
    """
    sc = schema(d.run_id)
    columns: dict[str, np.ndarray] = {}
    for name, arr in d.params.items():
        flat = arr.reshape(d.n_chain * d.n_draw, -1)
        for pos, (_, csv_name) in enumerate(spec_element_labels(name, sc["params"][name])):
            columns[csv_name] = flat[:, pos]
    missing = [name for name in names if name not in columns]
    if missing:
        raise RunError(
            f"run {d.run_id!r} does not contain constrained model parameters: {', '.join(missing)}"
        )
    if not names:
        return np.empty((d.n_chain * d.n_draw, 0))
    return np.column_stack([columns[name] for name in names])
