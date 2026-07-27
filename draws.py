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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import runstore
from errors import RunError

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Draws:
    """Draws as `(chain, draw, *var_dims)` arrays keyed by Stan variable name.

    The approximate methods have no chain structure and come back with one chain,
    rather than a special case for every caller to remember.
    """

    params: dict[str, np.ndarray]
    sampler_stats: dict[str, np.ndarray]
    n_chain: int
    n_draw: int
    truncated_to: int | None = None


def schema_path(run_id: str) -> Path:
    return runstore.run_dir(run_id) / "draws_schema.json"


def schema(run_id: str) -> dict:
    """The validated sidecar, without reading any draws.

    Separate from `load` so a summary sourced from stansummary can get names and
    dims without parsing megabytes of CSV that stansummary has already read.
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
    got = sc.get("schema")
    if got != SCHEMA_VERSION:
        raise RunError(
            f"run {run_id!r} has draws_schema version {got!r}, but this build reads "
            f"version {SCHEMA_VERSION} — re-run the model"
        )
    return sc


def csv_files(run_id: str) -> list[str]:
    """The run's draws CSVs, in the order the run produced them.

    By record, not by glob: `laplace` writes a mode CSV beside its draws CSV, and
    chain order must not depend on a filename sort (cmdstanpy's `from_csv` sorts by
    filename and thereby permutes chains relative to the original run).
    """
    d = runstore.run_dir(run_id) / "csv"
    return [str(d / name) for name in schema(run_id)["files"]]


def _read_csv(path: Path) -> np.ndarray:
    """`(draw, column)` floats from one CmdStan CSV."""
    body = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    if len(body) < 2:
        raise RunError(f"draws CSV {path.name!r} has no draws")
    # body[0] is the column header: CmdStan always writes it as the first
    # non-comment line.
    return np.asarray([[float(v) for v in ln.split(",")] for ln in body[1:]], dtype=float)


def load(run_id: str) -> Draws:
    sc = schema(run_id)
    csvdir = runstore.run_dir(run_id) / "csv"
    per_chain = []
    for name in sc["files"]:
        p = csvdir / name
        if not p.exists():
            raise RunError(f"run {run_id!r} is missing its draws CSV {name!r}")
        per_chain.append(_read_csv(p))

    lengths = [a.shape[0] for a in per_chain]
    n_draw = min(lengths)
    # A (chain, draw, ...) array cannot represent ragged chains, which a cancelled
    # run produces. Truncate to the shortest and RECORD it: silently dropping the
    # excess is the failure mode this layer exists to remove.
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
        params=build("params"),
        sampler_stats=build("sampler_stats"),
        n_chain=arr.shape[0],
        n_draw=n_draw,
        truncated_to=truncated_to,
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


def matrix(d: Draws) -> tuple[np.ndarray, list[str]]:
    """`(draws, elements)` design matrix over every parameter element, plus labels.

    Parameters only — never sampler stats.
    """
    cols, names = [], []
    for name, arr in d.params.items():
        for label, values in elements(name, arr):
            cols.append(values.reshape(-1))
            names.append(label)
    if not cols:
        raise RunError("run has no parameters to build a draw matrix from")
    return np.column_stack(cols), names
