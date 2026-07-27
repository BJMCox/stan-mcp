"""Run store: directory layout and persistence under STAN_MCP_HOME/runs/<run-id>/.

Persists, per run:

- `model.stan` / `data.json` — the exact inputs, for reproduction.
- `csv/` — the raw CmdStan draws CSVs. The ONLY durable record of a run's
  draws, and what `generate_quantities` reuses as `previous_fit`.
- `draws_schema.json` — the draws sidecar (each Stan variable's dims and its
  row-major CSV column indices, plus chain order); draws-producing methods
  only, written by `diagnostics.persist_run` and read by `draws.py`. A few
  hundred bytes to a few KB, holding no draws itself.
- `estimate.json` — the stored point estimate, `optimize` only (instead of the
  two above).
- `live/` — where CmdStan streams output during the run; deleted once `csv/`
  holds the durable copy.
- `meta.json` — model_id, method, method_args, status, created_at/finished_at/
  duration_s timestamps, stderr on failure. `list_runs` reads this across run
  dirs; failed runs are persisted with status="failed" and captured stderr.

There is no `inference.nc`: the arviz conversion that produced it is gone, and
the sidecar above replaces it. See `idata_path`.

Implemented during the build (see design "Run store").
"""

import json
import shutil
from datetime import UTC
from pathlib import Path

import config
from errors import RunError


def new_run_id(method: str, seq: int) -> str:
    return f"{method}-{seq:04d}"


def run_dir(run_id: str, create: bool = False) -> Path:
    d = config.runs_dir() / run_id
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _meta_path(run_id: str, create: bool = False) -> Path:
    return run_dir(run_id, create=create) / "meta.json"


def write_meta(run_id: str, meta: dict) -> None:
    m = dict(meta)
    m["run_id"] = run_id
    path = _meta_path(run_id, create=True)
    # Write atomically: a concurrent reader must never observe a truncated file.
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(m, indent=2))
    tmp.replace(path)


def read_meta(run_id: str) -> dict:
    """Read a run's meta.json, raising RunError (not a raw FileNotFoundError
    naming an internal path) when the run_id is unknown — every tool funnels
    through here, so the mapping belongs in one place."""
    path = _meta_path(run_id)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RunError(f"unknown run_id {run_id!r}") from exc
    except json.JSONDecodeError as exc:
        raise RunError(f"run {run_id!r} has an unreadable meta.json: {exc}") from exc


def list_runs(
    model_id: str | None = None,
    limit: int = 50,
    status: str | None = None,
    method: str | None = None,
) -> list[dict]:
    rows = []
    for d in config.runs_dir().iterdir():
        mp = d / "meta.json"
        if not mp.is_file():
            continue
        try:
            meta = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue  # skip a run whose meta.json is mid-write / corrupt
        if model_id is not None and meta.get("model_id") != model_id:
            continue
        if status is not None and meta.get("status") != status:
            continue
        if method is not None and meta.get("method") != method:
            continue
        # A failed run's stderr is a full traceback; returning 50 of them would
        # swamp the caller's context. Keep the first line (as get_run_status
        # does) and point at get_run_status/await_run for the rest.
        if meta.get("stderr"):
            meta["stderr"] = str(meta["stderr"]).splitlines()[0]
        rows.append(meta)
    rows.sort(key=lambda m: m.get("seq", 0), reverse=True)
    return rows[:limit]


def max_seq() -> int:
    """Highest run sequence number persisted on disk (0 if none).

    Lets a fresh RunManager continue the counter instead of restarting at 1
    and overwriting runs from a prior process.
    """
    best = 0
    runs = config.runs_dir()
    for d in runs.iterdir():
        mp = d / "meta.json"
        if not mp.is_file():
            continue
        try:
            seq = json.loads(mp.read_text()).get("seq", 0)
        except json.JSONDecodeError:
            continue
        if isinstance(seq, int) and seq > best:
            best = seq
    return best


def idata_path(run_id: str) -> Path:
    """Where the retired arviz `inference.nc` used to be written.

    Nothing writes or reads this any more. It survives ONLY as the regression
    guard that persist time never reintroduces the netcdf write: one test asserts
    this path does not exist after `persist_run`. Deleting it would remove the
    ability to make that assertion, which is why it is not dead code.
    """
    return run_dir(run_id) / "inference.nc"


def live_dir(run_id: str, create: bool = False) -> Path:
    """Directory CmdStan streams live CSV output into during a run.

    Routing output here (a) gives each run's chain subprocesses a command line
    containing this path, so cancel can target only that run's processes, and
    (b) lets get_run_progress count draws written so far."""
    d = run_dir(run_id, create=create) / "live"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def persist_csvs(run_id: str, csv_files) -> None:
    dest = run_dir(run_id, create=True) / "csv"
    dest.mkdir(parents=True, exist_ok=True)
    for f in csv_files:
        shutil.copy(f, dest / Path(f).name)


def clear_live(run_id: str) -> None:
    """Drop the live/ streaming dir once its CSVs are safely copied into csv/.

    Without this every finished run keeps two full copies of its draws on
    disk, which for a large sample run is gigabytes of pure duplication."""
    live = live_dir(run_id)
    if live.is_dir():
        shutil.rmtree(live, ignore_errors=True)


def csv_paths(run_id: str) -> list[str]:
    dest = run_dir(run_id) / "csv"
    if not dest.is_dir():
        return []
    return sorted(str(p) for p in dest.glob("*.csv"))


def persist_estimate(run_id: str, estimate: dict) -> None:
    (run_dir(run_id, create=True) / "estimate.json").write_text(json.dumps(estimate, indent=2))


def load_estimate(run_id: str) -> dict:
    return json.loads((run_dir(run_id) / "estimate.json").read_text())


def persist_inputs(run_id: str, source: str, data) -> None:
    """Persist the exact model source and data so a run can be reproduced."""
    d = run_dir(run_id, create=True)
    (d / "model.stan").write_text(source)
    if data is not None:
        (d / "data.json").write_text(json.dumps(data, sort_keys=True, indent=2))


def load_inputs(run_id: str) -> dict:
    """Return the persisted source and data for a run (for reproduction)."""
    d = run_dir(run_id)
    if not d.is_dir():
        # Otherwise a typo'd run_id looks like a real run with nothing stored.
        raise RunError(f"unknown run_id {run_id!r}")
    src = d / "model.stan"
    dat = d / "data.json"
    return {
        "source": src.read_text() if src.is_file() else None,
        "data": json.loads(dat.read_text()) if dat.is_file() else None,
    }


def prune_runs(keep: int | None = None, older_than_days: float | None = None) -> list[str]:
    """Delete stored runs, returning the run_ids removed.

    keep: retain only the newest `keep` runs (by seq). older_than_days: remove
    runs whose finished_at (or created_at) is older than the cutoff. At least
    one criterion must be given.
    """
    if keep is None and older_than_days is None:
        raise ValueError("prune_runs requires keep or older_than_days")
    if keep is not None and keep < 1:
        raise ValueError("keep must be >= 1; refusing to delete every run")
    rows = list_runs(limit=10**9)  # newest first
    victims: list[dict] = []
    if keep is not None:
        victims.extend(rows[keep:])
    if older_than_days is not None:
        from datetime import datetime

        cutoff = datetime.now(UTC).timestamp() - older_than_days * 86400
        for m in rows:
            stamp = m.get("finished_at") or m.get("created_at")
            if not stamp:
                continue
            try:
                ts = datetime.fromisoformat(stamp).timestamp()
            except ValueError:
                continue
            if ts < cutoff:
                victims.append(m)
    seen: set[str] = set()
    removed: list[str] = []
    for m in victims:
        rid = m.get("run_id")
        if not rid or rid in seen:
            continue
        if m.get("status") == "running":
            continue  # never delete a run still in flight (rmtree would race its writes)
        seen.add(rid)
        d = run_dir(rid)
        if d.is_dir():
            shutil.rmtree(d)
            removed.append(rid)
    return removed
