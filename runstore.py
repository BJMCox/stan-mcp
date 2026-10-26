"""Run store: directory layout and persistence under STAN_MCP_HOME/runs/<run-id>/.

Persists, per run:

- `model.stan` / `data.json` / `inits.json` / `source_manifest.json` — resolved
  inputs and compiler settings, for reproduction.
- `csv/` — the raw CmdStan draws CSVs. The ONLY durable record of a run's
  draws, and what `generate_quantities` reuses as `previous_fit`.
- `draws_schema.json` — the draws sidecar (each Stan variable's dims and its
  row-major CSV column indices, plus chain order); draws-producing methods
  only, written by `diagnostics.persist_run` and read by `draws.py`. A few
  hundred bytes to a few KB, holding no draws itself.
- `estimate.json` / `initialization.json` — the flattened point estimate and
  shaped parameter initialization, `optimize` only (alongside its raw CSV).
- `input_csv/` — owned fitted-parameter snapshots for GQ and Laplace mode reuse.
- `live/` — where CmdStan streams output during the run; deleted once `csv/`
  holds the durable copy.
- `meta.json` — model_id, method, method_args, status, created_at/finished_at/
  duration_s timestamps, stderr on failure. `list_runs` reads this across run
  dirs; failed runs are persisted with status="failed" and captured stderr.

There is no `inference.nc`: the arviz conversion that produced it is gone, and
the sidecar above replaces it. See `idata_path`.

Implemented during the build (see design "Run store").
"""

import fcntl
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import config
from errors import RunError, ValidationError


def new_run_id(method: str, seq: int) -> str:
    return f"{method}-{seq:04d}"


def reserve_run(method: str) -> tuple[str, int]:
    """Reserve an exclusive run directory and globally ordered sequence."""
    runs = config.runs_dir()
    with (runs / ".allocation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        counter = runs / ".sequence"
        try:
            high_water = int(counter.read_text())
            if high_water < 0:
                raise ValueError("negative sequence")
        except FileNotFoundError:
            high_water = max_seq()
        except ValueError as exc:
            raise RunError("run allocation counter is unreadable") from exc
        seq = high_water + 1
        while True:
            # Publish before mkdir: even a crashed or failed submission consumes
            # its ID, and pruning can never make an old ID available again.
            with tempfile.NamedTemporaryFile(mode="w", dir=runs, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    stream.write(f"{seq}\n")
                    stream.flush()
                    temporary.replace(counter)
                finally:
                    temporary.unlink(missing_ok=True)
            run_id = new_run_id(method, seq)
            try:
                run_dir(run_id).mkdir(exist_ok=False)
                return run_id, seq
            except FileExistsError:
                seq += 1


def process_owner() -> dict[str, int | float]:
    import psutil

    process = psutil.Process(os.getpid())
    return {"pid": process.pid, "create_time": process.create_time()}


def reconcile_abandoned() -> None:
    """Finalize crashed owners; another live server retains its own runs."""
    import psutil

    for meta in list_runs(limit=10**9):
        if meta.get("status") not in {"running", "cancelling"}:
            continue
        owner = meta.get("owner")
        if owner:
            try:
                process = psutil.Process(owner["pid"])
                if process.create_time() == owner["create_time"] and process.is_running():
                    continue
            except (psutil.NoSuchProcess, KeyError):
                pass
            except psutil.AccessDenied:
                continue  # Unable to prove that this owner died.
        meta["status"] = "failed"
        meta["finished_at"] = datetime.now(UTC).isoformat()
        meta["stderr"] = (
            "Run abandoned: owning server exited; surviving subprocesses are not adopted."
            if owner
            else "Run abandoned: no verifiable owning server was recorded."
        )
        write_meta(meta["run_id"], meta)


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
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(m, stream, indent=2)
            stream.flush()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


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
        except (FileNotFoundError, json.JSONDecodeError):
            continue  # Concurrent pruning or corrupt metadata.
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
    """Highest reserved or persisted sequence on disk (0 if none)."""
    best = 0
    runs = config.runs_dir()
    for d in runs.iterdir():
        suffix = d.name.rsplit("-", 1)[-1]
        if d.is_dir() and suffix.isdigit():
            best = max(best, int(suffix))
        mp = d / "meta.json"
        if not mp.is_file():
            continue
        try:
            seq = json.loads(mp.read_text()).get("seq", 0)
        except (FileNotFoundError, json.JSONDecodeError):
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


def persist_csvs(run_id: str, csv_files: list[str]) -> None:
    dest = run_dir(run_id, create=True) / "csv"
    dest.mkdir(parents=True, exist_ok=True)
    for f in csv_files:
        shutil.copy(f, dest / Path(f).name)


def persist_input_csvs(run_id: str, csv_files: str | list[str]) -> list[str]:
    """Snapshot external fitted-parameter CSVs in caller-supplied chain order."""
    paths = [csv_files] if isinstance(csv_files, str) else csv_files
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise ValidationError("csv_files must be a CSV path or a nonempty list of CSV paths")
    sources = [Path(p).expanduser().resolve() for p in paths]
    for source in sources:
        if source.suffix != ".csv" or not source.is_file():
            raise ValidationError(f"fitted-parameter CSV file not found: {source}")
    dest = run_dir(run_id, create=True) / "input_csv"
    staging = Path(tempfile.mkdtemp(prefix=".input_csv-", dir=dest.parent))
    try:
        for i, source in enumerate(sources):
            # CmdStanPy sorts CSV paths. Numbered names retain chain order.
            shutil.copyfile(source, staging / f"{i:08d}.csv")
        staging.replace(dest)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return [str((dest / f"{i:08d}.csv").resolve()) for i in range(len(sources))]


def persist_initialization(run_id: str, initialization: dict) -> None:
    (run_dir(run_id, create=True) / "initialization.json").write_text(
        json.dumps(initialization, indent=2)
    )


def load_initialization(run_id: str) -> dict | None:
    path = run_dir(run_id) / "initialization.json"
    return json.loads(path.read_text()) if path.is_file() else None


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


def persist_inputs(
    run_id: str,
    source: str,
    data: object,
    inits: dict | list | float | int | str | None = None,
    source_manifest: dict | None = None,
) -> None:
    """Persist the exact model source and data so a run can be reproduced."""
    d = run_dir(run_id, create=True)
    (d / "model.stan").write_text(source)
    if data is not None:
        (d / "data.json").write_text(json.dumps(data, sort_keys=True, indent=2))
    if inits is not None:
        (d / "inits.json").write_text(json.dumps(inits, sort_keys=True, indent=2))
    if source_manifest is not None:
        (d / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2))


def load_inputs(run_id: str) -> dict:
    """Return the persisted source and data for a run (for reproduction)."""
    d = run_dir(run_id)
    if not d.is_dir():
        # Otherwise a typo'd run_id looks like a real run with nothing stored.
        raise RunError(f"unknown run_id {run_id!r}")
    src = d / "model.stan"
    dat = d / "data.json"
    init = d / "inits.json"
    manifest = d / "source_manifest.json"
    return {
        "source": src.read_text() if src.is_file() else None,
        "data": json.loads(dat.read_text()) if dat.is_file() else None,
        "inits": json.loads(init.read_text()) if init.is_file() else None,
        "source_manifest": json.loads(manifest.read_text()) if manifest.is_file() else None,
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
        if m.get("status") in {"running", "cancelling"}:
            continue  # never delete a run still in flight (rmtree would race its writes)
        seen.add(rid)
        d = run_dir(rid)
        if d.is_dir():
            shutil.rmtree(d)
            removed.append(rid)
    return removed
