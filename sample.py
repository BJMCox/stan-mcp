"""Background run manager for all CmdStanPy inference methods."""

import asyncio
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime

import compile as _compile
import config
import diagnostics
import methods
import runstore
from errors import RunError


def _now() -> str:
    return datetime.now(UTC).isoformat()


# Bound on the failure text get_run_status returns. The stored stderr is a full
# traceback (kilobytes), but its FIRST LINE alone is routinely useless — CmdStan
# leads with "Error during sampling:" and puts the actual cause (the per-chain
# retcodes) further down. list_runs still shows one line per row, since it is an
# index; this is the detail view, so it gets a bounded excerpt instead.
_ERROR_EXCERPT_CHARS = 800

# CmdStan writes draws at 6 significant figures by default, which is lossy enough
# that a constrained draw read back from the CSV misses its constraint: a simplex
# sums to 1 only to ~1e-8, exactly stan::math::simplex_free's tolerance, so
# BridgeStan rejects about 1.5% of draws (measured: 15/1000) and run_geometry can
# never unconstrain such a model. 17 is the digit count at which an IEEE-754 double
# round-trips exactly, so the error becomes machine epsilon (measured max |sum-1|
# 2.2e-16) and stays clear of the tolerance at ANY simplex dimension -- 10 would
# suffice for simplex[3] (1e-10) but the error grows with the number of components.
# Costs ~1.9x the CSV bytes (measured 76,631 -> 146,710 on a 2-chain 1000-draw run);
# override via method_args.sig_figs when a large run's disk footprint matters more.
_OUTPUT_SIG_FIGS = 17


def _error_excerpt(stderr: str) -> str:
    text = stderr.strip()
    if len(text) <= _ERROR_EXCERPT_CHARS:
        return text
    return text[:_ERROR_EXCERPT_CHARS].rstrip() + " […truncated; full text in meta.json]"


def _count_csv_draws(paths) -> int | None:
    """Total draws across CmdStan CSVs (None if there are none). CmdStan CSV:
    `#` comment lines, one header row, then one line per draw.

    Streams line by line: a long run's CSVs can be hundreds of MB and
    get_run_progress may be polled repeatedly, so they must never be slurped
    whole into memory."""
    total = 0
    found = False
    for csv in paths:
        found = True
        rows = 0
        try:
            with open(csv, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        rows += 1
        except OSError:
            continue
        total += max(rows - 1, 0)  # drop the header row
    return total if found else None


def _count_live_draws(run_id: str) -> int | None:
    """Draws written so far for a run: from live/ while it is in flight, and
    from the persisted csv/ copy once live/ has been cleaned up."""
    live = runstore.live_dir(run_id)
    if live.is_dir():
        count = _count_csv_draws(sorted(p for p in live.glob("*.csv") if "-opt-" not in p.name))
        if count is not None:
            return count
    return _count_csv_draws([p for p in runstore.csv_paths(run_id) if "-opt-" not in p])


def _kill_run_processes(run_id: str) -> list[int]:
    """SIGKILL the CmdStan subprocesses of `run_id`, identified by the run's live
    output path appearing in their command line. Returns the PIDs killed."""
    import os

    import psutil

    marker = str(runstore.live_dir(run_id))
    killed: list[int] = []
    try:
        me = psutil.Process(os.getpid())
    except psutil.Error:
        return killed
    for child in me.children(recursive=True):
        try:
            cmdline = child.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if any(marker in arg for arg in cmdline):
            try:
                child.kill()
                killed.append(child.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return killed


class _Handle:
    def __init__(self):
        self.event = threading.Event()
        self.aio_events: list[tuple] = []  # (loop, asyncio.Event)
        self.future: Future | None = None  # set after submit
        self.cancelled = False  # set by cancel_run so _execute relabels the run


class RunManager:
    def __init__(self, max_workers: int | None = None):
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers if max_workers is not None else config.max_workers()
        )
        self._handles: dict[str, _Handle] = {}
        # Continue the sequence past any runs persisted by a prior process so
        # run_ids never collide with (and overwrite) earlier persisted runs.
        self._seq = runstore.max_seq()
        self._lock = threading.Lock()

    # --- injectable seams (replaced in tests) ---
    def _compile(self, source, stanc_args):
        return _compile.cmdstan_model(source, stanc_args)

    def _runner(self, method, model, data, kwargs):
        if method == "generate_quantities":
            kwargs = dict(kwargs)
            from_run_id = kwargs.pop("from_run_id", None)
            if not from_run_id:
                raise RunError("generate_quantities requires method_args.from_run_id")
            csvs = runstore.csv_paths(from_run_id)
            if not csvs:
                raise RunError(f"run {from_run_id!r} has no persisted draws CSVs to reuse")
            return model.generate_quantities(data=data, previous_fit=csvs, **kwargs)
        if method == "variational":
            kwargs = dict(kwargs)
            # ADVI is approximate; don't hard-fail the run on the convergence
            # gate (draws are still returned and summarized from the CSVs).
            kwargs.setdefault("require_converged", False)
        if method == "sample":
            kwargs = dict(kwargs)
            # CmdStanPy would otherwise default parallel_chains to
            # min(cpu_count(), chains) per run, so `max_workers` concurrent
            # runs would oversubscribe the machine several times over.
            kwargs.setdefault("parallel_chains", config.max_parallel_chains())
            kwargs.setdefault("sig_figs", _OUTPUT_SIG_FIGS)
        cmd = methods.METHODS[method]["cmd"]
        return getattr(model, cmd)(data=data, **kwargs)

    def _persist(self, run_id, method, fit):
        # diagnostics.persist_run lands in Task 12; tests monkeypatch this seam.
        diagnostics.persist_run(run_id, method, fit)  # pyright: ignore[reportAttributeAccessIssue]

    @staticmethod
    def _stamp_finish(meta: dict) -> None:
        meta["finished_at"] = _now()
        started = meta.get("created_at")
        if started:
            try:
                delta = datetime.fromisoformat(meta["finished_at"]) - datetime.fromisoformat(
                    started
                )
                meta["duration_s"] = round(delta.total_seconds(), 3)
            except ValueError:
                pass

    # --- public API ---
    def start_run(
        self,
        stan,
        data,
        method="sample",
        seed=None,
        inits=None,
        method_args=None,
        inits_from=None,
    ) -> dict:
        if inits_from is not None and inits is None:
            # Warm-start at another run's posterior/estimate mean.
            inits = diagnostics.posterior_mean_inits(inits_from)
        kwargs = methods.build_kwargs(method, seed, inits, method_args or {})
        source = _compile.resolve_source(stan)
        data = _compile.canonical_data(data)  # resolve dict / data_id / path / json
        with self._lock:
            self._seq += 1
            run_id = runstore.new_run_id(method, self._seq)
        runstore.persist_inputs(run_id, source, data)
        runstore.write_meta(
            run_id,
            {
                "run_id": run_id,
                "method": method,
                "status": "running",
                "model_id": _compile.model_id(source, None),
                "seq": self._seq,
                "method_args": method_args or {},
                "seed": seed,
                "created_at": _now(),
            },
        )
        handle = _Handle()
        self._handles[run_id] = handle
        handle.future = self._pool.submit(
            self._execute, run_id, source, data, method, kwargs, handle
        )
        return {"run_id": run_id, "method": method, "status": "running"}

    def rerun(self, run_id) -> dict:
        """Start a fresh run from a prior run's persisted source, data, method,
        seed, and method_args. (inits are not persisted and are not carried over.)"""
        meta = runstore.read_meta(run_id)
        inputs = runstore.load_inputs(run_id)
        if inputs["source"] is None:
            raise RunError(f"run {run_id!r} has no persisted inputs to rerun")
        return self.start_run(
            inputs["source"],
            inputs["data"],
            method=meta.get("method", "sample"),
            seed=meta.get("seed"),
            method_args=meta.get("method_args"),
        )

    def _wake_waiters(self, handle) -> None:
        handle.event.set()
        for loop, aio_event in handle.aio_events:
            try:
                loop.call_soon_threadsafe(aio_event.set)
            except RuntimeError:
                pass

    def cancel_run(self, run_id) -> dict:
        """Cancel a run. Only queued (not-yet-started) runs can be stopped: once
        CmdStan is executing, its subprocess cannot be interrupted from here."""
        handle = self._handles.get(run_id)
        if handle is None:
            meta = runstore.read_meta(run_id)
            return {
                "run_id": run_id,
                "cancelled": False,
                "status": meta.get("status"),
                "reason": "run already finished",
            }
        if handle.future is not None and handle.future.cancel():
            # Still queued: it will never run, so finalize the meta ourselves.
            handle.cancelled = True
            meta = runstore.read_meta(run_id)
            meta["status"] = "cancelled"
            self._stamp_finish(meta)
            runstore.write_meta(run_id, meta)
            self._wake_waiters(handle)
            self._handles.pop(run_id, None)
            return {"run_id": run_id, "cancelled": True, "status": "cancelled"}
        # Already executing: SIGKILL this run's CmdStan subprocess(es). The worker
        # then sees a nonzero exit, and _execute relabels the run "cancelled".
        # Only claim the run as cancelled if something was actually killed —
        # otherwise a run that goes on to fail on its own merits (or succeed)
        # would be mislabelled.
        killed = _kill_run_processes(run_id)
        handle.cancelled = bool(killed)
        return {
            "run_id": run_id,
            "cancelled": bool(killed),
            "status": "cancelling" if killed else "running",
            "killed_pids": killed,
            "reason": None
            if killed
            else "no live CmdStan process found (still compiling or between chains); retry",
        }

    def get_run_progress(self, run_id) -> dict:
        """Status, elapsed wall time, and (for a running sample) draws written so
        far vs expected, parsed from CmdStan's live CSV output."""
        meta = runstore.read_meta(run_id)
        out: dict = {"run_id": run_id, "status": meta.get("status")}
        if meta.get("duration_s") is not None:
            out["elapsed_s"] = meta["duration_s"]
        elif meta.get("created_at"):
            try:
                delta = datetime.now(UTC) - datetime.fromisoformat(meta["created_at"])
                out["elapsed_s"] = round(delta.total_seconds(), 1)
            except ValueError:
                pass
        written = _count_live_draws(run_id)
        if written is not None:
            out["draws_written"] = written
            if meta.get("method") == "sample":
                ma = meta.get("method_args", {})
                expected = int(ma.get("chains", 4)) * int(ma.get("iter_sampling", 1000))
                out["expected_draws"] = expected
                if expected:
                    out["progress"] = round(min(written / expected, 1.0), 3)
        return out

    def _execute(self, run_id, source, data, method, kwargs, handle):
        try:
            meta: dict = {"run_id": run_id, "method": method}
            try:
                meta = runstore.read_meta(run_id)
                model = self._compile(source, None)
                # Route CmdStan's live CSV output to a per-run dir so cancel can
                # target this run's subprocesses and progress can be measured.
                # generate_quantities spells the argument gq_output_dir.
                kwargs = dict(kwargs)
                out_key = "gq_output_dir" if method == "generate_quantities" else "output_dir"
                kwargs.setdefault(out_key, str(runstore.live_dir(run_id, create=True)))
                fit = self._runner(method, model, data, kwargs)
                self._persist(run_id, method, fit)
                # re-read: _persist may have written its own warning fields
                # (persist_warning / csv_warning) directly to meta.json, and
                # writing back the pre-persist `meta` below would silently
                # clobber them.
                meta = runstore.read_meta(run_id)
                meta["status"] = "done"
            except Exception as exc:  # noqa: BLE001 - persisted, not raised
                # Re-read: _persist may have written persist_warning / csv_warning
                # before raising; the pre-persist snapshot would clobber them.
                # Tolerate the re-read failing too (e.g. the meta was pruned out
                # from under us) — the run must never be left stuck "running".
                try:
                    meta = runstore.read_meta(run_id)
                except Exception:  # noqa: BLE001
                    pass
                meta["status"] = "cancelled" if handle.cancelled else "failed"
                meta["stderr"] = f"{exc}\n{traceback.format_exc()}"
            self._stamp_finish(meta)
            try:
                runstore.write_meta(run_id, meta)
            except Exception:  # noqa: BLE001 - nothing left to report it to
                pass
        finally:
            self._wake_waiters(handle)
            # Drop the handle now the run is finished; later await_run calls fall
            # back to reading status from disk. Prevents unbounded growth.
            self._handles.pop(run_id, None)

    def get_run_status(self, run_id) -> dict:
        meta = runstore.read_meta(run_id)
        out = {k: meta[k] for k in ("run_id", "method", "status") if k in meta}
        if meta.get("stderr"):
            out["error"] = _error_excerpt(meta["stderr"])
        # Surface persist warnings so a status="done" run whose draws failed to
        # persist does not look cleanly queryable.
        for w in ("persist_warning", "csv_warning"):
            if meta.get(w):
                out[w] = meta[w]
        return out

    async def await_run(self, run_id, timeout=None) -> dict:
        handle = self._handles.get(run_id)
        if handle is None:
            # already finished in a prior process/session: read from disk
            meta = runstore.read_meta(run_id)
            return {
                "status": meta["status"],
                "run_dir": str(runstore.run_dir(run_id)),
                "meta": meta,
            }
        loop = asyncio.get_running_loop()
        aio_event = asyncio.Event()
        # Register BEFORE checking is_set(): if we checked first, the worker
        # could set() and iterate the (still-empty) aio_events list entirely
        # in the gap before we append, losing the wakeup forever. Appending
        # first guarantees that whichever of the worker's event.set() or our
        # is_set() check "wins" the race, at least one side observes it.
        handle.aio_events.append((loop, aio_event))
        if handle.event.is_set():
            aio_event.set()
        try:
            await asyncio.wait_for(aio_event.wait(), timeout=timeout)
        except TimeoutError:
            return {"status": "running", "run_id": run_id}
        meta = runstore.read_meta(run_id)
        return {"status": meta["status"], "run_dir": str(runstore.run_dir(run_id)), "meta": meta}


MANAGER = RunManager()
