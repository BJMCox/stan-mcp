"""Background run manager for all CmdStanPy inference methods."""

import asyncio
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import compile as _compile
import diagnostics
import methods
import runstore
from errors import RunError


class _Handle:
    def __init__(self):
        self.event = threading.Event()
        self.aio_events: list[tuple] = []  # (loop, asyncio.Event)


class RunManager:
    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._handles: dict[str, _Handle] = {}
        self._seq = 0
        self._lock = threading.Lock()

    # --- injectable seams (replaced in tests) ---
    def _compile(self, source, stanc_args):
        return _compile.cmdstan_model(source, stanc_args)

    def _runner(self, method, model, data, kwargs):
        cmd = methods.METHODS[method]["cmd"]
        return getattr(model, cmd)(data=data, **kwargs)

    def _persist(self, run_id, method, fit):
        # diagnostics.persist_run lands in Task 12; tests monkeypatch this seam.
        diagnostics.persist_run(run_id, method, fit)  # pyright: ignore[reportAttributeAccessIssue]

    # --- public API ---
    def start_run(self, stan, data, method="sample", seed=None, inits=None,
                  method_args=None) -> dict:
        kwargs = methods.build_kwargs(method, seed, inits, method_args or {})
        source = _compile.resolve_source(stan)
        with self._lock:
            self._seq += 1
            run_id = runstore.new_run_id(method, self._seq)
        runstore.write_meta(run_id, {
            "run_id": run_id, "method": method, "status": "running",
            "model_id": _compile.model_id(source, None), "seq": self._seq,
            "method_args": method_args or {},
        })
        handle = _Handle()
        self._handles[run_id] = handle
        self._pool.submit(self._execute, run_id, source, data, method, kwargs, handle)
        return {"run_id": run_id, "method": method, "status": "running"}

    def _execute(self, run_id, source, data, method, kwargs, handle):
        meta = runstore.read_meta(run_id)
        try:
            model = self._compile(source, None)
            fit = self._runner(method, model, data, kwargs)
            self._persist(run_id, method, fit)
            meta["status"] = "done"
        except Exception as exc:  # noqa: BLE001 - persisted, not raised
            meta["status"] = "failed"
            meta["stderr"] = f"{exc}\n{traceback.format_exc()}"
        runstore.write_meta(run_id, meta)
        handle.event.set()
        for loop, aio_event in handle.aio_events:
            loop.call_soon_threadsafe(aio_event.set)

    def get_run_status(self, run_id) -> dict:
        try:
            meta = runstore.read_meta(run_id)
        except FileNotFoundError as exc:
            raise RunError(f"unknown run_id {run_id!r}") from exc
        return {k: meta[k] for k in ("run_id", "method", "status") if k in meta} | (
            {"error": meta["stderr"].splitlines()[0]} if meta.get("stderr") else {}
        )

    async def await_run(self, run_id, timeout=None) -> dict:
        handle = self._handles.get(run_id)
        if handle is None:
            # already finished in a prior process/session: read from disk
            meta = runstore.read_meta(run_id)
            return {"status": meta["status"], "run_dir": str(runstore.run_dir(run_id)),
                    "meta": meta}
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
        return {"status": meta["status"], "run_dir": str(runstore.run_dir(run_id)),
                "meta": meta}


MANAGER = RunManager()
