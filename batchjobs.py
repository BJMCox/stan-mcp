"""Owned, resumable batch jobs with a shared process-worker budget."""

import asyncio
import fcntl
import json
import multiprocessing
import os
import threading
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import batchstore
import config
import runstore
from errors import RunError, StanMcpError, ValidationError

_METHODS = {"evaluate": "evaluate_batch", "compare": "compare_batches"}
_ACTIVE = {"running", "cancelling"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _worker(plan: dict, directory: str, connection: Connection, parent: Connection) -> None:
    """Keep native models process-local, and exit if the owning server dies."""
    import batcheval

    def watch_parent() -> None:
        try:
            parent.recv()
        except EOFError:
            os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        while (index := connection.recv()) is not None:
            try:
                connection.send(
                    {"descriptor": batcheval.execute_shard(plan, index, Path(directory))}
                )
            except Exception:
                connection.send({"error": traceback.format_exc()})
                return
    except EOFError:
        pass
    finally:
        connection.close()


class _Handle:
    def __init__(self, lease: Any) -> None:
        self.lease = lease
        self.cancelled = threading.Event()
        self.failed = threading.Event()
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.processes: list[Any] = []
        self.owners: dict[int, dict] = {}
        self.finished = False


class BatchManager:
    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = config.max_workers() if max_workers is None else max_workers
        if self.max_workers < 1:
            raise ValidationError("batch worker cap must be positive")
        self._slots = threading.BoundedSemaphore(self.max_workers)
        self._worker_target: Callable[[dict, str, Connection, Connection], None] = _worker
        self._handles: dict[str, _Handle] = {}
        self._lock = threading.RLock()
        runstore.reconcile_abandoned()

    @staticmethod
    def _directory(job_id: str) -> Path:
        return runstore.run_dir(job_id) / "batch"

    @staticmethod
    def _lease(directory: Path) -> Any:
        lease = (directory / ".lease").open("a")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lease.close()
            raise RunError("batch job is owned by an active coordinator") from exc
        return lease

    @staticmethod
    def _prepare(kind: str, request: dict, directory: Path) -> dict:
        if kind == "evaluate":
            import batcheval

            return batcheval.prepare(request, directory)
        import batchcompare

        return batchcompare.prepare(request, directory)

    @staticmethod
    def _snapshot_request(kind: str, request: dict, directory: Path) -> dict:
        request = dict(request)
        for name in ("points",) if kind == "evaluate" else ("left", "right"):
            if not isinstance(request.get(name), str):
                raise ValidationError(f"batch request requires a {name} manifest path")
            request[name] = str(batchstore.snapshot(request[name], directory / name))
        if kind == "evaluate":
            import compile as compilation

            if not isinstance(request.get("stan"), str):
                raise ValidationError("batch request requires stan source or a model path")
            source = compilation.resolve_source(request["stan"], request.get("stanc_args"))
            request["stan"] = str(source)
            request["_source_manifest"] = compilation.model_manifest(source)
            request["data"] = compilation.canonical_data(request.get("data"))
        elif isinstance(request.get("finite_difference"), dict):
            fd = dict(request["finite_difference"])
            if not isinstance(fd.get("stencils"), str):
                raise ValidationError("finite_difference requires a stencils manifest path")
            fd["stencils"] = str(batchstore.snapshot(fd["stencils"], directory / "stencils"))
            request["finite_difference"] = fd
        return request

    def start(self, kind: str, request_file: str) -> dict:
        if kind not in _METHODS:
            raise ValidationError("batch kind must be evaluate or compare")
        path = Path(request_file).expanduser().resolve()
        try:
            request = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValidationError(f"cannot read batch request: {exc}") from exc
        if not isinstance(request, dict):
            raise ValidationError("batch request must be a JSON object")
        workers = request.get("workers", 1)
        if (
            isinstance(workers, bool)
            or not isinstance(workers, int)
            or not 1 <= workers <= self.max_workers
        ):
            raise ValidationError(f"workers must be between 1 and {self.max_workers}")
        job_id, seq = runstore.reserve_run(kind)
        directory = self._directory(job_id)
        directory.mkdir()
        lease = self._lease(directory)
        try:
            inputs = directory / "inputs"
            inputs.mkdir()
            request = self._snapshot_request(kind, request, inputs)
            batchstore.atomic_json(directory / "request.json", request)
            runstore.write_meta(
                job_id,
                {
                    "method": _METHODS[kind],
                    "kind": kind,
                    "seq": seq,
                    "status": "running",
                    "created_at": _now(),
                    "owner": runstore.process_owner(),
                    "workers": workers,
                    "processed_shards": 0,
                    "processed_rows": 0,
                },
            )
            self._launch(job_id, lease)
        except Exception as exc:
            lease.close()
            import shutil

            shutil.rmtree(runstore.run_dir(job_id))
            if isinstance(exc, OSError) and not isinstance(exc, StanMcpError):
                raise ValidationError(f"cannot snapshot batch submission: {exc}") from exc
            raise
        return {"run_id": job_id, "job_id": job_id, "status": "running"}

    def _launch(self, job_id: str, lease: Any) -> None:
        handle = _Handle(lease)
        thread = threading.Thread(target=self._execute, args=(job_id, handle), daemon=True)
        handle.thread = thread
        with self._lock:
            self._handles[job_id] = handle
        try:
            thread.start()
        except Exception:
            with self._lock:
                if self._handles.get(job_id) is handle:
                    self._handles.pop(job_id)
            raise

    def _update(self, job_id: str, handle: _Handle, **values: Any) -> None:
        with handle.lock:
            meta = runstore.read_meta(job_id)
            meta.update(values)
            runstore.write_meta(job_id, meta)

    def _acquire(self, handle: _Handle) -> bool:
        while not handle.cancelled.is_set() and not handle.failed.is_set():
            if self._slots.acquire(timeout=0.05):
                return True
        return False

    @staticmethod
    def _input_hashes(directory: Path) -> dict[str, str]:
        return {
            str(path.relative_to(directory)): batchstore.file_hash(path)
            for path in sorted((directory / "inputs").rglob("*"))
            if path.is_file()
        }

    def _execute(self, job_id: str, handle: _Handle) -> None:
        directory = self._directory(job_id)
        output = directory / "results"
        try:
            output.mkdir(exist_ok=True)
            meta = runstore.read_meta(job_id)
            kind = meta["kind"]
            plan_path = directory / "plan.json"
            if plan_path.is_file():
                plan = json.loads(plan_path.read_text())
            else:
                request = json.loads((directory / "request.json").read_text())
                prepared = directory / "inputs" / "prepared"
                prepared.mkdir(exist_ok=True)
                plan = self._prepare(kind, request, prepared)
                batchstore.atomic_json(plan_path, plan)
                identity = {
                    "plan": batchstore.fingerprint(plan),
                    "inputs": self._input_hashes(directory),
                    "request": batchstore.file_hash(directory / "request.json"),
                }
                batchstore.atomic_json(directory / "identity.json", identity)
            if kind == "evaluate" and "source_manifest" in plan:
                self._update(job_id, handle, model_id=plan["source_manifest"]["model_id"])
            if kind == "evaluate":
                result = self._evaluate(job_id, handle, plan, output, meta["workers"])
            else:
                result = self._compare(job_id, handle, plan, output)
            with handle.lock:
                self._update(
                    job_id,
                    handle,
                    result=result,
                    status="cancelled" if handle.cancelled.is_set() else "done",
                    finished_at=_now(),
                )
                handle.finished = True
        except Exception:
            with handle.lock:
                self._update(
                    job_id,
                    handle,
                    status="cancelled" if handle.cancelled.is_set() else "failed",
                    stderr=traceback.format_exc(),
                    finished_at=_now(),
                )
                handle.finished = True
        finally:
            with self._lock:
                handle.lease.close()
                if self._handles.get(job_id) is handle:
                    self._handles.pop(job_id, None)

    def _evaluate(
        self, job_id: str, handle: _Handle, plan: dict, output: Path, workers: int
    ) -> dict:
        manifest = batchstore.load(plan["input_manifest"])
        checkpoint = output.parent / "checkpoint.json"
        completed = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        pending = iter(
            index for index in range(len(manifest["shards"])) if str(index) not in completed
        )
        errors: list[str] = []

        def publish() -> None:
            descriptors = [completed[key] for key in sorted(completed, key=int)]
            batchstore.atomic_json(checkpoint, completed)
            if descriptors:
                batchstore.publish(output, plan["output_metadata"], descriptors)
            self._update(
                job_id,
                handle,
                processed_shards=len(descriptors),
                total_shards=len(manifest["shards"]),
                total_rows=manifest["count"],
                processed_rows=sum(item["rows"] for item in descriptors),
            )

        def consume() -> None:
            if not self._acquire(handle):
                return
            parent = child = watch_child = watch_parent = None
            process = None
            try:
                context = multiprocessing.get_context("spawn")
                parent, child = context.Pipe()
                watch_child, watch_parent = context.Pipe(duplex=False)
                process = context.Process(
                    target=self._worker_target, args=(plan, str(output), child, watch_child)
                )
                with handle.lock:
                    if handle.cancelled.is_set() or handle.failed.is_set():
                        return
                    process.start()
                    handle.processes.append(process)
                    import psutil

                    pid = process.pid
                    assert pid is not None
                    handle.owners[pid] = {
                        "pid": pid,
                        "create_time": psutil.Process(pid).create_time(),
                    }
                    self._update(job_id, handle, worker_owners=list(handle.owners.values()))
                child.close()
                watch_child.close()
                while not handle.cancelled.is_set() and not handle.failed.is_set():
                    with handle.lock:
                        index = next(pending, None)
                    if index is None:
                        break
                    parent.send(index)
                    while not parent.poll(0.05):
                        if handle.cancelled.is_set() or handle.failed.is_set():
                            return
                        if not process.is_alive():
                            raise RunError(f"batch worker exited with code {process.exitcode}")
                    reply = parent.recv()
                    if "error" in reply:
                        raise RunError(reply["error"])
                    with handle.lock:
                        completed[str(index)] = reply["descriptor"]
                        publish()
                if process.is_alive():
                    parent.send(None)
            except Exception:
                with handle.lock:
                    errors.append(traceback.format_exc())
                    handle.failed.set()
            finally:
                try:
                    if process is not None and process.pid is not None:
                        if process.is_alive():
                            process.terminate()
                        process.join()
                        with handle.lock:
                            if process in handle.processes:
                                handle.processes.remove(process)
                            handle.owners.pop(process.pid, None)
                            self._update(job_id, handle, worker_owners=list(handle.owners.values()))
                except Exception:
                    with handle.lock:
                        errors.append(traceback.format_exc())
                        handle.failed.set()
                finally:
                    for connection in (parent, child, watch_parent, watch_child):
                        if connection is not None:
                            connection.close()
                    self._slots.release()

        with handle.lock:
            publish()
        threads = [
            threading.Thread(target=consume)
            for _ in range(min(workers, len(manifest["shards"]) - len(completed)))
        ]
        started: list[threading.Thread] = []
        try:
            for thread in threads:
                thread.start()
                started.append(thread)
        except Exception:
            handle.failed.set()
            raise
        finally:
            for thread in started:
                thread.join()
        if errors and not handle.cancelled.is_set():
            raise RunError(errors[0])
        if not handle.cancelled.is_set() and len(completed) != len(manifest["shards"]):
            raise RunError("batch workers stopped before all shards were completed")
        result: dict[str, Any] = {"completed_shards": len(completed)}
        if completed:
            result["manifest"] = str(output / "manifest.json")
        return result

    def _compare(self, job_id: str, handle: _Handle, plan: dict, output: Path) -> dict:
        import batchcompare

        if not self._acquire(handle):
            return {"cancelled": True}
        try:
            checkpoint = output / "comparison-checkpoint.json"
            if checkpoint.exists():
                plan = {**plan, "completed": json.loads(checkpoint.read_text())["completed"]}
            right = batchstore.read_json(plan["right_manifest"])
            prefix = [0]
            for descriptor in right["shards"]:
                prefix.append(prefix[-1] + descriptor["rows"])

            def progress(done: int, total: int) -> None:
                self._update(
                    job_id,
                    handle,
                    processed_shards=done,
                    total_shards=total,
                    processed_rows=prefix[done],
                    total_rows=prefix[-1],
                )

            progress(len(plan.get("completed", [])), len(right["shards"]))
            result = batchcompare.execute(
                plan,
                output,
                handle.cancelled.is_set,
                progress,
            )
            completed = result.pop("completed")
            result["completed_shards"] = len(completed)
            return result
        finally:
            self._slots.release()

    def status(self, job_id: str) -> dict:
        meta = runstore.read_meta(job_id)
        if meta.get("method") not in _METHODS.values():
            raise RunError(f"run {job_id!r} is not a batch job")
        result = {key: meta[key] for key in ("run_id", "method", "status")}
        if meta.get("stderr"):
            result["error"] = meta["stderr"][-800:]
        return result

    def progress(self, job_id: str) -> dict:
        result = self.status(job_id)
        meta = runstore.read_meta(job_id)
        result.update(
            {
                key: meta[key]
                for key in ("processed_shards", "total_shards", "processed_rows", "total_rows")
                if key in meta
            }
        )
        return result

    def result(self, job_id: str) -> dict:
        result = self.progress(job_id)
        meta = runstore.read_meta(job_id)
        result.update(meta.get("result", {}))
        manifest = self._directory(job_id) / "results" / "manifest.json"
        if manifest.exists():
            result["manifest"] = str(manifest)
        return result

    def cancel(self, job_id: str) -> dict:
        status = self.status(job_id)
        with self._lock:
            handle = self._handles.get(job_id)
        if handle is None:
            return {
                **status,
                "cancelled": status["status"] == "cancelled",
                "reason": "owned by another server" if status["status"] in _ACTIVE else "finished",
            }
        with handle.lock:
            if handle.finished:
                status = self.status(job_id)
                return {**status, "cancelled": status["status"] == "cancelled"}
            handle.cancelled.set()
            self._update(job_id, handle, status="cancelling")
            for process in handle.processes:
                if process.is_alive():
                    process.terminate()
        return {"run_id": job_id, "status": "cancelling", "cancelled": True}

    def resume(self, job_id: str) -> dict:
        self.status(job_id)
        directory = self._directory(job_id)
        lease = self._lease(directory)
        try:
            meta = runstore.read_meta(job_id)
            previous_meta = dict(meta)
            if meta["status"] == "done":
                raise RunError("completed batch jobs cannot be resumed")
            if meta["status"] in _ACTIVE:
                import psutil

                owner = meta.get("owner", {})
                try:
                    process = psutil.Process(owner["pid"])
                    if (
                        process.is_running()
                        and process.create_time() == owner["create_time"]
                        and owner != runstore.process_owner()
                    ):
                        raise RunError("batch job still has a live owner")
                except (psutil.NoSuchProcess, KeyError):
                    pass
            if not (directory / "identity.json").is_file():
                raise RunError(
                    "batch preparation was interrupted before an owned plan was published"
                )
            identity = json.loads((directory / "identity.json").read_text())
            self._drain_abandoned_workers(meta)
            plan = json.loads((directory / "plan.json").read_text())
            current = {
                "plan": batchstore.fingerprint(plan),
                "inputs": self._input_hashes(directory),
                "request": batchstore.file_hash(directory / "request.json"),
            }
            if current != identity:
                raise RunError("owned batch inputs or computation settings changed")
            if meta["kind"] == "evaluate":
                import batcheval

                batcheval.validate_plan(plan)
            self._validate_completed(directory, plan, meta["kind"])
            meta.update(status="running", owner=runstore.process_owner(), resumed_at=_now())
            meta.pop("finished_at", None)
            meta.pop("stderr", None)
            meta.pop("result", None)
            runstore.write_meta(job_id, meta)
            try:
                self._launch(job_id, lease)
            except Exception:
                runstore.write_meta(job_id, previous_meta)
                raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            lease.close()
            raise RunError(f"cannot verify owned batch computation: {exc}") from exc
        except Exception:
            lease.close()
            raise
        return {"run_id": job_id, "job_id": job_id, "status": "running"}

    @staticmethod
    def _drain_abandoned_workers(meta: dict) -> None:
        import psutil

        workers: list[psutil.Process] = []
        for owner in meta.get("worker_owners", []):
            try:
                worker = psutil.Process(owner["pid"])
                if worker.create_time() == owner["create_time"]:
                    worker.kill()
                    workers.append(worker)
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied as exc:
                raise RunError("cannot stop an abandoned batch worker") from exc
        _, alive = psutil.wait_procs(workers, timeout=5)
        if any(worker.status() != psutil.STATUS_ZOMBIE for worker in alive):
            raise RunError("abandoned batch workers have not stopped")

    @staticmethod
    def _validate_completed(directory: Path, plan: dict, kind: str) -> None:
        output = directory / "results"
        checkpoint = (
            directory / "checkpoint.json"
            if kind == "evaluate"
            else output / "comparison-checkpoint.json"
        )
        state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        descriptors = list(state.values()) if kind == "evaluate" else state.get("completed", [])
        if len({descriptor["path"] for descriptor in descriptors}) != len(descriptors):
            raise RunError("batch checkpoint repeats a completed shard")
        inputs = batchstore.load(plan["input_manifest"])
        if kind == "evaluate":
            indices = {str(i) for i in range(len(inputs["shards"]))}
            if any(key not in indices for key in state):
                raise RunError("batch checkpoint contains an unknown input shard")
        elif len(descriptors) > len(inputs["shards"]):
            raise RunError("batch checkpoint exceeds its comparison input shards")
        manifest = output / "manifest.json"
        if manifest.exists():
            saved = batchstore.load(manifest)
            committed = {descriptor["path"]: descriptor for descriptor in descriptors}
            for descriptor in saved["shards"]:
                checkpointed = committed.get(descriptor["path"], {})
                if any(
                    descriptor.get(key) != checkpointed.get(key)
                    for key in ("path", "sha256", "rows", "errors", "errors_sha256")
                ):
                    raise RunError("completed batch manifest differs from its checkpoint")
        owned = {"manifest.json", "comparison-checkpoint.json"}
        for descriptor in descriptors:
            for key, hash_key in (("path", "sha256"), ("errors", "errors_sha256")):
                if key not in descriptor:
                    continue
                path = (output / descriptor[key]).resolve()
                if not path.is_relative_to(output.resolve()):
                    raise RunError("completed batch descriptor escapes its output directory")
                if batchstore.file_hash(path) != descriptor.get(hash_key):
                    raise RunError("completed batch shard content changed")
                owned.add(str(path.relative_to(output.resolve())))
            arrays = batchstore.read_shard({"_root": str(output)}, descriptor)
            if len(arrays["row_id"]) != descriptor["rows"]:
                raise RunError("completed batch shard row count changed")
        import numpy as np

        entries = state.items() if kind == "evaluate" else enumerate(descriptors)
        for key, descriptor in entries:
            original = batchstore.read_shard(inputs, inputs["shards"][int(key)])
            saved_arrays = batchstore.read_shard({"_root": str(output)}, descriptor)
            if not np.array_equal(original["row_id"], saved_arrays["row_id"]):
                raise RunError("batch checkpoint shard does not match its input rows")
        # Only descriptors checkpointed by the coordinator commit completed work.
        # A worker killed during publication can leave an unfinished chunk behind.
        for path in output.rglob("*"):
            if path.is_file() and str(path.relative_to(output)) not in owned:
                path.unlink()

    async def await_job(self, job_id: str, timeout: float | None = None) -> dict:
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        while True:
            status = self.status(job_id)
            with self._lock:
                owned = job_id in self._handles
            if status["status"] not in _ACTIVE and not owned:
                return {**self.result(job_id), "meta": runstore.read_meta(job_id)}
            if deadline is not None and time.monotonic() >= deadline:
                return status
            await asyncio.sleep(
                0.05 if deadline is None else min(0.05, max(0, deadline - time.monotonic()))
            )

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            handles = list(self._handles.items())
        for job_id, _ in handles:
            self.cancel(job_id)
        if wait:
            for _, handle in handles:
                if handle.thread is not None:
                    handle.thread.join()


Manager = BatchManager
