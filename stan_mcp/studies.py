"""Resumable inference studies, coordinated through the shared run manager."""

import asyncio
import fcntl
import json
import os
import shutil
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from stan_mcp import compilation, draws, methods, runstore
from stan_mcp.batch.store import atomic_json, file_hash, fingerprint, read_json
from stan_mcp.errors import RunError, ValidationError
from stan_mcp.inference import RunManager

SCHEMA = "stan-mcp.study.v1"
_TERMINAL = {"done", "failed", "cancelled"}


def study_dir(study_id: str) -> Path:
    meta = runstore.read_meta(study_id)
    if meta.get("method") != "study":
        raise ValidationError(f"{study_id!r} is not a study")
    return runstore.run_dir(study_id) / "study"


def _path(value: str, root: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (root / path).resolve())


def _snapshot(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    stan = spec["stan"]
    if isinstance(stan, str) and "\n" not in stan and stan.endswith(".stan"):
        stan = _path(stan, root)
    source = compilation.resolve_source(stan)
    data = spec.get("data")
    if isinstance(data, str) and data.endswith(".json"):
        data = _path(data, root)
    inits = spec.get("inits")
    if isinstance(inits, str):
        inits = compilation.read_json_file(_path(inits, root), "initialization")
    elif isinstance(inits, list):
        inits = [
            compilation.read_json_file(_path(v, root), "initialization")
            if isinstance(v, str)
            else v
            for v in inits
        ]
    method = spec.get("method", "sample")
    args = spec.get("method_args", {})
    if not isinstance(args, dict):
        raise ValidationError("method_args must be an object")
    methods.build_kwargs(method, 1, inits, args)
    return {
        "source_manifest": compilation.model_manifest(source),
        "data": compilation.canonical_data(data),
        "method": method,
        "method_args": dict(args),
        "inits": inits,
    }


def _mapping(value: Any, name: str, *, required: bool = True) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or (required and not value)
        or not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in value.items())
    ):
        raise ValidationError(f"{name} must map target variable names to generated variable names")
    return value


def _inventory(request: dict[str, Any], root: Path) -> dict[str, Any]:
    if (
        set(request) - {"schema", "seed", "max_parallel", "replicates", "tasks"}
        or request.get("schema") != SCHEMA
    ):
        raise ValidationError(f"study requests require schema {SCHEMA!r} and supported fields")
    seed = request.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValidationError("study seed must be an integer in [0, 2**32)")
    parallel = request.get("max_parallel", 1)
    repeats = request.get("replicates", 1)
    if type(parallel) is not int or not 1 <= parallel <= 32:
        raise ValidationError("max_parallel must be between 1 and 32")
    tasks = request.get("tasks")
    if (
        type(repeats) is not int
        or repeats < 1
        or not isinstance(tasks, list)
        or not 1 <= len(tasks) * repeats <= 10000
    ):
        raise ValidationError("a study must expand to between 1 and 10000 tasks")
    templates = []
    names: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) - {
            "id",
            "stan",
            "data",
            "method",
            "method_args",
            "inits",
            "simulate",
        }:
            raise ValidationError("unsupported study task fields")
        name = task.get("id")
        if not isinstance(name, str) or not name or len(name) > 120 or name in names:
            raise ValidationError(
                "task IDs must be unique nonempty strings of at most 120 characters"
            )
        names.add(name)
        item = {"template_id": name, "fit": _snapshot(task, root)}
        if "simulate" in task:
            sim = task["simulate"]
            if not isinstance(sim, dict) or set(sim) - {
                "stan",
                "data",
                "data_variables",
                "truth_variables",
                "integer_data",
            }:
                raise ValidationError("unsupported simulation fields")
            item["simulate"] = {
                **_snapshot(
                    {
                        **sim,
                        "method": "fixed_param",
                        "method_args": {"chains": 1, "iter_sampling": 1},
                        "inits": 0,
                    },
                    root,
                ),
                "data_variables": _mapping(sim.get("data_variables"), "data_variables"),
                "truth_variables": _mapping(
                    sim.get("truth_variables", {}), "truth_variables", required=False
                ),
                "integer_data": sim.get("integer_data", []),
            }
            integers = item["simulate"]["integer_data"]
            if not isinstance(integers, list) or not all(
                isinstance(name, str) and name in item["simulate"]["data_variables"]
                for name in integers
            ):
                raise ValidationError("integer_data must name generated data targets")
        templates.append(item)
    rng = np.random.default_rng(seed)
    seeds = rng.choice(2**31 - 1, size=len(tasks) * repeats * 2, replace=False) + 1
    expanded = []
    for replica in range(repeats):
        for template in templates:
            index = len(expanded)
            expanded.append(
                {
                    **template,
                    "id": template["template_id"]
                    if repeats == 1
                    else f"{template['template_id']}:{replica + 1}",
                    "fit_seed": int(seeds[2 * index]),
                    "simulate_seed": int(seeds[2 * index + 1]),
                }
            )
    return {"schema": SCHEMA, "seed": seed, "max_parallel": parallel, "tasks": expanded}


def _source(spec: dict[str, Any]) -> compilation.SourceSnapshot:
    manifest = spec["source_manifest"]
    return compilation.restore_source(manifest["source"], manifest)


def _copy_csvs(paths: list[str], directory: Path) -> list[str]:
    directory.mkdir(parents=True)
    result = []
    for index, path in enumerate(paths):
        target = directory / f"{index:08d}.csv"
        shutil.copyfile(path, target)
        result.append(str(target))
    return result


def _verify_build(request: dict[str, Any]) -> None:
    for task in request["tasks"]:
        for stage in ("fit", "simulate"):
            if stage in task:
                source = _source(task[stage])
                if compilation.model_manifest(source) != source.manifest:
                    raise ValidationError("study build identity changed; start a new study")


def _artifact_hashes(run_id: str) -> dict[str, str]:
    directory = runstore.run_dir(run_id)
    # Analysis reports have their own lifecycle. Resume depends on the fitted
    # values and their provenance, not a concurrently written predictive plot.
    paths = [
        directory / name
        for name in (
            "meta.json",
            "model.stan",
            "data.json",
            "inits.json",
            "source_manifest.json",
            "draws_schema.json",
            "estimate.json",
            "initialization.json",
        )
    ]
    paths.extend((directory / "csv").glob("*.csv"))
    paths.extend((directory / "input_csv").glob("*.csv"))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise RunError(f"child {run_id!r} has no durable artifacts")
    return {str(path.relative_to(directory)): file_hash(path) for path in sorted(paths)}


def verify_artifacts(run_id: str, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = runstore.run_dir(run_id) / name
        if not path.is_file() or file_hash(path) != expected:
            raise RunError(f"study child artifact changed or was pruned: {run_id}/{name}")


def task_state(directory: Path, index: int) -> dict[str, Any]:
    path = directory / "tasks" / f"{index:05d}.json"
    if not path.exists():
        return {"status": "pending", "attempts": []}
    state = read_json(path)
    expected = state.pop("state_hash", None)
    if expected != fingerprint(state):
        raise RunError("study task checkpoint changed")
    return state


def _save(directory: Path, index: int, state: dict[str, Any]) -> None:
    atomic_json(
        directory / "tasks" / f"{index:05d}.json", {**state, "state_hash": fingerprint(state)}
    )


def _no_surviving_process(run_id: str) -> None:
    import psutil

    marker = str(runstore.live_dir(run_id))
    for process in psutil.process_iter():
        try:
            if process.uids().real != os.getuid():
                continue
            if any(marker in arg for arg in process.cmdline()):
                raise RunError(f"child {run_id!r} still has a live process; wait before resuming")
        except psutil.NoSuchProcess, psutil.ZombieProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RunError("cannot verify that interrupted study child processes stopped") from exc


@dataclass
class _Handle:
    lease: Any
    cancel: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)


class StudyManager:
    def __init__(self, runs: RunManager) -> None:
        self.runs = runs
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stan-study")
        self._handles: dict[str, _Handle] = {}
        self._lock = threading.Lock()
        self._token = uuid4().hex

    def start(self, request_file: str) -> dict[str, Any]:
        request = json.loads(
            json.dumps(
                _inventory(read_json(request_file), Path(request_file).resolve().parent),
                allow_nan=False,
            )
        )
        study_id, seq = runstore.reserve_run("study")
        directory = runstore.run_dir(study_id) / "study"
        directory.mkdir()
        try:
            # Snapshot external fitted parameters once, before any child starts.
            csv_cache: dict[tuple[str, ...], list[str]] = {}

            def snapshot_csvs(paths: list[str]) -> list[str]:
                key = tuple(_path(p, Path(request_file).resolve().parent) for p in paths)
                if key not in csv_cache:
                    csv_cache[key] = _copy_csvs(
                        list(key), directory / "inputs" / str(len(csv_cache))
                    )
                return csv_cache[key]

            for task in request["tasks"]:
                spec = task["fit"]
                args = spec["method_args"]
                if spec["method"] == "generate_quantities" and args.get("from_run_id"):
                    args["csv_files"] = runstore.csv_paths(args.pop("from_run_id"))
                if "csv_files" in args:
                    paths = args["csv_files"]
                    if isinstance(paths, str):
                        paths = [paths]
                    args["csv_files"] = snapshot_csvs(paths)
                if "mode" in args:
                    args["mode"] = snapshot_csvs([args["mode"]])[0]
            atomic_json(directory / "request.json", request)
            runstore.write_meta(
                study_id,
                {
                    "method": "study",
                    "status": "running",
                    "seq": seq,
                    "owner": runstore.process_owner(),
                    "created_at": datetime.now(UTC).isoformat(),
                    "request_hash": fingerprint(request),
                },
            )
            return self._launch(study_id)
        except Exception:
            shutil.rmtree(runstore.run_dir(study_id))
            raise

    def _launch(self, study_id: str) -> dict[str, Any]:
        directory = study_dir(study_id)
        lease = (directory / ".lease").open("a")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lease.close()
            raise RunError("another coordinator owns this study") from exc
        handle = _Handle(lease)
        with self._lock:
            if study_id in self._handles:
                lease.close()
                raise RunError("study already runs in this server")
            self._handles[study_id] = handle
        try:
            meta = runstore.read_meta(study_id)
            meta.update(
                status="running", owner=runstore.process_owner(), study_manager_token=self._token
            )
            meta.pop("stderr", None)
            runstore.write_meta(study_id, meta)
            self._pool.submit(self._execute, study_id, handle)
        except Exception:
            with self._lock:
                self._handles.pop(study_id, None)
            lease.close()
            raise
        return {"study_id": study_id, "run_id": study_id, "status": "running"}

    def _prepare(self, study_id: str, cancel: threading.Event) -> dict[str, Any]:
        directory = study_dir(study_id)
        initial = read_json(directory / "request.json")
        meta = runstore.read_meta(study_id)
        if fingerprint(initial) != meta["request_hash"]:
            raise RunError("study input snapshot changed")
        path = directory / "prepared.json"
        if path.exists():
            prepared = read_json(path)
            if fingerprint(prepared) != meta.get("prepared_hash"):
                raise RunError("prepared study snapshot changed")
            _verify_build(prepared)
            for name, expected in prepared["input_csv_hashes"].items():
                if file_hash(Path(name)) != expected:
                    raise RunError("study input CSV snapshot changed")
            return prepared
        if meta.get("prepared_hash"):
            raise RunError("committed prepared study snapshot is missing; start a new study")
        # Provision and compile before freezing the build. Later child stages
        # must use this identity, even when the original source files disappear.
        cache: dict[str, dict[str, Any]] = {}
        for task in initial["tasks"]:
            for stage in ("fit", "simulate"):
                if stage not in task:
                    continue
                if cancel.is_set():
                    raise CancelledError()
                source = _source(task[stage])
                key = fingerprint(source.manifest)
                if key not in cache:
                    compilation.cmdstan_model(source)
                    cache[key] = compilation.model_manifest(source)
                task[stage]["source_manifest"] = cache[key]
        initial["input_csv_hashes"] = {
            str(p): file_hash(p) for p in (directory / "inputs").rglob("*.csv")
        }
        meta["prepared_hash"] = fingerprint(initial)
        runstore.write_meta(study_id, meta)
        atomic_json(path, initial)
        return initial

    def _reconcile(self, study_id: str, request: dict[str, Any]) -> None:
        directory = study_dir(study_id)
        children: dict[str, list[dict[str, Any]]] = {}
        for child in reversed(runstore.list_runs(limit=10**9)):
            context = child.get("study", {})
            if context.get("run_id") == study_id:
                children.setdefault(context["task_id"], []).append(child)
        for index, task in enumerate(request["tasks"]):
            state = task_state(directory, index)
            for stage in ("simulate", "fit"):
                if stage in state and "hashes" in state[stage]:
                    verify_artifacts(state[stage]["run_id"], state[stage]["hashes"])
            if state["status"] == "done":
                continue
            for child in children.get(task["id"], []):
                context = child.get("study", {})
                stage = context["stage"]
                if state.get(stage, {}).get("status") == "done":
                    continue
                child["status"] = self.runs.get_run_status(child["run_id"])["status"]
                if child["status"] in {"running", "cancelling"}:
                    raise RunError(f"child {child['run_id']!r} is still owned by a live server")
                _no_surviving_process(child["run_id"])
                if stage not in state or state[stage].get("run_id") != child["run_id"]:
                    state[stage] = {"run_id": child["run_id"], "status": child["status"]}
                if not any(a["run_id"] == child["run_id"] for a in state["attempts"]):
                    state["attempts"].append({"stage": stage, "run_id": child["run_id"]})
            state["status"] = "pending"
            _save(directory, index, state)

    def resume(self, study_id: str) -> dict[str, Any]:
        # Validation and reconciliation run under the lease in the coordinator.
        study_dir(study_id)
        return self._launch(study_id)

    def _start_stage(
        self, study_id: str, task: dict[str, Any], stage: str, state: dict[str, Any]
    ) -> None:
        spec = task[stage]
        _verify_build({"tasks": [{stage: spec}]})
        data = spec["data"]
        if stage == "fit" and "generated_data" in state:
            data = {**(data or {}), **state["generated_data"]}
        child = self.runs.start_run(
            _source(spec),
            data,
            method=spec["method"],
            seed=task[f"{stage}_seed"],
            inits=spec["inits"],
            method_args=spec["method_args"],
            study={"run_id": study_id, "task_id": task["id"], "stage": stage},
        )
        state[stage] = {"run_id": child["run_id"], "status": "running"}
        state["status"] = "simulating" if stage == "simulate" else "fitting"
        state["attempts"].append({"stage": stage, "run_id": child["run_id"]})

    def _collect(self, task: dict[str, Any], stage: str, state: dict[str, Any]) -> bool:
        current = state[stage]
        result = self.runs.get_run_status(current["run_id"])
        if result["status"] not in _TERMINAL:
            return False
        current["status"] = result["status"]
        if result["status"] != "done":
            state["status"] = result["status"]
            state["error"] = result.get("error", f"{stage} {result['status']}")
            return True
        child_meta = runstore.read_meta(current["run_id"])
        if child_meta["model_id"] != task[stage]["source_manifest"]["model_id"]:
            raise RunError("child build differs from the frozen study build")
        current["hashes"] = _artifact_hashes(current["run_id"])
        if stage == "simulate":
            d = draws.load(current["run_id"])
            if d.n_chain != 1 or d.n_draw != 1:
                raise RunError("a study simulator must produce exactly one joint draw")
            spec = task["simulate"]

            def select(mapping: dict[str, str]) -> dict[str, Any]:
                values: dict[str, Any] = {}
                for target, name in mapping.items():
                    if name not in d.params:
                        raise RunError(f"simulator did not emit {name!r}")
                    value = d.params[name][0, 0]
                    if not np.isfinite(value).all():
                        raise RunError("simulator emitted nonfinite data or truth")
                    # CmdStan CSVs are numeric. Restore declared integer draws
                    # before they enter a fitted model's integer data block.
                    if mapping is spec["data_variables"] and target in spec["integer_data"]:
                        if (
                            not np.equal(value, np.floor(value)).all()
                            or np.any(value < -(2**31))
                            or np.any(value > 2**31 - 1)
                        ):
                            raise RunError("simulator emitted a noninteger declared int")
                        value = value.astype(np.int64)
                    values[target] = value.tolist()
                return values

            generated = select(spec["data_variables"])
            truth = select(spec["truth_variables"])
            state.update(generated_data=generated, truth=truth)
            state["status"] = "pending"
        else:
            state["status"] = "done"
            state.pop("error", None)
        current["ready"] = True
        return True

    def _execute(self, study_id: str, handle: _Handle) -> None:
        directory = runstore.run_dir(study_id) / "study"
        try:
            request = self._prepare(study_id, handle.cancel)
            self._reconcile(study_id, request)
            active: dict[int, str] = {}
            finished: set[int] = set()
            while len(finished) < len(request["tasks"]):
                for index, stage in list(active.items()):
                    state = task_state(directory, index)
                    if handle.cancel.is_set():
                        self.runs.cancel_run(state[stage]["run_id"])
                    try:
                        collected = self._collect(request["tasks"][index], stage, state)
                    except (RunError, ValidationError, ValueError, KeyError) as exc:
                        state.update(status="failed", error=str(exc)[:800])
                        collected = True
                    if collected:
                        _save(directory, index, state)
                        active.pop(index)
                        if state["status"] in _TERMINAL:
                            finished.add(index)
                for index, task in enumerate(request["tasks"]):
                    if index in active or index in finished:
                        continue
                    state = task_state(directory, index)
                    if state["status"] == "done":
                        finished.add(index)
                        continue
                    if handle.cancel.is_set():
                        state["status"] = "cancelled"
                        _save(directory, index, state)
                        finished.add(index)
                        continue
                    if len(active) >= request["max_parallel"]:
                        break
                    try:
                        stage = (
                            "simulate"
                            if "simulate" in task and not state.get("simulate", {}).get("ready")
                            else "fit"
                        )
                        if stage in state and state[stage]["status"] == "done":
                            # Recover a child that finished before its checkpoint.
                            self._collect(task, stage, state)
                            _save(directory, index, state)
                            if state["status"] == "done":
                                finished.add(index)
                                continue
                            stage = "fit"
                        self._start_stage(study_id, task, stage, state)
                        active[index] = stage
                    except (RunError, ValidationError, ValueError, KeyError) as exc:
                        state.update(status="failed", error=str(exc)[:800])
                        finished.add(index)
                    _save(directory, index, state)
                if active:
                    handle.cancel.wait(
                        0.1
                    ) if not handle.cancel.is_set() else threading.Event().wait(0.1)
            meta = runstore.read_meta(study_id)
            meta.update(
                status="cancelled" if handle.cancel.is_set() else "done",
                finished_at=datetime.now(UTC).isoformat(),
            )
            runstore.write_meta(study_id, meta)
        except CancelledError:
            request = read_json(directory / "request.json")
            for index in range(len(request["tasks"])):
                state = task_state(directory, index)
                if state["status"] != "done":
                    state["status"] = "cancelled"
                    _save(directory, index, state)
            meta = runstore.read_meta(study_id)
            meta.update(status="cancelled", finished_at=datetime.now(UTC).isoformat())
            runstore.write_meta(study_id, meta)
        except Exception as exc:
            handle.cancel.set()
            # Drain all children, including the submission/checkpoint gap. No
            # successor can start while a failed coordinator relinquishes its lease.
            self._drain(study_id)
            meta = runstore.read_meta(study_id)
            meta.update(
                status="failed", stderr=str(exc)[:800], finished_at=datetime.now(UTC).isoformat()
            )
            runstore.write_meta(study_id, meta)
        finally:
            with self._lock:
                handle.lease.close()
                self._handles.pop(study_id, None)
            handle.finished.set()

    def _drain(self, study_id: str) -> None:
        for child in runstore.list_runs(limit=10**9):
            if child.get("study", {}).get("run_id") != study_id:
                continue
            run_id = child["run_id"]
            while self.runs.get_run_status(run_id)["status"] not in _TERMINAL:
                cancelled = self.runs.cancel_run(run_id)
                if not cancelled["cancelled"]:
                    break  # Another live owner must finish its own computation.
                threading.Event().wait(0.1)

    def result(self, study_id: str) -> dict[str, Any]:
        directory = study_dir(study_id)
        with self._lock:
            handle = self._handles.get(study_id)
        meta = runstore.read_meta(study_id)
        request = read_json(directory / "request.json")
        counts: dict[str, int] = {}
        errors = []
        for index, task in enumerate(request["tasks"]):
            state = task_state(directory, index)
            counts[state["status"]] = counts.get(state["status"], 0) + 1
            if state.get("error") and len(errors) < 10:
                errors.append({"task_id": task["id"], "error": state["error"]})
        status = meta["status"]
        if (
            status not in _TERMINAL
            and handle is None
            and meta.get("study_manager_token") == self._token
        ):
            status = "failed"
            meta["stderr"] = "coordinator exited without committing terminal metadata"
        if status not in _TERMINAL and handle is not None and handle.cancel.is_set():
            status = "cancelling"
        return {
            "study_id": study_id,
            "run_id": study_id,
            "status": status,
            "planned": len(request["tasks"]),
            "counts": counts,
            "errors": errors,
            "error": meta.get("stderr"),
            "manifest_file": str(directory / "prepared.json"),
            "tasks_directory": str(directory / "tasks"),
        }

    def cancel(self, study_id: str) -> dict[str, Any]:
        with self._lock:
            handle = self._handles.get(study_id)
        if handle is None:
            return {**self.result(study_id), "cancelled": False}
        handle.cancel.set()
        for child in runstore.list_runs(limit=10**9):
            if child.get("study", {}).get("run_id") == study_id:
                self.runs.cancel_run(child["run_id"])
        return {"run_id": study_id, "status": "cancelling", "cancelled": True}

    async def await_job(self, study_id: str, timeout: float | None = None) -> dict[str, Any]:
        async def wait() -> dict[str, Any]:
            while True:
                result = await asyncio.to_thread(self.result, study_id)
                with self._lock:
                    active = study_id in self._handles
                if result["status"] in _TERMINAL and not active:
                    return result
                await asyncio.sleep(0.1)

        try:
            return await asyncio.wait_for(wait(), timeout)
        except TimeoutError:
            return self.result(study_id)

    def shutdown(self) -> None:
        with self._lock:
            handles = list(self._handles.values())
        for handle in handles:
            handle.cancel.set()
        self._pool.shutdown(wait=True)
