import asyncio
import fcntl
import json
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stan_mcp import compilation, runstore
from stan_mcp.batch import evaluation as batcheval
from stan_mcp.batch import jobs as batchjobs
from stan_mcp.batch import store as batchstore
from stan_mcp.errors import RunError, ValidationError


def test_packaged_worker_exits_cleanly_on_stop_sentinel(tmp_path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    watch_child, watch_parent = context.Pipe(duplex=False)
    process = context.Process(
        target=batchjobs._worker, args=({}, str(tmp_path), child, watch_child)
    )
    try:
        process.start()
        child.close()
        watch_child.close()
        parent.send(None)
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
        if process.pid is not None:
            process.join(10)
        for connection in (parent, child, watch_child, watch_parent):
            connection.close()


def _wait(predicate: Callable[[], Any]) -> Any:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.01)
    raise AssertionError("batch progress timed out")


def _fake_worker(plan: dict, directory: str, connection: Any, parent: Any) -> None:
    def watch_parent() -> None:
        try:
            parent.recv()
        except EOFError:
            os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    control = Path(plan["control"])
    manifest = batchstore.load(plan["input_manifest"])
    while (index := connection.recv()) is not None:
        with (control / "counter.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            counter = control / "counter.json"
            counts = (
                json.loads(counter.read_text()) if counter.exists() else {"active": 0, "max": 0}
            )
            counts["active"] += 1
            counts["max"] = max(counts["max"], counts["active"])
            batchstore.atomic_json(counter, counts)
        (control / f"started-{os.getpid()}-{index}").touch()
        release = control / f"release-{index}"
        while not release.exists():
            time.sleep(0.01)
        arrays = batchstore.read_shard(manifest, manifest["shards"][index])
        arrays.update(
            status=np.zeros(len(arrays["row_id"]), dtype=np.int64),
            log_density=-np.sum(arrays["points"] ** 2, axis=1) / 2,
        )
        if plan.get("point_failure") and index == 1:
            arrays["status"][0] = 1
            arrays["log_density"][0] = np.nan
        descriptor = batchstore.write_shard(Path(directory), index, arrays)
        with (control / "counter.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            counts = json.loads(counter.read_text())
            counts["active"] -= 1
            batchstore.atomic_json(counter, counts)
        connection.send({"descriptor": descriptor})
    parent.close()
    connection.close()


def _crash_owner(home: str, points: str, control: str, ready: Any) -> None:
    os.environ["STAN_MCP_HOME"] = home
    compilation._build_identity = lambda: {"test": "build"}
    manager = batchjobs.BatchManager(max_workers=1)
    manager._prepare = _prepare
    manager._worker_target = _fake_worker
    request = Path(control) / "crash-request.json"
    batchstore.atomic_json(request, {"stan": "model {}", "points": points, "control": control})
    ready.send(manager.start("evaluate", str(request))["run_id"])
    ready.close()
    threading.Event().wait()


def _prepare(kind: str, request: dict, directory: Path) -> dict:
    manifest = batchstore.load(request["points"])
    return {
        "input_manifest": request["points"],
        "control": request["control"],
        "point_failure": request.get("point_failure", False),
        "output_metadata": {
            "kind": "results",
            "point_space": "constrained",
            "parameter_names": manifest["parameter_names"],
            "quantities": ["log_density"],
            "target": {"propto": False, "jacobian": False},
        },
    }


@pytest.fixture
def batch_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(compilation, "_build_identity", lambda: {"test": "build"})
    monkeypatch.setattr(batcheval, "validate_plan", lambda plan: None)
    source = tmp_path / "points"
    descriptors = [
        batchstore.write_shard(
            source,
            index,
            {"row_id": np.array([index], dtype=np.int64), "points": np.array([[float(index)]])},
        )
        for index in range(3)
    ]
    points = batchstore.publish(
        source,
        {"kind": "points", "point_space": "constrained", "parameter_names": ["x"]},
        descriptors,
    )
    control = tmp_path / "control"
    control.mkdir()
    managers: list[batchjobs.BatchManager] = []

    def manager(cap: int = 1) -> batchjobs.BatchManager:
        result = batchjobs.BatchManager(max_workers=cap)
        monkeypatch.setattr(result, "_prepare", _prepare)
        monkeypatch.setattr(result, "_worker_target", _fake_worker)
        managers.append(result)
        return result

    def request(**extra: Any) -> str:
        path = tmp_path / f"request-{len(list(tmp_path.glob('request-*')))}.json"
        batchstore.atomic_json(
            path, {"stan": "model {}", "points": str(points), "control": str(control), **extra}
        )
        return str(path)

    yield manager, request, control, points
    for index in range(3):
        (control / f"release-{index}").touch()
    for value in managers:
        value.shutdown()


def test_background_start_and_wait_cancellation_are_independent(batch_setup, monkeypatch):
    create, request, control, _ = batch_setup
    manager = create()
    entered, release = threading.Event(), threading.Event()

    def prepare(kind: str, value: dict, directory: Path) -> dict:
        entered.set()
        assert release.wait(10)
        return _prepare(kind, value, directory)

    monkeypatch.setattr(manager, "_prepare", prepare)
    try:
        started = manager.start("evaluate", request())
        job_id = started["run_id"]
        assert entered.wait(5)

        async def wait() -> None:
            assert (await manager.await_job(job_id, timeout=0))["status"] == "running"
            pending = asyncio.create_task(manager.await_job(job_id))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

        asyncio.run(wait())
        manager.cancel(job_id)
        release.set()
        assert asyncio.run(manager.await_job(job_id, timeout=5))["status"] == "cancelled"
        assert not list(control.glob("started-*"))
        assert manager.cancel(job_id)["cancelled"]
    finally:
        release.set()


def test_cancel_resume_preserves_completed_shards_and_owns_inputs(batch_setup):
    create, request, control, points = batch_setup
    manager = create()
    (control / "release-0").touch()
    started = manager.start("evaluate", request(point_failure=True))
    job_id = started["run_id"]
    _wait(lambda: manager.progress(job_id).get("processed_shards") == 1)
    first = Path(manager.result(job_id)["manifest"]).parent / "00000000.safetensors"
    identity = first.stat().st_ino, batchstore.file_hash(first)
    assert manager.cancel(job_id)["cancelled"]
    assert asyncio.run(manager.await_job(job_id, timeout=10))["status"] == "cancelled"
    assert not manager._handles
    # Neither the request nor its original manifests are needed after submission.
    points.unlink()
    for index in (1, 2):
        (control / f"release-{index}").touch()
    manager.resume(job_id)
    completed = asyncio.run(manager.await_job(job_id, timeout=10))
    assert completed["status"] == "done"
    assert identity == (first.stat().st_ino, batchstore.file_hash(first))
    result = batchstore.load(completed["manifest"])
    assert result["count"] == 3
    assert batchstore.read_shard(result, result["shards"][1])["status"].tolist() == [1]
    assert completed["processed_shards"] == 3


def test_shared_worker_cap_and_parallel_order(batch_setup):
    create, request, control, _ = batch_setup
    manager = create(2)
    started = [manager.start("evaluate", request(workers=2)) for _ in range(2)]
    _wait(lambda: len(list(control.glob("started-*"))) == 2)
    counter = json.loads((control / "counter.json").read_text())
    assert counter["max"] == 2
    for index in (2, 1, 0):
        (control / f"release-{index}").touch()
    outputs = [asyncio.run(manager.await_job(value["run_id"], timeout=10)) for value in started]
    assert all(value["status"] == "done" for value in outputs)
    assert json.loads((control / "counter.json").read_text())["max"] == 2
    values = []
    for output in outputs:
        manifest = batchstore.load(output["manifest"])
        arrays = [batchstore.read_shard(manifest, descriptor) for descriptor in manifest["shards"]]
        assert [int(row["row_id"][0]) for row in arrays] == [0, 1, 2]
        values.append([row["log_density"][0] for row in arrays])
    assert values[0] == values[1] == [0, -0.5, -2]


@pytest.mark.parametrize(
    "changed",
    ["input", "settings", "completed", "checkpoint", "wrong_rows", "duplicate", "unknown_index"],
)
def test_resume_refuses_changed_owned_computation(batch_setup, changed):
    create, request, control, _ = batch_setup
    manager = create()
    (control / "release-0").touch()
    job_id = manager.start("evaluate", request())["run_id"]
    _wait(lambda: manager.progress(job_id).get("processed_shards") == 1)
    manager.cancel(job_id)
    assert asyncio.run(manager.await_job(job_id, timeout=10))["status"] == "cancelled"
    directory = runstore.run_dir(job_id) / "batch"
    if changed == "input":
        path = directory / "inputs" / "points" / "00000000.safetensors"
    elif changed == "settings":
        path = directory / "plan.json"
    elif changed == "completed":
        path = directory / "results" / "00000000.safetensors"
    else:
        path = directory / "checkpoint.json"
    if changed in {"checkpoint", "wrong_rows", "duplicate", "unknown_index"}:
        state = json.loads(path.read_text())
        if changed == "checkpoint":
            state = {}
        elif changed == "wrong_rows":
            state = {"1": state["0"]}
        elif changed == "duplicate":
            state["1"] = state["0"]
        else:
            state = {"3": state["0"]}
        batchstore.atomic_json(path, state)
    else:
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((RunError, ValidationError)):
        manager.resume(job_id)
    assert manager.status(job_id)["status"] == "cancelled"


def test_exclusive_lease_blocks_other_manager_resume(batch_setup):
    create, request, control, _ = batch_setup
    first = create()
    second = create()
    job_id = first.start("evaluate", request())["run_id"]
    _wait(lambda: list(control.glob("started-*")))
    with pytest.raises(RunError):
        second.resume(job_id)
    assert second.cancel(job_id)["reason"] == "owned by another server"
    first.cancel(job_id)
    assert asyncio.run(first.await_job(job_id, timeout=10))["status"] == "cancelled"


def test_worker_crash_keeps_checkpoint_and_repeats_only_unfinished_work(batch_setup):
    create, request, control, _ = batch_setup
    manager = create()
    (control / "release-0").touch()
    job_id = manager.start("evaluate", request())["run_id"]
    _wait(lambda: manager.progress(job_id).get("processed_shards") == 1)
    _wait(lambda: len(list(control.glob("started-*"))) == 2)
    handle = manager._handles[job_id]
    with handle.lock:
        handle.processes[0].kill()
    assert asyncio.run(manager.await_job(job_id, timeout=10))["status"] == "failed"
    # An interrupted publication was never checkpointed and may be replaced.
    output = runstore.run_dir(job_id) / "batch" / "results"
    (output / "00000001.safetensors").write_bytes(b"unfinished")
    (control / "release-1").touch()
    (control / "release-2").touch()
    manager.resume(job_id)
    completed = asyncio.run(manager.await_job(job_id, timeout=10))
    assert completed["status"] == "done"
    assert batchstore.load(completed["manifest"])["count"] == 3


def test_manifest_write_failure_preserves_checkpointed_shards(batch_setup, monkeypatch):
    create, request, control, _ = batch_setup
    manager = create()
    (control / "release-0").touch()
    job_id = manager.start("evaluate", request())["run_id"]
    _wait(lambda: manager.progress(job_id).get("processed_shards") == 1)
    publish = batchstore.publish
    failed = threading.Event()

    def disk_failure(directory: Path, metadata: dict, descriptors: list[dict]) -> Path:
        if directory.name == "results" and len(descriptors) == 2 and not failed.is_set():
            failed.set()
            raise OSError("manifest disk failure")
        return publish(directory, metadata, descriptors)

    monkeypatch.setattr(batchstore, "publish", disk_failure)
    (control / "release-1").touch()
    assert asyncio.run(manager.await_job(job_id, timeout=10))["status"] == "failed"
    output = runstore.run_dir(job_id) / "batch" / "results"
    committed = [
        (path, path.stat().st_ino, batchstore.file_hash(path))
        for path in sorted(output.glob("*.safetensors"))
    ]
    assert len(committed) == 2
    (control / "release-2").touch()
    manager.resume(job_id)
    finished = asyncio.run(manager.await_job(job_id, timeout=10))
    assert finished["status"] == "done"
    assert batchstore.load(finished["manifest"])["count"] == 3
    assert all(
        (path.stat().st_ino, batchstore.file_hash(path)) == (inode, digest)
        for path, inode, digest in committed
    )


@pytest.mark.parametrize("failure", ["directory", "pipe", "thread"])
def test_setup_failure_finalizes_job_and_releases_resources(batch_setup, monkeypatch, failure):
    create, request, control, _ = batch_setup
    manager = create(2)
    failed = threading.Event()
    if failure == "directory":
        mkdir = Path.mkdir

        def fail_directory(path: Path, *args: Any, **kwargs: Any) -> None:
            if path.name == "results" and not failed.is_set():
                failed.set()
                raise OSError("output directory failure")
            mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_directory)
    elif failure == "pipe":
        context = multiprocessing.get_context("spawn")
        pipe = context.Pipe

        def fail_pipe(*args: Any, **kwargs: Any) -> Any:
            if not failed.is_set():
                failed.set()
                raise OSError("pipe exhaustion")
            return pipe(*args, **kwargs)

        monkeypatch.setattr(context, "Pipe", fail_pipe)
    else:
        start = threading.Thread.start
        starts = 0

        def fail_thread(thread: threading.Thread) -> None:
            nonlocal starts
            starts += 1
            if starts == 3:
                raise RuntimeError("thread creation failure")
            start(thread)

        monkeypatch.setattr(threading.Thread, "start", fail_thread)
    job_id = manager.start("evaluate", request(workers=2))["run_id"]
    assert asyncio.run(manager.await_job(job_id, timeout=10))["status"] == "failed"
    for index in range(3):
        (control / f"release-{index}").touch()
    next_id = manager.start("evaluate", request())["run_id"]
    finished = asyncio.run(manager.await_job(next_id, timeout=10))
    assert finished["status"] == "done"
    assert batchstore.load(finished["manifest"])["count"] == 3


@pytest.mark.parametrize("operation", ["start", "resume"])
def test_coordinator_start_failure_releases_ownership_for_retry(
    batch_setup, monkeypatch, operation
):
    create, request, control, _ = batch_setup
    manager = create()
    job_id = None
    if operation == "resume":
        (control / "release-0").touch()
        job_id = manager.start("evaluate", request())["run_id"]
        _wait(lambda: manager.progress(job_id).get("processed_shards") == 1)
        manager.cancel(job_id)
        assert asyncio.run(manager.await_job(job_id, timeout=10))["status"] == "cancelled"
    start = threading.Thread.start
    failed = False

    def fail_start(thread: threading.Thread) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("coordinator thread creation failure")
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError):
        if job_id is None:
            manager.start("evaluate", request())
        else:
            manager.resume(job_id)
    manager.shutdown()
    if job_id is None:
        assert runstore.list_runs() == []
    else:
        assert asyncio.run(manager.await_job(job_id, timeout=0))["status"] == "cancelled"
    for index in range(3):
        (control / f"release-{index}").touch()
    retry = manager.start("evaluate", request()) if job_id is None else manager.resume(job_id)
    finished = asyncio.run(manager.await_job(retry["run_id"], timeout=10))
    assert finished["status"] == "done"
    assert batchstore.load(finished["manifest"])["count"] == 3


def test_missing_input_shard_rejects_submission_before_workers_start(batch_setup):
    create, request, control, points = batch_setup
    manager = create()
    (points.parent / "00000001.safetensors").unlink()
    with pytest.raises(ValidationError):
        manager.start("evaluate", request())
    assert runstore.list_runs() == []
    assert not list(control.glob("started-*"))


@pytest.mark.parametrize("bad_checkpoint", [False, True])
def test_comparison_cancel_resume_keeps_verdict_separate_from_job_status(
    batch_setup, monkeypatch, tmp_path, bad_checkpoint
):
    from stan_mcp.batch import comparison as batchcompare

    create, _, _, points = batch_setup
    source = batchstore.load(points)
    result_dir = tmp_path / "external"
    descriptors = []
    for index, descriptor in enumerate(source["shards"]):
        arrays = batchstore.read_shard(source, descriptor)
        arrays.update(
            status=np.zeros(len(arrays["row_id"]), dtype=np.int64),
            log_density=-np.sum(arrays["points"] ** 2, axis=1) / 2,
        )
        descriptors.append(batchstore.write_shard(result_dir, index, arrays))
    result = batchstore.publish(
        result_dir,
        {
            "kind": "results",
            "point_space": "constrained",
            "parameter_names": ["x"],
            "quantities": ["log_density"],
            "target": {"propto": False, "jacobian": False},
        },
        descriptors,
    )
    request_file = tmp_path / "comparison.json"
    batchstore.atomic_json(
        request_file,
        {
            "left": str(result),
            "right": str(result),
            "tolerances": {"log_density": {"atol": 0, "rtol": 0}},
        },
    )
    manager = create()
    monkeypatch.setattr(manager, "_prepare", batchjobs.BatchManager._prepare)
    execute = batchcompare.execute
    pause = threading.Event()
    release = threading.Event()

    def compare(plan: dict, directory: Path, cancelled: Any, progress: Any) -> dict:
        def progressed(done: int, total: int) -> None:
            progress(done, total)
            if done == 1 and not release.is_set():
                pause.set()
                assert release.wait(10)

        return execute(plan, directory, cancelled, progressed)

    monkeypatch.setattr(batchcompare, "execute", compare)
    try:
        job_id = manager.start("compare", str(request_file))["run_id"]
        assert pause.wait(10)
        assert manager.progress(job_id)["processed_rows"] == 1
        manager.cancel(job_id)
        release.set()
        assert asyncio.run(manager.await_job(job_id, timeout=10))["status"] == "cancelled"
        saved = runstore.run_dir(job_id) / "batch" / "results" / "00000000.safetensors"
        inode = saved.stat().st_ino
        result.unlink()
        if bad_checkpoint:
            output = saved.parent
            checkpoint = batchstore.read_json(output / "comparison-checkpoint.json")
            descriptor = batchstore.write_shard(output, 3, arrays)
            descriptor.update(
                comparison_summary=checkpoint["completed"][0]["comparison_summary"], offset=None
            )
            checkpoint["completed"].append(descriptor)
            batchstore.atomic_json(output / "comparison-checkpoint.json", checkpoint)
            with pytest.raises(RunError):
                manager.resume(job_id)
            assert manager.status(job_id)["status"] == "cancelled"
            return
        manager.resume(job_id)
        finished = asyncio.run(manager.await_job(job_id, timeout=10))
        assert finished["status"] == "done"
        assert finished["summary"]["verdict"] == "pass"
        assert finished["processed_rows"] == finished["completed_shards"] == 3
        assert "completed" not in finished and "completed" not in finished["meta"]["result"]
        assert saved.stat().st_ino == inode
    finally:
        release.set()


def test_dead_coordinator_is_discoverable_and_resumable(batch_setup, tmp_path):
    create, _, control, points = batch_setup
    (control / "release-0").touch()
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    owner = context.Process(
        target=_crash_owner, args=(str(tmp_path / "home"), str(points), str(control), send)
    )
    owner.start()
    try:
        assert receive.poll(10)
        job_id = receive.recv()
        _wait(lambda: runstore.read_meta(job_id).get("processed_shards") == 1)
        owner.kill()
        owner.join(10)
        manager = create()
        assert manager.status(job_id)["status"] == "failed"
        assert runstore.list_runs()[0]["run_id"] == job_id
        for index in (1, 2):
            (control / f"release-{index}").touch()
        manager.resume(job_id)
        finished = asyncio.run(manager.await_job(job_id, timeout=10))
        assert finished["status"] == "done"
        assert batchstore.load(finished["manifest"])["count"] == 3
    finally:
        if owner.is_alive():
            owner.kill()
        owner.join(10)
        receive.close()
        send.close()
