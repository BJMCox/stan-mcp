import asyncio
import json
import multiprocessing
import threading
from pathlib import Path

import pytest

import compile as compilation
import methods
import runstore
import sample
from tests.test_file_inputs import _fit_csv, _GeneratedModel


def _reserve_worker(home, ready, release, output):
    import os

    os.environ["STAN_MCP_HOME"] = home
    ready.put(True)
    release.wait(10)
    run_id, seq = runstore.reserve_run("sample")
    runstore.persist_inputs(run_id, f"model {{ // {seq}\n}}", {"seq": seq})
    output.put((run_id, seq))


def test_cross_process_allocation_preserves_both_inputs(tmp_path, monkeypatch):
    home = str(tmp_path / "home")
    monkeypatch.setenv("STAN_MCP_HOME", home)
    context = multiprocessing.get_context("spawn")
    ready, output = context.Queue(), context.Queue()
    release = context.Event()
    workers = [
        context.Process(target=_reserve_worker, args=(home, ready, release, output))
        for _ in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for _ in workers:
            assert ready.get(timeout=10)
        release.set()
        allocated = [output.get(timeout=10) for _ in workers]
        assert len({run_id for run_id, _ in allocated}) == 2
        for run_id, seq in allocated:
            assert runstore.load_inputs(run_id)["data"] == {"seq": seq}
    finally:
        release.set()
        for worker in workers:
            worker.join(10)
            if worker.is_alive():
                worker.terminate()
                worker.join()


def test_allocation_remembers_pruned_ids_without_rescanning(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    # Bootstrap a pre-counter store, then remove every submission record.
    old = "2000-01-01T00:00:00+00:00"
    runstore.write_meta("sample-0042", {"seq": 42, "status": "done", "finished_at": old})
    first, first_seq = runstore.reserve_run("sample")
    runstore.write_meta(first, {"seq": first_seq, "status": "done", "finished_at": old})
    assert first_seq == 43
    assert set(runstore.prune_runs(older_than_days=1)) == {"sample-0042", first}
    assert runstore.list_runs() == []

    def unexpected_scan() -> int:
        pytest.fail("allocation rescanned a store with an existing counter")

    monkeypatch.setattr(runstore, "max_seq", unexpected_scan)
    second, second_seq = runstore.reserve_run("sample")
    assert second_seq == 44
    assert second != first


@pytest.mark.parametrize("operation", ["list", "allocate"])
def test_run_index_skips_metadata_removed_during_prune(tmp_path, monkeypatch, operation):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    victim = "sample-0042"
    runstore.write_meta(victim, {"seq": 42, "status": "done"})
    disappearing = runstore.run_dir(victim) / "meta.json"
    read_text = Path.read_text

    def prune_before_read(path: Path, *args, **kwargs) -> str:
        if path == disappearing:
            path.unlink()
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", prune_before_read)
    if operation == "list":
        assert runstore.list_runs() == []
    else:
        run_id, seq = runstore.reserve_run("sample")
        assert seq == 43
        assert run_id != victim


def test_startup_recovers_dead_owners_and_keeps_live_server(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    live = runstore.process_owner()
    dead = {**live, "create_time": live["create_time"] - 1}
    for index, owner in enumerate((live, dead, None), 1):
        runstore.write_meta(
            f"sample-{index:04d}",
            {
                "status": "running",
                "seq": index,
                "owner": owner,
            },
        )
    manager = sample.RunManager()
    try:
        assert runstore.read_meta("sample-0001")["status"] == "running"
        for run_id in ("sample-0002", "sample-0003"):
            assert asyncio.run(manager.await_run(run_id))["status"] == "failed"
        assert manager.cancel_run("sample-0001")["reason"] == "owned by another server"
        assert set(runstore.prune_runs(keep=1)) == {"sample-0002"}
    finally:
        manager._pool.shutdown(wait=True)


@pytest.mark.parametrize("runner_fails", [False, True])
def test_successful_cancellation_survives_repeat_and_completion(
    tmp_path, monkeypatch, runner_fails
):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = sample.RunManager(max_workers=1)
    entered, release = threading.Event(), threading.Event()

    def runner(*args):
        entered.set()
        assert release.wait(5)
        if runner_fails:
            raise RuntimeError("killed")
        return object()

    monkeypatch.setattr(manager, "_compile", lambda *args: object())
    monkeypatch.setattr(manager, "_runner", runner)
    monkeypatch.setattr(manager, "_persist", lambda *args: None)
    kills = iter(([123], []))
    monkeypatch.setattr(sample, "_kill_run_processes", lambda run_id: next(kills))
    try:
        started = manager.start_run("model {}", None)
        assert entered.wait(5)
        assert manager.cancel_run(started["run_id"])["cancelled"]
        assert manager.cancel_run(started["run_id"])["cancelled"]
        release.set()
        assert asyncio.run(manager.await_run(started["run_id"], timeout=5))["status"] == "cancelled"
    finally:
        release.set()
        manager._pool.shutdown(wait=True)


def test_managed_gq_and_rerun_survive_pruning_source(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    prior = "sample-0001"
    runstore.write_meta(prior, {"status": "done", "method": "sample", "seq": 1})
    runstore.persist_csvs(prior, [str(_fit_csv(tmp_path / "input.csv", "sample"))])
    manager = sample.RunManager(max_workers=1)
    entered, release = threading.Event(), threading.Event()

    def compile_model(*args):
        entered.set()
        assert release.wait(5)
        return _GeneratedModel()

    monkeypatch.setattr(manager, "_compile", compile_model)
    try:
        first = manager.start_run(
            "model {}", None, method="generate_quantities", method_args={"from_run_id": prior}
        )
        assert entered.wait(5)
        assert runstore.prune_runs(keep=1) == [prior]
        release.set()
        assert asyncio.run(manager.await_run(first["run_id"], timeout=5))["status"] == "done"
        second = manager.rerun(first["run_id"])
        assert asyncio.run(manager.await_run(second["run_id"], timeout=5))["status"] == "done"
        assert runstore.read_meta(second["run_id"])["input_method"] == "sample"
    finally:
        release.set()
        manager._pool.shutdown(wait=True)


def test_cancellation_commits_before_killed_worker_can_finish(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = sample.RunManager(max_workers=1)
    entered, release, persisted = threading.Event(), threading.Event(), threading.Event()

    def runner(*args):
        entered.set()
        assert release.wait(5)
        return object()

    def kill(run_id):
        release.set()
        # Force the worker past successful persistence while the kill helper
        # has not yet returned its successful result to cancel_run.
        assert persisted.wait(5)
        return [123]

    monkeypatch.setattr(manager, "_compile", lambda *args: object())
    monkeypatch.setattr(manager, "_runner", runner)
    monkeypatch.setattr(manager, "_persist", lambda *args: persisted.set())
    monkeypatch.setattr(sample, "_kill_run_processes", kill)
    try:
        started = manager.start_run("model {}", None)
        assert entered.wait(5)
        assert manager.cancel_run(started["run_id"])["cancelled"]
        assert asyncio.run(manager.await_run(started["run_id"], timeout=5))["status"] == "cancelled"
    finally:
        release.set()
        manager._pool.shutdown(wait=True)


def test_failed_submission_cleans_owned_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    inputs = [_fit_csv(tmp_path / f"input-{i}.csv", "sample") for i in range(2)]
    copyfile = runstore.shutil.copyfile

    def fail_second(source, destination):
        if Path(source) == inputs[1]:
            raise OSError("disk failure")
        return copyfile(source, destination)

    monkeypatch.setattr(runstore.shutil, "copyfile", fail_second)
    manager = sample.RunManager()
    try:
        with pytest.raises(OSError):
            manager.start_run(
                "model {}",
                None,
                method="generate_quantities",
                method_args={"csv_files": [str(path) for path in inputs]},
            )
        assert list((tmp_path / "home" / "runs").glob("generate_quantities-*")) == []
    finally:
        manager._pool.shutdown(wait=True)


async def test_waiters_deregister_after_timeout_and_task_cancellation(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = sample.RunManager(max_workers=1)
    release = threading.Event()
    monkeypatch.setattr(manager, "_compile", lambda *args: release.wait(5))
    monkeypatch.setattr(manager, "_runner", lambda *args: object())
    monkeypatch.setattr(manager, "_persist", lambda *args: None)
    try:
        started = manager.start_run("model {}", None)
        handle = manager._handles[started["run_id"]]
        assert (await manager.await_run(started["run_id"], timeout=0))["status"] == "running"
        assert not handle.aio_events
        pending = asyncio.create_task(manager.await_run(started["run_id"]))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not handle.aio_events
        release.set()
        assert (await manager.await_run(started["run_id"], timeout=5))["status"] == "done"
    finally:
        release.set()
        manager._pool.shutdown(wait=True)


def test_rerun_owns_resolved_inits_and_laplace_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    initialization = tmp_path / "init.json"
    initialization.write_text(json.dumps({"theta": [1, 2]}))
    mode = _fit_csv(tmp_path / "mode.csv", "optimize")
    original_mode = mode.read_text()
    calls = []
    manager = sample.RunManager(max_workers=1)
    monkeypatch.setattr(manager, "_compile", lambda *args: object())
    monkeypatch.setattr(
        manager, "_runner", lambda method, model, data, kw: calls.append((method, kw))
    )
    monkeypatch.setattr(manager, "_persist", lambda *args: None)
    try:
        first = manager.start_run("model {}", None, method="optimize", inits=str(initialization))
        assert asyncio.run(manager.await_run(first["run_id"], timeout=5))["status"] == "done"
        initialization.unlink()
        second = manager.rerun(first["run_id"])
        assert asyncio.run(manager.await_run(second["run_id"], timeout=5))["status"] == "done"
        assert [kw["inits"] for _, kw in calls] == [{"theta": [1, 2]}] * 2
        laplace = manager.start_run(
            "model {}", None, method="laplace", method_args={"mode": str(mode), "jacobian": False}
        )
        assert asyncio.run(manager.await_run(laplace["run_id"], timeout=5))["status"] == "done"
        mode.unlink()
        repeated = manager.rerun(laplace["run_id"])
        assert asyncio.run(manager.await_run(repeated["run_id"], timeout=5))["status"] == "done"
        assert all(Path(kw["mode"]).read_text() == original_mode for _, kw in calls[2:])
        assert all(kw["jacobian"] is False for _, kw in calls)
        assert methods.build_kwargs("laplace", None, None, {})["jacobian"] is True
    finally:
        manager._pool.shutdown(wait=True)


def test_rerun_retains_expanded_includes_and_compile_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    included = tmp_path / "parameters.stan"
    included.write_text("parameters {real theta;}")
    model = tmp_path / "model.stan"
    model.write_text("#include parameters.stan\nmodel {theta ~ normal(0, 1);}")
    source = compilation.resolve_source(str(model), ["--O1"])
    seen = []
    manager = sample.RunManager(max_workers=1)
    monkeypatch.setattr(
        manager,
        "_compile",
        lambda source, args: seen.append(
            (str(source), compilation.model_manifest(source)["stanc_args"])
        ),
    )
    monkeypatch.setattr(manager, "_runner", lambda *args: object())
    monkeypatch.setattr(manager, "_persist", lambda *args: None)
    try:
        first = manager.start_run(source, None)
        assert asyncio.run(manager.await_run(first["run_id"], timeout=5))["status"] == "done"
        included.unlink()
        model.unlink()
        second = manager.rerun(first["run_id"])
        assert asyncio.run(manager.await_run(second["run_id"], timeout=5))["status"] == "done"
        assert seen[0] == seen[1]
        assert seen[0][1] == ["--O1"]
        assert "parameters {real theta;}" in seen[0][0]
    finally:
        manager._pool.shutdown(wait=True)
