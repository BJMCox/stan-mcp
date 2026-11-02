import asyncio
import json
import threading
from pathlib import Path

import pytest

from stan_mcp import runstore, studies
from stan_mcp.batch.store import file_hash
from stan_mcp.errors import RunError
from tests.fakes import analytic_study_manager


def _request(tmp_path, **extra):
    fit = tmp_path / "fit.stan"
    fit.write_text(
        "data {real y;} parameters {real theta;} model {theta ~ normal(0,1); y ~ normal(theta,1);}"
    )
    sim = tmp_path / "sim.stan"
    sim.write_text(
        "generated quantities {real theta_sim = normal_rng(0,1); "
        "real y_sim = normal_rng(theta_sim,1);}"
    )
    request = tmp_path / "study.json"
    request.write_text(
        json.dumps(
            {
                "schema": studies.SCHEMA,
                "seed": 42,
                "replicates": 2,
                "max_parallel": 2,
                "tasks": [
                    {
                        "id": "normal",
                        "stan": str(fit),
                        "simulate": {
                            "stan": str(sim),
                            "data_variables": {"y": "y_sim"},
                            "truth_variables": {"theta": "theta_sim"},
                        },
                    }
                ],
                **extra,
            }
        )
    )
    return request


def _wait(manager, study_id):
    result = asyncio.run(manager.await_job(study_id, 10))
    assert result["status"] in {"done", "cancelled", "failed"}, result
    return result


def test_resume_reuses_simulator_and_preserves_success(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    request = _request(tmp_path)
    runner = manager.runs._runner
    persist = manager.runs._persist

    def with_report(run_id, method, fit):
        persist(run_id, method, fit)
        report = runstore.run_dir(run_id) / "predictive" / "report.json"
        report.parent.mkdir()
        report.write_text("{}")

    monkeypatch.setattr(manager.runs, "_persist", with_report)
    failed = False

    def once(method, model, data, kwargs):
        nonlocal failed
        if method == "sample" and not failed:
            failed = True
            raise RuntimeError("one failed fit")
        return runner(method, model, data, kwargs)

    monkeypatch.setattr(manager.runs, "_runner", once)
    study_id = manager.start(str(request))["study_id"]
    result = _wait(manager, study_id)
    assert result["counts"] == {"failed": 1, "done": 1}
    directory = studies.study_dir(study_id)
    states = [studies.task_state(directory, i) for i in range(2)]
    good = next(i for i, state in enumerate(states) if state["status"] == "done")
    good_file = directory / "tasks" / f"{good:05d}.json"
    original = good_file.read_bytes(), good_file.stat().st_ino
    report = runstore.run_dir(states[good]["fit"]["run_id"]) / "predictive" / "report.json"
    report.write_text('{"new_analysis":true}')
    simulator_ids = [state["simulate"]["run_id"] for state in states]
    truths = [state["truth"] for state in states]
    request.unlink()
    (tmp_path / "fit.stan").unlink()
    (tmp_path / "sim.stan").unlink()
    manager.resume(study_id)
    assert _wait(manager, study_id)["counts"] == {"done": 2}
    assert (good_file.read_bytes(), good_file.stat().st_ino) == original
    states = [studies.task_state(directory, i) for i in range(2)]
    assert [state["simulate"]["run_id"] for state in states] == simulator_ids
    assert [state["truth"] for state in states] == truths
    bad = 1 - good
    assert len(states[bad]["attempts"]) == 3
    child_file = Path(next(iter(states[good]["fit"]["hashes"])))
    child_file = runstore.run_dir(states[good]["fit"]["run_id"]) / child_file
    child_file.write_text("changed")
    manager.resume(study_id)
    assert _wait(manager, study_id)["status"] == "failed"
    assert "changed or was pruned" in manager.result(study_id)["error"]
    manager.shutdown()


def test_cancel_during_compile_never_starts_inference_and_lease_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    request = _request(tmp_path, replicates=1)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def compile(source, args):
        entered.set()
        assert release.wait(5)
        return source

    monkeypatch.setattr(manager.runs, "_compile", compile)
    monkeypatch.setattr(manager.runs, "_runner", lambda *args: calls.append(args))
    study_id = manager.start(str(request))["study_id"]
    assert entered.wait(5)
    other = studies.StudyManager(manager.runs)
    with pytest.raises(RunError):
        other.resume(study_id)
    assert manager.cancel(study_id)["cancelled"]
    release.set()
    assert _wait(manager, study_id)["status"] == "cancelled"
    assert calls == []
    assert manager.result(study_id)["counts"] == {"cancelled": 1}
    other.shutdown()
    manager.shutdown()


def test_recovery_after_child_submission_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    request = _request(tmp_path, replicates=1)
    save = studies._save
    failed = False

    def fail(directory, index, state):
        nonlocal failed
        if state["status"] == "simulating" and not failed:
            failed = True
            assert (
                asyncio.run(manager.runs.await_run(state["simulate"]["run_id"], 5))["status"]
                == "done"
            )
            raise OSError("checkpoint disk error")
        return save(directory, index, state)

    monkeypatch.setattr(studies, "_save", fail)
    study_id = manager.start(str(request))["study_id"]
    assert _wait(manager, study_id)["status"] == "failed"
    before = runstore.list_runs(method="fixed_param")
    assert len(before) == 1
    # A completed child is discoverable from ownership written before submission.
    assert before[0]["study"]["run_id"] == study_id
    manager.resume(study_id)
    assert _wait(manager, study_id)["counts"] == {"done": 1}
    after = runstore.list_runs(method="fixed_param")
    assert len(after) == 1
    manager.shutdown()


def test_changed_build_blocks_resume_and_keeps_completed_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    study_id = manager.start(str(_request(tmp_path, replicates=1)))["study_id"]
    assert _wait(manager, study_id)["counts"] == {"done": 1}
    directory = studies.study_dir(study_id)
    checkpoint = directory / "tasks" / "00000.json"
    digest = file_hash(checkpoint)
    monkeypatch.setenv("STAN_THREADS", "true")
    manager.resume(study_id)
    assert _wait(manager, study_id)["status"] == "failed"
    assert file_hash(checkpoint) == digest
    manager.shutdown()


@pytest.mark.parametrize("target", ["child", "study"])
def test_terminal_metadata_failure_is_visible_and_resume_is_safe(tmp_path, monkeypatch, target):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    write = runstore.write_meta

    def fail_terminal(run_id, meta):
        selected = (
            run_id.startswith("sample-") if target == "child" else run_id.startswith("study-")
        )
        if selected and meta.get("status") in {"done", "failed", "cancelled"}:
            raise OSError("terminal metadata disk failure")
        return write(run_id, meta)

    monkeypatch.setattr(runstore, "write_meta", fail_terminal)
    study_id = manager.start(str(_request(tmp_path, replicates=1)))["study_id"]
    try:
        first = _wait(manager, study_id)
        if target == "study":
            assert first["status"] == "failed"
            assert first["counts"] == {"done": 1}
        else:
            assert first["counts"] == {"failed": 1}
        monkeypatch.setattr(runstore, "write_meta", write)
        manager.resume(study_id)
        assert _wait(manager, study_id)["counts"] == {"done": 1}
    finally:
        manager.shutdown()


def test_missing_prepared_manifest_cannot_mix_builds(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    study_id = manager.start(str(_request(tmp_path, replicates=1)))["study_id"]
    try:
        assert _wait(manager, study_id)["counts"] == {"done": 1}
        directory = studies.study_dir(study_id)
        checkpoint = (directory / "tasks" / "00000.json").read_bytes()
        (directory / "prepared.json").unlink()
        monkeypatch.setenv("STAN_THREADS", "true")
        manager.resume(study_id)
        result = _wait(manager, study_id)
        assert result["status"] == "failed"
        assert (directory / "tasks" / "00000.json").read_bytes() == checkpoint
    finally:
        manager.shutdown()


def test_replicated_external_csv_inputs_share_one_owned_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    csv = tmp_path / "input.csv"
    csv.write_text("theta\n1\n")
    request = tmp_path / "study.json"
    request.write_text(
        json.dumps(
            {
                "schema": studies.SCHEMA,
                "seed": 1,
                "replicates": 3,
                "tasks": [
                    {
                        "id": "gq",
                        "stan": "model {}",
                        "data": {"y": 0},
                        "method": "generate_quantities",
                        "method_args": {"csv_files": str(csv)},
                    }
                ],
            }
        )
    )
    study_id = manager.start(str(request))["study_id"]
    try:
        csv.unlink()
        directory = studies.study_dir(study_id)
        owned = list((directory / "inputs").rglob("*.csv"))
        assert len(owned) == 1 and owned[0].read_text() == "theta\n1\n"
        snapshot = json.loads((directory / "request.json").read_text())
        assert all(
            task["fit"]["method_args"]["csv_files"] == [str(owned[0])] for task in snapshot["tasks"]
        )
    finally:
        manager.shutdown()


@pytest.mark.parametrize("unverifiable", [False, True])
def test_resume_refuses_surviving_or_unverifiable_child_process(
    tmp_path, monkeypatch, unverifiable
):
    import os
    from types import SimpleNamespace

    import psutil

    process_probe = studies._no_surviving_process
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    runner = manager.runs._runner
    failed = False

    def once(method, *args):
        nonlocal failed
        if method == "sample" and not failed:
            failed = True
            raise RuntimeError("interrupted fit")
        return runner(method, *args)

    monkeypatch.setattr(manager.runs, "_runner", once)
    study_id = manager.start(str(_request(tmp_path, replicates=1)))["study_id"]
    try:
        assert _wait(manager, study_id)["counts"] == {"failed": 1}
        state = studies.task_state(studies.study_dir(study_id), 0)
        marker = str(runstore.live_dir(state["fit"]["run_id"]))

        def command():
            if unverifiable:
                raise psutil.AccessDenied()
            return ["stan", marker]

        monkeypatch.setattr(
            psutil,
            "process_iter",
            lambda: [
                SimpleNamespace(uids=lambda: SimpleNamespace(real=os.getuid()), cmdline=command)
            ],
        )
        monkeypatch.setattr(studies, "_no_surviving_process", process_probe)
        children = len(runstore.list_runs(method="sample"))
        manager.resume(study_id)
        assert _wait(manager, study_id)["status"] == "failed"
        assert len(runstore.list_runs(method="sample")) == children
    finally:
        manager.shutdown()


def test_invalid_truth_mapping_never_launches_fit_after_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    request = _request(tmp_path, replicates=1)
    contents = json.loads(request.read_text())
    contents["tasks"][0]["simulate"]["truth_variables"] = {"theta": "missing"}
    request.write_text(json.dumps(contents))
    study_id = manager.start(str(request))["study_id"]
    try:
        assert _wait(manager, study_id)["counts"] == {"failed": 1}
        manager.resume(study_id)
        assert _wait(manager, study_id)["counts"] == {"failed": 1}
        assert runstore.list_runs(method="sample") == []
        assert len(runstore.list_runs(method="fixed_param")) == 1
    finally:
        manager.shutdown()


def test_other_live_child_owner_does_not_trap_resume_coordinator(tmp_path, monkeypatch):
    from stan_mcp.inference import RunManager

    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    write = runstore.write_meta

    def fail_child(run_id, meta):
        if run_id.startswith("sample-") and meta.get("status") in {"done", "failed"}:
            raise OSError("child finalization failed")
        return write(run_id, meta)

    monkeypatch.setattr(runstore, "write_meta", fail_child)
    study_id = manager.start(str(_request(tmp_path, replicates=1)))["study_id"]
    try:
        assert _wait(manager, study_id)["counts"] == {"failed": 1}
        monkeypatch.setattr(runstore, "write_meta", write)
        other = studies.StudyManager(RunManager())
        try:
            other.resume(study_id)
            result = _wait(other, study_id)
            assert result["status"] == "failed"
            assert "live server" in result["error"]
        finally:
            other.shutdown()
    finally:
        manager.shutdown()


def test_cancel_preparation_stops_before_compiling_the_next_model(tmp_path, monkeypatch):
    from stan_mcp import compilation

    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    request = _request(tmp_path, replicates=1)
    entered, release = threading.Event(), threading.Event()
    compiled = []

    def compile(source, args=None):
        compiled.append(source)
        entered.set()
        assert release.wait(5)
        return source

    monkeypatch.setattr(compilation, "cmdstan_model", compile)
    study_id = manager.start(str(request))["study_id"]
    try:
        assert entered.wait(5)
        manager.cancel(study_id)
        release.set()
        assert _wait(manager, study_id)["counts"] == {"cancelled": 1}
        assert len(compiled) == 1
        assert runstore.list_runs(method="fixed_param") == []
        assert runstore.list_runs(method="sample") == []
    finally:
        release.set()
        manager.shutdown()
