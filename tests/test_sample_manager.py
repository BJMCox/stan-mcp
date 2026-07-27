import asyncio
from pathlib import Path

import pytest

import diagnostics
import errors
import runstore
import sample


def test_start_and_await(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    mgr = sample.RunManager()

    # Fake compile + runner + persist so no toolchain is needed.
    monkeypatch.setattr(mgr, "_compile", lambda src, args: object())
    monkeypatch.setattr(mgr, "_runner", lambda method, model, data, kw: {"fake": "fit"})
    monkeypatch.setattr(
        mgr,
        "_persist",
        lambda run_id, method, fit: runstore.persist_estimate(run_id, {"lp__": -1.0}),
    )

    started = mgr.start_run("model {}", None, method="optimize")
    assert started["status"] == "running"
    run_id = started["run_id"]

    result = asyncio.run(mgr.await_run(run_id, timeout=5))
    assert result["status"] == "done"
    assert runstore.read_meta(run_id)["status"] == "done"


def test_await_failed_run(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    mgr = sample.RunManager()
    monkeypatch.setattr(mgr, "_compile", lambda src, args: object())

    def boom(method, model, data, kw):
        raise RuntimeError("sampler exploded")

    monkeypatch.setattr(mgr, "_runner", boom)
    started = mgr.start_run("model {}", None, method="sample")
    result = asyncio.run(mgr.await_run(started["run_id"], timeout=5))
    assert result["status"] == "failed"
    assert "exploded" in runstore.read_meta(started["run_id"])["stderr"]


def test_generate_quantities_missing_from_run_id_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    mgr = sample.RunManager()
    monkeypatch.setattr(mgr, "_compile", lambda src, args: object())
    # Real _runner is exercised (not monkeypatched) so it hits the RunError itself.

    started = mgr.start_run("model {}", None, method="generate_quantities", method_args={})
    result = asyncio.run(mgr.await_run(started["run_id"], timeout=5))
    assert result["status"] == "failed"
    assert "from_run_id" in runstore.read_meta(started["run_id"])["stderr"]


def test_sidecar_write_failure_fails_the_run(tmp_path, monkeypatch):
    """persist_run raising on a sidecar-write failure must land the run at
    status="failed", not "done" -- the draws sidecar (diagnostics.py) is now
    the only way to read a run's draws, so a run that never got one must not
    look like an unqualified success. Exercises the real (unmonkeypatched)
    `_persist` -> `diagnostics.persist_run` path; only `_sidecar_from_fit`
    itself is forced to fail, to simulate a real write failure without
    needing a specific malformed fit."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    mgr = sample.RunManager()
    monkeypatch.setattr(mgr, "_compile", lambda src, args: object())

    def runner(method, model, data, kw):
        live = Path(kw["output_dir"])
        (live / "model_1.csv").write_text("# c\nlp__,theta\n-1,0.5\n")
        return object()

    def boom(fit):
        raise RuntimeError("boom")

    monkeypatch.setattr(mgr, "_runner", runner)
    monkeypatch.setattr(diagnostics, "_sidecar_from_fit", boom)

    started = mgr.start_run("model {}", None, method="sample")
    result = asyncio.run(mgr.await_run(started["run_id"], timeout=5))
    assert result["status"] == "failed"
    meta = runstore.read_meta(started["run_id"])
    assert "draws schema" in meta["stderr"]
    # the CSVs must survive the failure -- they're still worth keeping
    assert runstore.csv_paths(started["run_id"])


def test_await_run_unknown_run_id_raises_run_error(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    mgr = sample.RunManager()

    with pytest.raises(errors.RunError):
        asyncio.run(mgr.await_run("does-not-exist"))
