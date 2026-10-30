"""Tests for run management."""

import pytest

from stan_mcp import errors, methods, runstore
from stan_mcp import inference as sample
from tests.fakes import _write_csv_run


def test_inputs_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.persist_inputs("sample-0001", "model {}", {"N": 3})
    got = runstore.load_inputs("sample-0001")
    assert got["source"] == "model {}"
    assert got["data"] == {"N": 3}


def test_max_seq(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    assert runstore.max_seq() == 0
    runstore.write_meta("sample-0005", {"method": "sample", "seq": 5})
    runstore.write_meta("optimize-0003", {"method": "optimize", "seq": 3})
    assert runstore.max_seq() == 5


def test_prune_keep(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    for i in (1, 2, 3):
        runstore.write_meta(f"sample-000{i}", {"method": "sample", "seq": i, "status": "done"})
    removed = runstore.prune_runs(keep=1)
    assert set(removed) == {"sample-0001", "sample-0002"}
    assert runstore.run_dir("sample-0003").is_dir()
    assert not runstore.run_dir("sample-0001").is_dir()


def test_prune_older_than(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta(
        "old-0001", {"method": "sample", "seq": 1, "created_at": "2000-01-01T00:00:00+00:00"}
    )
    runstore.write_meta("new-0002", {"method": "sample", "seq": 2, "created_at": sample._now()})
    removed = runstore.prune_runs(older_than_days=1)
    assert removed == ["old-0001"]


def test_prune_requires_criterion():
    with pytest.raises(ValueError):
        runstore.prune_runs()


def test_prune_refuses_keep_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("sample-0001", {"method": "sample", "seq": 1, "status": "done"})
    with pytest.raises(ValueError):
        runstore.prune_runs(keep=0)
    assert runstore.run_dir("sample-0001").is_dir()  # nothing deleted


def test_list_runs_status_and_method_filters(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("sample-0001", {"method": "sample", "seq": 1, "status": "done"})
    runstore.write_meta("sample-0002", {"method": "sample", "seq": 2, "status": "failed"})
    runstore.write_meta("optimize-0003", {"method": "optimize", "seq": 3, "status": "done"})
    assert [r["run_id"] for r in runstore.list_runs(status="done")] == [
        "optimize-0003",
        "sample-0001",
    ]
    assert [r["run_id"] for r in runstore.list_runs(method="sample")] == [
        "sample-0002",
        "sample-0001",
    ]
    assert [r["run_id"] for r in runstore.list_runs(status="done", method="sample")] == [
        "sample-0001"
    ]


def test_prune_skips_running(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    old = "2000-01-01T00:00:00+00:00"
    runstore.write_meta(
        "run-0001", {"method": "sample", "seq": 1, "status": "running", "created_at": old}
    )
    runstore.write_meta(
        "done-0002", {"method": "sample", "seq": 2, "status": "done", "created_at": old}
    )
    removed = runstore.prune_runs(older_than_days=1)
    assert removed == ["done-0002"]  # running run skipped
    assert runstore.run_dir("run-0001").is_dir()


def test_rerun_reuses_persisted_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta(
        "optimize-0001",
        {"method": "optimize", "status": "done", "seed": 7, "method_args": {"iter": 10}},
    )
    runstore.persist_inputs("optimize-0001", "model {}", {"N": 3})
    mgr = sample.RunManager()
    captured = {}
    monkeypatch.setattr(
        mgr, "start_run", lambda stan, data, **kw: captured.update(stan=stan, data=data, **kw)
    )
    mgr.rerun("optimize-0001")
    assert captured["stan"] == "model {}"
    assert captured["data"] == {"N": 3}
    assert captured["seed"] == 7
    assert captured["method"] == "optimize"
    assert captured["method_args"] == {"iter": 10}


def test_rerun_unknown_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    with pytest.raises(errors.RunError):
        sample.RunManager().rerun("nope")


def test_max_workers_env(monkeypatch):
    from stan_mcp import config

    monkeypatch.setenv("STAN_MCP_MAX_WORKERS", "5")
    assert config.max_workers() == 5
    assert sample.RunManager()._pool._max_workers == 5


def test_progress_reports_elapsed(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done", "duration_s": 1.5})
    out = sample.RunManager().get_run_progress("sample-0001")
    assert out["status"] == "done"
    assert out["elapsed_s"] == 1.5


def test_progress_unknown_run_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    with pytest.raises(errors.RunError):
        sample.RunManager().get_run_progress("nope")


def test_cancel_finished_run_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done"})
    res = sample.RunManager().cancel_run("sample-0001")
    assert res["cancelled"] is False
    assert res["status"] == "done"


def test_cancel_unknown_run_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    with pytest.raises(errors.RunError):
        sample.RunManager().cancel_run("nope")


def test_read_meta_unknown_run_raises_run_error(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    with pytest.raises(errors.RunError):
        runstore.read_meta("nope")


def test_load_inputs_unknown_run_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    # must not look like a real run that simply has nothing persisted
    with pytest.raises(errors.RunError):
        runstore.load_inputs("nope")


def test_list_runs_truncates_stderr(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta(
        "sample-0001",
        {"method": "sample", "seq": 1, "status": "failed", "stderr": "boom\nTraceback\n  line"},
    )
    assert runstore.list_runs()[0]["stderr"] == "boom"
    # ...but the detail view keeps enough to diagnose: CmdStan's first line is
    # "Error during sampling:" and the real cause sits below it.
    assert sample.RunManager().get_run_status("sample-0001")["error"] == "boom\nTraceback\n  line"


def test_get_run_status_bounds_long_stderr(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta(
        "sample-0001", {"method": "sample", "status": "failed", "stderr": "x" * 5000}
    )
    err = sample.RunManager().get_run_status("sample-0001")["error"]
    assert len(err) < 1000 and err.endswith("[…truncated; full text in meta.json]")


def test_progress_counts_persisted_csvs_after_live_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_csv_run("sample-0001")  # 2 draws, no live/
    out = sample.RunManager().get_run_progress("sample-0001")
    assert out["draws_written"] == 2


def test_sample_caps_parallel_chains(monkeypatch):
    from stan_mcp import config

    monkeypatch.setenv("STAN_MCP_MAX_PARALLEL_CHAINS", "3")
    seen = {}

    class _Model:
        def sample(self, **kw):
            seen.update(kw)

    sample.RunManager()._runner("sample", _Model(), None, {"chains": 8})
    assert seen["parallel_chains"] == config.max_parallel_chains() == 3


def test_explicit_parallel_chains_is_respected(monkeypatch):
    kw = methods.build_kwargs("sample", None, None, {"chains": 8, "parallel_chains": 2})
    assert kw["parallel_chains"] == 2


def test_sample_defaults_to_round_tripping_output_precision():
    """CmdStan's default 6 sig figs loses enough precision that a constrained draw
    read back misses its constraint; 17 is where a double round-trips exactly."""
    from stan_mcp import inference as _sample

    seen = {}

    class _Model:
        def sample(self, **kw):
            seen.update(kw)
            return "fit"

    mgr = _sample.RunManager()
    mgr._runner("sample", _Model(), {}, {"chains": 2})
    assert seen["sig_figs"] == 17

    # an explicit value still wins: a huge run may prefer the disk saving
    seen.clear()
    mgr._runner("sample", _Model(), {}, {"chains": 2, "sig_figs": 8})
    assert seen["sig_figs"] == 8
