"""Tests for the post-review enhancements (handles, repro, diagnostics, cancel)."""

import json

import numpy as np
import pytest
from cmdstanpy.stanfit.metadata import InferenceMetadata

import compile as c
import diagnostics
import errors
import methods
import runstore
import sample


# --- methods: inits gating (laplace / generate_quantities reject inits) ---
def test_inits_rejected_for_laplace():
    with pytest.raises(errors.ValidationError):
        methods.build_kwargs("laplace", None, {"theta": 0.0}, {})


def test_inits_rejected_for_generate_quantities():
    with pytest.raises(errors.ValidationError):
        methods.build_kwargs("generate_quantities", None, {"theta": 0.0}, {})


def test_inits_accepted_for_sample():
    kw = methods.build_kwargs("sample", None, {"theta": 0.0}, {})
    assert kw["inits"] == {"theta": 0.0}


# --- compile: model_id / data_id handles + path validation ---
def test_model_id_handle_resolves_to_source(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    src = "model {}"
    c.stan_file_for(src)  # writes compiled/<model_id>.stan
    assert c.resolve_source(c.model_id(src)) == src


def test_unknown_model_id_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    with pytest.raises(errors.ValidationError):
        c.resolve_source("0" * 16)


def test_missing_stan_file_raises():
    with pytest.raises(errors.ValidationError):
        c.resolve_source("does/not/exist.stan")


def test_register_data_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    did = c.register_data({"N": 3})
    assert c.canonical_data(did) == {"N": 3}


def test_unknown_data_id_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    with pytest.raises(errors.ValidationError):
        c.canonical_data("f" * 16)


def test_missing_json_file_raises():
    with pytest.raises(errors.ValidationError):
        c.canonical_data("does/not/exist.json")


def test_compiled_so_builds_once_under_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import sys
    import threading
    import time
    from pathlib import Path

    monkeypatch.setattr(c.toolchain, "ensure_bridgestan", lambda: None)
    calls = []

    class _FakeBS:
        @staticmethod
        def compile_model(stan_path, stanc_args=None):
            calls.append(1)
            time.sleep(0.15)  # widen the window a naive impl would race in
            so = Path(str(stan_path).replace(".stan", "_model.so"))
            so.write_text("so")
            return str(so)

    monkeypatch.setitem(sys.modules, "bridgestan", _FakeBS)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(c.compiled_so("model {}"))) for _ in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1  # 5 concurrent callers, one build
    assert all(str(r).endswith("_model.so") for r in results)


# --- runstore: repro inputs, max_seq, prune ---
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


# --- diagnostics: extract_draws stable shape + thin validation ---
def _write_sample_run(run_id):
    import arviz as az

    idata = az.from_dict(
        {
            "posterior": {
                "theta": np.arange(20.0).reshape(2, 10),
                "beta": np.arange(2 * 10 * 3, dtype=float).reshape(2, 10, 3),
            }
        }
    )
    runstore.write_meta(run_id, {"method": "sample", "status": "done"})
    idata.to_netcdf(str(runstore.idata_path(run_id)))


def test_extract_draws_scalar_is_flat_list(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_sample_run("sample-0001")
    out = diagnostics.extract_draws("sample-0001", "theta")
    assert len(out["draws"]) == 20
    assert isinstance(out["draws"][0], float)


def test_extract_draws_vector_is_list_of_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_sample_run("sample-0001")
    out = diagnostics.extract_draws("sample-0001", "beta")
    assert len(out["draws"]) == 20
    assert isinstance(out["draws"][0], list) and len(out["draws"][0]) == 3


def test_extract_draws_single_draw_stays_a_list(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_sample_run("sample-0001")
    out = diagnostics.extract_draws("sample-0001", "theta", thin=999)
    assert isinstance(out["draws"], list)  # not a bare float
    assert len(out["draws"]) == 1


def test_extract_draws_rejects_bad_thin(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_sample_run("sample-0001")
    with pytest.raises(errors.ValidationError):
        diagnostics.extract_draws("sample-0001", "theta", thin=0)


# --- diagnostics: verdict logic ---
def test_verdict_pass():
    assert diagnostics._verdict({"max_rhat": 1.0, "min_ess_bulk": 500})["verdict"] == "pass"


def test_verdict_flags_issues():
    v = diagnostics._verdict({"max_rhat": 1.05, "divergences": 3, "min_ess_bulk": 500})
    assert v["verdict"] == "warn"
    assert len(v["issues"]) == 2


def test_extract_draws_by_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_sample_run("sample-0001")
    out = diagnostics.extract_draws("sample-0001", "theta", by_chain=True)
    assert out["by_chain"] is True
    assert len(out["draws"]) == 2 and len(out["draws"][0]) == 10  # 2 chains x 10 draws


# --- diagnostics over a full idata with a sample_stats group (offline) ---
def _write_run_with_stats(run_id, *, diverging, tree_depth, max_td=10):
    import arviz as az

    rng = np.random.default_rng(0)
    nchain, ndraw = 2, 1000
    post = {"theta": rng.normal(size=(nchain, ndraw))}
    ss = {
        "diverging": np.full((nchain, ndraw), diverging, dtype=bool),
        "tree_depth": np.full((nchain, ndraw), tree_depth),
        "energy": rng.normal(size=(nchain, ndraw)),
        "acceptance_rate": np.full((nchain, ndraw), 0.9),
    }
    idata = az.from_dict({"posterior": post, "sample_stats": ss})
    runstore.write_meta(
        run_id, {"method": "sample", "status": "done", "method_args": {"max_treedepth": max_td}}
    )
    idata.to_netcdf(str(runstore.idata_path(run_id)))


def test_diagnostics_clean_run_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_run_with_stats("sample-0001", diverging=False, tree_depth=3)
    diag = diagnostics.run_diagnostics("sample-0001")
    assert diag["divergences"] == 0
    assert diag["treedepth_saturations"] == 0
    assert diag["max_rhat"] < 1.01
    assert diag["min_ess_bulk"] >= 400
    assert diag["verdict"] == "pass"


def test_diagnostics_bad_run_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_run_with_stats("sample-0001", diverging=True, tree_depth=10, max_td=10)
    diag = diagnostics.run_diagnostics("sample-0001")
    assert diag["divergences"] == 2000
    assert diag["treedepth_saturations"] == 2000
    assert diag["verdict"] == "warn"
    joined = " ".join(diag["issues"])
    assert "divergent" in joined and "max_treedepth" in joined


def test_method_diagnostics_extraction():
    class _VB:
        eta = 0.5

    class _PF:
        is_resampled = True

    class _Mode:
        optimized_params_dict = {"lp__": -3.0}

    class _LP:
        mode = _Mode()

    assert diagnostics._method_diagnostics("variational", _VB()) == {"eta": 0.5}
    assert diagnostics._method_diagnostics("pathfinder", _PF()) == {"is_resampled": True}
    assert diagnostics._method_diagnostics("laplace", _LP()) == {"mode_lp": -3.0}


def test_persist_run_copies_csvs_from_live(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    run_id = "pathfinder-0001"
    runstore.write_meta(run_id, {"method": "pathfinder", "status": "running"})
    live = runstore.live_dir(run_id, create=True)
    (live / "chain.csv").write_text("a,b\n1,2\n")

    class _FakePathfinder:  # no .runset, like CmdStanPathfinder
        is_resampled = True
        metadata = InferenceMetadata({"raw_header": "a,b"})

    diagnostics.persist_run(run_id, "pathfinder", _FakePathfinder())
    # CSVs copied from live/ despite no fit.runset
    assert runstore.csv_paths(run_id)
    # and method diagnostics captured
    assert runstore.read_meta(run_id).get("method_diagnostics") == {"is_resampled": True}


def test_run_diagnostics_falls_back_without_idata(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    # VI/pathfinder/laplace have no inference.nc; run_diagnostics must not raise.
    runstore.write_meta(
        "variational-0001",
        {"method": "variational", "status": "done", "method_diagnostics": {"eta": 1.0}},
    )
    diag = diagnostics.run_diagnostics("variational-0001")
    assert diag["method"] == "variational"
    assert diag["eta"] == 1.0
    assert "verdict" not in diag  # no HMC verdict without draws


def _write_csv_run(run_id):
    runstore.write_meta(run_id, {"method": "variational", "status": "done"})
    csvdir = runstore.run_dir(run_id, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "c.csv").write_text(
        "# comment\nlp__,theta,beta.1,beta.2\n0,0.5,1.0,2.0\n0,0.7,1.5,2.5\n"
    )


def test_csv_summary_for_non_idata_run(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_csv_run("variational-0001")
    s = diagnostics.run_summary("variational-0001")
    assert s["source"] == "csv"
    assert {r["param"] for r in s["summary"]} == {"theta", "beta.1", "beta.2"}  # lp__ excluded
    theta = next(r for r in s["summary"] if r["param"] == "theta")
    assert abs(theta["mean"] - 0.6) < 1e-9


def test_csv_extract_draws_for_non_idata_run(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_csv_run("variational-0001")
    scalar = diagnostics.extract_draws("variational-0001", "theta")
    assert scalar["source"] == "csv" and scalar["draws"] == [0.5, 0.7]
    vector = diagnostics.extract_draws("variational-0001", "beta")
    assert vector["draws"] == [[1.0, 2.0], [1.5, 2.5]]


def test_csv_extract_multi_param(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_csv_run("variational-0001")
    out = diagnostics.extract_draws("variational-0001", params=["theta", "beta"])
    assert set(out["draws"]) == {"theta", "beta"}
    assert out["draws"]["theta"] == [0.5, 0.7]
    assert out["draws"]["beta"] == [[1.0, 2.0], [1.5, 2.5]]


def test_extract_draws_requires_a_param(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_csv_run("variational-0001")
    with pytest.raises(errors.ValidationError):
        diagnostics.extract_draws("variational-0001")


def test_posterior_mean_inits_from_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_csv_run("variational-0001")
    inits = diagnostics.posterior_mean_inits("variational-0001")
    assert abs(inits["theta"] - 0.6) < 1e-9
    assert inits["beta"] == [1.25, 2.25]  # means of (1.0,1.5) and (2.0,2.5)


def test_posterior_mean_inits_from_optimize(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("optimize-0001", {"method": "optimize", "status": "done"})
    runstore.persist_estimate("optimize-0001", {"theta": 0.7, "lp__": -2.0})
    inits = diagnostics.posterior_mean_inits("optimize-0001")
    assert inits == {"theta": 0.7}  # lp__ dropped


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


def test_summary_numbers_coerced_to_float():
    r = diagnostics._coerce_numbers({"param": "theta", "mean": "0.585", "sd": "0.1"})
    assert r["param"] == "theta"
    assert r["mean"] == 0.585 and isinstance(r["mean"], float)


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


# --- rerun ---
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


# --- compiled cache pruning ---
def test_prune_compiled_keep(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import config

    d = config.compiled_dir()
    for mid in ("a" * 16, "b" * 16):
        (d / f"{mid}.stan").write_text("model {}")
        (d / f"{mid}_model.so").write_text("x")
    res = c.prune_compiled(keep=1)
    assert len(res["removed_models"]) == 1
    remaining = {p.stem for p in d.glob("*.stan")}
    assert len(remaining) == 1


def test_prune_compiled_requires_criterion(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    with pytest.raises(errors.ValidationError):
        c.prune_compiled()


# --- configurable pool size ---
def test_max_workers_env(monkeypatch):
    import config

    monkeypatch.setenv("STAN_MCP_MAX_WORKERS", "5")
    assert config.max_workers() == 5
    assert sample.RunManager()._pool._max_workers == 5


# --- sample: progress + cancel of a finished/unknown run ---
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


# --- unknown run_ids surface as RunError, not a raw FileNotFoundError ---
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


# --- live/ is not kept alongside csv/ ---
def test_persist_run_clears_live_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    run_id = "pathfinder-0002"
    runstore.write_meta(run_id, {"method": "pathfinder", "status": "running"})
    live = runstore.live_dir(run_id, create=True)
    (live / "chain.csv").write_text("a,b\n1,2\n")

    class _Fake:
        is_resampled = True
        metadata = InferenceMetadata({"raw_header": "a,b"})

    diagnostics.persist_run(run_id, "pathfinder", _Fake())
    assert runstore.csv_paths(run_id)  # durable copy kept
    assert not live.is_dir()  # duplicate dropped


def test_progress_counts_persisted_csvs_after_live_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_csv_run("sample-0001")  # 2 draws, no live/
    out = sample.RunManager().get_run_progress("sample-0001")
    assert out["draws_written"] == 2


# --- extract_draws: by_chain over CSVs, single load, size cap ---
def test_csv_extract_by_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    csvdir = runstore.run_dir("variational-0001", create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "c-1.csv").write_text("lp__,theta\n0,0.1\n0,0.2\n")
    (csvdir / "c-2.csv").write_text("lp__,theta\n0,0.3\n0,0.4\n")
    out = diagnostics.extract_draws("variational-0001", "theta", by_chain=True)
    assert out["by_chain"] is True
    assert out["draws"] == [[0.1, 0.2], [0.3, 0.4]]  # chain structure preserved


def test_extract_draws_multi_param_loads_source_once(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_sample_run("sample-0001")
    calls = []
    real = diagnostics._load_idata
    monkeypatch.setattr(diagnostics, "_load_idata", lambda rid: (calls.append(rid), real(rid))[1])
    diagnostics.extract_draws("sample-0001", params=["theta", "beta"])
    assert len(calls) == 1  # not one netCDF open per parameter


def test_extract_draws_caps_result_size(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_sample_run("sample-0001")  # 2 chains x 10 draws = 20
    out = diagnostics.extract_draws("sample-0001", "theta", max_draws=5)
    assert len(out["draws"]) <= 5
    assert out["thin"] == 4
    assert len(diagnostics.extract_draws("sample-0001", "theta", max_draws=0)["draws"]) == 20


# run_summary is the tool you actually reach for (raw draws tell you little), so
# it is the one that most needs to stay small — and arviz emits one row per
# parameter ELEMENT, so one big vector is hundreds of rows.
def test_run_summary_caps_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import arviz as az

    idata = az.from_dict({"posterior": {"beta": np.zeros((2, 10, 50))}})
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done"})
    idata.to_netcdf(str(runstore.idata_path("sample-0001")))

    out = diagnostics.run_summary("sample-0001", max_rows=10)
    assert len(out["summary"]) == 10
    assert out["truncated"]["total"] == 50 and out["truncated"]["shown"] == 10
    full = diagnostics.run_summary("sample-0001", max_rows=0)
    assert len(full["summary"]) == 50 and "truncated" not in full


def test_run_summary_truncation_keeps_worst_converged(tmp_path, monkeypatch):
    """A cap that keeps the first n silently drops the tail of a big vector; the
    rows worth keeping are the ones you would act on."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rows = [{"param": f"z[{i}]", "r_hat": 1.0, "ess_bulk": 900.0} for i in range(10)]
    rows[7]["r_hat"] = 1.4  # the broken one, last in arviz order among the tail
    rows[9]["ess_bulk"] = 12.0  # worst ESS, but r_hat is fine
    out = diagnostics._cap_rows(rows, 2)
    assert [r["param"] for r in out["summary"]] == ["z[7]", "z[9]"]
    assert out["truncated"]["selected_by"].startswith("worst r_hat")


def test_run_summary_truncation_without_convergence_columns(tmp_path, monkeypatch):
    """The CSV fallback has no r_hat/ess_bulk, so order is all there is."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rows = [{"param": f"z[{i}]", "mean": 0.0} for i in range(5)]
    out = diagnostics._cap_rows(rows, 2)
    assert [r["param"] for r in out["summary"]] == ["z[0]", "z[1]"]
    assert out["truncated"]["selected_by"] == "order"


def test_run_summary_csv_fallback_caps_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    csvdir = runstore.run_dir("variational-0001", create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    cols = ",".join(f"beta.{i}" for i in range(20))
    row = ",".join("0.5" for _ in range(20))
    (csvdir / "c-1.csv").write_text(f"lp__,{cols}\n0,{row}\n0,{row}\n")
    out = diagnostics.run_summary("variational-0001", max_rows=5)
    assert out["source"] == "csv" and len(out["summary"]) == 5
    assert out["truncated"]["total"] == 20


# The ceiling bounds what the WHOLE call returns. Capping per chain per
# parameter instead would let it through multiplied by chains x params x
# vector-length, which is how a "capped" call still overran a tool result.
def test_max_draws_scales_with_vector_width(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_sample_run("sample-0001")  # beta is (2 chains, 10 draws, 3)
    out = diagnostics.extract_draws("sample-0001", "beta", max_draws=6)
    assert sum(len(d) for d in out["draws"]) <= 6  # 2 draws x 3 components


def test_max_draws_splits_across_chains(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_sample_run("sample-0001")
    out = diagnostics.extract_draws("sample-0001", "theta", by_chain=True, max_draws=4)
    assert sum(len(c) for c in out["draws"]) <= 4  # not 4 per chain


def test_max_draws_splits_across_params(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_sample_run("sample-0001")
    out = diagnostics.extract_draws("sample-0001", params=["theta", "beta"], max_draws=6)
    theta, beta = out["draws"]["theta"], out["draws"]["beta"]
    assert len(theta) + sum(len(d) for d in beta) <= 6  # not 6 per name


# --- compiled cache: mtime tracks last use, and it can be listed ---
def test_stan_file_touch_on_reuse_orders_prune_by_last_use(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    old, new = "model { real a; }", "model { real b; }"
    c.stan_file_for(old)
    c.stan_file_for(new)
    c.stan_file_for(old)  # reuse the older model -> it is now the freshest
    res = c.prune_compiled(keep=1)
    assert res["removed_models"] == [c.model_id(new)]


def test_stan_file_reuse_does_not_restamp_source(tmp_path, monkeypatch):
    """Last use must be recorded out of band, never by touching the .stan.

    cmdstanpy rebuilds whenever the source is newer than the exe, so restamping
    it on reuse relinks the binary on every call — and relinking one that a
    concurrent run is executing has the kernel kill its chains (retcode -1)."""
    import os

    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    src = "model { real a; }"
    path = c.stan_file_for(src)
    os.utime(path, (1_000_000, 1_000_000))
    assert c.stan_file_for(src).stat().st_mtime == 1_000_000  # source untouched
    assert c._last_used(path) > 1_000_000  # ...but last use still advanced


def test_list_compiled(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    c.stan_file_for("model {}")
    did = c.register_data({"N": 1})
    out = c.list_compiled()
    assert out["models"][0]["model_id"] == c.model_id("model {}")
    assert [d["data_id"] for d in out["data"]] == [did]


# --- sample: chain concurrency is capped per run ---
def test_sample_caps_parallel_chains(monkeypatch):
    import config

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


# --- divergence localization ---
def _write_divergent_run(run_id, n_div=20, planted=3.0, n_chain=2, n_draw=500, seed=0):
    """A run whose divergences are planted on `v` and nowhere else."""
    import arviz as az

    rng = np.random.default_rng(seed)
    n = n_chain * n_draw
    v = rng.normal(size=n)
    x = rng.normal(size=(n, 3))
    div = np.zeros(n, bool)
    if n_div:
        div[rng.choice(n, n_div, replace=False)] = True
    v[div] += planted
    idata = az.from_dict(
        {
            "posterior": {
                "v": v.reshape(n_chain, n_draw),
                "x": x.reshape(n_chain, n_draw, 3),
                "fixed": np.full((n_chain, n_draw), 2.5),  # zero variance
            },
            "sample_stats": {"diverging": div.reshape(n_chain, n_draw)},
        }
    )
    runstore.write_meta(run_id, {"method": "sample", "status": "done"})
    idata.to_netcdf(str(runstore.idata_path(run_id)))


def test_pathologies_finds_the_planted_parameter(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001")
    out = diagnostics.run_pathologies("sample-0001", permutations=300)
    assert out["n_divergent"] == 20
    assert out["ranking"][0]["param"] == "v"
    assert out["ranking"][0]["direction"] == "above"  # planted shift was positive
    assert out["separation"]["margin"] > 3
    assert out["separation"]["p_value"] < 0.05


def test_pathologies_skips_constant_parameters(tmp_path, monkeypatch):
    """A parameter that never varies carries no location information, and
    standardizing it would divide by zero."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001")
    out = diagnostics.run_pathologies("sample-0001", top_k=10, permutations=100)
    assert "fixed" not in {r["param"] for r in out["ranking"]}


def test_pathologies_no_divergences_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001", n_div=0)
    out = diagnostics.run_pathologies("sample-0001", permutations=100)
    assert out["n_divergent"] == 0 and out["ranking"] == []
    assert "no divergent transitions" in out["note"]


def test_pathologies_warns_when_divergences_are_few(tmp_path, monkeypatch):
    """Below a handful of divergences the ranking outruns its evidence, and the
    result has to say so rather than read as confident."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001", n_div=3)
    out = diagnostics.run_pathologies("sample-0001", permutations=100)
    assert "indicative at best" in out["warning"]


def test_pathologies_is_deterministic(tmp_path, monkeypatch):
    """Same run must give the same p-value; a diagnostic that wobbles between
    calls is not one you can act on."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001")
    a = diagnostics.run_pathologies("sample-0001", permutations=200)
    b = diagnostics.run_pathologies("sample-0001", permutations=200)
    assert a["separation"] == b["separation"]


def test_pathologies_rejects_non_sample_methods(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    with pytest.raises(errors.RunError):
        diagnostics.run_pathologies("variational-0001")


def test_pathologies_validates_top_k(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001")
    with pytest.raises(errors.ValidationError):
        diagnostics.run_pathologies("sample-0001", top_k=0)


def test_ebfmi_is_reported_not_swallowed(tmp_path, monkeypatch):
    """az.bfmi raises TypeError on an arviz 1.2 DataTree, and a bare `except:
    pass` turned that into E-BFMI silently absent from every sample run — for the
    diagnostic most specific to funnel geometry."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import arviz as az

    rng = np.random.default_rng(0)
    idata = az.from_dict(
        {
            "posterior": {"theta": rng.normal(size=(2, 200))},
            "sample_stats": {
                "energy": rng.normal(size=(2, 200)),
                "diverging": np.zeros((2, 200), dtype=bool),
            },
        }
    )
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done"})
    idata.to_netcdf(str(runstore.idata_path("sample-0001")))
    d = diagnostics.run_diagnostics("sample-0001")
    assert d.get("min_ebfmi") is not None
    assert "ebfmi_warning" not in d


# --- posterior geometry ---
def _geom_run(monkeypatch, tmp_path, cols: dict, n_chain=4, n_draw=500):
    """A run whose unconstrained draws are planted. `cols` maps name -> (draws,)
    array of length n_chain*n_draw. The BridgeStan round-trip is stubbed: what
    needs testing is the statistics, and the unconstraining needs a toolchain."""
    import arviz as az

    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    names = list(cols)
    u3 = np.stack([cols[n] for n in names], axis=-1).reshape(n_chain, n_draw, len(names))
    idata = az.from_dict({"posterior": {"dummy": np.zeros((n_chain, n_draw))}})
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done"})
    idata.to_netcdf(str(runstore.idata_path("sample-0001")))
    monkeypatch.setattr(
        diagnostics,
        "_unconstrained_draws",
        lambda rid, idata: (u3, names, {"scale": "unconstrained"}),
    )
    return "sample-0001"


def _correlated(rho, n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rho * a + np.sqrt(max(1 - rho**2, 1e-12)) * rng.normal(size=n)
    return a, b


def test_geometry_catches_negative_correlation(tmp_path, monkeypatch):
    """Fisher z carries the sign of r. Comparing the signed value against a
    positive threshold dropped every negatively correlated pair — which is the
    commonest form collinearity takes, two coefficients trading off."""
    n = 2000
    b1, b2 = _correlated(-0.99, n)
    rid = _geom_run(monkeypatch, tmp_path, {"b1": b1, "b2": b2, "c": _correlated(0, n, 5)[0]})
    g = diagnostics.run_geometry(rid)
    assert g["verdict"] == "warn"
    got = {(p["a"], p["b"]) for p in g["pairs"]}
    assert ("b1", "b2") in got or ("b2", "b1") in got
    assert g["pairs"][0]["r"] < -0.9  # sign preserved in the report
    assert g["pairs"][0]["z"] > 0  # evidence is a magnitude


def test_geometry_catches_positive_correlation(tmp_path, monkeypatch):
    n = 2000
    b1, b2 = _correlated(0.97, n)
    rid = _geom_run(monkeypatch, tmp_path, {"b1": b1, "b2": b2})
    g = diagnostics.run_geometry(rid)
    assert g["pairs"] and g["pairs"][0]["r"] > 0.9


def test_geometry_clean_on_independent_params(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    cols = {f"b{i}": rng.normal(size=2000) for i in range(4)}
    rid = _geom_run(monkeypatch, tmp_path, cols)
    g = diagnostics.run_geometry(rid)
    assert g["verdict"] == "pass" and g["pairs"] == []
    assert g["max_abs_r_cross_block"] < 0.3  # reported even when nothing fires


def test_geometry_collapses_within_block_pairs(tmp_path, monkeypatch):
    """theta[1..20] is 190 of the 210 pairs; itemizing them would crowd out the
    one cross-block pair that names the actual problem."""
    rng = np.random.default_rng(2)
    shared = rng.normal(size=2000)
    cols = {f"theta[{i}]": 0.9 * shared + 0.44 * rng.normal(size=2000) for i in range(20)}
    cols["sigma"] = rng.normal(size=2000)
    rid = _geom_run(monkeypatch, tmp_path, cols)
    g = diagnostics.run_geometry(rid)
    blocks = {w["block"]: w for w in g["within_block"]}
    assert blocks["theta"]["n_pairs"] == 190
    assert blocks["theta"]["max_abs_r"] > 0.5
    assert all(p["a"] != p["b"] for p in g["pairs"])


def test_geometry_eigen_suppressed_when_dims_rival_ess(tmp_path, monkeypatch):
    """A sample correlation matrix is ill-conditioned by construction when p is
    not small next to n_eff, so the condition number would be pure artifact."""
    rng = np.random.default_rng(3)
    cols = {f"b{i}": rng.normal(size=200) for i in range(80)}
    rid = _geom_run(monkeypatch, tmp_path, cols, n_chain=2, n_draw=100)
    g = diagnostics.run_geometry(rid)
    assert "eigen" not in g and "suppressed" in g["eigen_note"]


def test_geometry_reports_the_noise_condition_number(tmp_path, monkeypatch):
    """A condition number is only readable next to what noise alone would give:
    at p/n_eff = 0.25 independent parameters already produce about 8."""
    rng = np.random.default_rng(11)
    cols = {f"b{i}": rng.normal(size=2000) for i in range(6)}
    rid = _geom_run(monkeypatch, tmp_path, cols)
    eig = diagnostics.run_geometry(rid)["eigen"]
    assert eig["noise_condition_number"] > 1.0
    # 6 dims against ~2000 effective draws is a negligible ratio, so both the
    # observed and the structural figure must be small.
    assert eig["condition_number"] < 10 and eig["noise_condition_number"] < 2


def test_geometry_eigen_finds_three_way_degeneracy(tmp_path, monkeypatch):
    """x3 = x1 + x2 leaves every pair moderate while the posterior is near
    singular along x1 + x2 - x3: the blind spot pairwise cannot see."""
    rng = np.random.default_rng(4)
    n = 2000  # must equal n_chain * n_draw
    b1, b2 = rng.normal(size=n), rng.normal(size=n)
    b3 = b1 + b2 + 0.05 * rng.normal(size=n)
    rid = _geom_run(monkeypatch, tmp_path, {"b1": b1, "b2": b2, "b3": b3})
    g = diagnostics.run_geometry(rid)
    assert g["eigen"]["condition_number"] > 100
    assert {d["param"] for d in g["eigen"]["direction"]} == {"b1", "b2", "b3"}


def test_geometry_thresholds_are_overridable(tmp_path, monkeypatch):
    n = 2000
    b1, b2 = _correlated(0.2, n)
    rid = _geom_run(monkeypatch, tmp_path, {"b1": b1, "b2": b2})
    assert diagnostics.run_geometry(rid)["pairs"] == []  # below default bar
    loose = diagnostics.run_geometry(rid, min_abs_r=0, min_z=0)
    assert loose["pairs"]  # 0 = unlimited, per the codebase convention


def test_geometry_refuses_above_max_dims(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    cols = {f"b{i}": rng.normal(size=400) for i in range(12)}
    rid = _geom_run(monkeypatch, tmp_path, cols, n_chain=2, n_draw=200)
    with pytest.raises(errors.ValidationError):
        diagnostics.run_geometry(rid, max_dims=5)


def test_geometry_params_filter(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    cols = {"a": rng.normal(size=800), "b[0]": rng.normal(size=800), "b[1]": rng.normal(size=800)}
    rid = _geom_run(monkeypatch, tmp_path, cols, n_chain=2, n_draw=400)
    assert diagnostics.run_geometry(rid, params=["b"])["n_dims"] == 2  # block name matches
    with pytest.raises(errors.ValidationError):
        diagnostics.run_geometry(rid, params=["nope"])


def test_geometry_rejects_non_sample_methods(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    with pytest.raises(errors.RunError):
        diagnostics.run_geometry("variational-0001")


def test_geometry_reports_preparing_not_an_error(tmp_path, monkeypatch):
    """The first call on a freshly sampled model has no .so; the build happens off
    the caller's thread and the call must say so rather than fail."""
    import arviz as az

    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done"})
    az.from_dict({"posterior": {"x": np.zeros((2, 10))}}).to_netcdf(
        str(runstore.idata_path("sample-0001"))
    )

    def _pending(rid, idata):
        raise diagnostics._GeometryPending("building")

    monkeypatch.setattr(diagnostics, "_unconstrained_draws", _pending)
    assert diagnostics.run_geometry("sample-0001")["status"] == "preparing"


def test_geometry_falls_back_when_draws_cannot_be_unconstrained(tmp_path, monkeypatch):
    """Stan's *_free functions validate the constraint they invert, so a draw read
    back at limited CSV precision can be rejected: simplex_free's tolerance is 1e-8
    and a simplex written at 6 significant figures only sums to 1 to about that.

    One rejected draw used to abort the whole call, so get_run_geometry hard-failed
    on every mixture or LKJ model. It must degrade to the constrained scale and say
    so instead -- reported, not swallowed, and not fatal.
    """
    import arviz as az

    import density

    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    n_chain, n_draw = 2, 50
    rng = np.random.default_rng(4)
    idata = az.from_dict(
        {
            "posterior": {
                "theta": rng.random((n_chain, n_draw, 3)),
                "s": rng.random((n_chain, n_draw)),
            }
        }
    )
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done"})
    runstore.persist_inputs("sample-0001", "parameters { simplex[3] theta; real s; } model { }", {})
    idata.to_netcdf(str(runstore.idata_path("sample-0001")))

    class _Rejecting:
        def param_names(self):
            return ["theta.1", "theta.2", "theta.3", "s"]

        def param_unc_names(self):
            return ["theta.1", "theta.2", "s"]

        def param_unc_num(self):
            return 3

        def param_unconstrain(self, _row):
            raise RuntimeError(
                "param_unconstrain() failed with exception: Exception: "
                "stan::math::simplex_free: Simplex variable is not a valid simplex. "
                "sum(Simplex variable) = 0.99999999, but should be 1"
            )

    import compile as _compile

    monkeypatch.setattr(_compile, "request_so", lambda *a, **k: {"status": "ready"})
    monkeypatch.setattr(density, "get_model", lambda *a, **k: _Rejecting())

    g = diagnostics.run_geometry("sample-0001")
    assert "error" not in g, g
    assert g["scale"] == "constrained"
    # the actual Stan message must survive into the reason, not be flattened away
    assert "simplex_free" in g["scale_reason"]
    assert "Re-run" in g["scale_reason"]
    # and it still did the analysis, on all four constrained components
    assert g["n_dims"] == 4


def test_sample_defaults_to_round_tripping_output_precision():
    """CmdStan's default 6 sig figs loses enough precision that a constrained draw
    read back misses its constraint; 17 is where a double round-trips exactly."""
    import sample as _sample

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


def test_sidecar_records_cols_in_row_major_fill_order():
    """CmdStan's CSV header is column-major and numpy.reshape is row-major, so the
    sidecar must record the permutation. Recording the contiguous run instead
    transposes silently: L.2.1's values land in L[1,2].

    Uses a REAL cmdstanpy InferenceMetadata (not a hand-rolled fake) built from
    a single raw_header string, which is what production actually hands
    `_sidecar_from_fit`: `fit.column_names` munges names to array notation
    (`m[1,1]`, not `m.1.1`) and cannot satisfy this test at all -- a fake that
    sets `column_names` directly to dot-notation strings would exercise a
    branch that can never run for a real fit.
    """

    class _Fit:
        metadata = InferenceMetadata(
            {"raw_header": "lp__,divergent__,theta,m.1.1,m.2.1,m.1.2,m.2.2"}
        )

    sc = diagnostics._sidecar_from_fit(_Fit())
    assert sc["schema"] == diagnostics.draws.SCHEMA_VERSION
    assert sc["params"]["theta"] == {"dims": [], "cols": [2]}
    # row-major fill order for (2,2): m[1,1], m[1,2], m[2,1], m[2,2]
    #                          cols:      3        5        4        6
    assert sc["params"]["m"] == {"dims": [2, 2], "cols": [3, 5, 4, 6]}
    assert sc["sampler_stats"]["lp__"] == {"dims": [], "cols": [0]}
    assert sc["sampler_stats"]["divergent__"] == {"dims": [], "cols": [1]}
    assert "theta" not in sc["sampler_stats"]


def test_csv_columns_falls_back_to_column_names_without_cmdstan_config():
    """Belt-and-braces only: no real cmdstanpy fit reaches this branch (its
    InferenceMetadata always carries cmdstan_config['raw_header']), but a
    fake/mock that sets only `.column_names` must not crash `_sidecar_from_fit`
    outright."""

    class _Meta:
        pass  # no cmdstan_config

    class _Fit:
        metadata = _Meta()
        column_names = ("lp__", "theta")

    assert diagnostics._csv_columns(_Fit()) == ["lp__", "theta"]


def test_persist_run_writes_a_sidecar_and_no_netcdf(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    (live / "chain_1.csv").write_text("# c\nlp__,divergent__,theta\n-1,0,0.5\n")

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,divergent__,theta"})

    diagnostics.persist_run(rid, "sample", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["files"] == ["chain_1.csv"]
    assert sc["params"]["theta"]["cols"] == [2]
    assert not runstore.idata_path(rid).exists()
    # and the typed layer can read what was just written
    d = diagnostics.draws.load(rid)
    assert d.params["theta"].tolist() == [[0.5]]


def test_persist_run_excludes_laplace_mode_csv_from_sidecar_files(tmp_path, monkeypatch):
    """laplace with no explicit `mode=` runs its own internal optimize into the
    SAME output dir, dropping an `-opt-...` CSV of a different shape (one row,
    fewer columns) beside the draws CSV. Recording it in `files` makes
    `draws.load` stack arrays of different widths and die with a bare numpy
    ValueError instead of a RunError naming the run."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "laplace-0001"
    runstore.write_meta(rid, {"method": "laplace", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    (live / "model-1.csv").write_text("# c\nlog_p__,log_q__,theta\n-1,-2,0.5\n")
    (live / "model-opt-1.csv").write_text("# c\nlp__,theta\n-1,0.5\n")

    class _Fit:
        # no .runset -- matches the real CmdStanLaplace, exercising the
        # numeric/name-sort fallback in _ordered_draws_files
        metadata = InferenceMetadata({"raw_header": "log_p__,log_q__,theta"})

    diagnostics.persist_run(rid, "laplace", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["files"] == ["model-1.csv"]
    d = diagnostics.draws.load(rid)
    assert d.params["theta"].tolist() == [[0.5]]


def test_persist_run_orders_sidecar_files_by_chain_not_filename(tmp_path, monkeypatch):
    """A filename sort puts chain 10 before chain 2 once there are 10+ chains,
    silently permuting per-chain results. Exercises the fallback path (no
    `.runset`), which must sort numerically on its own."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    for n in (1, 2, 10):
        (live / f"model_{n}.csv").write_text(f"# c\nlp__,theta\n-1,{n / 10}\n")

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    diagnostics.persist_run(rid, "sample", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["files"] == ["model_1.csv", "model_2.csv", "model_10.csv"]


def test_persist_run_prefers_fit_runset_order_for_sidecar_files(tmp_path, monkeypatch):
    """fit.runset.csv_files is cmdstanpy's own per-chain-ordered list and must be
    used verbatim, not re-sorted -- guards against reintroducing a filename
    sort that happens to agree with chain order only below 10 chains."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    for n in (1, 2, 10):
        (live / f"model_{n}.csv").write_text(f"# c\nlp__,theta\n-1,{n / 10}\n")

    class _Runset:
        csv_files = [
            str(live / "model_2.csv"),
            str(live / "model_10.csv"),
            str(live / "model_1.csv"),
        ]

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})
        runset = _Runset()

    diagnostics.persist_run(rid, "sample", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["files"] == ["model_2.csv", "model_10.csv", "model_1.csv"]


def test_persist_run_fails_the_run_when_sidecar_write_fails(tmp_path, monkeypatch):
    """The sidecar is now the ONLY record of a run's dims/columns -- unlike the
    arviz conversion it replaces, whose failure still left CSV-derived
    fallbacks readable for VI/pathfinder/laplace. A run whose sidecar failed to
    write must not settle at status="done" with only a persist_warning field
    to notice by, so persist_run must raise -- the same failure path sample.py
    already uses for e.g. a sampler crash -- rather than swallow the error."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    (live / "model_1.csv").write_text("# c\nlp__,theta\n-1,0.5\n")

    def boom(fit):
        raise RuntimeError("boom")

    monkeypatch.setattr(diagnostics, "_sidecar_from_fit", boom)

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    with pytest.raises(errors.RunError, match="draws schema not written"):
        diagnostics.persist_run(rid, "sample", _Fit())

    # the CSVs must survive the failure -- they're still worth keeping
    assert runstore.csv_paths(rid) == [str(runstore.run_dir(rid) / "csv" / "model_1.csv")]
    meta = runstore.read_meta(rid)
    assert "boom" in meta["persist_warning"]
