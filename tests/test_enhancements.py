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


# --- diagnostics: extract_draws reads the typed draws layer ---
def _typed_run(tmp_path, monkeypatch, method="sample", ragged=False):
    """A run with a sidecar: one scalar and one length-2 vector, 2 chains x 2 draws.

    ragged=True drops chain b's second draw -- 2 draws vs 1, the shape a
    cancelled run leaves behind -- so `Draws.truncated_to` is set.
    """
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = f"{method}-0001"
    runstore.write_meta(rid, {"method": method, "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "a.csv").write_text("# c\nlp__,theta,z.1,z.2\n-1,0.1,10,20\n-2,0.2,11,21\n")
    b_body = "-3,0.3,12,22\n" if ragged else "-3,0.3,12,22\n-4,0.4,13,23\n"
    (csvdir / "b.csv").write_text(f"# c\nlp__,theta,z.1,z.2\n{b_body}")
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": diagnostics.draws.SCHEMA_VERSION,
                "files": ["a.csv", "b.csv"],
                "params": {"theta": {"dims": [], "cols": [1]}, "z": {"dims": [2], "cols": [2, 3]}},
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    return rid


def test_extract_draws_reads_the_typed_layer(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    got = diagnostics.extract_draws(rid, param="theta")
    assert got["draws"] == [0.1, 0.2, 0.3, 0.4]
    assert "source" not in got  # there is only one source now


def test_extract_draws_by_chain_uses_sidecar_chain_order(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    got = diagnostics.extract_draws(rid, param="theta", by_chain=True)
    assert got["draws"] == [[0.1, 0.2], [0.3, 0.4]]


def test_extract_draws_keeps_vector_shape(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    got = diagnostics.extract_draws(rid, param="z")
    assert got["draws"][0] == [10.0, 20.0]


def test_extract_draws_unknown_param_raises(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    with pytest.raises(errors.RunError, match="nope"):
        diagnostics.extract_draws(rid, param="nope")


def test_extract_draws_works_for_a_variational_run(tmp_path, monkeypatch):
    """One path for all six methods: no branch on whether inference.nc exists."""
    rid = _typed_run(tmp_path, monkeypatch, method="variational")
    assert diagnostics.extract_draws(rid, param="theta")["draws"] == [0.1, 0.2, 0.3, 0.4]


def test_extract_draws_single_draw_stays_a_list(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    out = diagnostics.extract_draws(rid, "theta", thin=999)
    assert isinstance(out["draws"], list)  # not a bare float
    assert len(out["draws"]) == 1


def test_extract_draws_rejects_bad_thin(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    with pytest.raises(errors.ValidationError):
        diagnostics.extract_draws(rid, "theta", thin=0)


def _typed_run_wide(tmp_path, monkeypatch, method="sample"):
    """2 chains x 10 draws, scalar theta + 3-vector beta -- big enough that the
    max_draws/thin math is non-trivial (the minimal 2x2 `_typed_run` above
    collapses every budget check to a no-op)."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = f"{method}-0001"
    runstore.write_meta(rid, {"method": method, "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    header = "lp__,theta,beta.1,beta.2,beta.3"
    for chain, offset in enumerate((0, 10)):
        rows = "\n".join(
            f"-1,{offset + t},{3 * (offset + t)},{3 * (offset + t) + 1},{3 * (offset + t) + 2}"
            for t in range(10)
        )
        (csvdir / f"chain{chain}.csv").write_text(f"# c\n{header}\n{rows}\n")
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": diagnostics.draws.SCHEMA_VERSION,
                "files": ["chain0.csv", "chain1.csv"],
                "params": {
                    "theta": {"dims": [], "cols": [1]},
                    "beta": {"dims": [3], "cols": [2, 3, 4]},
                },
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    return rid


def test_extract_draws_scalar_is_flat_list(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    out = diagnostics.extract_draws(rid, "theta")
    assert len(out["draws"]) == 20
    assert isinstance(out["draws"][0], float)


def test_extract_draws_vector_is_list_of_lists(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    out = diagnostics.extract_draws(rid, "beta")
    assert len(out["draws"]) == 20
    assert isinstance(out["draws"][0], list) and len(out["draws"][0]) == 3


# --- diagnostics: verdict logic ---
def test_verdict_pass():
    assert diagnostics._verdict({"max_rhat": 1.0, "min_ess_bulk": 500})["verdict"] == "pass"


def test_verdict_flags_issues():
    v = diagnostics._verdict({"max_rhat": 1.05, "divergences": 3, "min_ess_bulk": 500})
    assert v["verdict"] == "warn"
    assert len(v["issues"]) == 2


def test_verdict_unknown_when_nothing_was_measured():
    """An empty diagnostics dict means every reader failed, not that the run is
    clean -- reporting "pass" there would be a false-clean on the headline
    field, exactly what the readers' own *_warning keys exist to avoid."""
    assert diagnostics._verdict({}) == {"verdict": "unknown", "issues": []}
    # a *_warning key alone, with none of the measured keys, is still nothing
    # measured -- the warning key itself is not one of the checked keys.
    assert diagnostics._verdict({"convergence_warning": "boom"}) == {
        "verdict": "unknown",
        "issues": [],
    }


def test_verdict_stays_pass_when_a_warning_accompanies_measured_stats():
    """A *_warning key must not, by itself, flip a partially-measured run to
    "warn" -- that would read as "the model is pathological" when the truth is
    "one statistic could not be checked"."""
    v = diagnostics._verdict({"divergences": 0, "sampler_stats_warning": "boom"})
    assert v == {"verdict": "pass", "issues": []}


def test_extract_draws_by_chain(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    out = diagnostics.extract_draws(rid, "theta", by_chain=True)
    assert out["by_chain"] is True
    assert len(out["draws"]) == 2 and len(out["draws"][0]) == 10  # 2 chains x 10 draws


def test_convergence_reads_the_same_stansummary_output_as_the_summary(tmp_path, monkeypatch):
    """The summary and the verdict must not be able to disagree: before this
    migration one went through az.summary() and the other through
    az.summary(kind='diagnostics'); both now read the identical stansummary()
    call, via draws.schema/draws.element_labels for the name/dims lookup."""
    rid = _typed_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda paths, **kw: {
            "lp__": {"R_hat": 9.9, "ESS_bulk": 1.0, "ESS_tail": 1.0},  # must be ignored
            "theta": {"R_hat": 1.001, "ESS_bulk": 900.0, "ESS_tail": 500.0},
            "z.1": {"R_hat": 1.05, "ESS_bulk": 120.0, "ESS_tail": 148.5},
            "z.2": {"R_hat": 1.00, "ESS_bulk": 700.0, "ESS_tail": 600.0},
        },
    )
    got = diagnostics._convergence(rid)
    assert got["max_rhat"] == 1.05
    assert got["max_rhat_param"] == "z[1]"  # Stan syntax, and not lp__
    assert got["min_ess_bulk"] == 120.0
    assert got["min_ess_tail"] == 148.5


def test_convergence_reports_a_warning_when_stansummary_itself_raises(tmp_path, monkeypatch):
    """The missing-sidecar case (draws.schema raising) isn't the only way into
    _convergence's warning path -- a sidecar can exist and stansummary can
    still fail on its own (e.g. the binary crashing on a malformed CSV)."""
    rid = _typed_run(tmp_path, monkeypatch)

    def _boom(paths, **kw):
        raise errors.RunError("boom")

    monkeypatch.setattr(diagnostics.stansummary, "summarize", _boom)
    assert diagnostics._convergence(rid) == {"convergence_warning": "R-hat/ESS unavailable: boom"}


def test_sampler_stats_reads_typed_arrays_and_reports_ebfmi():
    """E-BFMI is not in stansummary (only in `diagnose`, as prose), so it stays
    computed from energy__ -- fed a clean array rather than a DataTree."""
    rng = np.random.default_rng(0)
    d = diagnostics.draws.Draws(
        params={},
        sampler_stats={
            "divergent__": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "treedepth__": np.array([[3.0, 10.0], [10.0, 4.0]]),
            "accept_stat__": np.array([[0.8, 0.9], [0.7, 1.0]]),
            "energy__": rng.normal(size=(2, 2)),
        },
        n_chain=2,
        n_draw=2,
    )
    got = diagnostics._sampler_stats(d, {"method_args": {"max_treedepth": 10}})
    assert got["divergences"] == 2
    assert got["treedepth_saturations"] == 2
    assert got["mean_accept_stat"] == 0.85
    assert "min_ebfmi" in got and "ebfmi_warning" not in got


# --- diagnostics over a typed sample run with sampler stats (offline) ---
def _write_run_with_stats(run_id, *, diverging, tree_depth, max_td=10):
    """A typed sample run: one scalar param (theta) plus the raw CmdStan sampler
    stats columns (`divergent__`, `treedepth__`, `accept_stat__`, `energy__`)
    run_diagnostics now reads directly off the typed draws layer, no idata."""
    rng = np.random.default_rng(0)
    nchain, ndraw = 2, 1000
    runstore.write_meta(
        run_id, {"method": "sample", "status": "done", "method_args": {"max_treedepth": max_td}}
    )
    csvdir = runstore.run_dir(run_id, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    theta = rng.normal(size=(nchain, ndraw))
    energy = rng.normal(size=(nchain, ndraw))
    files = []
    for chain in range(nchain):
        rows = "\n".join(
            f"{theta[chain, i]},{int(diverging)},{tree_depth},0.9,{energy[chain, i]}"
            for i in range(ndraw)
        )
        name = f"chain{chain}.csv"
        (csvdir / name).write_text(
            f"# c\ntheta,divergent__,treedepth__,accept_stat__,energy__\n{rows}\n"
        )
        files.append(name)
    diagnostics.draws.schema_path(run_id).write_text(
        json.dumps(
            {
                "schema": diagnostics.draws.SCHEMA_VERSION,
                "files": files,
                "params": {"theta": {"dims": [], "cols": [0]}},
                "sampler_stats": {
                    "divergent__": {"dims": [], "cols": [1]},
                    "treedepth__": {"dims": [], "cols": [2]},
                    "accept_stat__": {"dims": [], "cols": [3]},
                    "energy__": {"dims": [], "cols": [4]},
                },
            }
        )
    )


def test_diagnostics_clean_run_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_run_with_stats("sample-0001", diverging=False, tree_depth=3)
    # stansummary's binary is not available in the offline suite; R-hat/ESS are
    # what this test is actually about, so they are controlled directly.
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda paths, **kw: {"theta": {"R_hat": 1.001, "ESS_bulk": 900.0, "ESS_tail": 850.0}},
    )
    diag = diagnostics.run_diagnostics("sample-0001")
    assert diag["divergences"] == 0
    assert diag["treedepth_saturations"] == 0
    assert diag["max_rhat"] < 1.01
    assert diag["min_ess_bulk"] >= 400
    assert diag["verdict"] == "pass"


def test_diagnostics_bad_run_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_run_with_stats("sample-0001", diverging=True, tree_depth=10, max_td=10)
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda paths, **kw: {"theta": {"R_hat": 1.001, "ESS_bulk": 900.0, "ESS_tail": 850.0}},
    )
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


def test_run_diagnostics_for_approximate_method_has_no_hmc_verdict(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    # VI/pathfinder/laplace never reach the sample-only convergence/sampler-stats
    # branch, so they carry only their method-specific scalars, no verdict.
    runstore.write_meta(
        "variational-0001",
        {"method": "variational", "status": "done", "method_diagnostics": {"eta": 1.0}},
    )
    diag = diagnostics.run_diagnostics("variational-0001")
    assert diag["method"] == "variational"
    assert diag["eta"] == 1.0
    assert "verdict" not in diag  # no HMC verdict for a non-sample method


def test_run_diagnostics_on_sample_run_without_sidecar_does_not_raise(tmp_path, monkeypatch):
    """Before the migration, a run with no persisted draws (missing
    inference.nc) degraded to the method-diagnostics fallback rather than
    failing the call. A sample run with no draws_schema.json sidecar (predates
    the typed draws layer, or its persist_run write failed) must keep that
    behavior: both _convergence and _sampler_stats read the sidecar, and each
    must turn its RunError into a warning key rather than letting it propagate.

    The verdict must be "unknown", not "pass": nothing was actually measured
    here, and a silent "pass" on the headline field is a false-clean.
    """
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done"})
    diag = diagnostics.run_diagnostics("sample-0001")
    assert diag["method"] == "sample"
    assert "convergence_warning" in diag
    assert "sampler_stats_warning" in diag
    assert diag["verdict"] == "unknown"
    assert diag["issues"] == []


def _write_csv_run(run_id):
    runstore.write_meta(run_id, {"method": "variational", "status": "done"})
    csvdir = runstore.run_dir(run_id, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "c.csv").write_text(
        "# comment\nlp__,theta,beta.1,beta.2\n0,0.5,1.0,2.0\n0,0.7,1.5,2.5\n"
    )


def test_csv_summary_for_non_idata_run(tmp_path, monkeypatch):
    """Approximate methods get numpy moments over the typed draws layer -- Stan
    labels, not arviz's dot notation."""
    rid = _typed_run(tmp_path, monkeypatch, method="variational")
    s = diagnostics.run_summary(rid)
    assert {r["param"] for r in s["summary"]} == {"theta", "z[1]", "z[2]"}  # lp__ excluded
    theta = next(r for r in s["summary"] if r["param"] == "theta")
    assert abs(theta["mean"] - 0.25) < 1e-9
    assert theta["n"] == 4


def _write_typed_csv_run(run_id):
    """`_write_csv_run`'s single-file layout, plus the sidecar `extract_draws`
    now requires. Kept separate from `_write_csv_run` itself -- get_run_progress
    still reads that fixture straight off the untyped CSV and must keep working
    unmigrated."""
    _write_csv_run(run_id)
    diagnostics.draws.schema_path(run_id).write_text(
        json.dumps(
            {
                "schema": diagnostics.draws.SCHEMA_VERSION,
                "files": ["c.csv"],
                "params": {
                    "theta": {"dims": [], "cols": [1]},
                    "beta": {"dims": [2], "cols": [2, 3]},
                },
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )


def test_csv_extract_draws_for_non_idata_run(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_typed_csv_run("variational-0001")
    scalar = diagnostics.extract_draws("variational-0001", "theta")
    assert scalar["draws"] == [0.5, 0.7]
    vector = diagnostics.extract_draws("variational-0001", "beta")
    assert vector["draws"] == [[1.0, 2.0], [1.5, 2.5]]


def test_csv_extract_multi_param(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_typed_csv_run("variational-0001")
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
    _write_typed_csv_run("variational-0001")
    inits = diagnostics.posterior_mean_inits("variational-0001")
    assert abs(inits["theta"] - 0.6) < 1e-9
    assert inits["beta"] == [1.25, 2.25]  # means of (1.0,1.5) and (2.0,2.5)


def test_posterior_mean_inits_from_optimize(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("optimize-0001", {"method": "optimize", "status": "done"})
    runstore.persist_estimate("optimize-0001", {"theta": 0.7, "lp__": -2.0})
    inits = diagnostics.posterior_mean_inits("optimize-0001")
    assert inits == {"theta": 0.7}  # lp__ dropped


def test_posterior_mean_inits_handles_matrices(tmp_path, monkeypatch):
    """Previously anything with two or more dots in its CSV name was silently
    skipped and left to Stan's default init, because reassembling it needed the
    dims. With dims recorded it is arr.mean(axis=(0,1)).tolist()."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    # columns: lp__, s, m.1.1, m.2.1, m.1.2, m.2.2  (CmdStan column-major)
    (csvdir / "a.csv").write_text(
        "# c\nlp__,s,m.1.1,m.2.1,m.1.2,m.2.2\n-1,2.0,11,21,12,22\n-1,4.0,13,23,14,24\n"
    )
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": diagnostics.draws.SCHEMA_VERSION,
                "files": ["a.csv"],
                "params": {
                    "s": {"dims": [], "cols": [1]},
                    # row-major fill order: m[1,1], m[1,2], m[2,1], m[2,2]
                    "m": {"dims": [2, 2], "cols": [2, 4, 3, 5]},
                },
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    inits = diagnostics.posterior_mean_inits(rid)
    assert inits["s"] == pytest.approx(3.0)
    assert inits["m"] == [[12.0, 13.0], [22.0, 23.0]]
    assert "lp__" not in inits


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
    diagnostics.draws.schema_path("variational-0001").write_text(
        json.dumps(
            {
                "schema": diagnostics.draws.SCHEMA_VERSION,
                "files": ["c-1.csv", "c-2.csv"],
                "params": {"theta": {"dims": [], "cols": [1]}},
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    out = diagnostics.extract_draws("variational-0001", "theta", by_chain=True)
    assert out["by_chain"] is True
    assert out["draws"] == [[0.1, 0.2], [0.3, 0.4]]  # chain structure preserved


def test_extract_draws_multi_param_loads_source_once(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    calls = []
    real = diagnostics.draws.load
    monkeypatch.setattr(diagnostics.draws, "load", lambda r: (calls.append(r), real(r))[1])
    diagnostics.extract_draws(rid, params=["theta", "beta"])
    assert len(calls) == 1  # not one CSV parse per parameter


def test_extract_draws_caps_result_size(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)  # 2 chains x 10 draws = 20
    out = diagnostics.extract_draws(rid, "theta", max_draws=5)
    assert len(out["draws"]) <= 5
    assert out["thin"] == 4
    assert len(diagnostics.extract_draws(rid, "theta", max_draws=0)["draws"]) == 20


# run_summary is the tool you actually reach for (raw draws tell you little), so
# it is the one that most needs to stay small — and arviz emits one row per
# parameter ELEMENT, so one big vector is hundreds of rows.
def test_run_summary_caps_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "a.csv").write_text("# c\nlp__,beta.1\n-1,0\n")  # stansummary is mocked below
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": diagnostics.draws.SCHEMA_VERSION,
                "files": ["a.csv"],
                "params": {"beta": {"dims": [50], "cols": list(range(1, 51))}},
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda paths, **kw: {f"beta.{i}": {"Mean": 0.0} for i in range(1, 51)},
    )
    out = diagnostics.run_summary(rid, max_rows=10)
    assert len(out["summary"]) == 10
    assert out["truncated"]["total"] == 50 and out["truncated"]["shown"] == 10
    full = diagnostics.run_summary(rid, max_rows=0)
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
    rid = "variational-0001"
    runstore.write_meta(rid, {"method": "variational", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    cols = ",".join(f"beta.{i}" for i in range(1, 21))
    row = ",".join("0.5" for _ in range(20))
    (csvdir / "c-1.csv").write_text(f"lp__,{cols}\n0,{row}\n0,{row}\n")
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": diagnostics.draws.SCHEMA_VERSION,
                "files": ["c-1.csv"],
                "params": {"beta": {"dims": [20], "cols": list(range(1, 21))}},
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    out = diagnostics.run_summary(rid, max_rows=5)
    assert len(out["summary"]) == 5
    assert out["truncated"]["total"] == 20


def test_run_summary_for_sample_comes_from_stansummary(tmp_path, monkeypatch):
    """CmdStan is the reference implementation, and its names are relabelled into
    Stan syntax on the way out."""
    rid = _typed_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda paths, **kw: {
            "lp__": {"Mean": -2.5},
            "theta": {
                "Mean": 0.25,
                "StdDev": 0.13,
                "MAD": 0.1,
                "5%": 0.1,
                "50%": 0.25,
                "95%": 0.4,
                "ESS_bulk": 900.0,
                "ESS_tail": 403.871,
                "R_hat": 1.001,
                "MCSE": 0.004,
            },
            "z.1": {
                "Mean": 11.5,
                "StdDev": 1.3,
                "MAD": 1.0,
                "5%": 10.0,
                "50%": 11.5,
                "95%": 13.0,
                "ESS_bulk": 800.0,
                "ESS_tail": 350.0,
                "R_hat": 1.02,
                "MCSE": 0.05,
            },
            "z.2": {
                "Mean": 21.5,
                "StdDev": 1.3,
                "MAD": 1.0,
                "5%": 20.0,
                "50%": 21.5,
                "95%": 23.0,
                "ESS_bulk": 700.0,
                "ESS_tail": 300.0,
                "R_hat": 1.00,
                "MCSE": 0.05,
            },
        },
    )
    out = diagnostics.run_summary(rid)
    rows = {r["param"]: r for r in out["summary"]}
    # Stan syntax, 1-based; sampler stats excluded
    assert set(rows) == {"theta", "z[1]", "z[2]"}
    assert rows["theta"]["ess_tail"] == 403.871
    assert rows["theta"]["q5"] == 0.1 and rows["theta"]["q95"] == 0.4
    assert rows["z[1]"]["r_hat"] == 1.02
    assert "eti89_lb" not in rows["theta"]


def test_run_summary_params_filter_and_worst_first_truncation(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda paths, **kw: {
            "theta": {"Mean": 0.25, "ESS_bulk": 900.0, "R_hat": 1.001},
            "z.1": {"Mean": 11.5, "ESS_bulk": 800.0, "R_hat": 1.20},
            "z.2": {"Mean": 21.5, "ESS_bulk": 700.0, "R_hat": 1.00},
        },
    )
    assert [r["param"] for r in diagnostics.run_summary(rid, params=["theta"])["summary"]] == [
        "theta"
    ]
    out = diagnostics.run_summary(rid, max_rows=1)
    assert out["summary"][0]["param"] == "z[1]"  # worst r_hat kept
    assert out["truncated"]["total"] == 3


def test_run_summary_for_variational_never_calls_stansummary(tmp_path, monkeypatch):
    """stansummary does not refuse non-MCMC input: handed variational draws it
    reports R_hat 0.999997 for what are IID samples. The gate must be ours."""
    rid = _typed_run(tmp_path, monkeypatch, method="variational")

    def _boom(*a, **k):
        raise AssertionError("stansummary must not be called for variational runs")

    monkeypatch.setattr(diagnostics.stansummary, "summarize", _boom)
    out = diagnostics.run_summary(rid)
    rows = {r["param"]: r for r in out["summary"]}
    assert set(rows) == {"theta", "z[1]", "z[2]"}
    assert rows["theta"]["mean"] == pytest.approx(0.25)
    assert rows["theta"]["n"] == 4
    # not MCMC chains, so no convergence statistics are invented
    assert "r_hat" not in rows["theta"] and "ess_bulk" not in rows["theta"]


def test_run_summary_surfaces_ragged_chain_truncation(tmp_path, monkeypatch):
    """A cancelled run leaves chains of unequal length. `_summary_from_draws`
    must surface that truncation rather than silently reporting statistics
    over the shortened array with no sign anything was dropped."""
    rid = _typed_run(tmp_path, monkeypatch, method="variational", ragged=True)
    out = diagnostics.run_summary(rid)
    assert out["chains_truncated_to"] == 1


def test_run_summary_omits_chains_truncated_to_when_chains_are_even(tmp_path, monkeypatch):
    """Paired negative: a key that is always present would pass the test above
    while being wrong."""
    rid = _typed_run(tmp_path, monkeypatch, method="variational")
    out = diagnostics.run_summary(rid)
    assert "chains_truncated_to" not in out


# The ceiling bounds what the WHOLE call returns. Capping per chain per
# parameter instead would let it through multiplied by chains x params x
# vector-length, which is how a "capped" call still overran a tool result.
def test_max_draws_scales_with_vector_width(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)  # beta is (2 chains, 10 draws, 3)
    out = diagnostics.extract_draws(rid, "beta", max_draws=6)
    assert sum(len(d) for d in out["draws"]) <= 6  # 2 draws x 3 components


def test_max_draws_splits_across_chains(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    out = diagnostics.extract_draws(rid, "theta", by_chain=True, max_draws=4)
    assert sum(len(c) for c in out["draws"]) <= 4  # not 4 per chain


def test_max_draws_splits_across_params(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    out = diagnostics.extract_draws(rid, params=["theta", "beta"], max_draws=6)
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
    """A run whose divergences are planted on `v` and nowhere else.

    Writes typed CmdStan CSVs (one file per chain) plus a draws_schema.json
    sidecar rather than an arviz idata, so `draws.load` reads it through the
    same typed layer `run_pathologies` now uses. Flat length-`n` arrays are
    split into contiguous `n_draw`-sized chunks per chain -- the same layout
    `.reshape(n_chain, n_draw)` used to give in-memory -- so `div.reshape(-1)`
    recovers the identical flat order and the planted divergences land on the
    same draws as before.
    """
    rng = np.random.default_rng(seed)
    n = n_chain * n_draw
    v = rng.normal(size=n)
    x = rng.normal(size=(n, 3))
    fixed = np.full(n, 2.5)  # zero variance
    lp = rng.normal(size=n)
    div = np.zeros(n, bool)
    if n_div:
        div[rng.choice(n, n_div, replace=False)] = True
    v[div] += planted

    runstore.write_meta(run_id, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(run_id, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    files = []
    for chain in range(n_chain):
        lo, hi = chain * n_draw, (chain + 1) * n_draw
        # repr'd through a plain python float: full round-trip precision,
        # without numpy>=2's `np.float64(...)` repr wrapper that `float()`
        # in draws._read_csv cannot parse.
        rows = "\n".join(
            ",".join(
                repr(float(col))
                for col in (lp[i], int(div[i]), v[i], x[i, 0], x[i, 1], x[i, 2], fixed[i])
            )
            for i in range(lo, hi)
        )
        name = f"chain{chain}.csv"
        (csvdir / name).write_text(f"# c\nlp__,divergent__,v,x.1,x.2,x.3,fixed\n{rows}\n")
        files.append(name)
    diagnostics.draws.schema_path(run_id).write_text(
        json.dumps(
            {
                "schema": diagnostics.draws.SCHEMA_VERSION,
                "files": files,
                "params": {
                    "v": {"dims": [], "cols": [2]},
                    "x": {"dims": [3], "cols": [3, 4, 5]},
                    "fixed": {"dims": [], "cols": [6]},
                },
                "sampler_stats": {
                    "lp__": {"dims": [], "cols": [0]},
                    "divergent__": {"dims": [], "cols": [1]},
                },
            }
        )
    )


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
    """The method gate must fire before `draws.load` ever runs: a run with no
    draws_schema.json also raises RunError, on a different message, so this has
    to pin the gate's own text or it would pass even with the gate deleted."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    with pytest.raises(errors.RunError, match="divergence localization needs HMC sampler stats"):
        diagnostics.run_pathologies("variational-0001")


def test_pathologies_validates_top_k(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001")
    with pytest.raises(errors.ValidationError):
        diagnostics.run_pathologies("sample-0001", top_k=0)


def test_pathologies_labels_matrix_elements_in_stan_syntax(tmp_path, monkeypatch):
    """_draw_matrix labelled a matrix[2,2] as m[0]..m[3] -- a flat 0-based index
    corresponding to no index in anyone's Stan file."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    lines = ["# c", "lp__,divergent__,m.1.1,m.2.1,m.1.2,m.2.2"]
    for i in range(200):
        div = 1 if i < 10 else 0
        # m.2.2 sits far from the bulk exactly on the divergent draws
        vals = rng.normal(size=4)
        if div:
            vals[3] += 6.0
        lines.append(f"-1,{div}," + ",".join(f"{v:.6f}" for v in vals))
    (csvdir / "a.csv").write_text("\n".join(lines) + "\n")
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": diagnostics.draws.SCHEMA_VERSION,
                "files": ["a.csv"],
                "params": {"m": {"dims": [2, 2], "cols": [2, 4, 3, 5]}},
                "sampler_stats": {
                    "lp__": {"dims": [], "cols": [0]},
                    "divergent__": {"dims": [], "cols": [1]},
                },
            }
        )
    )
    got = diagnostics.run_pathologies(rid, permutations=200)
    assert got["n_divergent"] == 10
    assert got["ranking"][0]["param"] == "m[2,2]"


def test_ebfmi_is_reported_not_swallowed(tmp_path, monkeypatch):
    """az.bfmi raises TypeError on an arviz 1.2 DataTree, and a bare `except:
    pass` turned that into E-BFMI silently absent from every sample run — for the
    diagnostic most specific to funnel geometry. `_sampler_stats` now feeds it a
    plain (chain, draw) `energy__` array off the typed draws layer instead of an
    idata, so that DataTree can no longer reach it at all -- verified end to end
    through `run_diagnostics`, on a real (if minimal) sample run's sidecar and
    CSVs rather than a hand-built `Draws`.

    The `except Exception` around the az.bfmi call itself staying narrow enough
    to still report a genuine failure is covered separately, by
    `test_sampler_stats_reports_ebfmi_warning_when_bfmi_raises`.
    """
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_run_with_stats("sample-0001", diverging=False, tree_depth=3)
    # stansummary's binary is not available in the offline suite; this test is
    # about E-BFMI, not R-hat/ESS, so the summary call is stubbed out entirely.
    monkeypatch.setattr(diagnostics.stansummary, "summarize", lambda paths, **kw: {})
    d = diagnostics.run_diagnostics("sample-0001")
    assert d.get("min_ebfmi") is not None
    assert "ebfmi_warning" not in d


def test_sampler_stats_reports_ebfmi_warning_when_bfmi_raises(monkeypatch):
    """A diagnostic that quietly returns nothing on failure reads as "checked
    and fine" -- the exact bug this code's own comment argues against. Forces
    az.bfmi to raise (a stand-in for the arviz-1.2-DataTree TypeError this
    replaced) and checks the `except Exception` around it still surfaces a
    warning rather than swallowing it a second way."""
    import arviz as az

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(az, "bfmi", _boom)
    d = diagnostics.draws.Draws(
        params={},
        sampler_stats={"energy__": np.zeros((2, 50))},
        n_chain=2,
        n_draw=50,
    )
    got = diagnostics._sampler_stats(d, {})
    assert "min_ebfmi" not in got
    assert got["ebfmi_warning"] == "E-BFMI unavailable: boom"


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

    sc = diagnostics._sidecar_from_fit(_Fit(), "sample", "sample-0001")
    assert sc["schema"] == diagnostics.draws.SCHEMA_VERSION
    assert sc["params"]["theta"] == {"dims": [], "cols": [2]}
    # row-major fill order for (2,2): m[1,1], m[1,2], m[2,1], m[2,2]
    #                          cols:      3        5        4        6
    assert sc["params"]["m"] == {"dims": [2, 2], "cols": [3, 5, 4, 6]}
    assert sc["sampler_stats"]["lp__"] == {"dims": [], "cols": [0]}
    assert sc["sampler_stats"]["divergent__"] == {"dims": [], "cols": [1]}
    assert "theta" not in sc["sampler_stats"]
    assert sc["draw_row_offset"] == 0


def test_sidecar_records_draw_row_offset_one_for_variational_only():
    """variational's first data row is cmdstanpy's own mean-field mean, not a
    draw (cmdstanpy stanfit/vb.py: draws_np[0] is the mean, draws_np[1:] the
    sample) -- checked directly against cmdstanpy's row-assembly source for
    every method that reaches persist_run, not assumed from method_diagnostics
    or other constructor-visible similarities.

    generate_quantities is excluded from this table on purpose: its offset is
    NOT a fixed per-method constant (see
    test_draw_row_offset_for_generate_quantities_depends_on_source_method) --
    it is 0 unless its source run (method_args.from_run_id) was itself
    variational, in which case cmdstanpy re-evaluates the generated-quantities
    block at the source's mean row too, landing that same extra row in the GQ
    run's own output CSV.
    """

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    # run_id is never dereferenced for these methods -- only generate_quantities
    # looks it up (to resolve its source run's method) -- so any string is fine.
    for method in ("sample", "pathfinder", "laplace"):
        assert diagnostics._sidecar_from_fit(_Fit(), method, "irrelevant")["draw_row_offset"] == 0
    assert (
        diagnostics._sidecar_from_fit(_Fit(), "variational", "irrelevant")["draw_row_offset"] == 1
    )


def test_draw_row_offset_for_generate_quantities_depends_on_source_method(tmp_path, monkeypatch):
    """generate_quantities re-evaluates the generated-quantities block once per
    row of its source run -- including a variational source's mean row
    (cmdstanpy gq.py: _draws_start computes draw1=1 for a CmdStanVB
    previous_fit and applies it to the GQ run's OWN assembled draws, not just
    a slice of the source's) -- so a GQ run built from_run_id from a
    variational run inherits offset 1 in its own CSV; from any other method,
    offset 0. Resolved via the source run's persisted method, not
    isinstance(fit.previous_fit, CmdStanVB)."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done"})
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    runstore.write_meta(
        "generate_quantities-0001",
        {
            "method": "generate_quantities",
            "status": "done",
            "method_args": {"from_run_id": "sample-0001"},
        },
    )
    runstore.write_meta(
        "generate_quantities-0002",
        {
            "method": "generate_quantities",
            "status": "done",
            "method_args": {"from_run_id": "variational-0001"},
        },
    )
    assert diagnostics._draw_row_offset("generate_quantities", "generate_quantities-0001") == 0
    assert diagnostics._draw_row_offset("generate_quantities", "generate_quantities-0002") == 1


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

    def boom(fit, method, run_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(diagnostics, "_sidecar_from_fit", boom)

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    with pytest.raises(errors.RunError, match="draws schema not written"):
        diagnostics.persist_run(rid, "sample", _Fit())

    # the CSVs must survive the failure -- they're still worth keeping
    assert runstore.csv_paths(rid) == [str(runstore.run_dir(rid) / "csv" / "model_1.csv")]
    meta = runstore.read_meta(rid)
    # exact suffix, not `in`: a TypeError from a signature mismatch also
    # contains the function's own name ("boom() takes ... arguments") and
    # would satisfy an `in` check just as well, which is exactly how this
    # test went vacuous once already (a prior round's boom(fit) took a
    # TypeError for the intended RuntimeError without anyone noticing).
    assert meta["persist_warning"] == "draws schema not written: boom"


def test_persist_run_fails_when_no_draws_csvs_are_persisted(tmp_path, monkeypatch):
    """If CSV persistence produced nothing at all (e.g. its own csv_warning
    already fired, or the run's live/ never had any files), the sidecar must
    not write "successfully" with files: [] -- draws.load would hit min([])
    on an empty CSV list and die with a bare ValueError, not a RunError
    naming the run."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    # no live/ dir at all -> the CSV-persist step copies nothing, csv/ stays empty

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    with pytest.raises(errors.RunError, match="no persisted draws CSVs"):
        diagnostics.persist_run(rid, "sample", _Fit())

    meta = runstore.read_meta(rid)
    assert "persist_warning" in meta


def test_persist_run_writes_variational_sidecar_excluding_mean_row(tmp_path, monkeypatch):
    """End-to-end through persist_run/draws.load: variational's first CSV data
    row is cmdstanpy's own mean-field mean (stanfit/vb.py: draws_np[0]), not a
    draw. Verifies draw_row_offset is both written by persist_run AND honoured
    by draws.load, rather than each layer's unit test alone."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "variational-0001"
    runstore.write_meta(rid, {"method": "variational", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    # row 0 is cmdstanpy's variational mean (999.0 sentinel); rows 1.. are draws
    (live / "model.csv").write_text("# c\nlp__,theta\n0,999.0\n-1,0.5\n-1,0.7\n")

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    diagnostics.persist_run(rid, "variational", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["draw_row_offset"] == 1
    d = diagnostics.draws.load(rid)
    assert d.n_draw == 2
    assert d.params["theta"].tolist() == [[0.5, 0.7]]


def test_persist_run_gq_from_variational_excludes_leading_row(tmp_path, monkeypatch):
    """A generate_quantities run whose from_run_id names a variational run
    inherits that run's leading mean-evaluation row in its OWN output CSV
    (cmdstanpy gq.py's _draws_start/_assemble_generated_quantities), so its
    sidecar must record draw_row_offset=1 too -- not the 0 that would be
    right for a GQ run built from any other method."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    rid = "generate_quantities-0001"
    runstore.write_meta(
        rid,
        {
            "method": "generate_quantities",
            "status": "done",
            "method_args": {"from_run_id": "variational-0001"},
        },
    )
    live = runstore.live_dir(rid, create=True)
    # row 0 is GQ evaluated at the variational source's mean (999.0 sentinel)
    (live / "model.csv").write_text("# c\nlog_lik\n999.0\n0.1\n0.2\n")

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "log_lik"})

    diagnostics.persist_run(rid, "generate_quantities", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["draw_row_offset"] == 1
    d = diagnostics.draws.load(rid)
    assert d.n_draw == 2
    assert d.params["log_lik"].tolist() == [[0.1, 0.2]]
