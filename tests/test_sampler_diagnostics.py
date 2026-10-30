"""Tests for sampler diagnostics."""

import numpy as np

from stan_mcp import diagnostics, errors, fitstorage, runstore
from tests.fakes import (
    _typed_run,
    _write_run_with_stats,
)


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


def test_verdict_is_unknown_when_required_stats_are_unavailable():
    v = diagnostics._verdict({"divergences": 0, "sampler_stats_warning": "boom"})
    assert v == {"verdict": "unknown", "issues": []}


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
    got = diagnostics.convergence(rid)
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
    assert diagnostics.convergence(rid) == {"convergence_warning": "R-hat/ESS unavailable: boom"}


def test_convergence_degrades_on_a_non_run_error_too(tmp_path, monkeypatch):
    """The handler is deliberately broad, and it has to be.

    `cmdstanpy.cmdstan_path()` raises a bare ValueError when CmdStan is absent;
    catching only RunError here let that escape run_diagnostics AND run_geometry,
    breaking the documented promise of a "unknown" verdict when the six measured
    keys cannot be read. Any exception type must become a warning instead.
    """
    rid = _typed_run(tmp_path, monkeypatch)

    def _value_error(paths, **kw):
        raise ValueError("No CmdStan directory, path /nonexistent does not exist.")

    monkeypatch.setattr(diagnostics.stansummary, "summarize", _value_error)
    got = diagnostics.convergence(rid)
    assert "No CmdStan directory" in got["convergence_warning"]
    assert "max_rhat" not in got


def test_run_diagnostics_reports_unknown_when_cmdstan_is_absent(tmp_path, monkeypatch):
    """End-to-end on the documented contract: a sample run whose stansummary
    cannot run at all must come back with verdict "unknown", not raise."""
    rid = _typed_run(tmp_path, monkeypatch)

    def _value_error(paths, **kw):
        raise ValueError("No CmdStan installation found.")

    monkeypatch.setattr(diagnostics.stansummary, "summarize", _value_error)
    out = diagnostics.run_diagnostics(rid)
    assert out["verdict"] == "unknown"
    assert "No CmdStan installation" in out["convergence_warning"]


def test_sampler_stats_reads_typed_arrays_and_reports_ebfmi():
    """E-BFMI is not in stansummary (only in `diagnose`, as prose), so it stays
    computed from energy__ -- fed a clean array rather than a DataTree."""
    rng = np.random.default_rng(0)
    d = diagnostics.draws.Draws(
        run_id="sample-0001",
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

    assert fitstorage._method_diagnostics("variational", _VB()) == {"eta": 0.5}
    assert fitstorage._method_diagnostics("pathfinder", _PF()) == {"is_resampled": True}
    assert fitstorage._method_diagnostics("laplace", _LP()) == {"mode_lp": -3.0}


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


def test_convergence_warns_when_the_join_matches_nothing(tmp_path, monkeypatch):
    """The same wholesale miss on the diagnostics side. `_convergence` is a
    degrade-to-warning site, so the verdict must come back "unknown" WITH the
    reason, rather than silently omitting R-hat and ESS."""
    rid = _typed_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda paths, **kw: {"theta[1]": {"R_hat": 1.0}},
    )
    out = diagnostics.run_diagnostics(rid)
    assert out["verdict"] == "unknown"
    assert "matched a stansummary row" in out["convergence_warning"]


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
        run_id="sample-0001",
        params={},
        sampler_stats={"energy__": np.zeros((2, 50))},
        n_chain=2,
        n_draw=50,
    )
    got = diagnostics._sampler_stats(d, {})
    assert "min_ebfmi" not in got
    assert got["ebfmi_warning"] == "E-BFMI unavailable: boom"
