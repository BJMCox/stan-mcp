"""Tests for run summary."""

import json

import pytest

from stan_mcp import diagnostics, errors, runstore
from tests.fakes import (
    _typed_run,
)


def test_summary_for_an_approximate_method_uses_numpy_moments(tmp_path, monkeypatch):
    """Approximate methods get numpy moments over the typed draws layer -- 1-based
    Stan labels, not CmdStan's dot notation."""
    rid = _typed_run(tmp_path, monkeypatch, method="variational")
    s = diagnostics.run_summary(rid)
    assert {r["param"] for r in s["summary"]} == {"theta", "z[1]", "z[2]"}  # lp__ excluded
    theta = next(r for r in s["summary"] if r["param"] == "theta")
    assert abs(theta["mean"] - 0.25) < 1e-9
    assert theta["n"] == 4


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
                "schema": 2,
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
    rows[7]["r_hat"] = 1.4  # the broken one, sitting in the tail a first-N cap drops
    rows[9]["ess_bulk"] = 12.0  # worst ESS, but r_hat is fine
    out = diagnostics._cap_rows(rows, 2)
    assert [r["param"] for r in out["summary"]] == ["z[7]", "z[9]"]
    assert out["truncated"]["selected_by"].startswith("worst r_hat")


def test_run_summary_truncation_without_convergence_columns(tmp_path, monkeypatch):
    """The approximate methods' numpy summary reports no r_hat/ess_bulk, so
    order is all there is."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rows = [{"param": f"z[{i}]", "mean": 0.0} for i in range(5)]
    out = diagnostics._cap_rows(rows, 2)
    assert [r["param"] for r in out["summary"]] == ["z[0]", "z[1]"]
    assert out["truncated"]["selected_by"] == "order"


def test_run_summary_caps_rows_for_an_approximate_method(tmp_path, monkeypatch):
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
                "schema": 2,
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


def test_run_summary_keeps_significant_figures_not_decimal_places(tmp_path, monkeypatch):
    """`round(x, 6)` rounds decimal PLACES, which partly undoes --sig_figs 18: it
    collapses any small-magnitude statistic to a flat 0.0. A well-mixed
    parameter's MCSE is sd/sqrt(ESS), and a parameter whose sd is ~1e-5 has a
    mean, sd and quantiles all below the old cliff."""
    rid = _typed_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda paths, **kw: {
            "theta": {"Mean": 1.23456789e-8, "StdDev": 2.5e-9, "MCSE": 3.21e-11},
            "z.1": {"Mean": 0.5},
            "z.2": {"Mean": 0.5},
        },
    )
    theta = next(r for r in diagnostics.run_summary(rid)["summary"] if r["param"] == "theta")
    # every one of these read 0.0 under round(x, 6)
    assert theta["mean"] == pytest.approx(1.23457e-8)
    assert theta["sd"] == pytest.approx(2.5e-9)
    assert theta["mcse"] == pytest.approx(3.21e-11)


def test_summary_from_draws_keeps_significant_figures(tmp_path, monkeypatch):
    """Same cliff on the approximate-method path, which computes its own moments:
    a parameter whose draws sit around 1e-8 must not summarize as all zeros."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "variational-0001"
    runstore.write_meta(rid, {"method": "variational", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "c.csv").write_text("# c\nlp__,tiny\n0,1e-8\n0,3e-8\n")
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": ["c.csv"],
                "params": {"tiny": {"dims": [], "cols": [1]}},
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    row = diagnostics.run_summary(rid)["summary"][0]
    assert row["mean"] == pytest.approx(2e-8)
    assert row["sd"] > 0.0  # read 0.0 under round(x, 6)
    assert row["q50"] == pytest.approx(2e-8)


def test_run_summary_replaces_nan_with_null(tmp_path, monkeypatch):
    """stansummary reports `nan` for the MCSE/ESS/R_hat of a CONSTANT column
    (verified: `"fixed",2.5,nan,0,0,...,nan,nan,nan,nan`), and NaN is not valid
    JSON. A constant parameter is common, so this is a normal path."""
    rid = _typed_run(tmp_path, monkeypatch)
    nan = float("nan")
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda paths, **kw: {
            "theta": {"Mean": 2.5, "StdDev": 0.0, "MCSE": nan, "ESS_bulk": nan, "R_hat": nan},
            "z.1": {"Mean": 0.5},
            "z.2": {"Mean": 0.5},
        },
    )
    theta = next(r for r in diagnostics.run_summary(rid)["summary"] if r["param"] == "theta")
    assert theta["mcse"] is None and theta["ess_bulk"] is None and theta["r_hat"] is None
    assert theta["mean"] == 2.5  # the finite values are untouched
    # and the whole result really is JSON-serializable, which NaN is not
    json.dumps(diagnostics.run_summary(rid))


def test_run_summary_raises_when_the_join_matches_nothing(tmp_path, monkeypatch):
    """A wholesale join miss — a naming change on either side — would otherwise
    return `{"summary": []}` as a SUCCESS. The per-element `continue` stays for a
    genuinely partial match; only matching nothing at all is the failure."""
    rid = _typed_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda paths, **kw: {"theta[1]": {"Mean": 0.25}},  # munged, so nothing matches
    )
    with pytest.raises(errors.RunError, match="matched a stansummary row"):
        diagnostics.run_summary(rid)


def test_run_summary_tolerates_a_partial_join(tmp_path, monkeypatch):
    """Paired negative to the guard above: one missing row is not a wholesale
    miss, and must still return the rows that did match."""
    rid = _typed_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda paths, **kw: {"theta": {"Mean": 0.25}, "z.1": {"Mean": 11.5}},  # z.2 absent
    )
    assert [r["param"] for r in diagnostics.run_summary(rid)["summary"]] == ["theta", "z[1]"]


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
    """A malformed or partially-written CSV set leaves chains of unequal length —
    not a cancelled run, which never gets a sidecar to be read at all.
    `_summary_from_draws` must surface the truncation rather than silently
    reporting statistics over the shortened array with no sign anything was
    dropped."""
    rid = _typed_run(tmp_path, monkeypatch, method="variational", ragged=True)
    out = diagnostics.run_summary(rid)
    assert out["chains_truncated_to"] == 1


def test_run_summary_omits_chains_truncated_to_when_chains_are_even(tmp_path, monkeypatch):
    """Paired negative: a key that is always present would pass the test above
    while being wrong."""
    rid = _typed_run(tmp_path, monkeypatch, method="variational")
    out = diagnostics.run_summary(rid)
    assert "chains_truncated_to" not in out
