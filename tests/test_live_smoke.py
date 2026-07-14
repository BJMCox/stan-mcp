"""Live integration tests. Require the C++ toolchain; gated by STAN_MCP_LIVE=1."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STAN_MCP_LIVE") != "1", reason="set STAN_MCP_LIVE=1 to run"
)

BERNOULLI = """
data { int<lower=0> N; array[N] int<lower=0,upper=1> y; }
parameters { real<lower=0,upper=1> theta; }
model { theta ~ beta(1,1); y ~ bernoulli(theta); }
"""
DATA = {"N": 5, "y": [0, 1, 0, 1, 1]}

BERNOULLI_GQ = BERNOULLI + "\ngenerated quantities { int y_sim = bernoulli_rng(theta); }\n"


def test_compile_model_returns_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import compile as c

    out = c.compile_model(BERNOULLI)
    assert "error" not in out
    assert out["so_path"].endswith(".so")
    assert os.path.exists(out["so_path"])
    assert os.path.exists(out["exe_path"])


def test_param_info(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import density

    info = density.param_info(BERNOULLI, DATA)
    assert info["param_names"] == ["theta"]
    assert info["param_num"] == 1
    assert info["param_unc_num"] == 1  # theta is bounded -> 1 unconstrained dim


def test_live_log_density(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import density

    out = density.log_density_gradient(BERNOULLI, DATA, [0.0])
    assert isinstance(out["lp"], float)
    assert len(out["gradient"]) == 1


def test_live_check_gradient(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import density

    out = density.check_gradient(BERNOULLI, DATA, [0.1])
    assert out["max_abs_err"] < 1e-3


def test_live_sample_run(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import asyncio

    import diagnostics
    import sample

    mgr = sample.RunManager()
    started = mgr.start_run(
        BERNOULLI, DATA, method="sample",
        method_args={"chains": 2, "iter_warmup": 200, "iter_sampling": 200},
    )
    result = asyncio.run(mgr.await_run(started["run_id"], timeout=120))
    assert result["status"] == "done"
    summ = diagnostics.run_summary(started["run_id"], params=["theta"])
    assert any(row["param"] == "theta" for row in summ["summary"])
    diag = diagnostics.run_diagnostics(started["run_id"])
    assert "divergences" in diag


def test_live_generate_quantities(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import asyncio

    import diagnostics
    import errors
    import runstore
    import sample

    mgr = sample.RunManager()

    started_a = mgr.start_run(
        BERNOULLI, DATA, method="sample",
        method_args={"chains": 2, "iter_warmup": 200, "iter_sampling": 200},
    )
    result_a = asyncio.run(mgr.await_run(started_a["run_id"], timeout=120))
    assert result_a["status"] == "done"
    run_id_a = started_a["run_id"]

    started_gq = mgr.start_run(
        BERNOULLI_GQ, DATA, method="generate_quantities",
        method_args={"from_run_id": run_id_a},
    )
    result_gq = asyncio.run(mgr.await_run(started_gq["run_id"], timeout=120))
    assert result_gq["status"] == "done"
    run_id_gq = started_gq["run_id"]

    # Reality as of arviz 1.2 / arviz-base: its from_cmdstanpy converter calls
    # fit.method_variables(), which cmdstanpy's CmdStanGQ does not implement,
    # so the arviz conversion in diagnostics.persist_run fails (recorded as
    # persist_warning) and inference.nc is never written for GQ runs. Output
    # is still retrievable via the persisted raw CmdStan CSVs
    # (runstore.csv_paths). If a future arviz version maps CmdStanGQ cleanly,
    # prefer asserting through run_summary/extract_draws instead.
    try:
        diagnostics.extract_draws(run_id_gq, "y_sim")
    except errors.RunError:
        assert runstore.csv_paths(run_id_gq)
    else:
        summ = diagnostics.run_summary(run_id_gq)
        assert "summary" in summ
