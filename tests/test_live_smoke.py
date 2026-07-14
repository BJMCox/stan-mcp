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
