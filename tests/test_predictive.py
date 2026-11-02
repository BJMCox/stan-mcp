import json
from pathlib import Path

import numpy as np
import pytest

from stan_mcp import predictive
from stan_mcp.errors import ValidationError
from stan_mcp.validation_stats import mean_uncertainty
from tests.fakes import write_draws_run


def test_predictive_event_axes_ties_and_rng_tail_uncertainty(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path))
    write_draws_run(
        "rng-1", {"y_rep": np.array([[[0, 0], [1, 3], [1, 2], [3, 3]]])}, method="fixed_param"
    )
    observed = tmp_path / "observed.json"
    observed.write_text(json.dumps({"y": [1, 2]}))
    request = tmp_path / "check.json"
    request.write_text(
        json.dumps(
            {
                "run_id": "rng-1",
                "kind": "prior",
                "independent_draws": True,
                "observed": str(observed),
                "checks": [
                    {"observed": "y", "replicated": "y_rep", "statistic": "mean"},
                    {"observed": "y", "replicated": "y_rep", "statistic": "sd"},
                ],
            }
        )
    )
    result = predictive.check_predictive(str(request))
    row = result["checks"][0]
    assert row["observed_statistic"] == 1.5
    assert row["replicated_mean"] == 1.625
    assert row["tail_probability_ge"] == 0.75
    assert row["equality_fraction"] == 0.25
    assert row["tail_uncertainty"]["wilson_95"][0] < 0.75 < row["tail_uncertainty"]["wilson_95"][1]
    assert result["checks"][1]["observed_statistic"] == pytest.approx(np.sqrt(0.5))
    assert Path(result["plot"]).stat().st_size > 0
    assert json.loads(Path(result["report_file"]).read_text())["checks"] == result["checks"]
    observed.write_text(json.dumps({"y": [1]}))
    with pytest.raises(ValidationError):
        predictive.check_predictive(str(request))


def test_predictive_normal_shared_parameter_control(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path))
    rng = np.random.default_rng(18)
    theta = rng.normal(2, np.sqrt(0.5), size=(2, 3000, 1))
    y = theta + rng.normal(size=(2, 3000, 5))
    write_draws_run("rng-1", {"y_rep": y}, method="fixed_param")
    request = tmp_path / "check.json"
    request.write_text(
        json.dumps(
            {
                "run_id": "rng-1",
                "kind": "prior",
                "independent_draws": True,
                "observed": {"y": [2] * 5},
                "checks": [{"observed": "y", "replicated": "y_rep", "statistic": "mean"}],
            }
        )
    )
    row = predictive.check_predictive(str(request))["checks"][0]
    assert abs(row["replicated_mean"] - 2) < 4 * np.sqrt(0.7 / 6000)
    assert abs(row["replicated_quantiles"]["q95"] - (2 + 1.644853626951 * np.sqrt(0.7))) < 0.08


def test_mcse_accounts_for_dependence_and_constant_indicators():
    rng = np.random.default_rng(9)
    independent = rng.normal(size=(2, 2000))
    correlated = np.empty_like(independent)
    correlated[:, 0] = independent[:, 0]
    for draw in range(1, correlated.shape[1]):
        correlated[:, draw] = (
            0.95 * correlated[:, draw - 1] + np.sqrt(1 - 0.95**2) * independent[:, draw]
        )
    iid = mean_uncertainty(independent, "sample")
    dependent = mean_uncertainty(correlated, "sample")
    assert dependent["mcse"] > 3 * iid["mcse"]
    assert dependent["ess"] < iid["ess"] / 5
    assert mean_uncertainty(np.zeros((2, 20)), "sample")["mcse"] is None
    assert mean_uncertainty(independent, "variational")["mcse"] is None
    assert mean_uncertainty(independent, "fixed_param")["mcse"] is None
