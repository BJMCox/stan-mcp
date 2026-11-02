import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from stan_mcp import calibration, studies
from tests.fakes import analytic_study_manager


def test_randomized_ties_and_finite_draw_coverage():
    rng = np.random.default_rng(48)
    ranks = np.array(
        [calibration.rank_interval(np.ones(9), 1, 0.9, rng)["rank"] for _ in range(5000)]
    )
    counts = np.bincount(ranks, minlength=10)
    assert np.all(np.abs(counts - 500) < 100)
    one = calibration.rank_interval(np.array([0.0]), 1, 0.9, rng)
    assert one["covered"] == 0
    assert one["finite_draw_target"] == 0
    assert one["order_indices"] == [1, 1]


def test_conjugate_sbc_iid_control_shift_and_repeated_draws():
    rng = np.random.default_rng(561)
    size, n = 3000, 19
    truth = rng.normal(size=size)
    y = truth + rng.normal(size=size)
    posterior = rng.normal(y[:, None] / 2, np.sqrt(0.5), size=(size, n))
    exact = [
        calibration.rank_interval(values, target, 0.8, rng)
        for values, target in zip(posterior, truth, strict=True)
    ]
    counts = np.bincount([r["rank"] for r in exact], minlength=n + 1)
    assert np.sum((counts - size / (n + 1)) ** 2 / (size / (n + 1))) < 45
    target = exact[0]["finite_draw_target"]
    assert abs(np.mean([r["covered"] for r in exact]) - target) < 4 * np.sqrt(
        target * (1 - target) / size
    )
    shifted = [
        calibration.rank_interval(values + 3, target, 0.8, rng)["rank"]
        for values, target in zip(posterior, truth, strict=True)
    ]
    assert np.mean(np.array(shifted) == 0) > 0.8
    repeated = [
        calibration.rank_interval(np.repeat(values[0], n), target, 0.8, rng)["rank"]
        for values, target in zip(posterior, truth, strict=True)
    ]
    assert set(repeated) == {0, n}


def test_sbc_report_discrete_bins_and_missing_case_bounds(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)
    request = tmp_path / "study.json"
    request.write_text(
        json.dumps(
            {
                "schema": studies.SCHEMA,
                "seed": 7,
                "replicates": 3,
                "max_parallel": 2,
                "tasks": [
                    {
                        "id": "normal",
                        "stan": "parameters {real theta;} model {theta ~ normal(0,1);}",
                        "simulate": {
                            "stan": "generated quantities {real theta_sim = normal_rng(0,1);}",
                            "data_variables": {"y": "y_sim"},
                            "truth_variables": {"theta": "theta_sim"},
                        },
                    }
                ],
            }
        )
    )
    runner = manager.runs._runner
    failure = False

    def fail_once(method, model, data, kwargs):
        nonlocal failure
        if method == "sample" and not failure:
            failure = True
            raise RuntimeError("missing fit")
        return runner(method, model, data, kwargs)

    monkeypatch.setattr(manager.runs, "_runner", fail_once)
    study_id = manager.start(str(request))["study_id"]
    try:
        assert asyncio.run(manager.await_job(study_id, 10))["counts"] == {"failed": 1, "done": 2}
        check = tmp_path / "sbc.json"
        check.write_text(json.dumps({"study_id": study_id, "draws": 9, "bins": 4, "seed": 5}))
        result = calibration.check_sbc(str(check))
        assert result["planned_cases"] == result["attempted_cases"] == 3
        assert result["completed_cases"] == result["assessable_cases"] == 2
        row = result["quantities"][0]
        assert row["planned"] == 3 and row["unresolved"] == 1
        assert row["rank_histogram"]["iid_reference_probabilities"] == [0.3, 0.2, 0.3, 0.2]
        lower, upper = row["coverage"]["all_case_empirical_bounds"]
        assert upper - lower == pytest.approx(1 / 3)
        assert load_file(result["ranks_file"])["rank"].shape == (2,)
        assert Path(result["plot"]).stat().st_size
        again = calibration.check_sbc(str(check))
        assert again["quantities"] == result["quantities"]
        assert np.array_equal(
            load_file(again["ranks_file"])["rank"], load_file(result["ranks_file"])["rank"]
        )
    finally:
        manager.shutdown()


def test_bounded_sbc_response_keeps_full_report_and_sampling_law(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manager = analytic_study_manager(monkeypatch)

    def runner(method, model, data, kwargs):
        if method == "fixed_param":
            return {"theta_sim": np.ones((1, 1, 11)), "y_sim": np.zeros((1, 1))}
        return {"theta": np.ones((1, 20, 11))}

    monkeypatch.setattr(manager.runs, "_runner", runner)
    request = tmp_path / "study.json"
    request.write_text(
        json.dumps(
            {
                "schema": studies.SCHEMA,
                "seed": 7,
                "tasks": [
                    {
                        "id": "vector",
                        "stan": "model {}",
                        "method": "variational",
                        "simulate": {
                            "stan": "model {}",
                            "data_variables": {"y": "y_sim"},
                            "truth_variables": {"theta": "theta_sim"},
                        },
                    }
                ],
            }
        )
    )
    study_id = manager.start(str(request))["study_id"]
    try:
        assert asyncio.run(manager.await_job(study_id, 10))["counts"] == {"done": 1}
        request = tmp_path / "sbc.json"
        request.write_text(json.dumps({"study_id": study_id, "draws": 9, "bins": 4}))
        result = calibration.check_sbc(str(request))
        assert len(result["quantities"]) == 10
        assert result["quantity_count"] == 11 and result["omitted_quantities"] == 1
        full = json.loads(Path(result["report_file"]).read_text())
        assert len(full["quantities"]) == 11
        assert len(full["quantities"][0]["rank_histogram"]["pointwise_rate_intervals_95"]) == 4
        assert result["sampling_method_counts"] == {"variational": 1}
        assert "sampling_warning" in result
        assert result["quantities"][0]["ties"] == 1
    finally:
        manager.shutdown()
