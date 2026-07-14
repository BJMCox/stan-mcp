import pytest

import diagnostics
import errors
import runstore


def test_summary_optimize(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("optimize-0001", {"method": "optimize", "status": "done"})
    runstore.persist_estimate("optimize-0001", {"theta": 0.6, "lp__": -3.2})
    out = diagnostics.run_summary("optimize-0001")
    assert out["estimate"]["theta"] == 0.6


def test_extract_draws_rejects_optimize(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("optimize-0001", {"method": "optimize", "status": "done"})
    with pytest.raises(errors.RunError):
        diagnostics.extract_draws("optimize-0001", "theta")


def test_persist_optimize_writes_estimate(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("optimize-0001", {"method": "optimize", "status": "running"})

    class FakeMLE:
        optimized_params_dict = {"theta": 0.7, "lp__": -2.0}

    diagnostics.persist_run("optimize-0001", "optimize", FakeMLE())
    assert runstore.load_estimate("optimize-0001")["theta"] == 0.7
