"""Tests for initialization."""

import json

import pytest

from stan_mcp import diagnostics, runstore
from stan_mcp import inference as sample
from tests.fakes import (
    _write_typed_csv_run,
)


def test_posterior_mean_inits_from_typed_draws(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_typed_csv_run("variational-0001")
    inits = sample.posterior_mean_inits("variational-0001")
    assert abs(inits["theta"] - 0.6) < 1e-9
    assert inits["beta"] == [1.25, 2.25]  # means of (1.0,1.5) and (2.0,2.5)


def test_posterior_mean_inits_from_optimize(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("optimize-0001", {"method": "optimize", "status": "done"})
    runstore.persist_estimate("optimize-0001", {"theta": 0.7, "lp__": -2.0})
    inits = sample.posterior_mean_inits("optimize-0001")
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
                "schema": 2,
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
    inits = sample.posterior_mean_inits(rid)
    assert inits["s"] == pytest.approx(3.0)
    assert inits["m"] == [[12.0, 13.0], [22.0, 23.0]]
    assert "lp__" not in inits
