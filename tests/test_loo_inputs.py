"""Tests for loo inputs."""

import json

import pytest

from stan_mcp import diagnostics, errors, runstore
from tests.fakes import (
    _typed_run,
)


def test_idata_with_loglik_is_built_from_typed_arrays(tmp_path, monkeypatch):
    """PSIS-LOO is the only remaining consumer that needs an arviz object, so it
    constructs one on demand instead of reading inference.nc."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "a.csv").write_text(
        "# c\nlp__,theta,log_lik.1,log_lik.2\n-1,0.5,-0.7,-0.8\n-1,0.6,-0.6,-0.9\n"
    )
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": ["a.csv"],
                "params": {
                    "theta": {"dims": [], "cols": [1]},
                    "log_lik": {"dims": [2], "cols": [2, 3]},
                },
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    idata = diagnostics._idata_with_loglik(rid, "log_lik")
    assert "log_likelihood" in idata.children
    assert idata.log_likelihood["log_lik"].shape == (1, 2, 2)


def test_missing_loglik_still_raises_a_clear_error(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    with pytest.raises(errors.RunError, match="log_lik") as exc:
        diagnostics._idata_with_loglik(rid, "log_lik")
    # the names that ARE there are listed, so the caller can see the near-miss
    assert "theta" in str(exc.value)


def test_missing_loglik_on_a_run_with_no_parameters_reads_as_prose(tmp_path, monkeypatch):
    """`", ".join(sorted({}))` is the empty string, so the message used to end at
    "(have: )" — a trailing space and nothing after it."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "a.csv").write_text("# c\nlp__\n-1\n")
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": ["a.csv"],
                "params": {},
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    with pytest.raises(errors.RunError, match=r"have: no parameters at all"):
        diagnostics._idata_with_loglik(rid, "log_lik")
