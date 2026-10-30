"""Tests for draw extraction."""

import json

import pytest

from stan_mcp import diagnostics, draws, errors, runstore
from tests.fakes import (
    _typed_run,
    _typed_run_wide,
    _write_csv_run,
    _write_typed_csv_run,
)


def test_extract_draws_reads_the_typed_layer(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    got = draws.extract_draws(rid, param="theta")
    assert got["draws"] == [0.1, 0.2, 0.3, 0.4]
    assert "source" not in got  # there is only one source now


def test_extract_draws_by_chain_uses_sidecar_chain_order(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    got = draws.extract_draws(rid, param="theta", by_chain=True)
    assert got["draws"] == [[0.1, 0.2], [0.3, 0.4]]


def test_extract_draws_keeps_vector_shape(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    got = draws.extract_draws(rid, param="z")
    assert got["draws"][0] == [10.0, 20.0]


def test_extract_draws_unknown_param_raises(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    with pytest.raises(errors.RunError, match="nope"):
        draws.extract_draws(rid, param="nope")


def test_extract_draws_works_for_a_variational_run(tmp_path, monkeypatch):
    """One path for all six methods: no branch on the method at all."""
    rid = _typed_run(tmp_path, monkeypatch, method="variational")
    assert draws.extract_draws(rid, param="theta")["draws"] == [0.1, 0.2, 0.3, 0.4]


def test_extract_draws_single_draw_stays_a_list(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    out = draws.extract_draws(rid, "theta", thin=999)
    assert isinstance(out["draws"], list)  # not a bare float
    assert len(out["draws"]) == 1


def test_extract_draws_rejects_bad_thin(tmp_path, monkeypatch):
    rid = _typed_run(tmp_path, monkeypatch)
    with pytest.raises(errors.ValidationError):
        draws.extract_draws(rid, "theta", thin=0)


def test_extract_draws_scalar_is_flat_list(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    out = draws.extract_draws(rid, "theta")
    assert len(out["draws"]) == 20
    assert isinstance(out["draws"][0], float)


def test_extract_draws_vector_is_list_of_lists(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    out = draws.extract_draws(rid, "beta")
    assert len(out["draws"]) == 20
    assert isinstance(out["draws"][0], list) and len(out["draws"][0]) == 3


def test_extract_draws_by_chain(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    out = draws.extract_draws(rid, "theta", by_chain=True)
    assert out["by_chain"] is True
    assert len(out["draws"]) == 2 and len(out["draws"][0]) == 10  # 2 chains x 10 draws


def test_extract_draws_for_an_approximate_method(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_typed_csv_run("variational-0001")
    scalar = draws.extract_draws("variational-0001", "theta")
    assert scalar["draws"] == [0.5, 0.7]
    vector = draws.extract_draws("variational-0001", "beta")
    assert vector["draws"] == [[1.0, 2.0], [1.5, 2.5]]


def test_extract_draws_multi_param_for_an_approximate_method(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_typed_csv_run("variational-0001")
    out = draws.extract_draws("variational-0001", params=["theta", "beta"])
    assert set(out["draws"]) == {"theta", "beta"}
    assert out["draws"]["theta"] == [0.5, 0.7]
    assert out["draws"]["beta"] == [[1.0, 2.0], [1.5, 2.5]]


def test_extract_draws_requires_a_param(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_csv_run("variational-0001")
    with pytest.raises(errors.ValidationError):
        draws.extract_draws("variational-0001")


def test_extract_draws_by_chain_for_an_approximate_method(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    csvdir = runstore.run_dir("variational-0001", create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "c-1.csv").write_text("lp__,theta\n0,0.1\n0,0.2\n")
    (csvdir / "c-2.csv").write_text("lp__,theta\n0,0.3\n0,0.4\n")
    diagnostics.draws.schema_path("variational-0001").write_text(
        json.dumps(
            {
                "schema": 2,
                "files": ["c-1.csv", "c-2.csv"],
                "params": {"theta": {"dims": [], "cols": [1]}},
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    out = draws.extract_draws("variational-0001", "theta", by_chain=True)
    assert out["by_chain"] is True
    assert out["draws"] == [[0.1, 0.2], [0.3, 0.4]]  # chain structure preserved


def test_extract_draws_multi_param_loads_source_once(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    calls = []
    real = diagnostics.draws.load
    monkeypatch.setattr(diagnostics.draws, "load", lambda r: (calls.append(r), real(r))[1])
    draws.extract_draws(rid, params=["theta", "beta"])
    assert len(calls) == 1  # not one CSV parse per parameter


def test_extract_draws_caps_result_size(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)  # 2 chains x 10 draws = 20
    out = draws.extract_draws(rid, "theta", max_draws=5)
    assert len(out["draws"]) <= 5
    assert out["thin"] == 4
    assert len(draws.extract_draws(rid, "theta", max_draws=0)["draws"]) == 20


def test_max_draws_scales_with_vector_width(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)  # beta is (2 chains, 10 draws, 3)
    out = draws.extract_draws(rid, "beta", max_draws=6)
    assert sum(len(d) for d in out["draws"]) <= 6  # 2 draws x 3 components


def test_max_draws_splits_across_chains(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    out = draws.extract_draws(rid, "theta", by_chain=True, max_draws=4)
    assert sum(len(c) for c in out["draws"]) <= 4  # not 4 per chain


def test_max_draws_splits_across_params(tmp_path, monkeypatch):
    rid = _typed_run_wide(tmp_path, monkeypatch)
    out = draws.extract_draws(rid, params=["theta", "beta"], max_draws=6)
    theta, beta = out["draws"]["theta"], out["draws"]["beta"]
    assert len(theta) + sum(len(d) for d in beta) <= 6  # not 6 per name
