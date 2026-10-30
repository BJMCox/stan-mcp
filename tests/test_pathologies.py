"""Tests for pathologies."""

import json

import numpy as np
import pytest

from stan_mcp import diagnostics, errors, runstore


def _write_divergent_run(run_id, n_div=20, planted=3.0, n_chain=2, n_draw=500, seed=0):
    """A run whose divergences are planted on `v` and nowhere else.

    Writes typed CmdStan CSVs (one file per chain) plus a draws_schema.json
    sidecar rather than an arviz idata, so `draws.load` reads it through the
    same typed layer `run_pathologies` now uses. Flat length-`n` arrays are
    split into contiguous `n_draw`-sized chunks per chain -- the same layout
    `.reshape(n_chain, n_draw)` used to give in-memory -- so `div.reshape(-1)`
    recovers the identical flat order and the planted divergences land on the
    same draws as before.
    """
    rng = np.random.default_rng(seed)
    n = n_chain * n_draw
    v = rng.normal(size=n)
    x = rng.normal(size=(n, 3))
    fixed = np.full(n, 2.5)  # zero variance
    lp = rng.normal(size=n)
    div = np.zeros(n, bool)
    if n_div:
        div[rng.choice(n, n_div, replace=False)] = True
    v[div] += planted

    runstore.write_meta(run_id, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(run_id, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    files = []
    for chain in range(n_chain):
        lo, hi = chain * n_draw, (chain + 1) * n_draw
        # repr'd through a plain python float: full round-trip precision,
        # without numpy>=2's `np.float64(...)` repr wrapper that `float()`
        # in draws._read_csv cannot parse.
        rows = "\n".join(
            ",".join(
                repr(float(col))
                for col in (lp[i], int(div[i]), v[i], x[i, 0], x[i, 1], x[i, 2], fixed[i])
            )
            for i in range(lo, hi)
        )
        name = f"chain{chain}.csv"
        (csvdir / name).write_text(f"# c\nlp__,divergent__,v,x.1,x.2,x.3,fixed\n{rows}\n")
        files.append(name)
    diagnostics.draws.schema_path(run_id).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": files,
                "params": {
                    "v": {"dims": [], "cols": [2]},
                    "x": {"dims": [3], "cols": [3, 4, 5]},
                    "fixed": {"dims": [], "cols": [6]},
                },
                "sampler_stats": {
                    "lp__": {"dims": [], "cols": [0]},
                    "divergent__": {"dims": [], "cols": [1]},
                },
            }
        )
    )


def test_pathologies_finds_the_planted_parameter(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001")
    out = diagnostics.run_pathologies("sample-0001", permutations=300)
    assert out["n_divergent"] == 20
    assert out["ranking"][0]["param"] == "v"
    assert out["ranking"][0]["direction"] == "above"  # planted shift was positive
    assert out["separation"]["margin"] > 3
    assert out["separation"]["p_value"] < 0.05


def test_pathologies_skips_constant_parameters(tmp_path, monkeypatch):
    """A parameter that never varies carries no location information, and
    standardizing it would divide by zero."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001")
    out = diagnostics.run_pathologies("sample-0001", top_k=10, permutations=100)
    assert "fixed" not in {r["param"] for r in out["ranking"]}


def test_pathologies_no_divergences_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001", n_div=0)
    out = diagnostics.run_pathologies("sample-0001", permutations=100)
    assert out["n_divergent"] == 0 and out["ranking"] == []
    assert "no divergent transitions" in out["note"]


def test_pathologies_warns_when_divergences_are_few(tmp_path, monkeypatch):
    """Below a handful of divergences the ranking outruns its evidence, and the
    result has to say so rather than read as confident."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001", n_div=3)
    out = diagnostics.run_pathologies("sample-0001", permutations=100)
    assert "indicative at best" in out["warning"]


def test_pathologies_is_deterministic(tmp_path, monkeypatch):
    """Same run must give the same p-value; a diagnostic that wobbles between
    calls is not one you can act on."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001")
    a = diagnostics.run_pathologies("sample-0001", permutations=200)
    b = diagnostics.run_pathologies("sample-0001", permutations=200)
    assert a["separation"] == b["separation"]


def test_pathologies_rejects_non_sample_methods(tmp_path, monkeypatch):
    """The method gate must fire before `draws.load` ever runs: a run with no
    draws_schema.json also raises RunError, on a different message, so this has
    to pin the gate's own text or it would pass even with the gate deleted."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    with pytest.raises(errors.RunError, match="divergence localization needs HMC sampler stats"):
        diagnostics.run_pathologies("variational-0001")


def test_pathologies_validates_top_k(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    _write_divergent_run("sample-0001")
    with pytest.raises(errors.ValidationError):
        diagnostics.run_pathologies("sample-0001", top_k=0)


def test_pathologies_labels_matrix_elements_in_stan_syntax(tmp_path, monkeypatch):
    """_draw_matrix labelled a matrix[2,2] as m[0]..m[3] -- a flat 0-based index
    corresponding to no index in anyone's Stan file."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    lines = ["# c", "lp__,divergent__,m.1.1,m.2.1,m.1.2,m.2.2"]
    for i in range(200):
        div = 1 if i < 10 else 0
        # m.2.2 sits far from the bulk exactly on the divergent draws
        vals = rng.normal(size=4)
        if div:
            vals[3] += 6.0
        lines.append(f"-1,{div}," + ",".join(f"{v:.6f}" for v in vals))
    (csvdir / "a.csv").write_text("\n".join(lines) + "\n")
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": ["a.csv"],
                "params": {"m": {"dims": [2, 2], "cols": [2, 4, 3, 5]}},
                "sampler_stats": {
                    "lp__": {"dims": [], "cols": [0]},
                    "divergent__": {"dims": [], "cols": [1]},
                },
            }
        )
    )
    got = diagnostics.run_pathologies(rid, permutations=200)
    assert got["n_divergent"] == 10
    assert got["ranking"][0]["param"] == "m[2,2]"
