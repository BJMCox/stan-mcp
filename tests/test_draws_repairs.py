"""Draw identity, component typing, and bounded extraction regressions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from cmdstanpy.stanfit.metadata import InferenceMetadata

import diagnostics
import draws
import runstore
from errors import RunError, ValidationError


def _run(tmp_path, monkeypatch, header, rows, method="pathfinder"):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = f"{method}-0001"
    runstore.write_meta(rid, {"method": method, "status": "done"})
    csvdir = runstore.run_dir(rid) / "csv"
    csvdir.mkdir()
    fit = SimpleNamespace(metadata=InferenceMetadata({"raw_header": header}))
    sidecar = diagnostics._sidecar_from_fit(fit, method, rid)
    sidecar["files"] = []
    for chain, values in enumerate(rows):
        name = f"chain_{chain}.csv"
        sidecar["files"].append(name)
        body = "\n".join(",".join(str(v) for v in row) for row in values)
        (csvdir / name).write_text(f"# before header\n{header}\n# before draws\n{body}\n")
    draws.schema_path(rid).write_text(json.dumps(sidecar))
    return rid


@pytest.mark.parametrize("by_chain", [False, True])
def test_extraction_uses_common_rows_and_whole_call_cap(tmp_path, monkeypatch, by_chain):
    rows = [[[i, 10 * i, 10 * i + 1] for i in range(10 * c, 10 * c + 10)] for c in range(2)]
    rid = _run(tmp_path, monkeypatch, "x,v.1,v.2", rows)
    out = diagnostics.extract_draws(rid, params=["x", "v"], by_chain=by_chain, max_draws=12)
    x = np.asarray(out["draws"]["x"])
    v = np.asarray(out["draws"]["v"])
    assert x.size + v.size <= 12
    np.testing.assert_array_equal(v[..., 0], 10 * x)
    expected = np.asarray(rows)[..., 0]
    if by_chain:
        expected = expected[:, :: out["thin"]]
    else:
        expected = expected.reshape(-1)[:: out["thin"]]
    np.testing.assert_array_equal(x, expected)


def test_cap_requires_a_complete_row_across_chains(tmp_path, monkeypatch):
    rid = _run(tmp_path, monkeypatch, "v.1,v.2,v.3", [[[1, 2, 3]], [[4, 5, 6]]])
    with pytest.raises(ValidationError, match="cannot hold one retained row"):
        diagnostics.extract_draws(rid, "v", by_chain=True, max_draws=5)
    out = diagnostics.extract_draws(rid, "v", by_chain=True, max_draws=6)
    assert out["draws"] == [[[1, 2, 3]], [[4, 5, 6]]]


def test_extract_preserves_singleton_and_matrix_axes(tmp_path, monkeypatch):
    rid = _run(tmp_path, monkeypatch, "x,v.1,m.1.1,m.2.1,m.1.2,m.2.2", [[[1, 2, 11, 21, 12, 22]]])
    out = diagnostics.extract_draws(rid, params=["x", "v", "m"])
    assert out["draws"] == {"x": [1], "v": [[2]], "m": [[[11, 12], [21, 22]]]}


def test_complex_and_nested_tuple_leaves_keep_component_axes(tmp_path, monkeypatch):
    header = (
        "s.real,s.imag,z.1.real,z.1.imag,z.2.real,z.2.imag,"
        "t.1:1,t.1:2.1.real,t.1:2.1.imag,t.1:2.2.real,t.1:2.2.imag,"
        "t.2:1,t.2:2.1.real,t.2:2.1.imag,t.2:2.2.real,t.2:2.2.imag,"
        "u:1:1,u:1:2,u:2.1,u:2.2"
    )
    rid = _run(tmp_path, monkeypatch, header, [[list(range(20))]])
    d = draws.load(rid)
    assert d.params["s.real"].tolist() == [[0]]
    assert d.params["s.imag"].tolist() == [[1]]
    assert d.params["z.real"].tolist() == [[[2, 4]]]
    assert d.params["z.imag"].tolist() == [[[3, 5]]]
    assert d.params["t:1"].tolist() == [[[6, 11]]]
    assert d.params["t:2.real"].tolist() == [[[[7, 9], [12, 14]]]]
    assert d.params["t:2.imag"].tolist() == [[[[8, 10], [13, 15]]]]
    assert d.params["u:1:1"].tolist() == [[16]]
    assert d.params["u:1:2"].tolist() == [[17]]
    assert d.params["u:2"].tolist() == [[[18, 19]]]
    np.testing.assert_array_equal(
        draws.parameter_matrix(d, ["t.2:2.2.imag", "s.real", "u:1:2", "z.2.real"]),
        [[15, 0, 17, 4]],
    )
    sc = draws.schema(rid)
    assert sc["variables"]["t"] == {
        "type": "tuple",
        "dims": [2],
        "contents": [
            {"type": "scalar", "dims": [], "contents": []},
            {"type": "complex", "dims": [2], "contents": []},
        ],
    }
    assert list(draws.spec_element_labels("t:2.real", sc["params"]["t:2.real"])) == [
        ("t[1]:2[1].real", "t.1:2.1.real"),
        ("t[1]:2[2].real", "t.1:2.2.real"),
        ("t[2]:2[1].real", "t.2:2.1.real"),
        ("t[2]:2[2].real", "t.2:2.2.real"),
    ]
    matrix, labels = draws.matrix(d)
    assert dict(zip(labels, matrix[0], strict=True))["t[2]:2[1].real"] == 12
    assert dict(zip(labels, matrix[0], strict=True))["z[2].imag"] == 5
    summary = diagnostics.run_summary(rid, max_rows=0)
    assert len(summary["summary"]) == 20
    assert summary["summary"][8]["param"] == "t[1]:2[1].real"


def test_each_csv_must_match_the_complete_recorded_header(tmp_path, monkeypatch):
    rid = _run(tmp_path, monkeypatch, "a,b", [[[1, 2]], [[3, 4]]])
    path = runstore.run_dir(rid) / "csv" / "chain_1.csv"
    path.write_text("b,a\n4,3\n")
    with pytest.raises(RunError, match="header differs"):
        draws.load(rid)


def test_sample_summary_verifies_header_before_using_statistics(tmp_path, monkeypatch):
    rid = _run(tmp_path, monkeypatch, "a,b", [[[1, 2]], [[3, 4]]], method="sample")
    calls = []

    def summarize(paths):
        calls.append(paths)
        return {"a": {"Mean": 2}, "b": {"Mean": 3}}

    monkeypatch.setattr(diagnostics.stansummary, "summarize", summarize)
    assert diagnostics.run_summary(rid)["summary"] == [
        {"param": "a", "mean": 2},
        {"param": "b", "mean": 3},
    ]
    path = runstore.run_dir(rid) / "csv" / "chain_1.csv"
    path.write_text("b,a\n4,3\n")
    with pytest.raises(RunError, match="header differs"):
        diagnostics.run_summary(rid)
    assert len(calls) == 1


def test_summary_computes_only_displayed_rows_with_joint_quantiles(tmp_path, monkeypatch):
    rid = _run(tmp_path, monkeypatch, "v.1,v.2,v.3", [[[1, 2, 3], [3, 4, 5]]])
    quantile = np.quantile
    calls = []

    def tracked_quantile(a, q):
        calls.append(q)
        return quantile(a, q)

    monkeypatch.setattr(np, "quantile", tracked_quantile)
    out = diagnostics.run_summary(rid, max_rows=1)
    assert calls == [[0.05, 0.5, 0.95]]
    assert out["summary"] == [
        {"param": "v[1]", "mean": 2, "sd": 1.41421, "q5": 1.1, "q50": 2, "q95": 2.9, "n": 2}
    ]
    assert out["truncated"]["total"] == 3


def test_legacy_sidecar_explicitly_reports_unverified_header(tmp_path, monkeypatch):
    rid = _run(tmp_path, monkeypatch, "x", [[[1]]])
    sidecar = draws.schema(rid)
    sidecar["schema"] = 2
    del sidecar["header"]
    draws.schema_path(rid).write_text(json.dumps(sidecar))
    assert diagnostics.extract_draws(rid, "x")["header_warning"]
    assert diagnostics.run_summary(rid)["header_warning"]


def test_streaming_parser_names_corrupt_numeric_csv(tmp_path, monkeypatch):
    rid = _run(tmp_path, monkeypatch, "x", [[[1]]])
    path = runstore.run_dir(rid) / "csv" / "chain_0.csv"
    path.write_text("x\nbroken\n")
    with pytest.raises(RunError, match="chain_0.csv.*invalid numeric rows"):
        draws.load(rid)
