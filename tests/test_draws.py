"""Tests for the typed draws layer.

Fixtures write a sidecar directly rather than a realistic CmdStan CSV: the sidecar
IS the contract, and cmdstanpy's own reader (rejected as the read engine) is the
only thing that needs CmdStan's config header.
"""

import json

import pytest

import draws
import errors
import runstore


def _run(tmp_path, monkeypatch, *, files, params, stats, rows_per_chain, draw_row_offset=None):
    """A run directory with hand-written CSVs and a matching sidecar.

    `rows_per_chain` maps a filename to a list of rows (each a list of floats).
    `draw_row_offset` defaults to None, meaning OMITTED from the sidecar (not
    written as 0) -- most fixtures here predate that field, so leaving it out
    by default keeps them exercising `load()`'s `sc.get(..., 0)` fallback
    rather than a value nobody asked for.
    """
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    ncol = max(len(r) for rows in rows_per_chain.values() for r in rows)
    header = ",".join(f"c{i}" for i in range(ncol))
    for name, rows in rows_per_chain.items():
        body = "\n".join(",".join(str(v) for v in r) for r in rows)
        (csvdir / name).write_text(f"# comment\n{header}\n{body}\n")
    sidecar = {
        "schema": draws.SCHEMA_VERSION,
        "files": files,
        "params": params,
        "sampler_stats": stats,
    }
    if draw_row_offset is not None:
        sidecar["draw_row_offset"] = draw_row_offset
    draws.schema_path(rid).write_text(json.dumps(sidecar))
    return rid


def test_scalar_and_vector_shapes(tmp_path, monkeypatch):
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["a.csv", "b.csv"],
        params={"theta": {"dims": [], "cols": [1]}, "z": {"dims": [2], "cols": [2, 3]}},
        stats={"lp__": {"dims": [], "cols": [0]}},
        rows_per_chain={
            "a.csv": [[-1.0, 0.1, 10.0, 20.0], [-2.0, 0.2, 11.0, 21.0]],
            "b.csv": [[-3.0, 0.3, 12.0, 22.0], [-4.0, 0.4, 13.0, 23.0]],
        },
    )
    d = draws.load(rid)
    assert d.n_chain == 2 and d.n_draw == 2 and d.truncated_to is None
    assert d.params["theta"].shape == (2, 2)
    assert d.params["z"].shape == (2, 2, 2)
    assert d.sampler_stats["lp__"].shape == (2, 2)
    # chain order follows `files`, not a filename sort of the directory
    assert d.params["theta"][0].tolist() == [0.1, 0.2]
    assert d.params["theta"][1].tolist() == [0.3, 0.4]
    assert d.params["z"][0, 0].tolist() == [10.0, 20.0]


def test_row_major_cols_place_matrix_elements_correctly(tmp_path, monkeypatch):
    """The sidecar records cols in row-major fill order, so a plain reshape is
    correct. CmdStan's header is column-major, so recording the contiguous run
    instead would transpose silently -- the failure this design exists to remove."""
    # one draw; columns 1..4 hold, in CmdStan's column-major order:
    # m.1.1=11, m.2.1=21, m.1.2=12, m.2.2=22
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["a.csv"],
        # row-major fill order for (2,2): m[1,1], m[1,2], m[2,1], m[2,2]
        params={"m": {"dims": [2, 2], "cols": [1, 3, 2, 4]}},
        stats={},
        rows_per_chain={"a.csv": [[0.0, 11.0, 21.0, 12.0, 22.0]]},
    )
    m = draws.load(rid).params["m"]
    assert m.shape == (1, 1, 2, 2)
    assert m[0, 0].tolist() == [[11.0, 12.0], [21.0, 22.0]]


def test_draw_row_offset_skips_variational_mean_row(tmp_path, monkeypatch):
    """variational's first data row is cmdstanpy's own mean-field mean
    (stanfit/vb.py: draws_np[0]), not a draw. draw_row_offset records how many
    leading rows to skip so it is never counted as one."""
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["a.csv"],
        params={"theta": {"dims": [], "cols": [0]}},
        stats={},
        # row 0 is the mean (999.0, a sentinel far from the real draws below)
        rows_per_chain={"a.csv": [[999.0], [1.0], [2.0], [3.0]]},
        draw_row_offset=1,
    )
    d = draws.load(rid)
    assert d.n_draw == 3
    assert d.params["theta"].tolist() == [[1.0, 2.0, 3.0]]


def test_draw_row_offset_absent_from_sidecar_reads_every_row(tmp_path, monkeypatch):
    """load() must not require draw_row_offset to be present -- a real,
    version-2 sidecar always has it (diagnostics._sidecar_from_fit always sets
    it), but load()'s own `.get(..., 0)` default is what's under test here, so
    the field is genuinely OMITTED (not passed as an explicit 0)."""
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["a.csv"],
        params={"theta": {"dims": [], "cols": [0]}},
        stats={},
        rows_per_chain={"a.csv": [[1.0], [2.0]]},
        # draw_row_offset omitted entirely -- see _run's docstring
    )
    assert "draw_row_offset" not in json.loads(draws.schema_path(rid).read_text())
    assert draws.load(rid).params["theta"].tolist() == [[1.0, 2.0]]


def test_ragged_chains_truncate_to_shortest_and_say_so(tmp_path, monkeypatch):
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["a.csv", "b.csv"],
        params={"theta": {"dims": [], "cols": [0]}},
        stats={},
        rows_per_chain={"a.csv": [[1.0], [2.0], [3.0]], "b.csv": [[4.0], [5.0]]},
    )
    d = draws.load(rid)
    assert d.n_draw == 2 and d.truncated_to == 2
    assert d.params["theta"].shape == (2, 2)


def test_missing_sidecar_raises_and_says_to_rerun(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done"})
    with pytest.raises(errors.RunError, match="re-run"):
        draws.load("sample-0001")


def test_schema_version_mismatch_raises(tmp_path, monkeypatch):
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["a.csv"],
        params={"theta": {"dims": [], "cols": [0]}},
        stats={},
        rows_per_chain={"a.csv": [[1.0]]},
    )
    p = draws.schema_path(rid)
    p.write_text(json.dumps({**json.loads(p.read_text()), "schema": 99}))
    with pytest.raises(errors.RunError, match="99"):
        draws.load(rid)


def test_stale_sidecar_referencing_a_missing_column_raises(tmp_path, monkeypatch):
    """A column index past the CSV's width is the one way a stale sidecar could
    mislabel data, so it must fail rather than mis-slice."""
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["a.csv"],
        params={"theta": {"dims": [], "cols": [7]}},
        stats={},
        rows_per_chain={"a.csv": [[1.0, 2.0]]},
    )
    with pytest.raises(errors.RunError, match="stale"):
        draws.load(rid)


def test_malformed_sidecar_names_the_missing_key_and_the_run(tmp_path, monkeypatch):
    """`schema()` promises a VALIDATED sidecar. Checking only the version left a
    sidecar missing `files`/`params`/`sampler_stats`, or a spec missing
    `cols`/`dims`, to raise a bare KeyError from inside `load` — naming a dict key
    rather than the run, and not a RunError at all."""
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["a.csv"],
        params={"theta": {"dims": [], "cols": [0]}},
        stats={},
        rows_per_chain={"a.csv": [[1.0]]},
    )
    good = json.loads(draws.schema_path(rid).read_text())

    for key in ("files", "params", "sampler_stats"):
        draws.schema_path(rid).write_text(json.dumps({k: v for k, v in good.items() if k != key}))
        with pytest.raises(errors.RunError, match=rf"no '{key}'"):
            draws.load(rid)

    # a spec missing `cols` is the same class of gap, one level down
    draws.schema_path(rid).write_text(json.dumps({**good, "params": {"theta": {"dims": []}}}))
    with pytest.raises(errors.RunError, match="params.theta is missing cols"):
        draws.load(rid)

    # every message must name the run, not just the key
    draws.schema_path(rid).write_text(json.dumps({k: v for k, v in good.items() if k != "files"}))
    with pytest.raises(errors.RunError, match=rid):
        draws.csv_files(rid)


def test_matrix_error_names_the_run(tmp_path, monkeypatch):
    """`matrix` was the one failure in this module whose message identified
    nothing — the function has no run_id of its own, so `Draws` carries it."""
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["a.csv"],
        params={},
        stats={"lp__": {"dims": [], "cols": [0]}},
        rows_per_chain={"a.csv": [[1.0]]},
    )
    d = draws.load(rid)
    assert d.run_id == rid
    with pytest.raises(errors.RunError, match=rid):
        draws.matrix(d)


def test_element_labels_are_stan_syntax_and_paired_with_cmdstan_names():
    assert list(draws.element_labels("theta", ())) == [("theta", "theta")]
    assert list(draws.element_labels("z", (2,))) == [("z[1]", "z.1"), ("z[2]", "z.2")]
    assert list(draws.element_labels("m", (2, 2))) == [
        ("m[1,1]", "m.1.1"),
        ("m[1,2]", "m.1.2"),
        ("m[2,1]", "m.2.1"),
        ("m[2,2]", "m.2.2"),
    ]


def test_elements_walks_row_major_matching_element_labels(tmp_path, monkeypatch):
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["a.csv"],
        params={"m": {"dims": [2, 2], "cols": [1, 3, 2, 4]}},
        stats={},
        rows_per_chain={"a.csv": [[0.0, 11.0, 21.0, 12.0, 22.0]]},
    )
    arr = draws.load(rid).params["m"]
    got = [(label, vals[0, 0]) for label, vals in draws.elements("m", arr)]
    assert got == [("m[1,1]", 11.0), ("m[1,2]", 12.0), ("m[2,1]", 21.0), ("m[2,2]", 22.0)]
    # the label order must agree with element_labels, which the summary relies on
    assert [g[0] for g in got] == [s for s, _ in draws.element_labels("m", (2, 2))]


def test_matrix_returns_draws_by_element_with_labels(tmp_path, monkeypatch):
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["a.csv", "b.csv"],
        params={"theta": {"dims": [], "cols": [1]}, "z": {"dims": [2], "cols": [2, 3]}},
        stats={"lp__": {"dims": [], "cols": [0]}},
        rows_per_chain={
            "a.csv": [[-1.0, 0.1, 10.0, 20.0], [-2.0, 0.2, 11.0, 21.0]],
            "b.csv": [[-3.0, 0.3, 12.0, 22.0], [-4.0, 0.4, 13.0, 23.0]],
        },
    )
    X, names = draws.matrix(draws.load(rid))
    assert X.shape == (4, 3)  # 2 chains x 2 draws, 3 elements
    assert names == ["theta", "z[1]", "z[2]"]  # sampler stats excluded


def test_csv_files_come_from_the_sidecar_not_a_glob(tmp_path, monkeypatch):
    """laplace writes a mode CSV alongside its draws CSV. Selecting by record
    rather than by filename substring is what keeps it out."""
    rid = _run(
        tmp_path,
        monkeypatch,
        files=["b.csv"],
        params={"theta": {"dims": [], "cols": [0]}},
        stats={},
        rows_per_chain={"a.csv": [[9.0]], "b.csv": [[1.0]]},
    )
    assert [p.rsplit("/", 1)[-1] for p in draws.csv_files(rid)] == ["b.csv"]
    assert draws.load(rid).params["theta"].tolist() == [[1.0]]
