"""Tests for fit storage."""

import json

import pytest
from cmdstanpy.stanfit.metadata import InferenceMetadata

from stan_mcp import diagnostics, errors, fitstorage, runstore


def test_persist_run_copies_csvs_from_live(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    run_id = "pathfinder-0001"
    runstore.write_meta(run_id, {"method": "pathfinder", "status": "running"})
    live = runstore.live_dir(run_id, create=True)
    (live / "chain.csv").write_text("a,b\n1,2\n")

    class _FakePathfinder:  # no .runset, like CmdStanPathfinder
        is_resampled = True
        metadata = InferenceMetadata({"raw_header": "a,b"})

    fitstorage.persist_run(run_id, "pathfinder", _FakePathfinder())
    # CSVs copied from live/ despite no fit.runset
    assert runstore.csv_paths(run_id)
    # and method diagnostics captured
    assert runstore.read_meta(run_id).get("method_diagnostics") == {"is_resampled": True}


def test_persist_run_clears_live_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    run_id = "pathfinder-0002"
    runstore.write_meta(run_id, {"method": "pathfinder", "status": "running"})
    live = runstore.live_dir(run_id, create=True)
    (live / "chain.csv").write_text("a,b\n1,2\n")

    class _Fake:
        is_resampled = True
        metadata = InferenceMetadata({"raw_header": "a,b"})

    fitstorage.persist_run(run_id, "pathfinder", _Fake())
    assert runstore.csv_paths(run_id)  # durable copy kept
    assert not live.is_dir()  # duplicate dropped


def test_sidecar_records_cols_in_row_major_fill_order():
    """CmdStan's CSV header is column-major and numpy.reshape is row-major, so the
    sidecar must record the permutation. Recording the contiguous run instead
    transposes silently: L.2.1's values land in L[1,2].

    Uses a REAL cmdstanpy InferenceMetadata (not a hand-rolled fake) built from
    a single raw_header string, which is what production actually hands
    `_sidecar_from_fit`: `fit.column_names` munges names to array notation
    (`m[1,1]`, not `m.1.1`) and cannot satisfy this test at all -- a fake that
    sets `column_names` directly to dot-notation strings would exercise a
    branch that can never run for a real fit.
    """

    class _Fit:
        metadata = InferenceMetadata(
            {"raw_header": "lp__,divergent__,theta,m.1.1,m.2.1,m.1.2,m.2.2"}
        )

    sc = fitstorage._sidecar_from_fit(_Fit(), "sample", "sample-0001")
    assert sc["schema"] == diagnostics.draws.SCHEMA_VERSION
    assert sc["params"]["theta"] == {"dims": [], "cols": [2]}
    # row-major fill order for (2,2): m[1,1], m[1,2], m[2,1], m[2,2]
    #                          cols:      3        5        4        6
    assert sc["params"]["m"] == {"dims": [2, 2], "cols": [3, 5, 4, 6]}
    assert sc["sampler_stats"]["lp__"] == {"dims": [], "cols": [0]}
    assert sc["sampler_stats"]["divergent__"] == {"dims": [], "cols": [1]}
    assert "theta" not in sc["sampler_stats"]
    assert sc["draw_row_offset"] == 0


def test_sidecar_records_draw_row_offset_one_for_variational_only():
    """variational's first data row is cmdstanpy's own mean-field mean, not a
    draw (cmdstanpy stanfit/vb.py: draws_np[0] is the mean, draws_np[1:] the
    sample) -- checked directly against cmdstanpy's row-assembly source for
    every method that reaches persist_run, not assumed from method_diagnostics
    or other constructor-visible similarities.

    generate_quantities is excluded from this table on purpose: its offset is
    NOT a fixed per-method constant (see
    test_draw_row_offset_for_generate_quantities_depends_on_source_method) --
    it is 0 unless its source run (method_args.from_run_id) was itself
    variational, in which case cmdstanpy re-evaluates the generated-quantities
    block at the source's mean row too, landing that same extra row in the GQ
    run's own output CSV.
    """

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    # run_id is never dereferenced for these methods -- only generate_quantities
    # looks it up (to resolve its source run's method) -- so any string is fine.
    for method in ("sample", "pathfinder", "laplace"):
        assert fitstorage._sidecar_from_fit(_Fit(), method, "irrelevant")["draw_row_offset"] == 0
    assert fitstorage._sidecar_from_fit(_Fit(), "variational", "irrelevant")["draw_row_offset"] == 1


def test_draw_row_offset_for_generate_quantities_depends_on_source_method(tmp_path, monkeypatch):
    """generate_quantities re-evaluates the generated-quantities block once per
    row of its source run -- including a variational source's mean row
    (cmdstanpy gq.py: _draws_start computes draw1=1 for a CmdStanVB
    previous_fit and applies it to the GQ run's OWN assembled draws, not just
    a slice of the source's) -- so a GQ run built from_run_id from a
    variational run inherits offset 1 in its own CSV; from any other method,
    offset 0. Resolved via the source run's persisted method, not
    isinstance(fit.previous_fit, CmdStanVB)."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done"})
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    runstore.write_meta(
        "generate_quantities-0001",
        {
            "method": "generate_quantities",
            "status": "done",
            "method_args": {"from_run_id": "sample-0001"},
        },
    )
    runstore.write_meta(
        "generate_quantities-0002",
        {
            "method": "generate_quantities",
            "status": "done",
            "method_args": {"from_run_id": "variational-0001"},
        },
    )
    assert fitstorage._draw_row_offset("generate_quantities", "generate_quantities-0001") == 0
    assert fitstorage._draw_row_offset("generate_quantities", "generate_quantities-0002") == 1


def test_csv_columns_falls_back_to_column_names_without_cmdstan_config():
    """Belt-and-braces only: no real cmdstanpy fit reaches this branch (its
    InferenceMetadata always carries cmdstan_config['raw_header']), but a
    fake/mock that sets only `.column_names` must not crash `_sidecar_from_fit`
    outright. The source name comes back too, so a lookup miss can say which
    header it missed against."""

    class _Meta:
        pass  # no cmdstan_config

    class _Fit:
        metadata = _Meta()
        column_names = ("lp__", "theta")

    assert fitstorage._csv_columns(_Fit()) == (["lp__", "theta"], "column_names")


def test_csv_columns_reports_raw_header_as_its_source():
    """The paired positive: `raw_header` must not be mislabelled as the fallback,
    or the error message above would send a reader to the wrong cause."""

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    assert fitstorage._csv_columns(_Fit()) == (["lp__", "theta"], "raw_header")


def test_sidecar_from_fit_names_the_element_it_cannot_place(tmp_path, monkeypatch):
    """The `column_names` fallback is munged (`m[1,1]`, not `m.1.1`), so a
    non-scalar variable cannot be placed from it at all.

    `pixi.toml` pins cmdstanpy loosely, so one upstream rename of
    cmdstan_config['raw_header'] drops every real fit onto this branch. It used
    to raise a bare `KeyError: 'm[1,1]'`, which `persist_run` re-raised — landing
    a run whose SAMPLING SUCCEEDED at status="failed" with a cryptic message. The
    error must instead name the variable, the element, and the header source.
    """
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})

    class _Meta:
        # no cmdstan_config -> the column_names fallback
        stan_vars = InferenceMetadata({"raw_header": "lp__,m.1.1,m.2.1,m.1.2,m.2.2"}).stan_vars

    class _Fit:
        metadata = _Meta()
        column_names = ("lp__", "m[1,1]", "m[2,1]", "m[1,2]", "m[2,2]")

    with pytest.raises(errors.RunError) as exc:
        fitstorage._sidecar_from_fit(_Fit(), "sample", rid)
    msg = str(exc.value)
    assert "m.1.1" in msg  # the element it could not place
    assert "'m'" in msg and "[2, 2]" in msg  # the variable and its dims
    assert "column_names" in msg  # which header source was used
    assert "Sampling succeeded" in msg  # not a sampling failure


def test_persist_run_writes_a_sidecar_and_no_netcdf(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    (live / "chain_1.csv").write_text("# c\nlp__,divergent__,theta\n-1,0,0.5\n")

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,divergent__,theta"})

    fitstorage.persist_run(rid, "sample", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["files"] == ["chain_1.csv"]
    assert sc["params"]["theta"]["cols"] == [2]
    assert not runstore.idata_path(rid).exists()
    # and the typed layer can read what was just written
    d = diagnostics.draws.load(rid)
    assert d.params["theta"].tolist() == [[0.5]]


def test_persist_run_excludes_laplace_mode_csv_from_sidecar_files(tmp_path, monkeypatch):
    """laplace with no explicit `mode=` runs its own internal optimize into the
    SAME output dir, dropping an `-opt-...` CSV of a different shape (one row,
    fewer columns) beside the draws CSV. Recording it in `files` makes
    `draws.load` stack arrays of different widths and die with a bare numpy
    ValueError instead of a RunError naming the run."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "laplace-0001"
    runstore.write_meta(rid, {"method": "laplace", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    (live / "model-1.csv").write_text("# c\nlog_p__,log_q__,theta\n-1,-2,0.5\n")
    (live / "model-opt-1.csv").write_text("# c\nlp__,theta\n-1,0.5\n")

    class _Fit:
        # no .runset -- matches the real CmdStanLaplace, exercising the
        # numeric/name-sort fallback in _ordered_draws_files
        metadata = InferenceMetadata({"raw_header": "log_p__,log_q__,theta"})

    fitstorage.persist_run(rid, "laplace", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["files"] == ["model-1.csv"]
    d = diagnostics.draws.load(rid)
    assert d.params["theta"].tolist() == [[0.5]]


def test_persist_run_orders_sidecar_files_by_chain_not_filename(tmp_path, monkeypatch):
    """A filename sort puts chain 10 before chain 2 once there are 10+ chains,
    silently permuting per-chain results. Exercises the fallback path (no
    `.runset`), which must sort numerically on its own."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    for n in (1, 2, 10):
        (live / f"model_{n}.csv").write_text(f"# c\nlp__,theta\n-1,{n / 10}\n")

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    fitstorage.persist_run(rid, "sample", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["files"] == ["model_1.csv", "model_2.csv", "model_10.csv"]


def test_persist_run_prefers_fit_runset_order_for_sidecar_files(tmp_path, monkeypatch):
    """fit.runset.csv_files is cmdstanpy's own per-chain-ordered list and must be
    used verbatim, not re-sorted -- guards against reintroducing a filename
    sort that happens to agree with chain order only below 10 chains."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    for n in (1, 2, 10):
        (live / f"model_{n}.csv").write_text(f"# c\nlp__,theta\n-1,{n / 10}\n")

    class _Runset:
        csv_files = [
            str(live / "model_2.csv"),
            str(live / "model_10.csv"),
            str(live / "model_1.csv"),
        ]

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})
        runset = _Runset()

    fitstorage.persist_run(rid, "sample", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["files"] == ["model_2.csv", "model_10.csv", "model_1.csv"]


def test_persist_run_fails_the_run_when_sidecar_write_fails(tmp_path, monkeypatch):
    """The sidecar is now the ONLY record of a run's dims/columns -- unlike the
    arviz conversion it replaces, whose failure still left CSV-derived
    fallbacks readable for VI/pathfinder/laplace. A run whose sidecar failed to
    write must not settle at status="done" with only a persist_warning field
    to notice by, so persist_run must raise -- the same failure path sample.py
    already uses for e.g. a sampler crash -- rather than swallow the error."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    (live / "model_1.csv").write_text("# c\nlp__,theta\n-1,0.5\n")

    def boom(fit, method, run_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(fitstorage, "_sidecar_from_fit", boom)

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    with pytest.raises(errors.RunError, match="draws schema not written"):
        fitstorage.persist_run(rid, "sample", _Fit())

    # the CSVs must survive the failure -- they're still worth keeping
    assert runstore.csv_paths(rid) == [str(runstore.run_dir(rid) / "csv" / "model_1.csv")]
    meta = runstore.read_meta(rid)
    # exact suffix, not `in`: a TypeError from a signature mismatch also
    # contains the function's own name ("boom() takes ... arguments") and
    # would satisfy an `in` check just as well, which is exactly how this
    # test went vacuous once already (a prior round's boom(fit) took a
    # TypeError for the intended RuntimeError without anyone noticing).
    assert meta["persist_warning"] == "draws schema not written: boom"


def test_persist_run_fails_when_no_draws_csvs_are_persisted(tmp_path, monkeypatch):
    """If CSV persistence produced nothing at all (e.g. its own csv_warning
    already fired, or the run's live/ never had any files), the sidecar must
    not write "successfully" with files: [] -- draws.load would hit min([])
    on an empty CSV list and die with a bare ValueError, not a RunError
    naming the run."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    # no live/ dir at all -> the CSV-persist step copies nothing, csv/ stays empty

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    with pytest.raises(errors.RunError, match="no persisted draws CSVs"):
        fitstorage.persist_run(rid, "sample", _Fit())

    meta = runstore.read_meta(rid)
    assert "persist_warning" in meta


def test_persist_run_writes_variational_sidecar_excluding_mean_row(tmp_path, monkeypatch):
    """End-to-end through persist_run/draws.load: variational's first CSV data
    row is cmdstanpy's own mean-field mean (stanfit/vb.py: draws_np[0]), not a
    draw. Verifies draw_row_offset is both written by persist_run AND honoured
    by draws.load, rather than each layer's unit test alone."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "variational-0001"
    runstore.write_meta(rid, {"method": "variational", "status": "done"})
    live = runstore.live_dir(rid, create=True)
    # row 0 is cmdstanpy's variational mean (999.0 sentinel); rows 1.. are draws
    (live / "model.csv").write_text("# c\nlp__,theta\n0,999.0\n-1,0.5\n-1,0.7\n")

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "lp__,theta"})

    fitstorage.persist_run(rid, "variational", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["draw_row_offset"] == 1
    d = diagnostics.draws.load(rid)
    assert d.n_draw == 2
    assert d.params["theta"].tolist() == [[0.5, 0.7]]


def test_persist_run_gq_from_variational_excludes_leading_row(tmp_path, monkeypatch):
    """A generate_quantities run whose from_run_id names a variational run
    inherits that run's leading mean-evaluation row in its OWN output CSV
    (cmdstanpy gq.py's _draws_start/_assemble_generated_quantities), so its
    sidecar must record draw_row_offset=1 too -- not the 0 that would be
    right for a GQ run built from any other method."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    rid = "generate_quantities-0001"
    runstore.write_meta(
        rid,
        {
            "method": "generate_quantities",
            "status": "done",
            "method_args": {"from_run_id": "variational-0001"},
        },
    )
    live = runstore.live_dir(rid, create=True)
    # row 0 is GQ evaluated at the variational source's mean (999.0 sentinel)
    (live / "model.csv").write_text("# c\nlog_lik\n999.0\n0.1\n0.2\n")

    class _Fit:
        metadata = InferenceMetadata({"raw_header": "log_lik"})

    fitstorage.persist_run(rid, "generate_quantities", _Fit())
    sc = json.loads(diagnostics.draws.schema_path(rid).read_text())
    assert sc["draw_row_offset"] == 1
    d = diagnostics.draws.load(rid)
    assert d.n_draw == 2
    assert d.params["log_lik"].tolist() == [[0.1, 0.2]]
