"""Tests for geometry."""

import json

import numpy as np
import pytest

from stan_mcp import diagnostics, errors, geometry, runstore


def test_constrained_matrix_maps_bridgestan_names_via_recorded_dims(tmp_path, monkeypatch):
    """BridgeStan spells elements m.2.1 (1-based, column-major). ravel_multi_index
    still flattens it, but against the sidecar's recorded dims rather than arviz's
    dim ordering.

    `m` is non-square (2x3, matching the live SHAPED model): a square fixture
    catches a transposed reshape (every value differs) but not a stride error
    that only shows up when the two dims are unequal.
    """
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "a.csv").write_text(
        "# c\nlp__,s,m.1.1,m.2.1,m.1.2,m.2.2,m.1.3,m.2.3\n-1,9.0,11,21,12,22,13,23\n"
    )
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": ["a.csv"],
                "params": {
                    "s": {"dims": [], "cols": [1]},
                    "m": {"dims": [2, 3], "cols": [2, 4, 6, 3, 5, 7]},
                },
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    d = diagnostics.draws.load(rid)
    # BridgeStan's own order for this model, column-major
    bs_names = ["m.1.1", "m.2.1", "m.1.2", "m.2.2", "m.1.3", "m.2.3", "s"]
    X = geometry._constrained_matrix(d, bs_names)
    assert X.shape == (1, 7)
    assert X[0].tolist() == [11.0, 21.0, 12.0, 22.0, 13.0, 23.0, 9.0]


def _stub_stansummary(monkeypatch, rid, rhat=1.0):
    """Stub stansummary for a run, keyed off that run's own sidecar names.

    `run_geometry` reads max R-hat through `_convergence`, which shells out to the
    real `stansummary` binary. Without this the geometry tests are the only ones
    in this file that reach a live toolchain: on a box without CmdStan they fail
    outright, and on a box with one they silently run the binary against a
    two-line fixture CSV, making their behaviour depend on the machine.
    """
    sc = diagnostics.draws.schema(rid)
    rows = {
        cmdstan_name: {"R_hat": rhat, "ESS_bulk": 1000.0, "ESS_tail": 1000.0}
        for name, spec in sc["params"].items()
        for _, cmdstan_name in diagnostics.draws.element_labels(name, tuple(spec["dims"]))
    }
    monkeypatch.setattr(diagnostics.stansummary, "summarize", lambda paths, **kw: rows)


def _geom_run(monkeypatch, tmp_path, cols: dict, n_chain=4, n_draw=500):
    """A run whose unconstrained draws are planted. `cols` maps name -> (draws,)
    array of length n_chain*n_draw. The BridgeStan round-trip is stubbed: what
    needs testing is the statistics, and the unconstraining needs a toolchain.

    `run_geometry` only uses the loaded `Draws` to pass into `_unconstrained_draws`,
    which is stubbed below regardless of what it's given -- so the on-disk typed
    run just needs to exist and load, not carry the planted values itself.
    """
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    names = list(cols)
    u3 = np.stack([cols[n] for n in names], axis=-1).reshape(n_chain, n_draw, len(names))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "chain0.csv").write_text("# c\nlp__,dummy\n-1,0\n")
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": ["chain0.csv"],
                "params": {"dummy": {"dims": [], "cols": [1]}},
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    monkeypatch.setattr(
        geometry,
        "_unconstrained_draws",
        lambda rid, d, **kwargs: (u3, names, {"scale": "unconstrained"}),
    )
    _stub_stansummary(monkeypatch, rid)
    return rid


def _correlated(rho, n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rho * a + np.sqrt(max(1 - rho**2, 1e-12)) * rng.normal(size=n)
    return a, b


def test_geometry_catches_negative_correlation(tmp_path, monkeypatch):
    """Fisher z carries the sign of r. Comparing the signed value against a
    positive threshold dropped every negatively correlated pair — which is the
    commonest form collinearity takes, two coefficients trading off."""
    n = 2000
    b1, b2 = _correlated(-0.99, n)
    rid = _geom_run(monkeypatch, tmp_path, {"b1": b1, "b2": b2, "c": _correlated(0, n, 5)[0]})
    g = geometry.run_geometry(rid)
    assert g["verdict"] == "warn"
    got = {(p["a"], p["b"]) for p in g["pairs"]}
    assert ("b1", "b2") in got or ("b2", "b1") in got
    assert g["pairs"][0]["r"] < -0.9  # sign preserved in the report
    assert g["pairs"][0]["z"] > 0  # evidence is a magnitude


def test_geometry_catches_positive_correlation(tmp_path, monkeypatch):
    n = 2000
    b1, b2 = _correlated(0.97, n)
    rid = _geom_run(monkeypatch, tmp_path, {"b1": b1, "b2": b2})
    g = geometry.run_geometry(rid)
    assert g["pairs"] and g["pairs"][0]["r"] > 0.9


def test_geometry_clean_on_independent_params(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    cols = {f"b{i}": rng.normal(size=2000) for i in range(4)}
    rid = _geom_run(monkeypatch, tmp_path, cols)
    g = geometry.run_geometry(rid)
    assert g["verdict"] == "pass" and g["pairs"] == []
    assert g["max_abs_r_cross_block"] < 0.3  # reported even when nothing fires


def test_geometry_collapses_within_block_pairs(tmp_path, monkeypatch):
    """theta[1..20] is 190 of the 210 pairs; itemizing them would crowd out the
    one cross-block pair that names the actual problem."""
    rng = np.random.default_rng(2)
    shared = rng.normal(size=2000)
    cols = {f"theta[{i}]": 0.9 * shared + 0.44 * rng.normal(size=2000) for i in range(20)}
    cols["sigma"] = rng.normal(size=2000)
    rid = _geom_run(monkeypatch, tmp_path, cols)
    g = geometry.run_geometry(rid)
    blocks = {w["block"]: w for w in g["within_block"]}
    assert blocks["theta"]["n_pairs"] == 190
    assert blocks["theta"]["max_abs_r"] > 0.5
    assert all(p["a"] != p["b"] for p in g["pairs"])


def test_geometry_eigen_suppressed_when_dims_rival_ess(tmp_path, monkeypatch):
    """A sample correlation matrix is ill-conditioned by construction when p is
    not small next to n_eff, so the condition number would be pure artifact."""
    rng = np.random.default_rng(3)
    cols = {f"b{i}": rng.normal(size=200) for i in range(80)}
    rid = _geom_run(monkeypatch, tmp_path, cols, n_chain=2, n_draw=100)
    g = geometry.run_geometry(rid)
    assert "eigen" not in g and "suppressed" in g["eigen_note"]


def test_geometry_reports_the_noise_condition_number(tmp_path, monkeypatch):
    """A condition number is only readable next to what noise alone would give:
    at p/n_eff = 0.25 independent parameters already produce about 8."""
    rng = np.random.default_rng(11)
    cols = {f"b{i}": rng.normal(size=2000) for i in range(6)}
    rid = _geom_run(monkeypatch, tmp_path, cols)
    eig = geometry.run_geometry(rid)["eigen"]
    assert eig["noise_condition_number"] > 1.0
    # 6 dims against ~2000 effective draws is a negligible ratio, so both the
    # observed and the structural figure must be small.
    assert eig["condition_number"] < 10 and eig["noise_condition_number"] < 2


def test_geometry_eigen_finds_three_way_degeneracy(tmp_path, monkeypatch):
    """x3 = x1 + x2 leaves every pair moderate while the posterior is near
    singular along x1 + x2 - x3: the blind spot pairwise cannot see."""
    rng = np.random.default_rng(4)
    n = 2000  # must equal n_chain * n_draw
    b1, b2 = rng.normal(size=n), rng.normal(size=n)
    b3 = b1 + b2 + 0.05 * rng.normal(size=n)
    rid = _geom_run(monkeypatch, tmp_path, {"b1": b1, "b2": b2, "b3": b3})
    g = geometry.run_geometry(rid)
    assert g["eigen"]["condition_number"] > 100
    assert {d["param"] for d in g["eigen"]["direction"]} == {"b1", "b2", "b3"}


def test_geometry_thresholds_are_overridable(tmp_path, monkeypatch):
    n = 2000
    b1, b2 = _correlated(0.2, n)
    rid = _geom_run(monkeypatch, tmp_path, {"b1": b1, "b2": b2})
    assert geometry.run_geometry(rid)["pairs"] == []  # below default bar
    loose = geometry.run_geometry(rid, min_abs_r=0, min_z=0)
    assert loose["pairs"]  # 0 = unlimited, per the codebase convention


def test_geometry_refuses_above_max_dims(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    cols = {f"b{i}": rng.normal(size=400) for i in range(12)}
    rid = _geom_run(monkeypatch, tmp_path, cols, n_chain=2, n_draw=200)
    with pytest.raises(errors.ValidationError):
        geometry.run_geometry(rid, max_dims=5)


def test_geometry_params_filter(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    cols = {"a": rng.normal(size=800), "b[0]": rng.normal(size=800), "b[1]": rng.normal(size=800)}
    rid = _geom_run(monkeypatch, tmp_path, cols, n_chain=2, n_draw=400)
    assert geometry.run_geometry(rid, params=["b"])["n_dims"] == 2  # block name matches
    with pytest.raises(errors.ValidationError):
        geometry.run_geometry(rid, params=["nope"])


def test_geometry_rejects_non_sample_methods(tmp_path, monkeypatch):
    """The method gate must fire before `draws.load` ever runs: a run with no
    draws_schema.json also raises RunError, on a different message, so this has
    to pin the gate's own text or it would pass even with the gate deleted."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("variational-0001", {"method": "variational", "status": "done"})
    with pytest.raises(errors.RunError, match="geometry needs MCMC draws with chain structure"):
        geometry.run_geometry("variational-0001")


def test_geometry_reports_preparing_not_an_error(tmp_path, monkeypatch):
    """The first call on a freshly sampled model has no .so; the build happens off
    the caller's thread and the call must say so rather than fail."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "chain0.csv").write_text("# c\nlp__,x\n-1,0\n")
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": ["chain0.csv"],
                "params": {"x": {"dims": [], "cols": [1]}},
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )

    def _pending(rid, d, **kwargs):
        raise geometry._GeometryPending("building")

    monkeypatch.setattr(geometry, "_unconstrained_draws", _pending)
    assert geometry.run_geometry(rid)["status"] == "preparing"


def test_geometry_falls_back_when_draws_cannot_be_unconstrained(tmp_path, monkeypatch):
    """Stan's *_free functions validate the constraint they invert, so a draw read
    back at limited CSV precision can be rejected: simplex_free's tolerance is 1e-8
    and a simplex written at 6 significant figures only sums to 1 to about that.

    One rejected draw used to abort the whole call, so get_run_geometry hard-failed
    on every mixture or LKJ model. It must degrade to the constrained scale and say
    so instead -- reported, not swallowed, and not fatal.
    """
    from stan_mcp import density

    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = "sample-0001"
    n_chain, n_draw = 2, 50
    rng = np.random.default_rng(4)
    theta = rng.random((n_chain, n_draw, 3))
    s = rng.random((n_chain, n_draw))

    runstore.write_meta(rid, {"method": "sample", "status": "done"})
    runstore.persist_inputs(rid, "parameters { simplex[3] theta; real s; } model { }", {})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    files = []
    for chain in range(n_chain):
        rows = "\n".join(
            f"-1,{theta[chain, i, 0]},{theta[chain, i, 1]},{theta[chain, i, 2]},{s[chain, i]}"
            for i in range(n_draw)
        )
        name = f"chain{chain}.csv"
        (csvdir / name).write_text(f"# c\nlp__,theta.1,theta.2,theta.3,s\n{rows}\n")
        files.append(name)
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": files,
                "params": {
                    "theta": {"dims": [3], "cols": [1, 2, 3]},
                    "s": {"dims": [], "cols": [4]},
                },
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )

    class _Rejecting:
        def param_names(self):
            return ["theta.1", "theta.2", "theta.3", "s"]

        def param_unc_names(self):
            return ["theta.1", "theta.2", "s"]

        def param_unc_num(self):
            return 3

        def param_unconstrain(self, _row):
            raise RuntimeError(
                "param_unconstrain() failed with exception: Exception: "
                "stan::math::simplex_free: Simplex variable is not a valid simplex. "
                "sum(Simplex variable) = 0.99999999, but should be 1"
            )

    from stan_mcp import compilation as _compile

    monkeypatch.setattr(_compile, "request_so", lambda *a, **k: {"status": "ready"})
    monkeypatch.setattr(density, "get_model", lambda *a, **k: _Rejecting())
    _stub_stansummary(monkeypatch, rid)

    g = geometry.run_geometry(rid)
    assert "error" not in g, g
    assert g["scale"] == "constrained"
    # the actual Stan message must survive into the reason, not be flattened away
    assert "simplex_free" in g["scale_reason"]
    assert "Re-run" in g["scale_reason"]
    # and it still did the analysis, on all four constrained components
    assert g["n_dims"] == 4


def test_geometry_flags_provisional_when_rhat_is_high(tmp_path, monkeypatch):
    """With chains disagreeing there is not yet one posterior whose geometry this
    describes, so the result must say so. Pins that the R-hat comes from
    `_convergence` (stubbed here) rather than being invented or dropped."""
    rng = np.random.default_rng(21)
    cols = {"b1": rng.normal(size=2000), "b2": rng.normal(size=2000)}
    rid = _geom_run(monkeypatch, tmp_path, cols)
    _stub_stansummary(monkeypatch, rid, rhat=1.5)
    g = geometry.run_geometry(rid)
    assert "1.5" in g["provisional"] and "R-hat" in g["provisional"]


def test_geometry_reads_rhat_without_reentering_run_diagnostics(tmp_path, monkeypatch):
    """run_geometry used to call the whole of run_diagnostics for one number,
    which spawned a second stansummary subprocess and re-parsed every CSV for
    sampler stats it then discarded. Fail the test if that returns."""
    rng = np.random.default_rng(22)
    cols = {"b1": rng.normal(size=2000), "b2": rng.normal(size=2000)}
    rid = _geom_run(monkeypatch, tmp_path, cols)

    def _boom(*a, **kw):
        raise AssertionError("run_geometry must not re-enter run_diagnostics")

    monkeypatch.setattr(diagnostics, "run_diagnostics", _boom)
    g = geometry.run_geometry(rid)
    assert g["verdict"] in {"pass", "warn"}
    assert "provisional" not in g  # the stubbed R-hat is healthy
