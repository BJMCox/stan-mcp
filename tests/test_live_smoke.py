"""Live integration tests. Require the C++ toolchain; gated by STAN_MCP_LIVE=1."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STAN_MCP_LIVE") != "1", reason="set STAN_MCP_LIVE=1 to run"
)

BERNOULLI = """
data { int<lower=0> N; array[N] int<lower=0,upper=1> y; }
parameters { real<lower=0,upper=1> theta; }
model { theta ~ beta(1,1); y ~ bernoulli(theta); }
"""
DATA = {"N": 5, "y": [0, 1, 0, 1, 1]}

BERNOULLI_GQ = BERNOULLI + "\ngenerated quantities { int y_sim = bernoulli_rng(theta); }\n"

BERNOULLI_LL = BERNOULLI + (
    "\ngenerated quantities {\n"
    "  vector[N] log_lik;\n"
    "  for (n in 1:N) log_lik[n] = bernoulli_lpmf(y[n] | theta);\n"
    "}\n"
)


def _sample(mgr, stan, seed):
    import asyncio

    started = mgr.start_run(
        stan,
        DATA,
        method="sample",
        seed=seed,
        method_args={"chains": 2, "iter_warmup": 300, "iter_sampling": 300},
    )
    result = asyncio.run(mgr.await_run(started["run_id"], timeout=120))
    assert result["status"] == "done"
    return started["run_id"]


def test_live_full_diagnostics_verdict(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import diagnostics
    import sample

    run_id = _sample(sample.RunManager(), BERNOULLI, 1)
    diag = diagnostics.run_diagnostics(run_id)
    assert diag["verdict"] in {"pass", "warn"}
    assert "max_rhat" in diag and "min_ess_bulk" in diag
    assert diag["divergences"] == 0


def test_live_run_loo_and_compare(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import diagnostics
    import sample

    mgr = sample.RunManager()
    a = _sample(mgr, BERNOULLI_LL, 1)
    b = _sample(mgr, BERNOULLI_LL, 2)
    loo = diagnostics.run_loo(a)
    assert "elpd_loo" in loo and loo["n_high_pareto_k"] == 0
    cmp = diagnostics.compare_runs([a, b])
    assert {r["run_id"] for r in cmp["comparison"]} == {a, b}


def test_live_progress_and_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import runstore
    import sample

    mgr = sample.RunManager()
    run_id = _sample(mgr, BERNOULLI, 1)
    prog = mgr.get_run_progress(run_id)
    assert prog["status"] == "done" and prog["draws_written"] == 600
    got = runstore.load_inputs(run_id)
    assert "bernoulli" in got["source"] and got["data"] == DATA


def test_live_compile_pedantic_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import compile as c

    out = c.compile_model("parameters { real x; } model { }", warn_pedantic=True)
    assert "error" not in out
    assert any("was not used" in w for w in out.get("warnings", []))


def test_live_pedantic_does_not_change_model_id(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import compile as c

    plain = c.compile_model(BERNOULLI)
    pedantic = c.compile_model(BERNOULLI, warn_pedantic=True)
    assert plain["model_id"] == pedantic["model_id"]
    # and that canonical id matches what a run would compute
    assert pedantic["model_id"] == c.model_id(c.resolve_source(BERNOULLI), None)


def test_live_method_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import asyncio

    import diagnostics
    import runstore
    import sample

    mgr = sample.RunManager()
    expect = {"variational": "eta", "pathfinder": "is_resampled", "laplace": "mode_lp"}
    for method, key in expect.items():
        started = mgr.start_run(BERNOULLI, DATA, method=method, seed=7)
        result = asyncio.run(mgr.await_run(started["run_id"], timeout=120))
        assert result["status"] == "done", f"{method} did not finish"
        # run_diagnostics must not raise despite no inference.nc, and carry the scalar
        diag = diagnostics.run_diagnostics(started["run_id"])
        assert diag["method"] == method
        assert key in diag, f"{method} missing {key}: {diag}"
        # raw draws persisted (these fits have no .runset; must come from live/)
        assert runstore.csv_paths(started["run_id"]), f"{method} lost its CSVs"
        # summary + draws work via the CSV fallback (no inference.nc for these)
        summ = diagnostics.run_summary(started["run_id"], params=["theta"])
        assert summ["source"] == "csv"
        assert any(r["param"] == "theta" for r in summ["summary"])
        draws = diagnostics.extract_draws(started["run_id"], "theta")
        assert draws["source"] == "csv" and len(draws["draws"]) > 0


def test_live_inits_from_prior_run(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import asyncio

    import sample

    mgr = sample.RunManager()
    first = _sample(mgr, BERNOULLI, 1)
    # warm-start a second run from the first run's posterior mean
    started = mgr.start_run(
        BERNOULLI,
        DATA,
        method="sample",
        seed=2,
        method_args={"chains": 2, "iter_warmup": 300, "iter_sampling": 300},
        inits_from=first,
    )
    result = asyncio.run(mgr.await_run(started["run_id"], timeout=120))
    assert result["status"] == "done"


def test_live_concurrent_runs_same_model(tmp_path, monkeypatch):
    """Two concurrent runs of one model must not disturb each other's binary.

    Regression: recording last-use by touching the `.stan` made cmdstanpy see a
    source newer than the exe and relink it while the first run's chains were
    executing that very file, so the kernel killed them all
    (`retcodes=[-1, -1, -1, -1]`)."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import asyncio

    import sample

    mgr = sample.RunManager(max_workers=2)
    args = {"chains": 2, "iter_warmup": 300, "iter_sampling": 1500}
    a = mgr.start_run(BERNOULLI, DATA, method="sample", seed=1, method_args=args)
    # starts while `a` is still sampling, and resolves to the same model_id
    b = mgr.start_run(BERNOULLI, DATA, method="sample", seed=2, method_args=args)
    for started in (a, b):
        result = asyncio.run(mgr.await_run(started["run_id"], timeout=180))
        assert result["status"] == "done", result["meta"].get("stderr", "")[:500]


FUNNEL = """
parameters { real v; vector[9] x; }
model { v ~ normal(0, 3); x ~ normal(0, exp(v/2)); }
"""


def test_live_pathologies_localize_funnel_neck(tmp_path, monkeypatch):
    """Neal's funnel, centered. The divergences must point at the log-scale
    parameter `v` and downward — the neck is at low `v` — not at any of the nine
    `x` it governs."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import asyncio

    import diagnostics
    import sample

    mgr = sample.RunManager()
    started = mgr.start_run(
        FUNNEL,
        None,
        method="sample",
        seed=5,
        method_args={"chains": 4, "iter_warmup": 1000, "iter_sampling": 1000},
    )
    result = asyncio.run(mgr.await_run(started["run_id"], timeout=180))
    assert result["status"] == "done", result["meta"].get("stderr", "")[:500]
    diag = diagnostics.run_diagnostics(started["run_id"])
    # E-BFMI is the diagnostic most specific to this geometry: it must be present
    # (it was silently swallowed for every run) and must flag the funnel.
    assert diag.get("min_ebfmi") is not None, diag.get("ebfmi_warning")
    assert diag["min_ebfmi"] < 0.3
    if diag.get("divergences", 0) == 0:
        pytest.skip("this seed produced no divergences; nothing to localize")
    out = diagnostics.run_pathologies(started["run_id"], permutations=400)
    assert out["ranking"][0]["param"] == "v"
    assert out["ranking"][0]["direction"] == "below"
    assert out["separation"]["margin"] > 3


COLLINEAR = """
data { int<lower=0> N; vector[N] x1; vector[N] x2; vector[N] y; }
parameters { real a; real b1; real b2; real<lower=0> sigma; }
model { y ~ normal(a + b1 * x1 + b2 * x2, sigma); }
"""


def test_live_geometry_finds_collinearity(tmp_path, monkeypatch):
    """Full unconstrained path: BridgeStan rebuild, the background .so build and
    its `preparing` status, the BridgeStan-name to arviz-dim index mapping, and
    the statistics. The offline tests stub all of that out.

    x2 = x1 + tiny noise, so b1 and b2 are near-degenerate and trade off — a
    NEGATIVE correlation, which is what the signed-Fisher-z bug silently dropped.
    """
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import asyncio
    import time

    import diagnostics
    import sample

    rng = np.random.default_rng(7)
    n = 100
    x1 = rng.normal(size=n)
    x2 = x1 + 0.01 * rng.normal(size=n)
    data = {
        "N": n,
        "x1": list(x1),
        "x2": list(x2),
        "y": list(x1 + x2 + 0.3 * rng.normal(size=n)),
    }
    mgr = sample.RunManager()
    started = mgr.start_run(
        COLLINEAR,
        data,
        method="sample",
        seed=1,
        method_args={"chains": 4, "iter_warmup": 500, "iter_sampling": 500},
    )
    result = asyncio.run(mgr.await_run(started["run_id"], timeout=300))
    assert result["status"] == "done", result["meta"].get("stderr", "")[:500]

    saw_preparing = False
    for _ in range(60):
        geom = diagnostics.run_geometry(started["run_id"])
        if geom.get("status") != "preparing":
            break
        saw_preparing = True
        time.sleep(3)
    else:
        pytest.fail("BridgeStan artifact never finished building")

    # A run builds only the CmdStan exe, so the first call must have deferred.
    assert saw_preparing
    assert geom["scale"] == "unconstrained"
    assert geom["verdict"] == "warn"
    pair = geom["pairs"][0]
    assert {pair["a"], pair["b"]} == {"b1", "b2"}
    assert pair["r"] < -0.9  # trading off: negative, and it must survive the filter
    assert pair["z"] > 4
    assert geom["eigen"]["condition_number"] > 100
    assert {d["param"] for d in geom["eigen"]["direction"][:2]} == {"b1", "b2"}


def test_live_validate_data(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import density
    import errors

    ok = density.validate_data(BERNOULLI, DATA)
    assert ok["ok"] is True and ok["param_unc_num"] == 1
    # missing required data (N, y) should be rejected at build time
    with pytest.raises((errors.StanMcpError, Exception)):
        density.validate_data(BERNOULLI, {})


def test_compile_model_returns_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import compile as c

    out = c.compile_model(BERNOULLI)
    assert "error" not in out
    assert out["so_path"].endswith(".so")
    assert os.path.exists(out["so_path"])
    assert os.path.exists(out["exe_path"])


def test_param_info(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import density

    info = density.param_info(BERNOULLI, DATA)
    assert info["param_names"] == ["theta"]
    assert info["param_num"] == 1
    assert info["param_unc_num"] == 1  # theta is bounded -> 1 unconstrained dim


def test_live_log_density(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import density

    out = density.log_density_gradient(BERNOULLI, DATA, [0.0])
    assert isinstance(out["lp"], float)
    assert len(out["gradient"]) == 1


def test_live_check_gradient(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import density

    out = density.check_gradient(BERNOULLI, DATA, [0.1])
    assert out["max_abs_err"] < 1e-3


def test_live_sample_run(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import asyncio

    import diagnostics
    import sample

    mgr = sample.RunManager()
    started = mgr.start_run(
        BERNOULLI,
        DATA,
        method="sample",
        method_args={"chains": 2, "iter_warmup": 200, "iter_sampling": 200},
    )
    result = asyncio.run(mgr.await_run(started["run_id"], timeout=120))
    assert result["status"] == "done"
    summ = diagnostics.run_summary(started["run_id"], params=["theta"])
    assert any(row["param"] == "theta" for row in summ["summary"])
    diag = diagnostics.run_diagnostics(started["run_id"])
    assert "divergences" in diag


def test_live_generate_quantities(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import asyncio

    import diagnostics
    import errors
    import runstore
    import sample

    mgr = sample.RunManager()

    started_a = mgr.start_run(
        BERNOULLI,
        DATA,
        method="sample",
        method_args={"chains": 2, "iter_warmup": 200, "iter_sampling": 200},
    )
    result_a = asyncio.run(mgr.await_run(started_a["run_id"], timeout=120))
    assert result_a["status"] == "done"
    run_id_a = started_a["run_id"]

    started_gq = mgr.start_run(
        BERNOULLI_GQ,
        DATA,
        method="generate_quantities",
        method_args={"from_run_id": run_id_a},
    )
    result_gq = asyncio.run(mgr.await_run(started_gq["run_id"], timeout=120))
    assert result_gq["status"] == "done"
    run_id_gq = started_gq["run_id"]

    # Reality as of arviz 1.2 / arviz-base: its from_cmdstanpy converter calls
    # fit.method_variables(), which cmdstanpy's CmdStanGQ does not implement,
    # so the arviz conversion in diagnostics.persist_run fails (recorded as
    # persist_warning) and inference.nc is never written for GQ runs. Output
    # is still retrievable via the persisted raw CmdStan CSVs
    # (runstore.csv_paths). If a future arviz version maps CmdStanGQ cleanly,
    # prefer asserting through run_summary/extract_draws instead.
    try:
        diagnostics.extract_draws(run_id_gq, "y_sim")
    except errors.RunError:
        assert runstore.csv_paths(run_id_gq)
    else:
        summ = diagnostics.run_summary(run_id_gq)
        assert "summary" in summ


# matrix elements get distinct, well-separated means (11,12,13 / 21,22,23) so a
# row/column-major mix-up in the name-to-dim mapping is unmistakable rather than
# merely a slightly-off number; the simplex makes the unconstrained dimension
# differ from the constrained one (3 -> 2).
SHAPED = """
parameters {
  matrix[2, 3] m;
  simplex[3] theta;
  real<lower=0> s;
}
model {
  for (i in 1:2) { for (j in 1:3) { m[i, j] ~ normal(10 * i + j, 0.1); } }
  theta ~ dirichlet([2.0, 5.0, 9.0]');
  s ~ lognormal(0, 0.5);
}
"""


def test_live_bridgestan_name_mapping_on_shaped_params(tmp_path, monkeypatch):
    """The BridgeStan `m.2.1` name to arviz array-dim mapping, on a model that
    actually has multi-dimensional and dimension-changing parameters.

    This is the one line in run_geometry whose failure is silent: get it wrong
    and every reported pair is labelled with the wrong parameter name. Nothing
    else covers it -- the offline tests monkeypatch _unconstrained_draws away,
    and the collinearity test's params are all scalars, so `pos` is always 0.

    Checked two independent ways: against the prior means the draws must
    concentrate on, and against CmdStan's own CSV column names, which use the
    same `m.i.j` convention and are produced by code that shares nothing with
    the mapping under test.
    """
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import asyncio
    import time

    import density
    import diagnostics
    import runstore
    import sample

    mgr = sample.RunManager()
    started = mgr.start_run(
        SHAPED,
        {},
        method="sample",
        seed=2,
        method_args={"chains": 2, "iter_warmup": 500, "iter_sampling": 500},
    )
    rid = started["run_id"]
    result = asyncio.run(mgr.await_run(rid, timeout=300))
    assert result["status"] == "done", result["meta"].get("stderr", "")[:500]

    inputs = runstore.load_inputs(rid)
    model = density.get_model(inputs["source"], inputs["data"])
    bs_names = list(model.param_names())
    # 6 matrix elements + 3 simplex components + 1 scalar
    assert len(bs_names) == 10, bs_names

    idata = diagnostics._load_idata(rid)
    X = diagnostics._constrained_matrix(idata, bs_names)
    assert X.shape == (1000, 10)

    # (1) each matrix element must land on its own prior mean, 10*i + j
    for k, nm in enumerate(bs_names):
        if not nm.startswith("m."):
            continue
        i, j = (int(p) for p in nm.split(".")[1:])
        got = float(X[:, k].mean())
        assert abs(got - (10 * i + j)) < 0.5, f"{nm} resolved to {got}, wants {10 * i + j}"

    # (2) independent oracle: CmdStan writes its CSV header with the same
    # `m.i.j` spelling, so column nm from the CSV and column k here are the same
    # quantity. Compared as multisets, since the two paths need not agree on the
    # order in which chains are concatenated.
    df = diagnostics._draws_df(rid)
    for k, nm in enumerate(bs_names):
        assert nm in df.columns, f"{nm} missing from CmdStan CSV: {list(df.columns)[:12]}"
        mine = np.sort(X[:, k])
        theirs = np.sort(df[nm].to_numpy(dtype=float))
        assert np.allclose(mine, theirs), f"{nm} mismatch vs CmdStan CSV"

    # (3) the unconstrained space is genuinely a different size: the simplex
    # contributes 2 dims, not 3, so param_unc_names() cannot be assumed to line
    # up with param_names().
    assert model.param_unc_num() == 9
    for _ in range(60):
        geom = diagnostics.run_geometry(rid)
        if geom.get("status") != "preparing":
            break
        time.sleep(3)
    else:
        pytest.fail("BridgeStan artifact never finished building")
    assert geom["scale"] == "unconstrained"
    assert geom["n_dims"] == 9, geom
