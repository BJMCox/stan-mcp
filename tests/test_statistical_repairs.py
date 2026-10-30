import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from stan_mcp import compilation as compiler
from stan_mcp import density, diagnostics, draws, fitstorage, geometry, runstore
from stan_mcp import inference as sample
from stan_mcp.errors import RunError, ValidationError


def store(tmp_path, monkeypatch, method="sample", columns="lp__,x,g", rows="-1,1,2\n-1,-1,-2\n"):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path))
    rid = f"{method}-0001"
    runstore.write_meta(rid, {"method": method, "status": "done"})
    runstore.persist_inputs(rid, "parameters {real x;} model {}", {})
    live = runstore.live_dir(rid, create=True)
    (live / "a.csv").write_text(f"{columns}\n{rows}")
    fit = SimpleNamespace(
        runset=SimpleNamespace(csv_files=[str(live / "a.csv")]),
        metadata=SimpleNamespace(
            cmdstan_config={"raw_header": columns},
            stan_vars={"x": SimpleNamespace(dimensions=()), "g": SimpleNamespace(dimensions=())},
        ),
    )
    fitstorage.persist_run(rid, method, fit)
    return rid


def test_undefined_convergence_cannot_pass(tmp_path, monkeypatch):
    rid = store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda files: {
            "x": {"R_hat": 1.0, "ESS_bulk": 500, "ESS_tail": 500},
            "g": {"R_hat": float("nan"), "ESS_bulk": float("nan"), "ESS_tail": 0},
        },
    )
    result = diagnostics.run_diagnostics(rid)
    assert result["verdict"] == "warn"  # finite zero ESS remains a measured failure
    assert result["undefined_diagnostics"]["R_hat"]["parameters"] == ["g"]
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda files: {
            name: {"R_hat": 1.0 if name == "x" else float("nan"), "ESS_bulk": 500, "ESS_tail": 500}
            for name in ("x", "g")
        },
    )
    assert diagnostics.run_diagnostics(rid)["verdict"] == "unknown"


def test_loo_rejects_approximate_law_before_arviz(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path))
    import arviz as az

    unsafe = Mock(side_effect=AssertionError("approximate draws reached ordinary LOO"))
    monkeypatch.setattr(az, "loo", unsafe)
    monkeypatch.setattr(az, "compare", unsafe)
    for method in ("variational", "laplace", "pathfinder"):
        rid = f"{method}-0001"
        runstore.write_meta(rid, {"method": method})
        with pytest.raises(RunError):
            diagnostics.run_loo(rid)
        with pytest.raises(RunError):
            diagnostics.compare_runs([rid, "sample-0002"])
    assert not unsafe.called


def test_loo_accepts_verified_gq_snapshot_after_source_removed(tmp_path, monkeypatch):
    rid = store(tmp_path, monkeypatch, method="generate_quantities")
    runstore.write_meta(rid, {"method": "generate_quantities", "input_method": "sample"})
    tree: Any = diagnostics._idata_with_loglik(rid, "g")
    assert np.array_equal(tree["log_likelihood"]["g"].values, [[2, -2]])
    runstore.write_meta(rid, {"method": "generate_quantities", "input_method": "variational"})
    with pytest.raises(RunError):
        diagnostics._idata_with_loglik(rid, "g")


def test_failed_csv_copy_retains_every_original(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path))
    rid = "sample-0001"
    runstore.write_meta(rid, {"method": "sample"})
    live = runstore.live_dir(rid, create=True)
    paths = []
    for n in range(2):
        path = live / f"{n}.csv"
        path.write_text("lp__,x\n-1,2\n")
        paths.append(str(path))
    original = runstore.shutil.copy

    def fail_second(source, target):
        if str(source).endswith("1.csv"):
            raise OSError("disk full")
        return original(source, target)

    monkeypatch.setattr(runstore.shutil, "copy", fail_second)
    fit = SimpleNamespace(runset=SimpleNamespace(csv_files=paths))
    with pytest.raises(RunError):
        fitstorage.persist_run(rid, "sample", fit)
    assert all(__import__("pathlib").Path(path).read_text() == "lp__,x\n-1,2\n" for path in paths)
    assert not draws.schema_path(rid).exists()


def test_optimize_persists_csv_and_shaped_warm_start(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path))
    rid = "optimize-0001"
    runstore.write_meta(rid, {"method": "optimize"})
    live = runstore.live_dir(rid, create=True)
    csv = live / "opt.csv"
    csv.write_text("lp__,v.1,v.2\n-1,2,3\n")
    fit = SimpleNamespace(
        runset=SimpleNamespace(csv_files=[str(csv)]),
        optimized_params_dict={"lp__": -1, "v.1": 2, "v.2": 3},
        stan_variables=lambda: {"v": np.array([2.0, 3.0])},
    )
    fitstorage.persist_run(rid, "optimize", fit)
    assert sample.posterior_mean_inits(rid) == {"v": [2.0, 3.0]}
    assert len(runstore.csv_paths(rid)) == 1
    assert not live.exists()


def test_geometry_excludes_gq_and_checks_size_before_transforms(tmp_path, monkeypatch):
    rid = store(tmp_path, monkeypatch)
    model = SimpleNamespace(
        param_names=lambda: ["x"],
        param_unc_names=lambda: ["x"],
        param_unc_num=lambda: 1,
        param_unconstrain=Mock(side_effect=RuntimeError("constraint")),
    )
    monkeypatch.setattr(compiler, "request_so", lambda source: {"status": "ready"})
    monkeypatch.setattr(density, "get_model", lambda *args: model)
    result, names, scale = geometry._unconstrained_draws(rid, draws.load(rid))
    assert names == ["x"] and result.shape == (1, 2, 1)
    assert scale["scale"] == "constrained"
    model.param_unc_names = lambda: ["x", "y"]
    model.param_unconstrain.reset_mock()
    with pytest.raises(ValidationError):
        geometry._unconstrained_draws(rid, draws.load(rid), max_dims=1)
    assert not model.param_unconstrain.called
    monkeypatch.setattr(
        compiler, "request_so", lambda source: {"status": "failed", "error": "no compiler"}
    )
    with pytest.raises(RunError):
        geometry._unconstrained_draws(rid, draws.load(rid))


def test_invalid_unit_vector_mean_uses_valid_draw(tmp_path, monkeypatch):
    rid = store(tmp_path, monkeypatch)
    model = SimpleNamespace(
        param_names=lambda: ["x"],
        param_unconstrain_json=lambda value: np.array([json.loads(value)["x"]]),
        param_constrain=lambda value: value / np.linalg.norm(value),
    )
    monkeypatch.setattr(density, "get_model", lambda *args: model)
    with np.errstate(invalid="ignore"):
        assert sample.posterior_mean_inits(rid) == {"x": 1.0}


def test_published_data_and_initialization_schema_and_strict_json():
    from stan_mcp import server

    schemas = {
        tool.name: tool.model_dump(by_alias=True)["inputSchema"]
        for tool in asyncio.run(server.mcp.list_tools())
    }
    variants = schemas["start_run"]["properties"]["data"]["anyOf"]
    assert {part["type"] for part in variants} == {"object", "string", "null"}
    variants = schemas["start_run"]["properties"]["inits"]["anyOf"]
    assert {part["type"] for part in variants} >= {"object", "array", "string", "number", "null"}
    result = server._safe(
        lambda: {
            "nested": [{"value": np.float64(float("inf"))}],
            "array": np.array([float("nan"), 1]),
        }
    )
    assert json.loads(json.dumps(result, allow_nan=False)) == {
        "nested": [{"value": None}],
        "array": [None, 1],
    }


def test_undefined_bfmi_cannot_hide_behind_healthy_convergence(tmp_path, monkeypatch):
    rid = store(tmp_path, monkeypatch, columns="lp__,x,g,energy__", rows="-1,1,2,3\n-1,-1,-2,3\n")
    monkeypatch.setattr(
        diagnostics.stansummary,
        "summarize",
        lambda files: {name: {"R_hat": 1, "ESS_bulk": 500, "ESS_tail": 500} for name in ("x", "g")},
    )
    import arviz as az

    monkeypatch.setattr(az, "bfmi", lambda energy: np.array([np.nan, 1.0]))
    result = diagnostics.run_diagnostics(rid)
    assert result["verdict"] == "unknown" and result["min_ebfmi"] == 1.0
    assert "1" in result["ebfmi_warning"]
    monkeypatch.setattr(az, "bfmi", lambda energy: np.array([0.1, 1.0]))
    assert diagnostics.run_diagnostics(rid)["verdict"] == "warn"
