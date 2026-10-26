import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import cmdstanpy
import pytest
from cmdstanpy.stanfit.metadata import InferenceMetadata

import density
import diagnostics
import runstore
import sample
import server
from tests.fakes import FakeModel


def _tool_result(result) -> dict:
    return json.loads(next(block.text for block in result.content if block.type == "text"))


@pytest.mark.parametrize(
    "name,expected",
    [
        ("log_density", {"lp": -2.5}),
        ("log_density_gradient", {"lp": -2.5, "gradient": [-1.0, -2.0]}),
        (
            "log_density_hessian",
            {"lp": -2.5, "gradient": [-1.0, -2.0], "hessian": [[-1.0, 0.0], [0.0, -1.0]]},
        ),
        ("param_constrain", {"constrained": {"a": 1.0, "b": 2.0}}),
        ("check_gradient", {"max_abs_err": 0.0, "max_rel_err": 0.0, "mismatches": []}),
        ("param_unconstrain", {"theta_unc": [1.0, 2.0]}),
    ],
)
async def test_parameter_files_through_mcp(tmp_path, monkeypatch, name, expected):
    monkeypatch.setattr(density, "get_model", lambda *a, **k: FakeModel(2, ["a", "b"]))
    field = "constrained" if name == "param_unconstrain" else "theta_unc"
    payload = {"a": 1.0, "b": 2.0} if field == "constrained" else [1.0, 2.0]
    path = tmp_path / "params.json"
    path.write_text(json.dumps(payload))
    for value in (str(path), payload):
        result = _tool_result(
            await server.mcp.call_tool(name, {"stan": "model {}", "data": "{}", field: value})
        )
        if name == "check_gradient":
            assert result["max_abs_err"] < 1e-8
            assert result["mismatches"] == []
        else:
            assert result == expected


@pytest.mark.parametrize(
    "field,content",
    [
        ("theta_unc", None),
        ("theta_unc", "["),
        ("theta_unc", "{}"),
        ("theta_unc", "[1, 2]"),
        ("constrained", "[]"),
    ],
)
async def test_parameter_file_errors_return_without_cpp(tmp_path, monkeypatch, field, content):
    model = FakeModel()
    monkeypatch.setattr(density, "get_model", lambda *a, **k: model)
    monkeypatch.setattr(
        model, "log_density", lambda *a, **k: pytest.fail("invalid input reached C++")
    )
    monkeypatch.setattr(
        model, "param_unconstrain_json", lambda *a, **k: pytest.fail("invalid input reached C++")
    )
    path = tmp_path / "bad.json"
    if content is not None:
        path.write_text(content)
    name = "param_unconstrain" if field == "constrained" else "log_density"
    result = _tool_result(
        await server.mcp.call_tool(name, {"stan": "model {}", "data": "{}", field: str(path)})
    )
    assert result["error_type"] == "ValidationError"


def _fit_csv(path, method, extra_rows=False, values=(1.0, 2.0)):
    config = {"model": "input", "method": method, "id": 1}
    header = "lp__,x\n"
    body = "".join(f"-1,{v}\n" for v in values)
    if method == "sample":
        config.update(
            algorithm="hmc",
            engine="nuts",
            metric="diag_e",
            max_depth=10,
            num_samples=2,
            num_warmup=1,
            save_warmup=int(extra_rows),
            thin=1,
        )
        header = "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,x\n"
        body = "-1,1,1,1,1,0,1,999\n" if extra_rows else ""
        body += (
            "# Adaptation terminated\n# Step size = 1\n"
            "# Diagonal elements of inverse mass matrix:\n# 1\n"
        )
        body += "".join(f"-1,1,1,1,1,0,1,{v}\n" for v in values)
        body += (
            "#  Elapsed Time: 0 seconds (Warm-up)\n#  0 seconds (Sampling)\n#  0 seconds (Total)\n"
        )
    elif method == "variational":
        config.update(
            algorithm="meanfield",
            iter=10,
            grad_samples=1,
            elbo_samples=1,
            eta=1,
            tol_rel_obj=0.01,
            eval_elbo=1,
            output_samples=2,
        )
        body = "# Stepsize adaptation complete.\n# eta = 1\n-1,999\n" + body
    else:
        config.update(algorithm="lbfgs", save_iterations=int(extra_rows))
        body = ("-1,999\n" if extra_rows else "") + f"-1,{values[-1]}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"# {k} = {v}\n" for k, v in config.items()) + header + body)
    return path


class _GeneratedModel:
    def generate_quantities(self, data, previous_fit, gq_output_dir, **kwargs):
        if isinstance(previous_fit, list):
            previous_fit = cmdstanpy.from_csv(previous_fit)
        assert isinstance(
            previous_fit, (cmdstanpy.CmdStanMCMC, cmdstanpy.CmdStanMLE, cmdstanpy.CmdStanVB)
        )
        outputs = []
        for i, path in enumerate(previous_fit.runset.csv_files):
            rows = [
                ln for ln in Path(path).read_text().splitlines() if ln and not ln.startswith("#")
            ]
            target = Path(gq_output_dir) / f"gq_{i + 1}.csv"
            target.write_text(
                "g\n" + "".join(f"{2 * float(ln.split(',')[-1])}\n" for ln in rows[1:])
            )
            outputs.append(str(target))
        return SimpleNamespace(
            previous_fit=previous_fit,
            metadata=InferenceMetadata({"raw_header": "g"}),
            runset=SimpleNamespace(csv_files=outputs),
        )


@pytest.mark.parametrize(
    "method,extra_rows,expected",
    [
        ("sample", False, [2.0, 4.0]),
        ("sample", True, [2.0, 4.0]),
        ("variational", False, [2.0, 4.0]),
        ("optimize", False, [4.0]),
        ("optimize", True, [4.0]),
    ],
)
async def test_external_gq_csvs_keep_only_draws(
    tmp_path, monkeypatch, method, extra_rows, expected
):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    csv = _fit_csv(tmp_path / "fit.csv", method, extra_rows)
    mgr = sample.RunManager(max_workers=1)
    monkeypatch.setattr(mgr, "_compile", lambda *a: _GeneratedModel())
    monkeypatch.setattr(server, "MANAGER", mgr)
    try:
        started = _tool_result(
            await server.mcp.call_tool(
                "start_run",
                {
                    "stan": "model {}",
                    "method": "generate_quantities",
                    "method_args": {"csv_files": str(csv)},
                },
            )
        )
        result = await mgr.await_run(started["run_id"], timeout=5)
        assert result["status"] == "done", result
        assert diagnostics.extract_draws(started["run_id"], "g")["draws"] == expected
    finally:
        mgr._pool.shutdown(wait=True)


def test_external_gq_snapshots_preserve_order_and_rerun(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    # Reversed names and duplicate basenames must not change caller-supplied chain order.
    paths = [
        _fit_csv(tmp_path / d / "fit.csv", "sample", values=vals)
        for d, vals in [("z", (3, 4)), ("a", (1, 2))]
    ]
    mgr = sample.RunManager(max_workers=1)
    monkeypatch.setattr(mgr, "_compile", lambda *a: _GeneratedModel())
    try:
        started = mgr.start_run(
            "model {}",
            None,
            method="generate_quantities",
            method_args={"csv_files": [str(p) for p in paths]},
        )
        for path in paths:
            path.unlink()
        first = asyncio.run(mgr.await_run(started["run_id"], timeout=5))
        assert first["status"] == "done", first
        rerun = mgr.rerun(started["run_id"])
        second = asyncio.run(mgr.await_run(rerun["run_id"], timeout=5))
        assert second["status"] == "done", second
        for run in (started, rerun):
            got = diagnostics.extract_draws(run["run_id"], "g", by_chain=True)
            assert got["draws"] == [[6.0, 8.0], [2.0, 4.0]]
    finally:
        mgr._pool.shutdown(wait=True)


async def test_external_gq_rejects_conflicting_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    result = await server.start_run(
        "model {}",
        method="generate_quantities",
        method_args={"from_run_id": "sample-0001", "csv_files": "fit.csv"},
    )
    assert result["error_type"] == "ValidationError"
    assert runstore.list_runs() == []


def test_external_gq_rejects_multiple_non_mcmc_csvs(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    paths = [_fit_csv(tmp_path / f"fit-{i}.csv", "optimize") for i in range(2)]
    mgr = sample.RunManager(max_workers=1)
    monkeypatch.setattr(mgr, "_compile", lambda *a: _GeneratedModel())
    try:
        started = mgr.start_run(
            "model {}",
            None,
            method="generate_quantities",
            method_args={"csv_files": [str(p) for p in paths]},
        )
        result = asyncio.run(mgr.await_run(started["run_id"], timeout=5))
        assert result["status"] == "failed"
        assert runstore.csv_paths(started["run_id"]) == []
    finally:
        mgr._pool.shutdown(wait=True)


@pytest.mark.parametrize("csv_files", [[], [1], "missing.csv"])
async def test_invalid_csv_paths_do_not_start_a_run(tmp_path, monkeypatch, csv_files):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    result = await server.start_run(
        "model {}", method="generate_quantities", method_args={"csv_files": csv_files}
    )
    assert result["error_type"] == "ValidationError"
    assert runstore.list_runs() == []


def test_external_gq_rejects_non_fit_csvs(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    path = tmp_path / "not-a-fit.csv"
    path.write_text("# model = input\n# method = generate_quantities\nx\n1\n")
    mgr = sample.RunManager(max_workers=1)
    monkeypatch.setattr(mgr, "_compile", lambda *a: _GeneratedModel())
    try:
        started = mgr.start_run(
            "model {}", None, method="generate_quantities", method_args={"csv_files": str(path)}
        )
        result = asyncio.run(mgr.await_run(started["run_id"], timeout=5))
        assert result["status"] == "failed"
        assert runstore.csv_paths(started["run_id"]) == []
    finally:
        mgr._pool.shutdown(wait=True)
