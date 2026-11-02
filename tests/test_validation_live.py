"""File-first validation workflows through an actual stdio MCP server."""

import json
import os
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from safetensors.numpy import load_file

pytestmark = pytest.mark.skipif(
    os.environ.get("STAN_MCP_LIVE") != "1", reason="requires native compiler"
)


async def test_stdio_predictive_study_sbc_and_fixed_parameter_law(tmp_path, caplog):
    project = Path(__file__).resolve().parents[1]
    assets = project / "examples" / "validation"
    data = tmp_path / "data.json"
    data.write_text('{"N":20}')
    observed = tmp_path / "observed.json"
    observed.write_text('{"y":11}')
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "stan_mcp"],
        cwd=str(project),
        env={**os.environ, "STAN_MCP_HOME": str(tmp_path / "home")},
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer, read_timeout_seconds=300) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            assert len(tools) == 37
            for tool in tools:
                Draft202012Validator.check_schema(tool.input_schema)

            async def call(name, args, allow_error=False):
                result = await session.call_tool(name, args)
                value = json.loads(
                    next(part.text for part in result.content if part.type == "text")
                )
                assert not result.is_error and (allow_error or "error_type" not in value), value
                return value

            prior = await call(
                "start_run",
                {
                    "stan": str(assets / "simulate.stan"),
                    "data": str(data),
                    "method": "fixed_param",
                    "seed": 9,
                    "method_args": {"chains": 2, "iter_sampling": 300},
                },
            )
            assert (await call("await_run", {"run_id": prior["run_id"], "timeout": 240}))[
                "status"
            ] == "done"
            diag = await call("get_run_diagnostics", {"run_id": prior["run_id"]})
            assert diag == {"method": "fixed_param"}
            request = tmp_path / "predictive.json"
            request.write_text(
                json.dumps(
                    {
                        "run_id": prior["run_id"],
                        "kind": "prior",
                        "independent_draws": True,
                        "observed": str(observed),
                        "checks": [{"observed": "y", "replicated": "y_sim", "statistic": "mean"}],
                    }
                )
            )
            checked = await call("check_predictive", {"request_file": str(request)})
            assert 0 < checked["checks"][0]["tail_probability_ge"] < 1
            assert Path(checked["plot"]).is_file()
            request = tmp_path / "study.json"
            request.write_text(
                json.dumps(
                    {
                        "schema": "stan-mcp.study.v1",
                        "seed": 13,
                        "replicates": 4,
                        "max_parallel": 2,
                        "tasks": [
                            {
                                "id": "beta-binomial",
                                "stan": str(assets / "model.stan"),
                                "data": str(data),
                                "method_args": {
                                    "chains": 2,
                                    "iter_warmup": 150,
                                    "iter_sampling": 150,
                                },
                                "simulate": {
                                    "stan": str(assets / "simulate.stan"),
                                    "data": str(data),
                                    "data_variables": {"y": "y_sim"},
                                    "integer_data": ["y"],
                                    "truth_variables": {"theta": "theta_sim", "joint": "joint_sim"},
                                },
                            }
                        ],
                    }
                )
            )
            started = await call("start_study", {"request_file": str(request)})
            study = await call("await_run", {"run_id": started["study_id"], "timeout": 240})
            assert study["status"] == "done" and study["counts"] == {"done": 4}, study
            case = json.loads((Path(study["tasks_directory"]) / "00000.json").read_text())
            observed.write_text(json.dumps(case["generated_data"]))
            request = tmp_path / "posterior.json"
            request.write_text(
                json.dumps(
                    {
                        "run_id": case["fit"]["run_id"],
                        "observed": str(observed),
                        "checks": [{"observed": "y", "replicated": "y_rep", "statistic": "mean"}],
                    }
                )
            )
            checked = await call("check_predictive", {"request_file": str(request)})
            assert checked["sampling_method"] == "sample"
            request = tmp_path / "sbc.json"
            request.write_text(
                json.dumps(
                    {"study_id": started["study_id"], "draws": 40, "thin": 5, "bins": 7, "seed": 18}
                )
            )
            calibrated = await call("check_sbc", {"request_file": str(request)})
            assert calibrated["assessable_cases"] == calibrated["planned_cases"] == 4
            assert {row["quantity"] for row in calibrated["quantities"]} == {"theta", "joint"}
            assert load_file(calibrated["ranks_file"])["rank"].shape == (8,)
            await call("resume_study", {"study_id": started["study_id"]})
            resumed = await call("await_run", {"run_id": started["study_id"], "timeout": 30})
            assert resumed["counts"] == {"done": 4}, resumed
            frozen = await call(
                "start_run",
                {
                    "stan": str(assets / "model.stan"),
                    "data": {"N": 20, "y": 11},
                    "method": "fixed_param",
                    "seed": 7,
                    "method_args": {"chains": 1, "iter_sampling": 5},
                },
            )
            frozen_result = await call("await_run", {"run_id": frozen["run_id"], "timeout": 30})
            assert frozen_result["status"] == "done", frozen_result
            csvs = sorted(
                str(path) for path in (Path(frozen_result["run_dir"]) / "csv").glob("*.csv")
            )
            generated = await call(
                "start_run",
                {
                    "stan": str(assets / "model.stan"),
                    "data": {"N": 20, "y": 11},
                    "method": "generate_quantities",
                    "method_args": {"csv_files": csvs},
                },
            )
            generated_result = await call(
                "await_run", {"run_id": generated["run_id"], "timeout": 30}
            )
            assert generated_result["status"] == "done", generated_result
            assert generated_result["meta"]["input_method"] == "fixed_param"
            loo = await call("run_loo", {"run_id": generated["run_id"]}, allow_error=True)
            assert loo["error_type"] == "RunError"
    assert not any(
        record.name == "mcp.client.stdio" and record.levelname == "ERROR"
        for record in caplog.records
    )
