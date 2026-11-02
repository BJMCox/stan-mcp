"""Write file-first predictive-check and SBC study examples.

Run: pixi run python examples/model_validation.py /absolute/demo
Submit prior.json with start_run, then run this script with --prior-run RUN_ID
for check_predictive. Submit study.json through start_study, then use
--study-id STUDY_ID for check_sbc. Inputs and results stay on the server host.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def main(directory: Path, prior_run: str | None = None, study_id: str | None = None) -> None:
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "validation"
    for name in ("model.stan", "simulate.stan"):
        shutil.copyfile(source / name, directory / name)

    def write(name: str, value: Any) -> None:
        (directory / name).write_text(json.dumps(value, indent=2) + "\n")

    write("data.json", {"N": 20})
    write("observed.json", {"y": 11})
    write(
        "prior.json",
        {
            "stan": str(directory / "simulate.stan"),
            "data": str(directory / "data.json"),
            "method": "fixed_param",
            "seed": 81,
            "method_args": {"chains": 2, "iter_sampling": 500},
        },
    )
    write(
        "study.json",
        {
            "schema": "stan-mcp.study.v1",
            "seed": 49,
            "replicates": 20,
            "max_parallel": 2,
            "tasks": [
                {
                    "id": "beta-binomial",
                    "stan": str(directory / "model.stan"),
                    "data": str(directory / "data.json"),
                    "method_args": {"chains": 2, "iter_warmup": 300, "iter_sampling": 300},
                    "simulate": {
                        "stan": str(directory / "simulate.stan"),
                        "data": str(directory / "data.json"),
                        "data_variables": {"y": "y_sim"},
                        "integer_data": ["y"],
                        "truth_variables": {"theta": "theta_sim", "joint": "joint_sim"},
                    },
                }
            ],
        },
    )
    if prior_run:
        write(
            "predictive.json",
            {
                "run_id": prior_run,
                "kind": "prior",
                "independent_draws": True,
                "observed": str(directory / "observed.json"),
                "checks": [{"observed": "y", "replicated": "y_sim", "statistic": "mean"}],
            },
        )
    if study_id:
        write(
            "sbc.json",
            {
                "study_id": study_id,
                "draws": 100,
                "thin": 5,
                "bins": 10,
                "interval": 0.9,
                "seed": 17,
            },
        )
    print(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prior-run")
    parser.add_argument("--study-id")
    args = parser.parse_args()
    main(args.directory, args.prior_run, args.study_id)
