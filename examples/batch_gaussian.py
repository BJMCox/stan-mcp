"""Write a file-based batch example; optionally compare an analytic reference.

Run with `pixi run python examples/batch_gaussian.py /absolute/demo`.
Submit demo/evaluate.json through evaluate_batch. Then pass the result manifest
as --stan-result to write an analytic reference and demo/compare.json.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _artifact(directory: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    shard = directory / "00000000.safetensors"
    save_file(arrays, str(shard))
    manifest = directory / "manifest.json"
    _json(
        manifest,
        {
            "schema": "stan-mcp.batch.v1",
            **metadata,
            "shards": [
                {
                    "path": shard.name,
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                    "rows": len(arrays["row_id"]),
                }
            ],
        },
    )
    return manifest


def main(directory: Path, stan_result: Path | None = None) -> None:
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if stan_result is None:
        model = directory / "model.stan"
        model.write_text("parameters { vector[2] x; } model { x ~ std_normal(); }\n")
        _json(directory / "data.json", {})
        points = _artifact(
            directory / "points",
            {
                "row_id": np.array([2, 7, 15], dtype=np.int64),
                "points": np.array([[0.0, 0.0], [1.0, -2.0], [0.25, 0.5]]),
                "vectors": np.tile(np.eye(2), (3, 1, 1)),
            },
            {"kind": "points", "point_space": "constrained", "parameter_names": ["x.1", "x.2"]},
        )
        _json(
            directory / "evaluate.json",
            {
                "stan": str(model),
                "data": str(directory / "data.json"),
                "points": str(points),
                "quantities": ["log_density", "gradient", "hessian", "hvp"],
            },
        )
        print(directory / "evaluate.json")
        return
    stan_result = stan_result.expanduser().resolve()
    manifest = json.loads(stan_result.read_text())
    if manifest["parameter_names"] != ["x.1", "x.2"]:
        raise ValueError("Use the result from this example's two-dimensional Gaussian")
    shards = [load_file(stan_result.parent / part["path"]) for part in manifest["shards"]]
    points = np.concatenate([part["points"] for part in shards])
    vectors = np.concatenate([part["vectors"] for part in shards])
    arrays = {
        "row_id": np.concatenate([part["row_id"] for part in shards]),
        "points": points,
        "vectors": vectors,
        "status": np.zeros(len(points), dtype=np.int64),
        "log_density": -np.sum(points**2, axis=1) / 2 - np.log(2 * np.pi),
        "gradient": -points,
        "hessian": np.tile(-np.eye(2), (len(points), 1, 1)),
        "hvp": -vectors,
    }
    # This model has the identity transform, so analytic and native coordinates agree.
    reference = _artifact(
        directory / "reference",
        arrays,
        {
            key: manifest[key]
            for key in (
                "kind",
                "point_space",
                "parameter_names",
                "quantities",
                "target",
                "derivative_coordinates",
            )
        },
    )
    _json(
        directory / "compare.json",
        {
            "left": str(stan_result),
            "right": str(reference),
            "tolerances": {q: {"atol": 1e-12, "rtol": 1e-12} for q in manifest["quantities"]},
        },
    )
    print(directory / "compare.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--stan-result", type=Path)
    args = parser.parse_args()
    main(args.directory, args.stan_result)
