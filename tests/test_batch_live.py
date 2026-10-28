"""Live server evaluation and external comparison across spawned native workers."""

import os
from pathlib import Path

import numpy as np
import pytest

import batchjobs
import batchstore

pytestmark = pytest.mark.skipif(
    os.environ.get("STAN_MCP_LIVE") != "1", reason="requires native compiler"
)


@pytest.mark.asyncio
async def test_server_native_parallel_evaluation_and_external_comparison(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    import server

    manager = batchjobs.BatchManager(max_workers=2)
    monkeypatch.setattr(server, "BATCH_MANAGER", manager)
    points = np.array([[0.2, -0.7], [1.0, 2.0], [-0.5, 0.3], [0.0, 0.0]])
    mean = np.array([0.5, -0.25])
    ids = np.array([11, 15, 26, 43], dtype=np.int64)
    vectors = np.broadcast_to([[1.0, -2.0], [0.25, 1.5]], (4, 2, 2)).copy()
    point_dir = tmp_path / "points"
    descriptors = [
        batchstore.write_shard(
            point_dir,
            index,
            {
                "row_id": ids[start : start + 2],
                "points": points[start : start + 2],
                "vectors": vectors[start : start + 2],
            },
        )
        for index, start in enumerate((0, 2))
    ]
    point_file = batchstore.publish(
        point_dir,
        {
            "schema": batchstore.SCHEMA,
            "kind": "points",
            "point_space": "constrained",
            "parameter_names": ["x.1", "x.2"],
        },
        descriptors,
    )
    source = tmp_path / "model.stan"
    source.write_text(
        "data { vector[2] mu; } parameters { vector[2] x; } model { x ~ normal(mu,1); }"
    )
    data = tmp_path / "data.json"
    batchstore.atomic_json(data, {"mu": mean.tolist()})
    quantities = ["log_density", "gradient", "hessian", "hvp"]
    request = tmp_path / "evaluate.json"
    batchstore.atomic_json(
        request,
        {
            "stan": str(source),
            "data": str(data),
            "points": str(point_file),
            "quantities": quantities,
            "workers": 2,
        },
    )
    try:
        started = await server.evaluate_batch(str(request))
        job_id = started["job_id"]
        assert (await server.await_run(job_id, timeout=120))["status"] == "done"
        assert (await server.get_run_status(job_id))["status"] == "done"
        progress = await server.get_run_progress(job_id)
        assert progress["processed_shards"] == progress["total_shards"] == 2
        assert progress["processed_rows"] == 4
        evaluated = await server.get_batch_result(job_id)
        result = batchstore.load(evaluated["manifest"])
        shards = [batchstore.read_shard(result, descriptor) for descriptor in result["shards"]]
        actual = {name: np.concatenate([shard[name] for shard in shards]) for name in shards[0]}
        expected = {
            "row_id": ids,
            "points": points,
            "vectors": vectors,
            "status": np.zeros(4, dtype=np.int64),
            "log_density": -np.log(2 * np.pi) - np.sum((points - mean) ** 2, axis=1) / 2,
            "gradient": mean - points,
            "hessian": np.broadcast_to(-np.eye(2), (4, 2, 2)).copy(),
            "hvp": -vectors,
        }
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-13, atol=1e-13)
        expected["gradient"][1, 0] += 0.1
        external_dir = tmp_path / "external"
        external_descriptors = [
            batchstore.write_shard(
                external_dir, index, {name: values[start:stop] for name, values in expected.items()}
            )
            for index, (start, stop) in enumerate(((0, 3), (3, 4)))
        ]
        metadata = {
            key: result[key]
            for key in (
                "schema",
                "kind",
                "point_space",
                "parameter_names",
                "quantities",
                "target",
                "derivative_coordinates",
            )
        }
        external = batchstore.publish(external_dir, metadata, external_descriptors)
        compare_request = tmp_path / "compare.json"
        batchstore.atomic_json(
            compare_request,
            {
                "left": evaluated["manifest"],
                "right": str(external),
                "tolerances": {name: {"atol": 1e-12, "rtol": 1e-12} for name in quantities},
            },
        )
        compared = await server.compare_batches(str(compare_request))
        assert (await server.await_run(compared["job_id"], timeout=30))["status"] == "done"
        comparison = await server.get_batch_result(compared["job_id"])
        assert comparison["summary"]["verdict"] == "fail"
        summaries = comparison["summary"]["quantities"]
        assert summaries["gradient"]["outside_tolerance"] == 1
        assert summaries["gradient"]["worst"][0]["row_id"] == 15
        assert all(
            summaries[name]["verdict"] == "pass" for name in quantities if name != "gradient"
        )
        assert Path(comparison["manifest"]).is_file()
    finally:
        manager.shutdown()
