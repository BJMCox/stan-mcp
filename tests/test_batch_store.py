"""Artifact identity and bounded publication through the shared batch store."""

import json
from pathlib import Path

import numpy as np
import pytest

import batchstore
from errors import ValidationError


def artifact(directory: Path) -> Path:
    arrays = {
        "row_id": np.array([2, 7], dtype=np.int64),
        "points": np.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    descriptor = batchstore.write_shard(directory, 0, arrays)
    return batchstore.publish(
        directory,
        {"kind": "points", "point_space": "constrained", "parameter_names": ["v.1", "v.2"]},
        [descriptor],
    )


def test_snapshot_owns_consumed_arrays_and_preserves_shapes(tmp_path):
    source = artifact(tmp_path / "source")
    owned = batchstore.snapshot(source, tmp_path / "owned")
    source.unlink()
    manifest = batchstore.load(owned)
    values = batchstore.read_shard(manifest, manifest["shards"][0])
    np.testing.assert_array_equal(values["row_id"], [2, 7])
    np.testing.assert_array_equal(values["points"], [[1.0, 2.0], [3.0, 4.0]])
    assert manifest["count"] == 2


def test_changed_content_and_wrong_axes_cannot_relabel_points(tmp_path):
    path = artifact(tmp_path)
    manifest = json.loads(path.read_text())
    batchstore.load(path)
    manifest["count"] = 3
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError):
        batchstore.snapshot(path, tmp_path / "bad-count")
    manifest["count"] = 2
    manifest["parameter_names"] = ["x"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError):
        batchstore.load(path)
    manifest["parameter_names"] = ["v.1", "v.2"]
    path.write_text(json.dumps(manifest))
    binary = tmp_path / manifest["shards"][0]["path"]
    binary.write_bytes(binary.read_bytes()[:-1] + b"x")
    with pytest.raises(ValidationError):
        batchstore.load(path)
    with pytest.raises(ValidationError):
        batchstore.snapshot(path, tmp_path / "copy")


def test_sidecar_failure_cannot_publish_a_completed_shard(tmp_path, monkeypatch):
    arrays = {"row_id": np.array([0], dtype=np.int64), "points": np.array([[0.0]])}
    original = batchstore.atomic_json
    monkeypatch.setattr(
        batchstore, "atomic_json", lambda *args: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        batchstore.write_shard(tmp_path, 0, arrays, {"0": {"error": "domain"}})
    assert not (tmp_path / "00000000.safetensors").exists()
    monkeypatch.setattr(batchstore, "atomic_json", original)
    descriptor = batchstore.write_shard(tmp_path, 0, arrays, {"0": {"error": "domain"}})
    before = (tmp_path / descriptor["path"]).read_bytes()
    with pytest.raises(ValidationError):
        batchstore.write_shard(tmp_path, 0, arrays)
    assert (tmp_path / descriptor["path"]).read_bytes() == before


def test_row_identity_survives_shards_and_duplicate_ids_fail(tmp_path):
    path = artifact(tmp_path)
    manifest = json.loads(path.read_text())
    arrays = {"row_id": np.array([8], dtype=np.int64), "points": np.array([[5.0, 6.0]])}
    manifest["shards"].append(batchstore.write_shard(tmp_path, 1, arrays))
    manifest.pop("count")
    path.write_text(json.dumps(manifest))
    assert batchstore.load(path)["count"] == 3
    arrays["row_id"][0] = 7
    other = batchstore.write_shard(tmp_path, 2, arrays)
    manifest["shards"][1] = other
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError):
        batchstore.load(path)


def test_rechunk_preserves_tensor_axes_precision_and_errors(tmp_path):
    arrays = {
        "row_id": np.array([2, 7], dtype=np.int64),
        "points": np.array([[np.nextafter(1.0, 2.0)], [np.pi]]),
        "directions": np.arange(8, dtype=np.float64).reshape(2, 2, 2),
    }
    descriptor = batchstore.write_shard(tmp_path / "source", 0, arrays, {"7": {"error": "domain"}})
    path = batchstore.publish(
        tmp_path / "source",
        {"kind": "points", "point_space": "constrained", "parameter_names": ["x"]},
        [descriptor],
    )
    owned = batchstore.load(batchstore.rechunk(path, tmp_path / "split", 1))
    assert [entry["rows"] for entry in owned["shards"]] == [1, 1]
    for index, entry in enumerate(owned["shards"]):
        values = batchstore.read_shard(owned, entry)
        for name in arrays:
            np.testing.assert_array_equal(values[name], arrays[name][index : index + 1])
    assert "errors" not in owned["shards"][0]
    errors = batchstore.read_json(batchstore.resolve(owned, owned["shards"][1]["errors"]))
    assert errors == {"7": {"error": "domain"}}


def test_log_only_null_coordinates_and_hvp_vector_dtype(tmp_path):
    arrays = {
        "row_id": np.array([0], dtype=np.int64),
        "points": np.array([[0.0]]),
        "status": np.array([0], dtype=np.int64),
        "log_density": np.array([-1.0]),
        "vectors": np.ones((1, 1, 1), dtype=np.int64),
        "hvp": np.array([[[-1.0]]]),
    }
    descriptor = batchstore.write_shard(tmp_path, 0, arrays)
    metadata = {
        "kind": "results",
        "point_space": "constrained",
        "parameter_names": ["x"],
        "quantities": ["log_density"],
        "target": {"propto": False, "jacobian": False},
        "derivative_coordinates": None,
    }
    assert batchstore.load(batchstore.publish(tmp_path, metadata, [descriptor]))["count"] == 1
    metadata["quantities"] = ["hvp"]
    metadata["derivative_coordinates"] = {"kind": "stan_unconstrained", "names": ["x"]}
    with pytest.raises(ValidationError, match="HVP"):
        batchstore.load(batchstore.publish(tmp_path, metadata, [descriptor]))
