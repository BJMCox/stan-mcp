"""Numerical and ownership contracts for batched native evaluations."""

import os
from pathlib import Path

import bridgestan
import numpy as np
import pytest
from numpy.typing import DTypeLike

import batcheval
import batchstore
import compile as compilation
import density
import errors
import toolchain


class Gaussian:
    def param_names(self):
        return ["x.1", "x.2"]

    def param_unc_names(self):
        return self.param_names()

    def model_info(self):
        return "BRIDGESTAN_AD_HESSIAN=true"

    def param_unconstrain(self, point):
        return point

    def param_constrain(self, theta):
        return theta

    def log_density_hessian(self, theta, *, propto, jacobian):
        assert not propto and not jacobian
        if theta[0] == 999:
            raise RuntimeError("intentional point failure")
        hessian = np.array([[-2.0, 0.5], [0.5, -3.0]])
        return -np.log(2 * np.pi) + theta @ hessian @ theta / 2, hessian @ theta, hessian

    def log_density_gradient(self, theta, **flags):
        return self.log_density_hessian(theta, **flags)[:2]

    def log_density(self, theta, **flags):
        return self.log_density_hessian(theta, **flags)[0]

    def log_density_hessian_vector_product(self, theta, vector, **flags):
        lp, _, hessian = self.log_density_hessian(theta, **flags)
        return lp, hessian @ vector


def _points(
    tmp_path,
    points,
    *,
    names=None,
    space="constrained",
    vectors=None,
    split=None,
    dtype: DTypeLike = np.float64,
    vector_dtype: DTypeLike = np.float64,
):
    points = np.asarray(points, dtype=dtype)
    directory = tmp_path / "points"
    descriptors = []
    for start in range(0, len(points), split or len(points)):
        end = start + (split or len(points))
        arrays = {
            "row_id": np.arange(len(points), dtype=np.int64)[start:end] * 10 + 7,
            "points": points[start:end],
        }
        if vectors is not None:
            arrays["vectors"] = np.asarray(vectors, dtype=vector_dtype)[start:end]
        descriptors.append(batchstore.write_shard(directory, len(descriptors), arrays))
    return batchstore.publish(
        directory,
        {
            "schema": "stan-mcp.batch.v1",
            "kind": "points",
            "point_space": space,
            "parameter_names": names or ["x.1", "x.2"],
        },
        descriptors,
    )


@pytest.fixture
def fake_native(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    binary = tmp_path / "native.so"
    binary.write_bytes(b"immutable native library")
    monkeypatch.setattr(compilation, "compiled_so", lambda *args, **kwargs: binary)
    loads = []

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return Gaussian()

    monkeypatch.setattr(density, "_new_model", load)
    monkeypatch.setattr(density, "get_model", lambda *args, **kwargs: pytest.fail("public wrapper"))
    monkeypatch.setattr(batcheval, "_CONTEXT", None)
    return loads


def _evaluate(plan, tmp_path):
    descriptors = [
        batcheval.execute_shard(plan, index, tmp_path / "output")
        for index in range(plan["shard_count"])
    ]
    result = batchstore.load(
        batchstore.publish(tmp_path / "output", plan["output_metadata"], descriptors)
    )
    return [batchstore.read_shard(result, descriptor) for descriptor in result["shards"]]


def test_constrained_names_axes_failures_and_worker_reuse(tmp_path, fake_native):
    points = [[2, 1], [1, 999], [4, -1]]
    vectors = np.array([[[1, 0], [0, 2]]] * 3)
    manifest = _points(tmp_path, points, names=["x.2", "x.1"], vectors=vectors, split=2)
    plan = batcheval.prepare(
        {
            "stan": "model {}",
            "points": str(manifest),
            "quantities": ["log_density", "gradient", "hessian", "hvp"],
        },
        tmp_path / "owned",
    )
    results = _evaluate(plan, tmp_path)
    assert len(fake_native) == 2  # coordinator preflight, then one owned worker context
    actual = {name: np.concatenate([shard[name] for shard in results]) for name in results[0]}
    np.testing.assert_array_equal(actual["row_id"], [7, 17, 27])
    np.testing.assert_array_equal(actual["status"], [0, 1, 0])
    np.testing.assert_array_equal(actual["points"], [[1, 2], [999, 1], [-1, 4]])
    hessian = np.array([[-2, 0.5], [0.5, -3]])
    np.testing.assert_allclose(actual["gradient"][[0, 2]], np.array([[1, 2], [-1, 4]]) @ hessian)
    np.testing.assert_allclose(
        actual["hvp"][[0, 2]], np.broadcast_to(vectors[0] @ hessian, (2, 2, 2))
    )
    assert np.isnan(actual["hessian"][1]).all()
    np.testing.assert_allclose(actual["hessian"][[0, 2]], np.broadcast_to(hessian, (2, 2, 2)))
    assert plan["source_manifest"]["make_args"] == ["BRIDGESTAN_AD_HESSIAN=true"]


def test_hvp_without_dense_hessian_and_unconstrained_identity(tmp_path, fake_native, monkeypatch):
    manifest = _points(tmp_path, [[1, 2]], space="unconstrained", vectors=[[[2, -1]]])
    plan = batcheval.prepare(
        {"stan": "model {}", "points": str(manifest), "quantities": ["hvp"]}, tmp_path / "owned"
    )
    calls = []
    original = Gaussian.log_density_hessian_vector_product

    def hvp(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Gaussian, "log_density_hessian_vector_product", hvp)
    result = _evaluate(plan, tmp_path)[0]
    np.testing.assert_allclose(result["hvp"], [[[-4.5, 4]]])
    np.testing.assert_array_equal(result["points"], [[1, 2]])
    assert len(calls) == 1
    assert plan["derivative_coordinates"]["names"] == ["x.1", "x.2"]


def test_normalizing_inverse_cannot_change_constrained_points(tmp_path, fake_native, monkeypatch):
    manifest = _points(tmp_path, [[0.5, 0], [1, 0]])
    monkeypatch.setattr(
        Gaussian, "param_unconstrain", lambda self, point: point / np.linalg.norm(point)
    )
    plan = batcheval.prepare({"stan": "model {}", "points": str(manifest)}, tmp_path / "owned")
    result = _evaluate(plan, tmp_path)[0]
    np.testing.assert_array_equal(result["status"], [1, 0])
    np.testing.assert_array_equal(result["points"], [[0.5, 0], [1, 0]])
    assert np.isnan(result["log_density"][0])


def test_requested_curvature_rechunks_before_evaluation(tmp_path, fake_native, monkeypatch):
    manifest = _points(tmp_path, np.zeros((5, 2)))
    monkeypatch.setattr(batchstore, "MAX_SHARD_BYTES", 16512)
    plan = batcheval.prepare(
        {"stan": "model {}", "points": str(manifest), "quantities": ["hessian"]}, tmp_path / "owned"
    )
    assert [len(shard["row_id"]) for shard in _evaluate(plan, tmp_path)] == [2, 2, 1]


def test_single_curvature_row_exceeding_limit_is_rejected(tmp_path, fake_native, monkeypatch):
    manifest = _points(tmp_path, [[1, 2]])
    monkeypatch.setattr(batchstore, "MAX_SHARD_BYTES", 16432)
    with pytest.raises(errors.ValidationError):
        batcheval.prepare(
            {"stan": "model {}", "points": str(manifest), "quantities": ["hessian"]},
            tmp_path / "owned",
        )


def test_structural_dimensions_rejected_before_target_evaluation(tmp_path, fake_native):
    manifest = _points(tmp_path, [[1, 2]], vectors=[[[1, 2, 3]]])
    with pytest.raises(errors.ValidationError):
        batcheval.prepare(
            {"stan": "model {}", "points": str(manifest), "quantities": ["hvp"]}, tmp_path / "owned"
        )


def test_integer_hvp_directions_are_rejected_before_evaluation(tmp_path, fake_native):
    manifest = _points(tmp_path, [[1, 2]], vectors=[[[1, 2]]], vector_dtype=np.int64)
    with pytest.raises(errors.ValidationError):
        batcheval.prepare(
            {"stan": "model {}", "points": str(manifest), "quantities": ["hvp"]}, tmp_path / "owned"
        )


def test_failure_records_and_chunk_rows_remain_bounded(tmp_path, fake_native, monkeypatch):
    manifest = _points(tmp_path, np.zeros((4100, 2)))
    plan = batcheval.prepare(
        {"stan": "model {}", "points": str(manifest), "chunk_rows": 100000}, tmp_path / "owned"
    )
    owned = batchstore.load(plan["input_manifest"])
    assert [descriptor["rows"] for descriptor in owned["shards"]] == [4096, 4]

    def fail(*args, **kwargs):
        raise RuntimeError("😀" * 10000)

    monkeypatch.setattr(Gaussian, "log_density", fail)
    batcheval.execute_shard(plan, 1, tmp_path / "output")
    records = batchstore.read_json(tmp_path / "output" / "00000001.errors.json")
    assert len(records) == 4
    assert all(record["truncated"] and len(record["error"]) <= 2048 for record in records.values())


def test_ad_build_identity_and_arguments_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(toolchain, "ensure_bridgestan", lambda: "installed")
    calls = []

    def build(path, **kwargs):
        calls.append(kwargs)
        output = Path(path).with_name(Path(path).stem + "_model.so")
        output.write_bytes(b"native")
        return output

    monkeypatch.setattr(bridgestan, "compile_model", build)
    environment = dict(os.environ)
    ordinary = compilation.resolve_source("model {}")
    ad = compilation.with_make_args(ordinary, ["BRIDGESTAN_AD_HESSIAN=true"])
    assert compilation.compiled_so(ad) != compilation.compiled_so(ordinary)
    assert compilation.compiled_so(ad).is_file()
    assert calls == [
        {"stanc_args": [], "make_args": ["BRIDGESTAN_AD_HESSIAN=true"]},
        {"stanc_args": []},
    ]
    assert dict(os.environ) == environment


def test_unapproved_finite_difference_library_is_refused(tmp_path, fake_native, monkeypatch):
    manifest = _points(tmp_path, [[1, 2]])
    monkeypatch.setattr(Gaussian, "model_info", lambda self: "BRIDGESTAN_AD_HESSIAN=false")
    with pytest.raises(errors.ValidationError):
        batcheval.prepare(
            {"stan": "model {}", "points": str(manifest), "quantities": ["hessian"]},
            tmp_path / "owned",
        )


def test_resume_verifies_native_binary_and_build_inputs(tmp_path, fake_native, monkeypatch):
    manifest = _points(tmp_path, [[1, 2]])
    plan = batcheval.prepare({"stan": "model {}", "points": str(manifest)}, tmp_path / "owned")
    batcheval.validate_plan(plan)
    with monkeypatch.context() as changed:
        changed.setenv("STAN_NO_RANGE_CHECKS", "true")
        with pytest.raises(errors.ValidationError):
            batcheval.validate_plan(plan)
    Path(plan["so_path"]).write_bytes(b"different native build")
    with pytest.raises(errors.ValidationError):
        batcheval.validate_plan(plan)


@pytest.mark.skipif(os.environ.get("STAN_MCP_LIVE") != "1", reason="requires native compiler")
def test_live_f32_unit_vector_rounding_preserves_native_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    points = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [0.5, 0], [1, 0]], dtype=np.float32)
    manifest = _points(tmp_path, points, dtype=np.float32)
    plan = batcheval.prepare(
        {
            "stan": "parameters { unit_vector[2] x; } model { target += x[1]; }",
            "points": str(manifest),
        },
        tmp_path / "owned",
    )
    result = _evaluate(plan, tmp_path)[0]
    np.testing.assert_array_equal(result["status"], [1, 1, 0])
    np.testing.assert_array_equal(result["points"], points.astype(np.float64))
    assert np.isnan(result["log_density"][:2]).all()
    assert result["log_density"][2] == pytest.approx(1)


@pytest.mark.skipif(os.environ.get("STAN_MCP_LIVE") != "1", reason="requires native compiler")
def test_live_unit_vector_points_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manifest = _points(tmp_path, [[0.5, 0], [1, 0]])
    plan = batcheval.prepare(
        {
            "stan": "parameters { unit_vector[2] x; } model { target += x[1]; }",
            "points": str(manifest),
        },
        tmp_path / "owned",
    )
    result = _evaluate(plan, tmp_path)[0]
    np.testing.assert_array_equal(result["status"], [1, 0])
    np.testing.assert_array_equal(result["points"], [[0.5, 0], [1, 0]])
    assert result["log_density"][1] == pytest.approx(1)


@pytest.mark.skipif(os.environ.get("STAN_MCP_LIVE") != "1", reason="requires native compiler")
def test_live_gaussian_gradient_and_native_hvp(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    manifest = _points(
        tmp_path, [[1, 2], [-2, 1]], space="unconstrained", vectors=[[[2, -3]], [[1, 4]]]
    )
    plan = batcheval.prepare(
        {
            "stan": "parameters { vector[2] x; } model { x[1] ~ normal(0,1); x[2] ~ normal(0,2); }",
            "points": str(manifest),
            "quantities": ["log_density", "gradient", "hvp"],
        },
        tmp_path / "owned",
    )
    result = _evaluate(plan, tmp_path)[0]
    np.testing.assert_allclose(result["log_density"], -np.log(4 * np.pi) - np.array([1, 2.125]))
    np.testing.assert_allclose(result["gradient"], [[-1, -0.5], [2, -0.25]], rtol=1e-14)
    np.testing.assert_allclose(result["hvp"], [[[-2, 0.75]], [[-1, -1]]], rtol=1e-14)


@pytest.mark.skipif(os.environ.get("STAN_MCP_LIVE") != "1", reason="requires native compiler")
def test_live_nonlinear_constraint_native_derivatives(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    points = np.array([0.3, 2.1], dtype=np.float32).astype(np.float64)
    manifest = _points(
        tmp_path, points[:, None], names=["x"], vectors=[[[2.0]], [[-3.0]]], dtype=np.float32
    )
    request = {
        "stan": "parameters { real<lower=0> x; } model { x ~ exponential(2); }",
        "points": str(manifest),
        "quantities": ["log_density", "gradient", "hessian", "hvp"],
    }
    plan = batcheval.prepare(request, tmp_path / "owned")
    result = _evaluate(plan, tmp_path)[0]
    np.testing.assert_array_equal(result["points"], points[:, None])
    np.testing.assert_allclose(result["log_density"], np.log(2) - 2 * points)
    np.testing.assert_allclose(result["gradient"], -2 * points[:, None], rtol=1e-14)
    np.testing.assert_allclose(result["hessian"], -2 * points[:, None, None], rtol=1e-14)
    np.testing.assert_allclose(result["hvp"], (-2 * points * [2, -3])[:, None, None], rtol=1e-14)
    request["target"] = {"propto": True, "jacobian": True}
    second = batcheval.prepare(request, tmp_path / "jacobian")
    result = _evaluate(second, tmp_path / "jacobian_result")[0]
    np.testing.assert_allclose(result["log_density"], -2 * points + np.log(points))
    np.testing.assert_allclose(result["gradient"], (1 - 2 * points)[:, None], rtol=1e-14)
