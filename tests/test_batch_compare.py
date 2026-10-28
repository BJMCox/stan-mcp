from pathlib import Path

import numpy as np
import pytest

import batchcompare
import batchstore
from errors import ValidationError

COORDINATES = {"kind": "stan_unconstrained", "names": ["x", "y"], "model_id": "analytic"}
QUANTITIES = ["log_density", "gradient", "hessian", "hvp"]


def _artifact(directory, arrays, splits=None, coordinates=None, quantities=None, kind="results"):
    count = len(arrays["row_id"])
    splits = splits or [count]
    descriptors = []
    start = 0
    for index, end in enumerate(splits):
        part = {key: value[start:end].copy() for key, value in arrays.items()}
        errors = {}
        if "status" in part:
            failed = part["status"] != 0
            for quantity in quantities or QUANTITIES:
                if quantity in part:
                    part[quantity][failed] = np.nan
            errors = {
                str(int(row)): {"error": "fixture evaluation failure"}
                for row in part["row_id"][failed]
            }
        descriptor = batchstore.write_shard(directory, index, part, errors or None)
        descriptors.append(descriptor)
        start = end
    metadata = {
        "schema": batchstore.SCHEMA,
        "kind": kind,
        "point_space": "constrained",
        "parameter_names": ["x", "y"] if arrays["points"].shape[1] == 2 else ["x"],
    }
    if kind == "results":
        metadata.update(
            target={"propto": False, "jacobian": False},
            quantities=quantities or QUANTITIES,
            derivative_coordinates=coordinates or COORDINATES,
        )
    return batchstore.publish(directory, metadata, descriptors)


def _gaussian(ids=(2, 5, 9, 14)):
    points = np.array([[0.2 * i, 0.1 - 0.3 * i] for i in ids])
    precision = np.array([[2.0, 0.5], [0.5, 3.0]])
    gradient = -points @ precision
    hessian = np.broadcast_to(-precision, (len(ids), 2, 2)).copy()
    vectors = np.broadcast_to([[1.0, -2.0], [0.25, 1.5]], (len(ids), 2, 2)).copy()
    return {
        "row_id": np.array(ids, dtype=np.int64),
        "points": points,
        "status": np.zeros(len(ids), dtype=np.int64),
        "log_density": np.sum(points * gradient, axis=1) / 2,
        "gradient": gradient,
        "hessian": hessian,
        "vectors": vectors,
        "hvp": vectors @ -precision,
    }


def _compare(tmp_path, left, right, **options):
    plan = batchcompare.prepare(
        {"left": str(left), "right": str(right), **options}, tmp_path / "owned"
    )
    return batchcompare.execute(plan, tmp_path / "output", lambda: False, lambda *_: None)


def _tolerances(quantities=QUANTITIES, atol=0.0, rtol=0.0):
    return {q: {"atol": atol, "rtol": rtol} for q in quantities}


def test_native_gaussian_aligns_ids_across_shard_boundaries(tmp_path):
    arrays = _gaussian()
    left = _artifact(tmp_path / "left", arrays, [1, 4])
    right = _artifact(tmp_path / "right", arrays, [3, 4])
    result = _compare(tmp_path, left, right, tolerances=_tolerances())
    assert result["summary"]["verdict"] == "pass"
    output = batchstore.load(result["manifest"])
    assert [s["rows"] for s in output["shards"]] == [3, 1]
    for quantity in QUANTITIES:
        summary = result["summary"]["quantities"][quantity]
        assert summary["comparable"] == 4
        assert summary["max_absolute_error"] == 0.0
    for descriptor in output["shards"]:
        shard = batchstore.read_shard(output, descriptor)
        assert np.all(shard["comparison_status_hvp"] == 0)
        assert np.all(shard["scaled_error_hessian"] == 0)


def test_comparison_rechunks_legal_inputs_to_fit_discrepancy_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(batchstore, "MAX_SHARD_BYTES", 32768)
    arrays = _gaussian(tuple(range(1, 81)))
    left = _artifact(tmp_path / "left", arrays)
    right = _artifact(tmp_path / "right", arrays)
    result = _compare(tmp_path, left, right, tolerances=_tolerances())
    assert result["summary"]["verdict"] == "pass"
    manifest = batchstore.load(result["manifest"])
    assert len(manifest["shards"]) == 2
    ids = np.concatenate([batchstore.read_shard(manifest, d)["row_id"] for d in manifest["shards"]])
    assert np.array_equal(ids, arrays["row_id"])
    assert all(
        (Path(manifest["_root"]) / d["path"]).stat().st_size <= 32768 for d in manifest["shards"]
    )


def test_report_only_and_symmetric_scale_at_extreme_magnitudes(tmp_path):
    left_values, right_values = _gaussian((1,)), _gaussian((1,))
    left_values["log_density"][0], right_values["log_density"][0] = 1.0e308, -1.0e308
    left = _artifact(tmp_path / "left", left_values)
    right = _artifact(tmp_path / "right", right_values)
    report = _compare(tmp_path / "report", left, right, quantities=["log_density"])
    item = report["summary"]["quantities"]["log_density"]
    assert report["summary"]["verdict"] == "report_only"
    assert item["max_scaled_error"] == 2.0
    assert item["max_absolute_error"] == "infinity"
    forward = _compare(
        tmp_path / "forward",
        left,
        right,
        quantities=["log_density"],
        tolerances=_tolerances(["log_density"], rtol=1.0),
    )
    reverse = _compare(
        tmp_path / "reverse",
        right,
        left,
        quantities=["log_density"],
        tolerances=_tolerances(["log_density"], rtol=1.0),
    )
    assert forward["summary"]["verdict"] == reverse["summary"]["verdict"] == "fail"
    assert forward["summary"]["quantities"]["log_density"]["max_scaled_error"] == 2.0


def test_fixed_offset_anchor_reports_constant_and_detects_drift(tmp_path):
    left_values, right_values = _gaussian(), _gaussian()
    left_values["log_density"] += 7.0
    left_values["log_density"][-1] += 0.25
    left = _artifact(tmp_path / "left", left_values, [1, 4])
    right = _artifact(tmp_path / "right", right_values, [2, 4])
    result = _compare(
        tmp_path,
        left,
        right,
        quantities=["log_density"],
        additive_constant=True,
        tolerances=_tolerances(["log_density"], atol=1e-12),
    )
    assert result["summary"]["offset"] == {"value": 7.0, "row_id": 2}
    item = result["summary"]["quantities"]["log_density"]
    assert item["verdict"] == "fail"
    assert item["outside_tolerance"] == 1
    assert item["max_absolute_error"] == pytest.approx(0.25)
    assert item["worst"][0]["row_id"] == 14


@pytest.mark.parametrize("mismatch", ["coordinates", "points", "vectors"])
def test_conventions_are_checked_before_comparison(tmp_path, mismatch):
    arrays = _gaussian()
    left = _artifact(tmp_path / "left", arrays)
    altered = {key: value.copy() for key, value in arrays.items()}
    coordinates = dict(COORDINATES)
    if mismatch == "coordinates":
        coordinates["names"] = ["y", "x"]
    elif mismatch == "points":
        altered["points"][0, 0] += 1.0
    else:
        altered["vectors"][0, 0, 0] += 1.0
    right = _artifact(tmp_path / "right", altered, coordinates=coordinates)
    with pytest.raises(ValidationError):
        batchcompare.prepare({"left": str(left), "right": str(right)}, tmp_path / "owned")
    assert not (tmp_path / "owned" / "comparison-checkpoint.json").exists()


def test_incomplete_failed_and_nonfinite_rows_cannot_pass(tmp_path):
    left_values, right_values = _gaussian((2, 5, 9, 14)), _gaussian((2, 5, 9, 20))
    left_values["status"][1] = 1
    left_values["gradient"][2] = np.nan
    left = _artifact(tmp_path / "left", left_values)
    right = _artifact(tmp_path / "right", right_values)
    result = _compare(
        tmp_path, left, right, quantities=["gradient"], tolerances=_tolerances(["gradient"])
    )
    item = result["summary"]["quantities"]["gradient"]
    assert item["verdict"] == "unknown"
    assert item["requested"] == 5
    assert item["comparable"] == 1
    assert item["failed"] == 2
    assert item["inconclusive"] == 2


def _nonlinear_fd(directory, steps, gradient_bias=0.0, target_constant=0.0):
    # Scalar constrained x=exp(u), target=-x²/2; derivatives are in external u.
    u, direction = 0.2, 2.3
    points = [np.exp(u)]
    stencil = {"row_id": np.array([0], dtype=np.int64), "points": np.array([[points[0]]])}
    for quantity in ("gradient", "hessian", "hvp"):
        tensor = np.empty(
            (1, len(steps), 1, 2) if quantity == "gradient" else (1, len(steps), 1, 1, 4),
            dtype=np.int64,
        )
        for step_index, h in enumerate(steps):
            displacements = (
                [-h, h]
                if quantity == "gradient"
                else (
                    [2 * h, 0.0, 0.0, -2 * h]
                    if quantity == "hessian"
                    else [
                        h * (1 + direction),
                        h * (1 - direction),
                        h * (-1 + direction),
                        -h * (1 + direction),
                    ]
                )
            )
            ids = []
            for displacement in displacements:
                ids.append(len(points))
                points.append(np.exp(u + displacement))
            tensor[(0, step_index)] = ids
        stencil[f"fd_{quantity}_rows"] = tensor
    points_array = np.array(points).reshape(-1, 1)
    left_arrays = {
        "row_id": np.arange(len(points), dtype=np.int64),
        "points": points_array,
        "status": np.zeros(len(points), dtype=np.int64),
        "log_density": -(points_array[:, 0] ** 2) / 2 + target_constant,
    }
    coordinates = {"kind": "external_log", "names": ["u"], "model_id": "nonlinear"}
    right_arrays = {
        "row_id": np.array([0], dtype=np.int64),
        "points": points_array[:1],
        "status": np.array([0], dtype=np.int64),
        "log_density": left_arrays["log_density"][:1],
        "gradient": np.array([[-np.exp(2 * u) + gradient_bias]]),
        "hessian": np.array([[[-2 * np.exp(2 * u)]]]),
        "hvp": np.array([[[-2 * direction * np.exp(2 * u)]]]),
        "vectors": np.array([[[direction]]]),
    }
    left = _artifact(directory / "left", left_arrays, [4, len(points)], quantities=["log_density"])
    right = _artifact(directory / "right", right_arrays, coordinates=coordinates)
    stencils = _artifact(directory / "stencils", stencil, kind="points")
    return (
        left,
        right,
        {"steps": steps, "stencils": str(stencils), "derivative_coordinates": coordinates},
    )


def test_fixed_central_stencils_validate_nonlinear_external_derivatives(tmp_path):
    left, right, fd = _nonlinear_fd(tmp_path, [0.002, 0.001, 0.0005])
    result = _compare(
        tmp_path, left, right, finite_difference=fd, tolerances=_tolerances(atol=0.001)
    )
    assert result["summary"]["verdict"] == "pass"
    for quantity in ("gradient", "hessian", "hvp"):
        item = result["summary"]["quantities"][quantity]
        assert item["estimated"] == item["comparable"] == 1
        assert item["max_absolute_error"] < 0.001
    output = batchstore.load(result["manifest"])
    arrays = batchstore.read_shard(output, output["shards"][0])
    assert np.all(arrays["stability_hvp"] > 0)


def test_finite_difference_instability_is_inconclusive(tmp_path):
    left, right, fd = _nonlinear_fd(tmp_path, [0.5, 0.25])
    result = _compare(
        tmp_path, left, right, finite_difference=fd, tolerances=_tolerances(atol=0.001)
    )
    assert result["summary"]["verdict"] == "unknown"
    for quantity in ("gradient", "hessian", "hvp"):
        item = result["summary"]["quantities"][quantity]
        assert item["estimated"] == item["inconclusive"] == 1
        assert item["comparable"] == 0


def test_finite_difference_detects_wrong_derivative(tmp_path):
    left, right, fd = _nonlinear_fd(tmp_path, [0.002, 0.001], gradient_bias=0.02)
    result = _compare(
        tmp_path, left, right, finite_difference=fd, tolerances=_tolerances(atol=0.001)
    )
    assert result["summary"]["verdict"] == "fail"
    assert result["summary"]["quantities"]["gradient"]["outside_tolerance"] == 1


def _cubic_fd(directory, derivative):
    points = np.array([[1.0], [0.8], [1.2], [0.9], [1.1]])
    left = _artifact(
        directory / "left",
        {
            "row_id": np.arange(5, dtype=np.int64),
            "points": points,
            "status": np.zeros(5, dtype=np.int64),
            "log_density": points[:, 0] ** 3,
        },
        quantities=["log_density"],
    )
    coordinates = {"kind": "external_identity", "names": ["u"], "model_id": "cubic"}
    right = _artifact(
        directory / "right",
        {
            "row_id": np.array([0], dtype=np.int64),
            "points": points[:1],
            "status": np.array([0], dtype=np.int64),
            "gradient": np.array([[derivative]]),
        },
        quantities=["gradient"],
        coordinates=coordinates,
    )
    stencils = _artifact(
        directory / "stencils",
        {
            "row_id": np.array([0], dtype=np.int64),
            "points": points[:1],
            "fd_gradient_rows": np.array([[[[1, 2]], [[3, 4]]]], dtype=np.int64),
        },
        kind="points",
    )
    return (
        left,
        right,
        {"steps": [0.2, 0.1], "stencils": str(stencils), "derivative_coordinates": coordinates},
    )


def test_finite_difference_uncertainty_overlap_is_inconclusive(tmp_path):
    left, right, fd = _cubic_fd(tmp_path, 3.0)
    result = _compare(
        tmp_path,
        left,
        right,
        quantities=["gradient"],
        finite_difference=fd,
        tolerances=_tolerances(["gradient"], atol=0.035),
    )
    item = result["summary"]["quantities"]["gradient"]
    assert item["comparable"] == 0
    assert item["inconclusive"] == 1
    assert item["max_absolute_error"] == pytest.approx(0.01)
    assert item["verdict"] == "unknown"
    manifest = batchstore.load(result["manifest"])
    shard = batchstore.read_shard(manifest, manifest["shards"][0])
    stability = shard["stability_gradient"][0, 0]
    assert stability == pytest.approx(0.03)
    assert stability + item["max_absolute_error"] > 0.035


def test_finite_difference_mismatch_beyond_uncertainty_fails(tmp_path):
    left, right, fd = _cubic_fd(tmp_path, 3.1)
    result = _compare(
        tmp_path,
        left,
        right,
        finite_difference=fd,
        tolerances=_tolerances(["gradient"], atol=0.035),
    )
    item = result["summary"]["quantities"]["gradient"]
    assert item["comparable"] == 1
    assert item["outside_tolerance"] == 1
    assert item["max_absolute_error"] == pytest.approx(0.09)
    assert item["verdict"] == "fail"


def test_finite_difference_without_tolerance_reports_estimate_as_inconclusive(tmp_path):
    left, right, fd = _nonlinear_fd(tmp_path, [0.002, 0.001])
    fd["quantities"] = ["gradient"]
    result = _compare(tmp_path, left, right, quantities=["gradient"], finite_difference=fd)
    item = result["summary"]["quantities"]["gradient"]
    assert item["verdict"] == "unknown"
    assert item["estimated"] == item["inconclusive"] == 1
    assert item["max_absolute_error"] is not None


def test_finite_difference_refuses_supplied_weights(tmp_path):
    left, right, fd = _nonlinear_fd(tmp_path, [0.002, 0.001])
    fd["weights"] = [0.0, 0.0]
    with pytest.raises(ValidationError):
        batchcompare.prepare(
            {"left": str(left), "right": str(right), "finite_difference": fd}, tmp_path / "owned"
        )


def test_finite_difference_roundoff_prevents_false_zero_agreement(tmp_path):
    left, right, fd = _nonlinear_fd(
        tmp_path, [0.002, 0.001], gradient_bias=np.exp(0.4), target_constant=1.0e30
    )
    fd["quantities"] = ["gradient"]
    result = _compare(
        tmp_path,
        left,
        right,
        quantities=["gradient"],
        finite_difference=fd,
        tolerances=_tolerances(["gradient"], atol=0.001),
    )
    item = result["summary"]["quantities"]["gradient"]
    assert item["max_absolute_error"] == item["max_empirical_stability"] == 0.0
    assert item["max_roundoff_estimate"] > 0.001
    assert item["verdict"] == "unknown"
    assert item["inconclusive"] == 1


def test_failed_nonfinite_canonical_points_remain_visible_in_report(tmp_path):
    arrays = _gaussian()
    arrays["points"][0] = np.nan
    arrays["status"][0] = 1
    left = _artifact(tmp_path / "left", arrays)
    right = _artifact(tmp_path / "right", arrays)
    result = _compare(tmp_path, left, right, tolerances=_tolerances())
    assert result["summary"]["verdict"] == "unknown"
    for quantity in QUANTITIES:
        item = result["summary"]["quantities"][quantity]
        assert item["failed"] == 1
        assert item["comparable"] == 3
        assert item["failed_rows"] == [2]


def test_cancellation_before_first_shard_does_not_publish_empty_artifact(tmp_path):
    left = _artifact(tmp_path / "left", _gaussian())
    right = _artifact(tmp_path / "right", _gaussian())
    plan = batchcompare.prepare({"left": str(left), "right": str(right)}, tmp_path / "owned")
    result = batchcompare.execute(plan, tmp_path / "output", lambda: True, lambda *_: None)
    assert result["cancelled"]
    assert result["manifest"] is None
    assert result["summary"]["verdict"] == "unknown"
    assert not (tmp_path / "output" / "manifest.json").exists()


def test_cancel_resume_keeps_shards_and_offset_anchor(tmp_path):
    left_values, right_values = _gaussian(), _gaussian()
    left_values["log_density"] += 3.0
    left = _artifact(tmp_path / "left", left_values)
    right = _artifact(tmp_path / "right", right_values, [2, 4])
    plan = batchcompare.prepare(
        {
            "left": str(left),
            "right": str(right),
            "quantities": ["log_density"],
            "additive_constant": True,
            "tolerances": _tolerances(["log_density"], atol=1e-12),
        },
        tmp_path / "owned",
    )
    progress = []
    partial = batchcompare.execute(
        plan, tmp_path / "output", lambda: bool(progress), lambda *p: progress.append(p)
    )
    assert partial["cancelled"]
    assert partial["summary"]["verdict"] == "unknown"
    first_path = Path(partial["manifest"]).parent / partial["completed"][0]["path"]
    before = first_path.read_bytes()
    checkpoint = batchstore.read_json(tmp_path / "output" / "comparison-checkpoint.json")
    resumed = batchcompare.execute(
        {**plan, "completed": checkpoint["completed"]},
        tmp_path / "output",
        lambda: False,
        lambda *_: None,
    )
    assert resumed["summary"]["verdict"] == "pass"
    assert resumed["summary"]["offset"] == {"value": 3.0, "row_id": 2}
    assert first_path.read_bytes() == before
    assert len(resumed["completed"]) == 2
