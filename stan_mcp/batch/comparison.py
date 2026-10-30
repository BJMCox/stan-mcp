"""Stream comparisons of immutable, explicitly declared numerical references.

Native derivatives require identical coordinate declarations. Opt-in finite
differences use caller-provided constrained perturbations and fixed central
stencils; no foreign coordinate transform or supplied weights are executed.
"""

from bisect import bisect_left
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from stan_mcp.batch import store as batchstore
from stan_mcp.errors import ValidationError

_QUANTITIES = ("log_density", "gradient", "hessian", "hvp")
_WORST_LIMIT = 8


class _Rows:
    """A shard range index with at most one loaded shard per input artifact."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.ends: list[int] = []
        self.indices: list[int] = []
        self.count = 0
        self.index = -1
        self.arrays: dict[str, np.ndarray] = {}
        for index, descriptor in enumerate(manifest["shards"]):
            self.count += descriptor["rows"]
            if descriptor["rows"]:
                self.ends.append(descriptor["last_id"])
                self.indices.append(index)

    def get(self, row_id: int) -> tuple[dict[str, np.ndarray], int] | None:
        position = bisect_left(self.ends, row_id)
        if position == len(self.ends):
            return None
        index = self.indices[position]
        if index != self.index:
            self.arrays = batchstore.read_shard(self.manifest, self.manifest["shards"][index])
            self.index = index
        offset = int(np.searchsorted(self.arrays["row_id"], row_id))
        ids = self.arrays["row_id"]
        if offset == len(ids) or ids[offset] != row_id:
            return None
        return self.arrays, offset


def _coordinates(manifest: dict[str, Any]) -> dict[str, Any]:
    coordinates = manifest.get("derivative_coordinates")
    if (
        not isinstance(coordinates, dict)
        or not isinstance(coordinates.get("kind"), str)
        or not coordinates["kind"]
        or not isinstance(coordinates.get("model_id"), str)
        or not coordinates["model_id"]
        or not isinstance(coordinates.get("names"), list)
        or any(not isinstance(name, str) for name in coordinates["names"])
        or len(set(coordinates["names"])) != len(coordinates["names"])
    ):
        raise ValidationError("Derivatives require explicit kind, names and model_id coordinates")
    return coordinates


def _tolerances(request: dict[str, Any], quantities: list[str]) -> dict[str, dict[str, float]]:
    supplied = request.get("tolerances", {})
    if not isinstance(supplied, dict) or set(supplied) - set(quantities):
        raise ValidationError("Tolerances must name requested quantities")
    result: dict[str, dict[str, float]] = {}
    for quantity, tolerance in supplied.items():
        if not isinstance(tolerance, dict) or set(tolerance) != {"atol", "rtol"}:
            raise ValidationError("Each tolerance must contain explicit atol and rtol")
        values = {}
        for key in ("atol", "rtol"):
            value = tolerance[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError("Tolerances must be finite nonnegative numbers")
            values[key] = float(value)
            if not np.isfinite(values[key]) or values[key] < 0:
                raise ValidationError("Tolerances must be finite nonnegative numbers")
        result[quantity] = values
    return result


def _fd_plan(
    request: dict[str, Any], directory: Path, right: dict[str, Any], quantities: list[str]
) -> dict[str, Any] | None:
    supplied = request.get("finite_difference")
    if supplied is None:
        return None
    if not isinstance(supplied, dict):
        raise ValidationError("finite_difference must declare steps, stencils and coordinates")
    if set(supplied) - {"steps", "stencils", "derivative_coordinates", "quantities"}:
        raise ValidationError(
            "Finite differences use fixed central stencils; supplied weights are unsupported"
        )
    steps = supplied.get("steps")
    if (
        not isinstance(steps, list)
        or len(steps) < 2
        or any(isinstance(h, bool) or not isinstance(h, (int, float)) for h in steps)
    ):
        raise ValidationError("Finite differences require at least two positive descending steps")
    steps = [float(h) for h in steps]
    if any(not np.isfinite(h) or h <= 0 for h in steps) or any(
        first <= second for first, second in zip(steps, steps[1:], strict=False)
    ):
        raise ValidationError("Finite differences require positive strictly descending steps")
    if supplied.get("derivative_coordinates") != _coordinates(right):
        raise ValidationError(
            "Finite-difference coordinates must equal right derivative coordinates"
        )
    selected = supplied.get("quantities", [q for q in quantities if q != "log_density"])
    if (
        not isinstance(selected, list)
        or not selected
        or len(set(selected)) != len(selected)
        or set(selected) - (set(quantities) - {"log_density"})
    ):
        raise ValidationError("Finite-difference quantities must name requested derivatives")
    path = batchstore.snapshot(supplied["stencils"], directory / "stencils")
    stencil = batchstore.load(path)
    if stencil["kind"] != "points" or stencil["point_space"] != "constrained":
        raise ValidationError(
            "Finite-difference stencils require canonical constrained base points"
        )
    if stencil["parameter_names"] != right["parameter_names"]:
        raise ValidationError("Stencil point names must equal right point names")
    return {"steps": steps, "quantities": selected, "manifest": str(path)}


def prepare(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Snapshot and validate comparison inputs before publishing discrepancies."""
    directory.mkdir(parents=True, exist_ok=True)
    left_path = batchstore.snapshot(request["left"], directory / "left")
    right_path = batchstore.snapshot(request["right"], directory / "right")
    left, right = batchstore.load(left_path), batchstore.load(right_path)
    for manifest in (left, right):
        if manifest["kind"] != "results" or manifest["point_space"] != "constrained":
            raise ValidationError("Comparison requires results with canonical constrained points")
        if manifest.get("target") != {"propto": False, "jacobian": False}:
            raise ValidationError("Comparison requires explicit propto=false and jacobian=false")
    if left["parameter_names"] != right["parameter_names"]:
        raise ValidationError("Comparison constrained parameter names and order must match")
    quantities = request.get("quantities", right.get("quantities", []))
    if (
        not isinstance(quantities, list)
        or not quantities
        or len(set(quantities)) != len(quantities)
        or set(quantities) - set(_QUANTITIES)
        or set(quantities) - set(right.get("quantities", []))
    ):
        raise ValidationError("Comparison quantities must be present in right results")
    tolerances = _tolerances(request, quantities)
    fd = _fd_plan(request, directory, right, quantities)
    fd_quantities = set(fd["quantities"]) if fd else set()
    native = set(quantities) - fd_quantities
    if native - set(left.get("quantities", [])):
        raise ValidationError("Native comparison quantities must be present in left results")
    if fd and "log_density" not in left.get("quantities", []):
        raise ValidationError("Finite differences require left log_density results")
    if native - {"log_density"} and _coordinates(left) != _coordinates(right):
        raise ValidationError("Native derivative coordinate declarations must match exactly")
    offset_mode = request.get("additive_constant", False)
    if not isinstance(offset_mode, bool) or (offset_mode and "log_density" not in quantities):
        raise ValidationError("additive_constant must be boolean and requires log_density")
    left_rows = _Rows(left)
    stencil_rows = _Rows(batchstore.load(fd["manifest"])) if fd else None
    matched = 0
    output_row_bytes = 0
    largest_shard_rows = 0
    for descriptor in right["shards"]:
        arrays = batchstore.read_shard(right, descriptor)
        largest_shard_rows = max(largest_shard_rows, descriptor["rows"])
        row_bytes = arrays["points"].shape[1] * arrays["points"].dtype.itemsize + 16
        if "hvp" in quantities:
            row_bytes += arrays["vectors"][0].nbytes
        for quantity in quantities:
            row_bytes += arrays[quantity][0].size * (32 if quantity in fd_quantities else 16) + 8
        output_row_bytes = max(output_row_bytes, row_bytes)
        for index, row_id in enumerate(arrays["row_id"]):
            pair = left_rows.get(int(row_id))
            if pair is not None:
                source, position = pair
                points_left, points_right = source["points"][position], arrays["points"][index]
                if np.all(np.isfinite(points_left)) and np.all(np.isfinite(points_right)):
                    if not np.array_equal(points_left, points_right):
                        raise ValidationError(f"Constrained points differ for row {row_id}")
                elif source["status"][position] == 0 and arrays["status"][index] == 0:
                    raise ValidationError(
                        f"Nonfinite canonical points require failed status for row {row_id}"
                    )
                matched += 1
                for quantity in native:
                    if source[quantity][position].shape != arrays[quantity][index].shape:
                        raise ValidationError(f"{quantity} axes differ for row {row_id}")
                if "hvp" in native and not np.array_equal(
                    source["vectors"][position], arrays["vectors"][index]
                ):
                    raise ValidationError(f"HVP vectors differ for row {row_id}")
            if stencil_rows is not None and fd is not None:
                stencil_pair = stencil_rows.get(int(row_id))
                if stencil_pair is None:
                    raise ValidationError(f"Missing finite-difference stencil base row {row_id}")
                stencil, position = stencil_pair
                points_stencil, points_right = stencil["points"][position], arrays["points"][index]
                if np.all(np.isfinite(points_stencil)) and np.all(np.isfinite(points_right)):
                    if not np.array_equal(points_stencil, points_right):
                        raise ValidationError(f"Stencil base points differ for row {row_id}")
                elif arrays["status"][index] == 0:
                    raise ValidationError(
                        f"Nonfinite stencil base requires failed status for row {row_id}"
                    )
                for quantity in fd["quantities"]:
                    key = f"fd_{quantity}_rows"
                    expected = (len(fd["steps"]), *arrays[quantity][index].shape)
                    expected += (2,) if quantity == "gradient" else (4,)
                    if key not in stencil or stencil[key][position].shape != expected:
                        raise ValidationError(f"Invalid fixed central {quantity} stencil axes")
                    ids = stencil[key][position]
                    if ids.dtype != np.int64 or np.any(ids < 0):
                        raise ValidationError("Stencil row references must be nonnegative int64")
                    for perturbation in ids.flat:
                        if left_rows.get(int(perturbation)) is None:
                            raise ValidationError(f"Missing perturbation result row {perturbation}")
    output_rows = min(4096, (batchstore.MAX_SHARD_BYTES - 16384) // max(output_row_bytes, 1))
    if output_rows < 1:
        raise ValidationError("One point's discrepancy tensors exceed the shard byte limit")
    if largest_shard_rows > output_rows:
        right_path = batchstore.rechunk(right_path, directory / "right-chunked", output_rows)
    metadata = {
        "schema": "stan-mcp.batch.v1",
        "kind": "comparison",
        "point_space": "constrained",
        "parameter_names": right["parameter_names"],
        "quantities": quantities,
        "target": right["target"],
        "tolerances": tolerances,
        "derivative_coordinates": right.get("derivative_coordinates"),
        "methods": {
            q: "central_finite_difference" if q in fd_quantities else "native" for q in quantities
        },
        "additive_constant": offset_mode,
        "offset_estimator": "first_valid_paired_left_minus_right" if offset_mode else None,
        "finite_difference": fd,
    }
    return {
        "left_manifest": str(left_path),
        "right_manifest": str(right_path),
        "input_manifest": str(right_path),
        "output_metadata": metadata,
        "unmatched_left": 0 if fd else left_rows.count - matched,
    }


def _errors(
    left: np.ndarray, right: np.ndarray, tolerance: dict[str, float] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Use a symmetric scale without overflowing its sum or the subtraction."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        absolute = np.abs(left - right)
        magnitude = np.maximum(np.abs(left), np.abs(right))
        normalizer = magnitude if tolerance is None else np.maximum(magnitude, tolerance["atol"])
        normalized = np.abs(
            np.divide(left, normalizer, out=np.zeros_like(left), where=normalizer != 0)
            - np.divide(right, normalizer, out=np.zeros_like(right), where=normalizer != 0)
        )
        normalized = np.where(
            np.isfinite(absolute),
            np.divide(absolute, normalizer, out=np.zeros_like(absolute), where=normalizer != 0),
            normalized,
        )
        if tolerance is None:
            scaled = normalized
        else:
            denominator = np.divide(
                tolerance["atol"],
                normalizer,
                out=np.zeros_like(magnitude),
                where=normalizer != 0,
            ) + tolerance["rtol"] * np.divide(
                magnitude, normalizer, out=np.zeros_like(magnitude), where=normalizer != 0
            )
            scaled = np.divide(
                normalized,
                denominator,
                out=np.full_like(normalized, np.inf),
                where=denominator != 0,
            )
        scaled = np.where(left == right, 0.0, scaled)
    return absolute, scaled


def _scaled_uncertainty(
    uncertainty: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    tolerance: dict[str, float],
) -> np.ndarray:
    magnitude = np.maximum(np.abs(left), np.abs(right))
    normalizer = np.maximum(magnitude, tolerance["atol"])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = np.divide(
            tolerance["atol"], normalizer, out=np.zeros_like(magnitude), where=normalizer != 0
        ) + tolerance["rtol"] * np.divide(
            magnitude, normalizer, out=np.zeros_like(magnitude), where=normalizer != 0
        )
        normalized = np.divide(
            uncertainty, normalizer, out=np.full_like(magnitude, np.inf), where=normalizer != 0
        )
        result = np.divide(
            normalized, denominator, out=np.full_like(magnitude, np.inf), where=denominator != 0
        )
    return np.where(uncertainty == 0, 0.0, result)


def _number(value: float) -> float | str:
    return float(value) if np.isfinite(value) else "infinity"


def _summary() -> dict[str, Any]:
    return {
        "processed": 0,
        "comparable": 0,
        "failed": 0,
        "inconclusive": 0,
        "estimated": 0,
        "outside_tolerance": 0,
        "max_absolute_error": None,
        "max_scaled_error": None,
        "max_empirical_stability": None,
        "max_roundoff_estimate": None,
        "worst": [],
        "failed_rows": [],
        "inconclusive_rows": [],
    }


def _record(summary: dict[str, Any], row_id: int, absolute: np.ndarray, scaled: np.ndarray) -> None:
    if absolute.size == 0:
        return
    maximum = float(np.max(absolute))
    score = float(np.max(scaled))
    for key, value in (("max_absolute_error", maximum), ("max_scaled_error", score)):
        previous = summary[key]
        if previous is None or (previous != "infinity" and value > previous):
            summary[key] = _number(value)
    # One component per row keeps the evidence bounded even for full Hessians.
    component = tuple(int(i) for i in np.unravel_index(int(np.argmax(scaled)), scaled.shape))
    summary["worst"].append(
        {
            "row_id": row_id,
            "component": list(component),
            "absolute_error": _number(float(absolute[component])),
            "scaled_error": _number(float(scaled[component])),
        }
    )
    summary["worst"].sort(key=_worst_score, reverse=True)
    del summary["worst"][_WORST_LIMIT:]


def _worst_score(evidence: dict[str, Any]) -> float:
    value = evidence["scaled_error"]
    return np.inf if value == "infinity" else float(value)


def _finite_difference(
    quantity: str,
    ids: np.ndarray,
    steps: list[float],
    rows: _Rows,
    cancelled: Callable[[], bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    estimates = np.empty(ids.shape[:-1], dtype=np.float64)
    roundoff_estimates = np.empty_like(estimates)
    for index in np.ndindex(ids.shape[:-1]):
        if cancelled():
            return None
        values = []
        roundoff = 0.0
        for row_id in ids[index]:
            pair = rows.get(int(row_id))
            if pair is None:
                return None
            arrays, position = pair
            value = float(arrays["log_density"][position])
            if arrays["status"][position] != 0 or not np.isfinite(value):
                return None
            values.append(value)
            roundoff += abs(value) * np.finfo(arrays["log_density"].dtype).eps
        h = steps[index[0]]
        # Pair the differences first so a large common additive constant cancels.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if quantity == "gradient":
                estimate = (values[1] - values[0]) / h / 2
                roundoff = roundoff / h
            else:
                estimate = ((values[0] - values[1]) - (values[2] - values[3])) / h / h / 4
                roundoff = roundoff / h / h / 2
        estimates[index] = estimate
        roundoff_estimates[index] = roundoff
    if not np.all(np.isfinite(estimates)):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        stability = np.max(np.abs(np.diff(estimates, axis=0)), axis=0)
    return estimates[-1], stability, np.max(roundoff_estimates, axis=0)


def _merge_summary(destination: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "processed",
        "comparable",
        "failed",
        "inconclusive",
        "estimated",
        "outside_tolerance",
    ):
        destination[key] += source[key]
    for key in (
        "max_absolute_error",
        "max_scaled_error",
        "max_empirical_stability",
        "max_roundoff_estimate",
    ):
        value, previous = source[key], destination[key]
        if value is not None and (
            previous is None
            or (previous != "infinity" and (value == "infinity" or value > previous))
        ):
            destination[key] = value
    destination["worst"].extend(source["worst"])
    destination["worst"].sort(key=_worst_score, reverse=True)
    del destination["worst"][_WORST_LIMIT:]
    for key in ("failed_rows", "inconclusive_rows"):
        destination[key].extend(source[key])
        del destination[key][_WORST_LIMIT:]


def execute(
    plan: dict[str, Any],
    directory: Path,
    cancelled: Callable[[], bool],
    progress: Callable[[int, int], None],
) -> dict[str, Any]:
    """Publish one discrepancy shard per right shard, retaining durable checkpoints."""
    directory.mkdir(parents=True, exist_ok=True)
    left = batchstore.load(plan["left_manifest"])
    right = batchstore.load(plan["right_manifest"])
    metadata = dict(plan["output_metadata"])
    quantities = metadata["quantities"]
    fd = metadata["finite_difference"]
    left_rows = _Rows(left)
    stencil_rows = _Rows(batchstore.load(fd["manifest"])) if fd else None
    fd_quantities = set(fd["quantities"]) if fd else set()
    completed = list(plan.get("completed", []))
    if len(completed) > len(right["shards"]):
        raise ValidationError("Completed comparison shards exceed input shard count")
    offset = None
    for descriptor in completed:
        batchstore.read_shard({"_root": str(directory)}, descriptor)
        if descriptor.get("offset") is not None:
            offset = descriptor["offset"]
    for shard_index in range(len(completed), len(right["shards"])):
        if cancelled():
            break
        reference = batchstore.read_shard(right, right["shards"][shard_index])
        arrays: dict[str, np.ndarray] = {
            "row_id": reference["row_id"],
            "points": reference["points"],
            "status": np.zeros(len(reference["row_id"]), dtype=np.int64),
        }
        shard_summary = {q: _summary() for q in quantities}
        if "hvp" in quantities:
            arrays["vectors"] = reference["vectors"]
        for quantity in quantities:
            shape = reference[quantity].shape
            arrays[f"abs_error_{quantity}"] = np.full(shape, np.nan)
            arrays[f"scaled_error_{quantity}"] = np.full(shape, np.nan)
            arrays[f"comparison_status_{quantity}"] = np.full(shape[0], 2, dtype=np.int64)
            if quantity in fd_quantities:
                arrays[f"stability_{quantity}"] = np.full(shape, np.nan)
                arrays[f"roundoff_{quantity}"] = np.full(shape, np.nan)
        for index, row_id in enumerate(reference["row_id"]):
            if cancelled():
                break
            pair = left_rows.get(int(row_id))
            for quantity in quantities:
                summary = shard_summary[quantity]
                summary["processed"] += 1
                tolerance = metadata["tolerances"].get(quantity)
                value_right = reference[quantity][index]
                value_left = np.asarray(np.nan)
                status = 0
                stability = None
                uncertainty_scale = None
                residual = None
                if reference["status"][index] != 0 or not np.all(np.isfinite(value_right)):
                    status = 1
                elif quantity in fd_quantities and stencil_rows is not None and fd is not None:
                    stencil_pair = stencil_rows.get(int(row_id))
                    if stencil_pair is None:
                        raise ValidationError("Owned stencil base row disappeared")
                    stencil, position = stencil_pair
                    estimate = _finite_difference(
                        quantity,
                        stencil[f"fd_{quantity}_rows"][position],
                        fd["steps"],
                        left_rows,
                        cancelled,
                    )
                    if estimate is None:
                        status = 1
                    else:
                        value_left, stability, roundoff = estimate
                        arrays[f"stability_{quantity}"][index] = stability
                        arrays[f"roundoff_{quantity}"][index] = roundoff
                        for key, value in (
                            ("max_empirical_stability", stability),
                            ("max_roundoff_estimate", roundoff),
                        ):
                            if value.size:
                                maximum = float(np.max(value))
                                previous = summary[key]
                                if previous is None or (
                                    previous != "infinity" and maximum > previous
                                ):
                                    summary[key] = _number(maximum)
                        summary["estimated"] += 1
                elif pair is None:
                    status = 2
                else:
                    source, position = pair
                    value_left = source[quantity][position]
                    if source["status"][position] != 0 or not np.all(np.isfinite(value_left)):
                        status = 1
                if status == 0:
                    if quantity == "log_density" and metadata["additive_constant"]:
                        if offset is None:
                            difference = float(value_left) - float(value_right)
                            if not np.isfinite(difference):
                                status = 1
                            else:
                                offset = {"value": difference, "row_id": int(row_id)}
                        if status == 0:
                            assert offset is not None
                            with np.errstate(over="ignore", invalid="ignore"):
                                residual = (value_left - value_right) - offset["value"]
                                value_left = value_left - offset["value"]
                            if not np.all(np.isfinite(value_left)) or not np.isfinite(residual):
                                status = 1
                    if status == 0:
                        absolute, scaled = _errors(value_left, value_right, tolerance)
                        if residual is not None:
                            absolute = np.abs(np.asarray(residual))
                            if tolerance is not None:
                                scaled = _scaled_uncertainty(
                                    absolute, value_left, value_right, tolerance
                                )
                            else:
                                magnitude = np.maximum(np.abs(value_left), np.abs(value_right))
                                scaled = np.divide(
                                    absolute,
                                    magnitude,
                                    out=np.zeros_like(absolute),
                                    where=magnitude != 0,
                                )
                        arrays[f"abs_error_{quantity}"][index] = absolute
                        arrays[f"scaled_error_{quantity}"][index] = scaled
                        _record(summary, int(row_id), absolute, scaled)
                        outside = tolerance is not None and np.any(scaled > 1)
                        if stability is not None:
                            if tolerance is None:
                                status = 2
                            else:
                                with np.errstate(over="ignore", invalid="ignore"):
                                    uncertainty = stability + roundoff
                                uncertainty_scale = _scaled_uncertainty(
                                    uncertainty, value_left, value_right, tolerance
                                )
                                if not np.all(uncertainty_scale <= 1):
                                    status = 2
                                    outside = False
                                else:
                                    lower = scaled - uncertainty_scale
                                    upper = scaled + uncertainty_scale
                                    outside = np.any(lower > 1)
                                    if np.any((upper > 1) & (lower <= 1)):
                                        status = 2
                        if outside:
                            summary["outside_tolerance"] += 1
                        if status == 0:
                            summary["comparable"] += 1
                if status == 1:
                    summary["failed"] += 1
                    arrays["status"][index] = 1
                    if len(summary["failed_rows"]) < _WORST_LIMIT:
                        summary["failed_rows"].append(int(row_id))
                elif status == 2:
                    summary["inconclusive"] += 1
                    if len(summary["inconclusive_rows"]) < _WORST_LIMIT:
                        summary["inconclusive_rows"].append(int(row_id))
                arrays[f"comparison_status_{quantity}"][index] = status
        if cancelled():
            break
        descriptor = batchstore.write_shard(directory, shard_index, arrays)
        descriptor["comparison_summary"] = shard_summary
        descriptor["offset"] = offset
        completed.append(descriptor)
        batchstore.atomic_json(directory / "comparison-checkpoint.json", {"completed": completed})
        progress(len(completed), len(right["shards"]))
    finished = len(completed) == len(right["shards"])
    summary = {q: _summary() for q in quantities}
    requested = sum(descriptor["rows"] for descriptor in right["shards"]) + plan["unmatched_left"]
    for descriptor in completed:
        for quantity in quantities:
            _merge_summary(summary[quantity], descriptor["comparison_summary"][quantity])
    for quantity in quantities:
        item = summary[quantity]
        item["requested"] = requested
        item["unmatched_left"] = plan["unmatched_left"]
        item["inconclusive"] += plan["unmatched_left"]
        if not finished or item["comparable"] != requested:
            item["verdict"] = "unknown"
        elif quantity not in metadata["tolerances"]:
            item["verdict"] = "report_only"
        else:
            item["verdict"] = "fail" if item["outside_tolerance"] else "pass"
    verdicts = [item["verdict"] for item in summary.values()]
    verdict = (
        "unknown"
        if "unknown" in verdicts
        else "fail"
        if "fail" in verdicts
        else "report_only"
        if "report_only" in verdicts
        else "pass"
    )
    report = {
        "verdict": verdict,
        "quantities": summary,
        "offset": offset,
        "complete": finished,
        "tolerances": metadata["tolerances"],
        "methods": metadata["methods"],
        "finite_difference_stability": "maximum adjacent-step absolute difference; finest estimate"
        if fd
        else None,
        "finite_difference_pass_budget": (
            "absolute discrepancy + empirical stability + estimated floating roundoff "
            "<= tolerance scale"
        )
        if fd
        else None,
        "finite_difference_roundoff": (
            "maximum over steps of 2*sum(eps_dtype*abs(logtarget))/central denominator; "
            "estimate, not rigorous bound"
        )
        if fd
        else None,
        "finite_difference_fail_budget": (
            "absolute discrepancy - empirical uncertainty > tolerance scale; "
            "overlap is inconclusive; estimate, not rigorous bound"
        )
        if fd
        else None,
    }
    metadata["summary"] = report
    manifest = batchstore.publish(directory, metadata, completed) if completed else None
    return {
        "manifest": str(manifest) if manifest is not None else None,
        "summary": report,
        "completed": completed,
        "cancelled": not finished,
    }
