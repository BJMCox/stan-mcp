"""Owned, bounded batches of native BridgeStan evaluations."""

import json
from pathlib import Path
from typing import Any

import numpy as np

import batchstore
import compile as compilation
import density
import errors

_CONTEXT: tuple[str, density._SerializedModel, dict[str, Any]] | None = None
_QUANTITIES = ("log_density", "gradient", "hessian", "hvp")


def validate_plan(plan: dict[str, Any]) -> None:
    """Refuse a resumed computation whose effective native build has changed."""
    source = compilation.restore_source(plan["source"], plan["source_manifest"])
    if compilation.refresh_source(source).manifest != source.manifest:
        raise errors.ValidationError("batch native build inputs have changed")
    binary = Path(plan["so_path"])
    if not binary.is_file() or batchstore.file_hash(binary) != plan["so_sha256"]:
        raise errors.ValidationError("batch native library has changed or is missing")


def _model(plan: dict[str, Any]) -> density._SerializedModel:
    global _CONTEXT
    key = plan["context_id"]
    if _CONTEXT is not None and _CONTEXT[0] == key:
        return _CONTEXT[1]
    validate_plan(plan)
    source = compilation.restore_source(plan["source"], plan["source_manifest"])
    model = density._new_model(source, plan["data"], seed=plan["seed"])
    if any(q in plan["quantities"] for q in ("hessian", "hvp")):
        if "BRIDGESTAN_AD_HESSIAN=true" not in model.model_info():
            raise errors.ValidationError("batch curvature requires an AD-enabled native library")
    manifest = batchstore.load(plan["input_manifest"])
    _CONTEXT = key, model, manifest
    return model


def prepare(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Snapshot a model, data and point collection and preflight every shard."""
    quantities = request.get("quantities", ["log_density"])
    if (
        not isinstance(quantities, list)
        or not quantities
        or any(q not in _QUANTITIES for q in quantities)
        or len(set(quantities)) != len(quantities)
    ):
        raise errors.ValidationError("quantities must be distinct native evaluation quantities")
    target = {"propto": False, "jacobian": False, **request.get("target", {})}
    if set(target) != {"propto", "jacobian"} or any(
        not isinstance(value, bool) for value in target.values()
    ):
        raise errors.ValidationError("target requires boolean propto and jacobian flags")
    seed = request.get("seed", 1234)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise errors.ValidationError("seed must be an unsigned 32-bit integer")
    chunk_rows = request.get("chunk_rows")
    if chunk_rows is not None and (
        not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0
    ):
        raise errors.ValidationError("chunk_rows must be a positive integer")
    directory.mkdir(parents=True, exist_ok=True)
    original = batchstore.load(batchstore.snapshot(request["points"], directory / "points"))
    if original["kind"] != "points":
        raise errors.ValidationError("evaluation input must be a points manifest")
    source = (
        compilation.restore_source(request["stan"], request["_source_manifest"])
        if "_source_manifest" in request
        else compilation.resolve_source(request["stan"], request.get("stanc_args"))
    )
    source = compilation.refresh_source(source)
    if any(q in quantities for q in ("hessian", "hvp")):
        source = compilation.with_make_args(source, ["BRIDGESTAN_AD_HESSIAN=true"])
    data = compilation.canonical_data(request.get("data"))
    model = density._new_model(source, data, seed=seed)
    source = compilation.refresh_source(source)
    if any(q in quantities for q in ("hessian", "hvp")):
        if "BRIDGESTAN_AD_HESSIAN=true" not in model.model_info():
            raise errors.ValidationError("batch curvature requires an AD-enabled native library")
    names: list[str] = model.param_names()
    unc_names: list[str] = model.param_unc_names()
    expected = names if original["point_space"] == "constrained" else unc_names
    provided = original["parameter_names"]
    if len(provided) != len(expected) or set(provided) != set(expected):
        raise errors.ValidationError("point parameter names do not match the native model")
    if original["point_space"] == "unconstrained" and provided != unc_names:
        raise errors.ValidationError("unconstrained points require native parameter order")
    order = [provided.index(name) for name in expected]
    dimensions = len(unc_names)
    directions: int | None = None
    descriptors: list[dict[str, Any]] = []
    rows = 0
    # Rechunk owned inputs once: workers never hash or rewrite files per point.
    owned = directory / "inputs"
    for descriptor in original["shards"]:
        arrays = batchstore.read_shard(original, descriptor)
        vectors = arrays.get("vectors")
        if "hvp" in quantities:
            if (
                vectors is None
                or vectors.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or vectors.ndim != 3
                or vectors.shape[2] != dimensions
            ):
                raise errors.ValidationError(
                    "HVP requires F32/F64 vectors with axes [point,direction,parameter]"
                )
            if directions is not None and vectors.shape[1] != directions:
                raise errors.ValidationError("HVP direction count differs across point shards")
            directions = vectors.shape[1]
            if not directions or not np.isfinite(vectors).all():
                raise errors.ValidationError("HVP vectors must have finite, nonempty directions")
        width = len(names) + 2 + (directions or 0) * dimensions
        width += 1 if "log_density" in quantities else 0
        width += dimensions if "gradient" in quantities else 0
        width += dimensions**2 if "hessian" in quantities else 0
        width += (directions or 0) * dimensions if "hvp" in quantities else 0
        capacity = (batchstore.MAX_SHARD_BYTES - 16384) // (8 * max(width, 1))
        if capacity < 1:
            raise errors.ValidationError("one requested output row exceeds the shard size limit")
        count = min(capacity, chunk_rows or capacity, 4096)
        for start in range(0, len(arrays["row_id"]), count):
            part = {key: value[start : start + count] for key, value in arrays.items()}
            part["points"] = np.ascontiguousarray(part["points"][:, order])
            descriptors.append(batchstore.write_shard(owned, len(descriptors), part))
            rows += len(part["row_id"])
    input_metadata = {key: value for key, value in original.items() if not key.startswith("_")}
    input_metadata["parameter_names"] = expected
    input_manifest = batchstore.publish(owned, input_metadata, descriptors)
    source_path = directory / "source.stan"
    source_path.write_text(str(source))
    source_manifest = directory / "source_manifest.json"
    batchstore.atomic_json(source_manifest, source.manifest)
    data_path = directory / "data.json"
    batchstore.atomic_json(data_path, data)
    binary = compilation.compiled_so(source)
    binary_hash = batchstore.file_hash(binary)
    coordinates = {
        "kind": "stan_unconstrained",
        "names": unc_names,
        "model_id": compilation.model_id(source),
    }
    metadata = {
        "schema": "stan-mcp.batch.v1",
        "kind": "results",
        "point_space": "constrained",
        "parameter_names": names,
        "quantities": quantities,
        "target": target,
        "derivative_coordinates": coordinates,
        "dtype": "float64",
        "axes": {
            "row_id": ["point"],
            "status": ["point"],
            "points": ["point", "parameter"],
            "log_density": ["point"],
            "gradient": ["point", "parameter"],
            "hessian": ["point", "parameter", "parameter"],
            "hvp": ["point", "direction", "parameter"],
            "vectors": ["point", "direction", "parameter"],
        },
        "provenance": {
            "model_id": compilation.model_id(source),
            "data_sha256": batchstore.fingerprint(data),
            "native_library_sha256": binary_hash,
            "seed": seed,
            "curvature_method": "autodiff"
            if any(q in quantities for q in ("hessian", "hvp"))
            else None,
        },
    }
    plan = {
        "input_manifest": str(input_manifest.resolve()),
        "source": str(source),
        "source_manifest": source.manifest,
        "source_path": str(source_path.resolve()),
        "source_manifest_path": str(source_manifest.resolve()),
        "data": data,
        "data_path": str(data_path.resolve()),
        "seed": seed,
        "target": target,
        "quantities": quantities,
        "derivative_coordinates": coordinates,
        "output_metadata": metadata,
        "rows": rows,
        "shard_count": len(descriptors),
        "so_path": str(binary.resolve()),
        "so_sha256": binary_hash,
    }
    plan["context_id"] = batchstore.fingerprint(plan)
    return plan


def execute_shard(plan: dict[str, Any], index: int, directory: Path) -> dict[str, Any]:
    """Evaluate one immutable owned input shard using a process-local native context."""
    model = _model(plan)
    assert _CONTEXT is not None
    manifest = _CONTEXT[2]
    inputs = batchstore.read_shard(manifest, manifest["shards"][index])
    count = len(inputs["row_id"])
    dimensions = len(plan["derivative_coordinates"]["names"])
    names = plan["output_metadata"]["parameter_names"]
    quantities = plan["quantities"]
    arrays = {
        "row_id": inputs["row_id"],
        "status": np.zeros(count, dtype=np.int64),
        "points": np.full((count, len(names)), np.nan),
    }
    shapes = {
        "log_density": (count,),
        "gradient": (count, dimensions),
        "hessian": (count, dimensions, dimensions),
    }
    if "hvp" in quantities:
        arrays["vectors"] = inputs["vectors"].astype(np.float64)
        shapes["hvp"] = arrays["vectors"].shape
    arrays.update({name: np.full(shapes[name], np.nan) for name in quantities})
    failures: dict[str, Any] = {}
    tolerance = 64 * np.finfo(np.float64).eps
    for row, point in enumerate(inputs["points"]):
        try:
            if not np.isfinite(point).all():
                raise ValueError("point parameters must be finite")
            if manifest["point_space"] == "constrained":
                arrays["points"][row] = point
                native_point = np.ascontiguousarray(point, dtype=np.float64)
                theta = np.ascontiguousarray(
                    model.param_unconstrain(native_point), dtype=np.float64
                )
                roundtrip = np.asarray(model.param_constrain(theta))
                # Native inverses can normalize unsupported constrained values.
                # Allow floating-point transform error, but preserve the test point.
                if not np.allclose(point, roundtrip, rtol=tolerance, atol=tolerance):
                    raise ValueError("constrained point does not survive the native transform")
            else:
                theta = np.ascontiguousarray(point, dtype=np.float64)
                arrays["points"][row] = model.param_constrain(theta)
            flags = plan["target"]
            if "hessian" in quantities:
                lp, gradient, hessian = model.log_density_hessian(theta, **flags)
                arrays["hessian"][row] = hessian
                if "gradient" in quantities:
                    arrays["gradient"][row] = gradient
            elif "gradient" in quantities:
                lp, gradient = model.log_density_gradient(theta, **flags)
                arrays["gradient"][row] = gradient
            elif "log_density" in quantities:
                lp = model.log_density(theta, **flags)
            else:
                lp = np.nan
            if "hvp" in quantities:
                for direction, vector in enumerate(arrays["vectors"][row]):
                    if "hessian" in quantities:
                        arrays["hvp"][row, direction] = arrays["hessian"][row] @ vector
                    else:
                        lp, product = model.log_density_hessian_vector_product(
                            theta, vector, **flags
                        )
                        arrays["hvp"][row, direction] = product
            if "log_density" in quantities:
                arrays["log_density"][row] = lp
        except (RuntimeError, ValueError) as exc:
            arrays["status"][row] = 1
            for name in quantities:
                arrays[name][row] = np.nan
            message = str(exc)
            detail = message[:2048]
            # Escaped Unicode can otherwise make bounded character counts large on disk.
            while len(json.dumps(detail)) > 8192:
                detail = detail[: len(detail) // 2]
            failures[str(int(inputs["row_id"][row]))] = {
                "error": detail,
                "error_type": type(exc).__name__,
                "truncated": len(detail) < len(message),
            }
    return batchstore.write_shard(directory, index, arrays, failures or None)
