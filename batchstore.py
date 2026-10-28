"""Bounded Safetensors artifacts shared by batch evaluation and comparison."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from errors import ValidationError

SCHEMA = "stan-mcp.batch.v1"
MAX_SHARD_BYTES = 64 * 1024 * 1024
QUANTITIES = ("log_density", "gradient", "hessian", "hvp")


def fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot read batch JSON {str(path)!r}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError("batch JSON must contain an object")
    return value


def resolve(manifest: dict[str, Any], path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(manifest["_root"]) / value


def _names(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValidationError(f"{label} must be a list of nonempty names")
    if len(set(value)) != len(value):
        raise ValidationError(f"{label} contains duplicate names")
    return value


def read_shard(manifest: dict[str, Any], descriptor: dict[str, Any]) -> dict[str, np.ndarray]:
    path = resolve(manifest, descriptor["path"])
    try:
        if path.stat().st_size > MAX_SHARD_BYTES:
            raise ValidationError(f"shard {path.name!r} exceeds {MAX_SHARD_BYTES} bytes; split it")
        return load_file(path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValidationError(f"cannot read batch shard {path.name!r}: {exc}") from exc


def _validate_arrays(manifest: dict[str, Any], arrays: dict[str, np.ndarray], rows: int) -> None:
    ids = arrays.get("row_id")
    points = arrays.get("points")
    if ids is None or ids.dtype != np.dtype("int64") or ids.shape != (rows,):
        raise ValidationError("row_id must be an int64 vector matching shard rows")
    if rows and (np.any(ids < 0) or np.any(ids[1:] <= ids[:-1])):
        raise ValidationError("row_id values must be nonnegative and strictly increasing")
    if (
        points is None
        or points.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or points.shape != (rows, len(manifest["parameter_names"]))
    ):
        raise ValidationError(
            "points must be a floating matrix with the declared parameter columns"
        )
    for name, values in arrays.items():
        if values.ndim < 1 or values.shape[0] != rows:
            raise ValidationError(f"tensor {name!r} must have shard rows as its first axis")
        if values.dtype not in (np.dtype("float32"), np.dtype("float64"), np.dtype("int64")):
            raise ValidationError(f"unsupported batch dtype {values.dtype} in {name!r}")
    status = arrays.get("status")
    if status is not None and (
        status.dtype != np.dtype("int64")
        or status.shape != (rows,)
        or np.any((status < 0) | (status > 1))
    ):
        raise ValidationError("status must be an int64 vector containing zero or one")
    if manifest["kind"] == "results":
        if status is None:
            raise ValidationError("result shards require a status vector")
        coordinates = manifest.get("derivative_coordinates") or {}
        dimension = len(coordinates.get("names", []))
        for name in manifest["quantities"]:
            values = arrays.get(name)
            shape = {
                "log_density": (rows,),
                "gradient": (rows, dimension),
                "hessian": (rows, dimension, dimension),
            }.get(name)
            if name == "hvp":
                vectors = arrays.get("vectors")
                if (
                    vectors is None
                    or vectors.dtype.kind != "f"
                    or vectors.ndim != 3
                    or vectors.shape[2] != dimension
                ):
                    raise ValidationError(
                        "HVP results require vectors shaped (point,direction,parameter)"
                    )
                shape = vectors.shape
            if values is None or values.shape != shape or values.dtype.kind != "f":
                raise ValidationError(f"result tensor {name!r} has the wrong shape or dtype")


def load(path: str | Path, verify: bool = True) -> dict[str, Any]:
    """Validate an artifact one bounded shard at a time; retain only its metadata."""
    path = Path(path).expanduser().resolve()
    manifest = read_json(path)
    if manifest.get("schema") != SCHEMA or manifest.get("kind") not in (
        "points",
        "results",
        "comparison",
    ):
        raise ValidationError("unsupported batch schema or artifact kind")
    manifest["_root"] = str(path.parent)
    _names(manifest.get("parameter_names"), "parameter_names")
    if manifest.get("point_space") not in ("constrained", "unconstrained"):
        raise ValidationError("point_space must be constrained or unconstrained")
    if manifest["kind"] == "results":
        quantities = manifest.get("quantities")
        if (
            not isinstance(quantities, list)
            or not quantities
            or any(q not in QUANTITIES for q in quantities)
            or len(set(quantities)) != len(quantities)
        ):
            raise ValidationError("result quantities must name supported numerical outputs")
        target = manifest.get("target", {})
        if not isinstance(target, dict) or any(
            type(target.get(k)) is not bool for k in ("propto", "jacobian")
        ):
            raise ValidationError("results require explicit propto and jacobian booleans")
        coordinates = manifest.get("derivative_coordinates")
        if coordinates is not None and not isinstance(coordinates, dict):
            raise ValidationError("derivative_coordinates must be an object or null")
        if any(q != "log_density" for q in quantities):
            if not isinstance(coordinates, dict) or not isinstance(coordinates.get("kind"), str):
                raise ValidationError("derivative results require declared coordinates")
            _names(coordinates.get("names"), "derivative coordinate names")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValidationError("a batch artifact requires at least one shard")
    total, previous = 0, -1
    paths: set[str] = set()
    for entry in shards:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValidationError("each shard requires a path")
        if entry["path"] in paths:
            raise ValidationError("duplicate shard path")
        paths.add(entry["path"])
        rows = entry.get("rows")
        if type(rows) is not int or rows <= 0:
            raise ValidationError("shard rows must be a positive integer")
        shard_path = resolve(manifest, entry["path"])
        try:
            if verify and file_hash(shard_path) != entry.get("sha256"):
                raise ValidationError(f"shard {entry['path']!r} content hash does not match")
            if entry.get("errors"):
                errors = resolve(manifest, entry["errors"])
                if errors.stat().st_size > MAX_SHARD_BYTES:
                    raise ValidationError("error sidecar exceeds the shard byte limit")
                if verify and file_hash(errors) != entry.get("errors_sha256"):
                    raise ValidationError("error sidecar content hash does not match")
                read_json(errors)
        except OSError as exc:
            raise ValidationError(f"cannot read batch artifact file: {exc}") from exc
        arrays = read_shard(manifest, entry)
        _validate_arrays(manifest, arrays, rows)
        ids = arrays["row_id"]
        if ids[0] <= previous:
            raise ValidationError("row IDs must increase across all shards")
        previous = int(ids[-1])
        entry["first_id"], entry["last_id"] = int(ids[0]), previous
        total += rows
    if "count" in manifest and manifest["count"] != total:
        raise ValidationError("artifact count does not match shard rows")
    manifest["count"] = total
    return manifest


def _copy_snapshot(source: Path, target: Path) -> str:
    """Hash the bytes actually copied, rather than rereading a mutable source."""
    digest = hashlib.sha256()
    total = 0
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            total += len(block)
            if total > MAX_SHARD_BYTES:
                raise ValidationError("input shard exceeds the byte limit; split it")
            digest.update(block)
            outgoing.write(block)
    return digest.hexdigest()


def snapshot(path: str | Path, directory: Path) -> Path:
    """Own an artifact's consumed bytes and verify the owned copy before use."""
    source = Path(path).expanduser().resolve()
    manifest = read_json(source)
    if manifest.get("schema") != SCHEMA or manifest.get("kind") not in (
        "points",
        "results",
        "comparison",
    ):
        raise ValidationError("unsupported batch schema or artifact kind")
    manifest["_root"] = str(source.parent)
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, entry in enumerate(manifest.get("shards", [])):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValidationError("invalid shard descriptor")
        target = directory / f"{index:08d}.safetensors"
        digest = _copy_snapshot(resolve(manifest, entry["path"]), target)
        if digest != entry.get("sha256"):
            raise ValidationError("input shard changed or does not match its content hash")
        descriptor = {**entry, "path": target.name, "sha256": digest}
        if entry.get("errors"):
            errors = directory / f"{index:08d}.errors.json"
            digest = _copy_snapshot(resolve(manifest, entry["errors"]), errors)
            if digest != entry.get("errors_sha256"):
                raise ValidationError("input error records changed")
            descriptor.update(errors=errors.name, errors_sha256=digest)
        saved.append(descriptor)
    # Preserve declared metadata so validation cannot repair an invalid input count.
    owned = {key: value for key, value in manifest.items() if not key.startswith("_")}
    owned["shards"] = saved
    result = directory / "manifest.json"
    atomic_json(result, owned)
    load(result)
    return result


def write_shard(
    directory: Path,
    index: int,
    arrays: dict[str, np.ndarray],
    errors: dict | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{index:08d}.safetensors"
    if path.exists():
        raise ValidationError(f"refusing to replace completed shard {path.name!r}")
    if sum(a.nbytes for a in arrays.values()) + 16384 > MAX_SHARD_BYTES:
        raise ValidationError("output shard exceeds the byte limit; use fewer rows")
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
    created = False
    error_path = directory / f"{index:08d}.errors.json"
    try:
        save_file({k: np.ascontiguousarray(v) for k, v in arrays.items()}, temporary)
        if temporary.stat().st_size > MAX_SHARD_BYTES:
            raise ValidationError("serialized shard exceeds the byte limit")
        # Link creates the destination exclusively, even across concurrent writers.
        os.link(temporary, path)
        created = True
        descriptor: dict[str, Any] = {
            "path": path.name,
            "sha256": file_hash(path),
            "rows": len(arrays["row_id"]),
        }
        if errors:
            atomic_json(error_path, errors)
            if error_path.stat().st_size > MAX_SHARD_BYTES:
                raise ValidationError("error sidecar exceeds the shard byte limit")
            descriptor.update(errors=error_path.name, errors_sha256=file_hash(error_path))
        return descriptor
    except Exception:
        if created:
            path.unlink(missing_ok=True)
            error_path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def publish(directory: Path, metadata: dict[str, Any], descriptors: list[dict]) -> Path:
    result = {
        k: v for k, v in metadata.items() if not k.startswith("_") and k not in ("shards", "count")
    }
    result.update(schema=SCHEMA, shards=descriptors, count=sum(d["rows"] for d in descriptors))
    path = directory / "manifest.json"
    atomic_json(path, result)
    return path


def rechunk(path: str | Path, directory: Path, max_rows: int) -> Path:
    """Split an owned artifact without changing rows, tensor axes or error records."""
    if max_rows < 1:
        raise ValidationError("one output row exceeds the shard byte limit")
    source = load(path)
    descriptors = []
    for descriptor in source["shards"]:
        arrays = read_shard(source, descriptor)
        errors = (
            read_json(resolve(source, descriptor["errors"])) if descriptor.get("errors") else {}
        )
        for start in range(0, len(arrays["row_id"]), max_rows):
            part = {key: value[start : start + max_rows] for key, value in arrays.items()}
            part_errors = {
                str(int(row)): errors[str(int(row))]
                for row in part["row_id"]
                if str(int(row)) in errors
            }
            descriptors.append(write_shard(directory, len(descriptors), part, part_errors or None))
    return publish(directory, source, descriptors)
