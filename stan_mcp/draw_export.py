"""Portable Safetensors exports of completed runs' typed draws."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from stan_mcp import draws, runstore
from stan_mcp.batch.store import file_hash
from stan_mcp.errors import RunError, ValidationError

FORMAT = "stan-mcp.draws.v1"


def _select(specs: dict[str, Any], requested: list[str] | None) -> list[str]:
    if requested is None:
        return list(specs)
    if not requested or any(not name for name in requested):
        raise ValidationError("params must contain one or more variable names")
    selected = set()
    for name in requested:
        matches = {key for key in specs if key == name or key.startswith((name + ".", name + ":"))}
        if not matches:
            raise ValidationError(f"unknown draw variable {name!r}")
        selected.update(matches)
    return [key for key in specs if key in selected]


def export_draws(
    run_id: str, output_file: str, params: list[str] | None = None, thin: int = 1
) -> dict[str, Any]:
    """Export a rectangular draw array per variable, using the recorded CSV order."""
    path = Path(output_file)
    if not path.is_absolute() or path.suffix != ".safetensors":
        raise ValidationError("output_file must be an absolute server-local .safetensors path")
    if not path.parent.is_dir():
        raise ValidationError("output_file's parent directory must exist on the server")
    if path.exists() or path.is_symlink():
        raise ValidationError(f"export already exists: {path}")
    if thin < 1:
        raise ValidationError("thin must be at least 1")

    meta = runstore.read_meta(run_id)
    if meta.get("status") != "done" or meta.get("method") == "optimize":
        raise RunError("export_draws requires a completed run with saved draws")
    schema = draws.schema(run_id)
    selected = _select(schema["params"], params)
    if not selected:
        raise RunError("run has no variables to export")
    fit = draws.load(run_id, params=selected, sampler_stats=(), _schema=schema)
    if fit.truncated_to is not None:
        raise RunError("export_draws requires equal-length chains")

    method = meta.get("method")
    sampling_method = method
    if method == "generate_quantities":
        try:
            sampling_method = draws.sampling_method(run_id)
        except RunError:
            # A legacy generated-quantities run may outlive its source run.
            sampling_method = None

    arrays = {
        "chain_id": np.arange(1, fit.n_chain + 1, dtype=np.int64),
        "draw_id": np.arange(1, fit.n_draw + 1, thin, dtype=np.int64),
    }
    variables = {}
    for name in selected:
        key = f"draws.{name}"
        values = np.ascontiguousarray(fit.params[name][:, ::thin])
        arrays[key] = values
        spec = schema["params"][name]
        variables[name] = {
            "tensor": key,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            **{field: spec[field] for field in ("labels", "column_names") if field in spec},
        }
    manifest = {
        "schema": FORMAT,
        "run_id": run_id,
        "model_id": meta.get("model_id"),
        "method": method,
        "sampling_method": sampling_method,
        "variables": variables,
        "axes": ["chain", "draw"],
        "source_files": schema["files"],
        "source_draws_schema": schema["schema"],
        "source_draw_row_offset": schema.get("draw_row_offset", 0),
        "thin": thin,
    }
    if fit.header_warning:
        manifest["source_header_warning"] = fit.header_warning
    del fit

    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".stan-draws-") as temporary:
            save_file(arrays, temporary.name, metadata={"stan_mcp": json.dumps(manifest)})
            digest = file_hash(Path(temporary.name))
            # Publish only the complete file and never replace a destination.
            os.link(temporary.name, path)
    except FileExistsError as exc:
        raise ValidationError(f"export already exists: {path}") from exc
    except OSError as exc:
        raise RunError(f"cannot write draw export: {exc}") from exc

    return {
        "format": FORMAT,
        "path": str(path),
        "sha256": digest,
        "run_id": run_id,
        "chains": arrays["chain_id"].size,
        "draws_per_chain": arrays["draw_id"].size,
        "variable_count": len(variables),
    }
