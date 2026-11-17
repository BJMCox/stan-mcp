"""Round-trip the portable draws file against known CSV values."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from cmdstanpy.stanfit.metadata import InferenceMetadata
from safetensors import safe_open

from stan_mcp import draw_export, draws, fitstorage, runstore
from stan_mcp.batch.store import file_hash
from stan_mcp.errors import RunError, ValidationError
from tests.fakes import _typed_run


def test_export_preserves_selected_shapes_chain_order_and_draw_positions(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    run = "sample-0001"
    runstore.write_meta(run, {"method": "sample", "status": "done", "model_id": "model-id"})
    header = "lp__,theta,m.1.1,m.2.1,m.1.2,m.2.2,z.real,z.imag,unused"
    fit = SimpleNamespace(metadata=InferenceMetadata({"raw_header": header}))
    sidecar = fitstorage._sidecar_from_fit(fit, "sample", run)
    files = ["z.csv", "a.csv"]
    sidecar["files"] = files
    sidecar_path = draws.schema_path(run)
    csv_dir = sidecar_path.parent / "csv"
    csv_dir.mkdir()
    theta = 0.12345678901234567
    for chain, filename in enumerate(files):
        rows = []
        for draw in range(3):
            k = 100 * chain + 10 * draw
            rows.append(
                ",".join(map(str, [-1, theta + k, k + 1, k + 2, k + 3, k + 4, k + 5, k + 6, 0]))
            )
        (csv_dir / filename).write_text(header + "\n" + "\n".join(rows) + "\n")
    sidecar_path.write_text(json.dumps(sidecar))

    path = tmp_path / "export.safetensors"
    result = draw_export.export_draws(run, str(path), params=["m", "theta", "z"], thin=2)
    assert result["sha256"] == file_hash(path)
    assert (result["chains"], result["draws_per_chain"], result["variable_count"]) == (2, 2, 4)
    with safe_open(path, framework="np") as exported:
        assert set(exported.keys()) == {
            "chain_id",
            "draw_id",
            "draws.theta",
            "draws.m",
            "draws.z.real",
            "draws.z.imag",
        }
        meta = json.loads(exported.metadata()["stan_mcp"])
        assert (meta["schema"], meta["run_id"], meta["model_id"], meta["sampling_method"]) == (
            "stan-mcp.draws.v1",
            run,
            "model-id",
            "sample",
        )
        assert meta["source_files"] == files
        assert meta["variables"]["m"] == {
            "tensor": "draws.m",
            "shape": [2, 2, 2, 2],
            "dtype": "float64",
        }
        assert meta["variables"]["z.imag"]["column_names"] == ["z.imag"]
        for name, expected in (("chain_id", [1, 2]), ("draw_id", [1, 3])):
            values = exported.get_tensor(name)
            assert values.dtype == np.int64
            np.testing.assert_array_equal(values, expected)
        values = exported.get_tensor("draws.theta")
        assert values.dtype == np.float64
        np.testing.assert_array_equal(values, [[theta, 20 + theta], [100 + theta, 120 + theta]])
        np.testing.assert_array_equal(
            exported.get_tensor("draws.m"),
            np.array([[0, 20], [100, 120]])[:, :, None, None] + [[1, 3], [2, 4]],
        )
        np.testing.assert_array_equal(exported.get_tensor("draws.z.imag"), [[6, 26], [106, 126]])


def test_legacy_generated_quantities_export_respects_draw_offset(tmp_path, monkeypatch):
    run = _typed_run(tmp_path, monkeypatch, method="generate_quantities")
    runstore.write_meta(
        run,
        {
            "method": "generate_quantities",
            "status": "done",
            "method_args": {"from_run_id": "variational-0042"},
        },
    )
    sidecar = draws.schema_path(run)
    spec = json.loads(sidecar.read_text())
    spec["draw_row_offset"] = 1
    sidecar.write_text(json.dumps(spec))
    path = tmp_path / "legacy.safetensors"
    draw_export.export_draws(run, str(path))
    with safe_open(path, framework="np") as exported:
        meta = json.loads(exported.metadata()["stan_mcp"])
        assert meta["sampling_method"] is None
        assert meta["source_draw_row_offset"] == 1
        assert meta["source_header_warning"]
        np.testing.assert_array_equal(exported.get_tensor("draw_id"), [1])
        np.testing.assert_array_equal(exported.get_tensor("draws.theta"), [[0.2], [0.4]])


def test_export_does_not_publish_ragged_chains(tmp_path, monkeypatch):
    run = _typed_run(tmp_path, monkeypatch, ragged=True)
    path = tmp_path / "ragged.safetensors"
    with pytest.raises(RunError, match="equal-length chains"):
        draw_export.export_draws(run, str(path))
    assert not path.exists()


def test_export_does_not_replace_a_concurrent_destination(tmp_path, monkeypatch):
    run = _typed_run(tmp_path, monkeypatch)
    path = tmp_path / "existing.safetensors"
    link = draw_export.os.link

    def racing_link(source, target):
        path.write_bytes(b"another writer")
        link(source, target)

    monkeypatch.setattr(draw_export.os, "link", racing_link)
    with pytest.raises(ValidationError, match="already exists"):
        draw_export.export_draws(run, str(path), params=["theta"])
    assert path.read_bytes() == b"another writer"
    assert not list(tmp_path.glob(".stan-draws-*"))


def test_failed_export_does_not_publish_incomplete_file(tmp_path, monkeypatch):
    run = _typed_run(tmp_path, monkeypatch)
    path = tmp_path / "incomplete.safetensors"

    def disk_full(tensors, filename, *, metadata):
        Path(filename).write_bytes(b"partial file")
        raise OSError("disk full")

    monkeypatch.setattr(draw_export, "save_file", disk_full)
    with pytest.raises(RunError, match="disk full"):
        draw_export.export_draws(run, str(path))
    assert not path.exists()
    assert not list(tmp_path.glob(".stan-draws-*"))
