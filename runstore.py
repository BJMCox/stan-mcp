"""Run store: directory layout and persistence under STAN_MCP_HOME/runs/<run-id>/.

Persists model.stan, data.json, the CmdStan CSVs, inference.nc (arviz
InferenceData, for draws-producing methods) or estimate.json (optimize),
meta.json (model_id, method, method_args, status, timestamps, stderr on
failure), and a cached summary.json. list_runs reads meta.json across run dirs;
failed runs are persisted with status="failed" and captured stderr.

Implemented during the build (see design "Run store").
"""

import json
from pathlib import Path

import config


def new_run_id(method: str, seq: int) -> str:
    return f"{method}-{seq:04d}"


def run_dir(run_id: str) -> Path:
    d = config.runs_dir() / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _meta_path(run_id: str) -> Path:
    return run_dir(run_id) / "meta.json"


def write_meta(run_id: str, meta: dict) -> None:
    m = dict(meta)
    m["run_id"] = run_id
    _meta_path(run_id).write_text(json.dumps(m, indent=2))


def read_meta(run_id: str) -> dict:
    return json.loads(_meta_path(run_id).read_text())


def list_runs(model_id: str | None = None, limit: int = 50) -> list[dict]:
    rows = []
    for d in config.runs_dir().iterdir():
        mp = d / "meta.json"
        if not mp.is_file():
            continue
        meta = json.loads(mp.read_text())
        if model_id is not None and meta.get("model_id") != model_id:
            continue
        rows.append(meta)
    rows.sort(key=lambda m: m.get("seq", 0), reverse=True)
    return rows[:limit]


def idata_path(run_id: str) -> Path:
    return run_dir(run_id) / "inference.nc"


def persist_estimate(run_id: str, estimate: dict) -> None:
    (run_dir(run_id) / "estimate.json").write_text(json.dumps(estimate, indent=2))


def load_estimate(run_id: str) -> dict:
    return json.loads((run_dir(run_id) / "estimate.json").read_text())
