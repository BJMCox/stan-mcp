"""Source resolution, hashing, and disk-cached compilation."""

import hashlib
import json
from pathlib import Path

import config


def resolve_source(stan: str) -> str:
    p = Path(stan)
    if stan.endswith(".stan") and p.is_file():
        return p.read_text()
    return stan


def _norm_args(stanc_args: list[str] | None) -> list[str]:
    return list(stanc_args or [])


def model_id(source: str, stanc_args: list[str] | None = None) -> str:
    h = hashlib.sha256()
    h.update(source.encode())
    h.update(b"\x00")
    h.update("\x00".join(_norm_args(stanc_args)).encode())
    return h.hexdigest()[:16]


def _canonical_data(data) -> object:
    if data is None:
        return None
    if isinstance(data, dict):
        return data
    s = str(data)
    if s.endswith(".json") and Path(s).is_file():
        return json.loads(Path(s).read_text())
    return json.loads(s)  # assume JSON string literal


def data_hash(data) -> str:
    canon = json.dumps(_canonical_data(data), sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def stan_file_for(source: str, stanc_args: list[str] | None = None) -> Path:
    path = config.compiled_dir() / f"{model_id(source, stanc_args)}.stan"
    if not path.exists() or path.read_text() != source:
        path.write_text(source)
    return path
