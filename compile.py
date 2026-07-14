"""Source resolution, hashing, and disk-cached compilation."""

import hashlib
import json
from pathlib import Path

import config
import errors
import toolchain


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


def compiled_so(source: str, stanc_args: list[str] | None = None) -> Path:
    toolchain.ensure_bridgestan()
    stan_path = stan_file_for(source, stanc_args)
    # bridgestan.compile_model returns the actual .so path; cache by existence.
    import bridgestan

    expected = stan_path.parent / f"{stan_path.stem}_model.so"
    if expected.exists():
        return expected
    built = bridgestan.compile_model(stan_path, stanc_args=_norm_args(stanc_args))
    return Path(built)


def cmdstan_model(source: str, stanc_args: list[str] | None = None):
    toolchain.ensure_cmdstan()
    stan_path = stan_file_for(source, stanc_args)
    import cmdstanpy

    opts = {}
    for a in _norm_args(stanc_args):
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v or True
    return cmdstanpy.CmdStanModel(stan_file=str(stan_path), stanc_options=opts or None)


def compile_model(stan: str, stanc_args: list[str] | None = None) -> dict:
    try:
        source = resolve_source(stan)
        so = compiled_so(source, stanc_args)
        cm = cmdstan_model(source, stanc_args)
        return {
            "model_id": model_id(source, stanc_args),
            "so_path": str(so),
            "exe_path": str(cm.exe_file),
        }
    except errors.StanMcpError as exc:
        return errors.as_error_dict(exc)
    except Exception as exc:  # noqa: BLE001 - compilation failures returned, not raised
        return {"error": str(exc), "error_type": "CompileError", "compile_log": str(exc)}
