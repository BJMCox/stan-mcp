"""Source resolution, hashing, and disk-cached compilation."""

import hashlib
import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import config
import errors
import toolchain

# Per-model-id compile locks: two concurrent runs / density calls that miss the
# cache for the same model must not drive bridgestan/cmdstan builds on the same
# artifact targets at once (concurrent make on one .so/exe corrupts it).
_COMPILE_LOCKS: dict[str, threading.Lock] = {}
_COMPILE_LOCKS_GUARD = threading.Lock()


def _compile_lock(key: str) -> threading.Lock:
    with _COMPILE_LOCKS_GUARD:
        lock = _COMPILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _COMPILE_LOCKS[key] = lock
        return lock


# Background BridgeStan builds. A run compiles only the CmdStan executable, so a
# sampled model has no .so until something asks for one — and building it inline
# would stall that caller for a whole compile. Requests kick a build off here and
# return immediately; the caller retries. Single worker, deliberately NOT the
# RunManager pool, so a build can never occupy a slot an inference run needs.
_SO_BUILDER: "ThreadPoolExecutor | None" = None
_SO_BUILDS: dict[str, "Future"] = {}
_SO_GUARD = threading.Lock()

if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor


def _so_builder():
    global _SO_BUILDER
    if _SO_BUILDER is None:
        from concurrent.futures import ThreadPoolExecutor

        _SO_BUILDER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="so-build")
    return _SO_BUILDER


def so_path_if_built(source: str, stanc_args: list[str] | None = None) -> Path | None:
    p = config.compiled_dir() / f"{model_id(source, stanc_args)}_model.so"
    return p if p.exists() else None


def request_so(source: str, stanc_args: list[str] | None = None) -> dict:
    """Ensure the BridgeStan .so exists, building it in the background if not.

    Returns {"status": "ready"|"building"|"failed", ...}. The build goes through
    compiled_so, so it takes the same per-model compile lock as a foreground
    compile_model and the two serialize rather than racing on one artifact."""
    if so_path_if_built(source, stanc_args) is not None:
        return {"status": "ready"}
    mid = model_id(source, stanc_args)
    with _SO_GUARD:
        fut = _SO_BUILDS.get(mid)
        if fut is not None and fut.done():
            _SO_BUILDS.pop(mid, None)
            if (exc := fut.exception()) is not None:
                return {"status": "failed", "error": str(exc)}
            if so_path_if_built(source, stanc_args) is not None:
                return {"status": "ready"}
            return {"status": "failed", "error": "build finished but no .so was produced"}
        if fut is None:
            _SO_BUILDS[mid] = _so_builder().submit(compiled_so, source, stanc_args)
    return {"status": "building"}


def _pedantic_check(stan_path: Path) -> list[str]:
    """Run `stanc --warn-pedantic` directly on the model and return its warnings.

    Done as a standalone stanc pass rather than via the compile step so that
    (a) --warn-pedantic never enters the model_id / .so cache key (it does not
    change generated code), and (b) warnings surface even when the binary is
    already cached. Returns [] if stanc is unavailable."""
    import cmdstanpy

    stanc = Path(cmdstanpy.cmdstan_path()) / "bin" / "stanc"
    if not stanc.exists():
        return []
    try:
        proc = subprocess.run(
            [str(stanc), "--warn-pedantic", "--o=" + os.devnull, str(stan_path)],
            capture_output=True,
            text=True,
            # Bounded: this runs on a thread-pool thread with no other ceiling,
            # so a wedged stanc would otherwise block that thread forever.
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return ["pedantic check timed out after 120s"]
    text = (proc.stderr or "") + (proc.stdout or "")
    if "Warning" not in text:
        return []
    blocks = re.findall(r"Warning in .*?(?=\nWarning in |\Z)", text, re.DOTALL)
    return [w.strip() for w in blocks] if blocks else [text.strip()]


def _is_hash_id(s: str) -> bool:
    return len(s) == 16 and all(c in "0123456789abcdef" for c in s)


def _data_cache_dir() -> Path:
    d = config.compiled_dir() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_source(stan: str) -> str:
    if stan.endswith(".stan"):
        p = Path(stan)
        if not p.is_file():
            raise errors.ValidationError(f"Stan file not found: {stan}")
        return p.read_text()
    # A bare model_id references source already persisted under compiled/ by a
    # prior compile — lets callers pass the short id instead of re-sending source.
    if _is_hash_id(stan):
        cached = config.compiled_dir() / f"{stan}.stan"
        if cached.is_file():
            return cached.read_text()
        raise errors.ValidationError(f"unknown model_id {stan!r} (compile the model first)")
    return stan


def register_data(data) -> str:
    """Cache data under compiled/data/<data_id>.json; return the data_id.

    The returned id can then be passed as `data` to any tool instead of
    re-sending the full data payload."""
    canon = canonical_data(data)
    did = data_hash(data)
    (_data_cache_dir() / f"{did}.json").write_text(json.dumps(canon, sort_keys=True))
    return did


def list_compiled(limit: int = 50) -> dict:
    """Enumerate the compiled-model and registered-data caches, newest use first.

    Without this a caller that loses a `model_id` / `data_id` has no way back to
    it short of re-sending the whole payload, even though the artifact is
    sitting in the cache."""
    d = config.compiled_dir()
    models = []
    for p in sorted(d.glob("*.stan"), key=_last_used, reverse=True)[:limit]:
        mid = p.stem
        models.append(
            {
                "model_id": mid,
                "last_used": _last_used(p),
                "so_built": (d / f"{mid}_model.so").exists(),
                "source_bytes": p.stat().st_size,
            }
        )
    data_dir = d / "data"
    data = []
    if data_dir.is_dir():
        for p in sorted(data_dir.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True)[
            :limit
        ]:
            data.append({"data_id": p.stem, "bytes": p.stat().st_size})
    return {"models": models, "data": data}


def prune_compiled(keep: int | None = None, older_than_days: float | None = None) -> dict:
    """Prune the compiled/ cache (recomputable). Removes whole model artifact sets
    (.stan/.so/exe) and, when older_than_days is given, stale data-cache entries.

    keep: retain the newest `keep` models by mtime. older_than_days: also drop
    models and data older than the cutoff. keep=0 is allowed here (unlike runs)
    since everything recompiles on demand."""
    import time

    if keep is None and older_than_days is None:
        raise errors.ValidationError("prune_compiled requires keep or older_than_days")
    d = config.compiled_dir()
    stans = sorted(d.glob("*.stan"), key=_last_used, reverse=True)
    victims: list[Path] = []
    if keep is not None:
        victims.extend(stans[keep:])
    cutoff = time.time() - older_than_days * 86400 if older_than_days is not None else None
    if cutoff is not None:
        victims.extend(p for p in stans if _last_used(p) < cutoff)

    removed_models: list[str] = []
    seen: set[str] = set()
    for stan in victims:
        mid = stan.stem
        if mid in seen:
            continue
        seen.add(mid)
        for f in d.glob(f"{mid}*"):
            if f.is_file():
                f.unlink()
        removed_models.append(mid)

    removed_data: list[str] = []
    if cutoff is not None:
        data_dir = d / "data"
        if data_dir.is_dir():
            for f in data_dir.glob("*.json"):
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed_data.append(f.stem)
    return {"removed_models": removed_models, "removed_data": removed_data}


def _norm_args(stanc_args: list[str] | None) -> list[str]:
    return list(stanc_args or [])


def model_id(source: str, stanc_args: list[str] | None = None) -> str:
    h = hashlib.sha256()
    h.update(source.encode())
    h.update(b"\x00")
    h.update("\x00".join(_norm_args(stanc_args)).encode())
    return h.hexdigest()[:16]


def canonical_data(data) -> object:
    """Resolve data (dict, registered data_id, .json path, or JSON string)."""
    if data is None:
        return None
    if isinstance(data, dict):
        return data
    s = str(data)
    if _is_hash_id(s):
        cached = _data_cache_dir() / f"{s}.json"
        if cached.is_file():
            return json.loads(cached.read_text())
        raise errors.ValidationError(f"unknown data_id {s!r} (register the data first)")
    if s.endswith(".json"):
        p = Path(s)
        if not p.is_file():
            raise errors.ValidationError(f"data file not found: {s}")
        return json.loads(p.read_text())
    return json.loads(s)  # assume JSON string literal


def data_hash(data) -> str:
    canon = json.dumps(canonical_data(data), sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _touch_used(mid: str) -> None:
    """Record last use of a model in a `<model_id>.used` sidecar.

    Deliberately NOT `.stan.touch()`: cmdstanpy rebuilds whenever the source is
    newer than the exe, so touching the source on reuse forces a relink on every
    single call — and relinking an executable that a concurrent run is still
    executing has the kernel kill those processes (every chain returns -1).
    The sidecar is inert to both toolchains and is swept by prune_compiled's
    `{mid}*` glob."""
    try:
        (config.compiled_dir() / f"{mid}.used").touch()
    except OSError:  # last-use tracking is advisory; never fail a compile over it
        pass


def _last_used(stan_path: Path) -> float:
    """Last-use time of a cached model: the sidecar if present, else the .stan
    mtime (which for a model compiled before sidecars existed is first compile)."""
    try:
        return (stan_path.parent / f"{stan_path.stem}.used").stat().st_mtime
    except OSError:
        return stan_path.stat().st_mtime


def stan_file_for(source: str, stanc_args: list[str] | None = None) -> Path:
    mid = model_id(source, stanc_args)
    path = config.compiled_dir() / f"{mid}.stan"
    if not path.exists() or path.read_text() != source:
        path.write_text(source)
    # Track last use so prune_compiled(keep=N) evicts by last use rather than by
    # first compile — otherwise the model you run every day loses to a one-off
    # compiled an hour ago.
    _touch_used(mid)
    return path


def compiled_so(source: str, stanc_args: list[str] | None = None) -> Path:
    toolchain.ensure_bridgestan()
    # bridgestan.compile_model returns the actual .so path; cache by existence.
    import bridgestan

    with _compile_lock(model_id(source, stanc_args)):
        stan_path = stan_file_for(source, stanc_args)
        expected = stan_path.parent / f"{stan_path.stem}_model.so"
        if expected.exists():  # re-check under lock: another thread may have built it
            return expected
        built = bridgestan.compile_model(stan_path, stanc_args=_norm_args(stanc_args))
        return Path(built)


def cmdstan_model(source: str, stanc_args: list[str] | None = None):
    toolchain.ensure_cmdstan()
    import cmdstanpy

    opts = {}
    for a in _norm_args(stanc_args):
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v or True
    with _compile_lock(model_id(source, stanc_args)):
        stan_path = stan_file_for(source, stanc_args)
        return cmdstanpy.CmdStanModel(stan_file=str(stan_path), stanc_options=opts or None)


def compile_model(
    stan: str, stanc_args: list[str] | None = None, warn_pedantic: bool = False
) -> dict:
    try:
        source = resolve_source(stan)
        so = compiled_so(source, stanc_args)
        cm = cmdstan_model(source, stanc_args)
        out: dict = {
            "model_id": model_id(source, stanc_args),
            "so_path": str(so),
            "exe_path": str(cm.exe_file),
        }
        # Pedantic runs as a separate stanc pass so it never perturbs model_id.
        if warn_pedantic:
            warnings = _pedantic_check(stan_file_for(source, stanc_args))
            if warnings:
                out["warnings"] = warnings
        return out
    except errors.StanMcpError as exc:
        return errors.as_error_dict(exc)
    except Exception as exc:  # noqa: BLE001 - compilation failures returned, not raised
        return {"error": str(exc), "error_type": "CompileError", "compile_log": str(exc)}
