"""Source resolution, hashing, and disk-cached compilation."""

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from stan_mcp import config, errors, toolchain


class SourceSnapshot(str):
    """Self-contained Stan source with its immutable compilation identity."""

    manifest: dict[str, Any]

    def __new__(cls, source: str, manifest: dict[str, Any]) -> Self:
        obj = super().__new__(cls, source)
        obj.manifest = manifest
        return obj


_SOURCES: dict[tuple[str, str], SourceSnapshot] = {}
_SOURCES_LOCK = threading.Lock()
_BUILD_CACHE: tuple[tuple, dict[str, Any]] | None = None
_BUILD_LOCK = threading.Lock()
_VERSIONS = {p: importlib.metadata.version(p) for p in ("bridgestan", "cmdstanpy")}


def _build_identity() -> dict[str, Any]:
    """Describe inputs to native builds without invoking a provisioning helper."""
    global _BUILD_CACHE
    versions = _VERSIONS
    roots = {
        "bridgestan": Path(
            os.environ.get("BRIDGESTAN")
            or Path.home() / ".bridgestan" / f"bridgestan-{versions['bridgestan']}"
        ),
    }
    import cmdstanpy

    try:
        roots["cmdstan"] = Path(cmdstanpy.cmdstan_path())
    except ValueError:
        roots["cmdstan"] = Path(os.environ.get("CMDSTAN", Path.home() / ".cmdstan"))
    paths: dict[str, Path] = {}
    for name, root in roots.items():
        for relative in (
            "make/local",
            "makefile",
            "Makefile",
            "stan/src/stan/version.hpp",
            "bin/stanc",
        ):
            p = root / relative
            if p.is_file():
                paths[f"{name}/{relative}"] = p
    env = {
        k: os.environ.get(k)
        for k in (
            "CC",
            "CXX",
            "CXXFLAGS",
            "CPPFLAGS",
            "LDFLAGS",
            "STANCFLAGS",
            "STAN_THREADS",
            "STAN_OPENCL",
            "CXXFLAGS_O3",
            "MAKE",
            "STANC",
            "STAN_CPP_OPTIMS",
            "STAN_NO_RANGE_CHECKS",
            "STAN_MPI",
            "BRIDGESTAN_AD_HESSIAN",
            "BRIDGESTAN_AD_HESSIAN_VECTOR_PRODUCT",
            "PRECOMPILED_HEADERS",
            "O",
        )
    }
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler:
        paths["compiler"] = Path(compiler)
    if os.environ.get("STANC"):
        stanc = shutil.which(os.environ["STANC"])
        if stanc:
            paths["stanc_override"] = Path(stanc)
    stamps = {
        k: (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for k, p in paths.items()
    }
    root_names = {k: str(v.resolve()) for k, v in roots.items()}
    key = (tuple(env.items()), tuple(root_names.items()), tuple(stamps.items()))
    with _BUILD_LOCK:
        if _BUILD_CACHE is not None and _BUILD_CACHE[0] == key:
            return _BUILD_CACHE[1]
        files: dict[str, Any] = {
            k: list(stamp)
            if k == "compiler" or k.endswith("/stanc") or k == "stanc_override"
            else hashlib.sha256(paths[k].read_bytes()).hexdigest()
            for k, stamp in stamps.items()
        }
        identity = {
            "schema": 2,
            "versions": versions,
            "platform": platform.platform(),
            "roots": root_names,
            "files": files,
            "environment": env,
            "native_calls": "process-serialized",
        }
        _BUILD_CACHE = (key, identity)
        return identity


def _identity(manifest: dict[str, Any]) -> str:
    payload = {k: v for k, v in manifest.items() if k != "model_id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _expand_includes(
    source: str, origin: str | None, args: list[str]
) -> tuple[str, list[dict[str, str]]]:
    roots: list[Path] = []
    for index, arg in enumerate(args):
        if arg.startswith("--include-paths="):
            roots = [Path(p).expanduser().resolve() for p in arg.split("=", 1)[1].split(",")]
        elif arg in ("--include-paths", "-I") and index + 1 < len(args):
            roots = [Path(p).expanduser().resolve() for p in args[index + 1].split(",")]
    roots.append(Path(origin).parent if origin else Path.cwd())
    dependencies: list[dict[str, str]] = []

    def expand(text: str, stack: tuple[Path, ...]) -> str:
        lines: list[str] = []
        cursor = start = 0
        # Preserve all surrounding source. Stan recognizes include directives
        # anywhere between tokens, including after code or a closing comment.
        while cursor < len(text):
            if text.startswith("//", cursor):
                end = text.find("\n", cursor)
                cursor = len(text) if end == -1 else end + 1
                continue
            if text.startswith("/*", cursor):
                end = text.find("*/", cursor + 2)
                cursor = len(text) if end == -1 else end + 2
                continue
            if text[cursor] == '"':
                cursor += 1
                while cursor < len(text):
                    char = text[cursor]
                    cursor += 2 if char == "\\" else 1
                    if char == '"':
                        break
                continue
            match = (
                re.match(r'#include\s+(?:"([^"\n]+)"|<([^>\n]+)>|([^\s]+))', text[cursor:])
                if text.startswith("#include", cursor)
                else None
            )
            if match is None:
                cursor += 1
                continue
            name = next(group for group in match.groups() if group is not None)
            candidates = [root / name for root in roots]
            path = next((p.resolve() for p in candidates if p.is_file()), None)
            if path is None:
                raise errors.ValidationError(f"Stan include file not found: {name}")
            if path in stack:
                raise errors.ValidationError(f"recursive Stan include: {path}")
            included = path.read_text()
            dependencies.append({"path": str(path), "source": included})
            lines.extend((text[start:cursor], expand(included, (*stack, path)), "\n"))
            cursor += match.end()
            start = cursor
        lines.append(text[start:])
        return "".join(lines)

    return expand(source, ()), dependencies


def _snapshot(
    source: str,
    stanc_args: list[str] | None = None,
    origin: str | None = None,
    *,
    refresh_build: bool = False,
) -> SourceSnapshot:
    if isinstance(source, SourceSnapshot):
        build = _build_identity() if refresh_build else source.manifest["build"]
        args = source.manifest["stanc_args"] if stanc_args is None else stanc_args
        if args == source.manifest["stanc_args"] and build == source.manifest["build"]:
            return source
        manifest = {**source.manifest, "stanc_args": list(args), "build": build}
    else:
        args = _norm_args(stanc_args)
        expanded, dependencies = _expand_includes(source, origin, args)
        manifest = {
            "source": expanded,
            "original_source": source,
            "origin": origin,
            "dependencies": dependencies,
            "stanc_args": args,
            "build": _build_identity(),
        }
    manifest["model_id"] = _identity(manifest)
    return SourceSnapshot(manifest["source"], manifest)


def model_manifest(source: str, stanc_args: list[str] | None = None) -> dict[str, Any]:
    """Return a JSON-owned manifest suitable for durable run input snapshots."""
    return json.loads(json.dumps(refresh_source(source, stanc_args).manifest))


def refresh_source(source: str, stanc_args: list[str] | None = None) -> SourceSnapshot:
    """Validate the build context before compiling an immutable source snapshot."""
    refreshed = _snapshot(source, stanc_args, refresh_build=True)
    if isinstance(source, SourceSnapshot):
        with _SOURCES_LOCK:
            _SOURCES[(str(config.home() / "compiled"), source.manifest["model_id"])] = refreshed
    return refreshed


def restore_source(source: str, manifest: dict[str, Any]) -> SourceSnapshot:
    """Restore a persisted snapshot independently of the compiled cache."""
    if source != manifest.get("source") or _identity(manifest) != manifest.get("model_id"):
        raise errors.ValidationError("persisted Stan source does not match its manifest")
    return SourceSnapshot(source, manifest)


def with_make_args(source: str, make_args: list[str]) -> SourceSnapshot:
    """Bind explicit native build arguments to an immutable compilation identity."""
    source = refresh_source(resolve_source(source))
    manifest = {**source.manifest, "make_args": list(make_args)}
    manifest["model_id"] = _identity(manifest)
    return SourceSnapshot(str(source), manifest)


# Per-model-id compile locks: two concurrent runs / density calls that miss the
# cache for the same model must not drive bridgestan/cmdstan builds on the same
# artifact targets at once (concurrent make on one .so/exe corrupts it).
_COMPILE_LOCKS: dict[str, threading.Lock] = {}
_COMPILE_LOCKS_GUARD = threading.Lock()


@contextmanager
def _compile_lock(key: str) -> Iterator[None]:
    with _COMPILE_LOCKS_GUARD:
        lock = _COMPILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _COMPILE_LOCKS[key] = lock
    with lock:
        # A separate directory keeps lock inodes alive while model artifacts
        # are pruned. All server processes lock the same inode for this ID.
        directory = config.compiled_dir() / ".locks"
        directory.mkdir(exist_ok=True)
        with (directory / key).open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield


# Background BridgeStan builds. A run compiles only the CmdStan executable, so a
# sampled model has no .so until something asks for one — and building it inline
# would stall that caller for a whole compile. Requests kick a build off here and
# return immediately; the caller retries. Single worker, deliberately NOT the
# RunManager pool, so a build can never occupy a slot an inference run needs.
_SO_BUILDER: ThreadPoolExecutor | None = None
_SO_BUILDS: dict[str, Future] = {}
_SO_GUARD = threading.Lock()

if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor

    from cmdstanpy import CmdStanModel


def _so_builder() -> ThreadPoolExecutor:
    global _SO_BUILDER
    if _SO_BUILDER is None:
        from concurrent.futures import ThreadPoolExecutor

        _SO_BUILDER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="so-build")
    return _SO_BUILDER


def so_path_if_built(source: str, stanc_args: list[str] | None = None) -> Path | None:
    source = resolve_source(source, stanc_args)
    p = config.compiled_dir() / f"{model_id(source, stanc_args)}_model.so"
    return p if p.exists() else None


def request_so(source: str, stanc_args: list[str] | None = None) -> dict[str, Any]:
    """Ensure the BridgeStan .so exists, building it in the background if not.

    Returns {"status": "ready"|"building"|"failed", ...}. The build goes through
    compiled_so, so it takes the same per-model compile lock as a foreground
    compile_model and the two serialize rather than racing on one artifact."""
    source = resolve_source(source, stanc_args)
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


def resolve_source(stan: str, stanc_args: list[str] | None = None) -> SourceSnapshot:
    if isinstance(stan, SourceSnapshot):
        with _SOURCES_LOCK:
            stan = _SOURCES.get((str(config.home() / "compiled"), stan.manifest["model_id"]), stan)
        return _snapshot(stan, stanc_args)
    if stan.endswith(".stan"):
        p = Path(stan).expanduser().resolve()
        if not p.is_file():
            raise errors.ValidationError(f"Stan file not found: {stan}")
        return _snapshot(p.read_text(), stanc_args, str(p))
    # A bare model_id references source already persisted under compiled/ by a
    # prior compile — lets callers pass the short id instead of re-sending source.
    if _is_hash_id(stan):
        key = (str(config.home() / "compiled"), stan)
        with _SOURCES_LOCK:
            cached_source = _SOURCES.get(key)
        if cached_source is not None:
            return _snapshot(cached_source, stanc_args)
        cached = config.compiled_dir() / f"{stan}.stan"
        manifest = cached.with_suffix(".manifest.json")
        if cached.is_file() and manifest.is_file():
            snapshot = restore_source(cached.read_text(), json.loads(manifest.read_text()))
            if snapshot.manifest["model_id"] != stan:
                raise errors.ValidationError("cached model_id does not match its manifest")
            with _SOURCES_LOCK:
                _SOURCES[key] = snapshot
            return _snapshot(snapshot, stanc_args)
        if cached.is_file():
            raise errors.ValidationError(
                f"model_id {stan!r} predates verified manifests; compile its source again"
            )
        raise errors.ValidationError(f"unknown model_id {stan!r} (compile the model first)")
    return _snapshot(stan, stanc_args)


def register_data(data: object) -> str:
    """Cache data under compiled/data/<data_id>.json; return the data_id.

    The returned id can then be passed as `data` to any tool instead of
    re-sending the full data payload."""
    canon = canonical_data(data)
    payload = json.dumps(canon, sort_keys=True)
    did = hashlib.sha256(payload.encode()).hexdigest()[:16]
    _atomic_write(_data_cache_dir() / f"{did}.json", payload)
    return did


def list_compiled(limit: int = 50) -> dict[str, Any]:
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
                "recompile_required": not p.with_suffix(".manifest.json").is_file(),
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


def prune_compiled(keep: int | None = None, older_than_days: float | None = None) -> dict[str, Any]:
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
    return _snapshot(source, stanc_args).manifest["model_id"]


def read_json_file(path: str, label: str) -> Any:
    """Read a server-local JSON input with a useful validation error."""
    p = Path(path).expanduser()
    if p.suffix != ".json" or not p.is_file():
        raise errors.ValidationError(f"{label} JSON file not found: {path}")
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise errors.ValidationError(f"cannot read {label} JSON file {path!r}: {exc}") from exc


def canonical_data(data: object) -> object:
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
        return read_json_file(s, "data")
    return json.loads(s)  # assume JSON string literal


def data_hash(data: object) -> str:
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
    source = _snapshot(source, stanc_args)
    mid = model_id(source)
    path = config.compiled_dir() / f"{mid}.stan"
    if not path.exists():
        _atomic_write(path, source)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        _atomic_write(manifest_path, json.dumps(source.manifest, sort_keys=True))
    with _SOURCES_LOCK:
        _SOURCES[(str(path.parent), mid)] = source
    # Track last use so prune_compiled(keep=N) evicts by last use rather than by
    # first compile — otherwise the model you run every day loses to a one-off
    # compiled an hour ago.
    _touch_used(mid)
    return path


def _atomic_write(path: Path, text: str) -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def compiled_so(source: str, stanc_args: list[str] | None = None) -> Path:
    toolchain.ensure_bridgestan()
    # bridgestan.compile_model returns the actual .so path; cache by existence.
    import bridgestan

    source = refresh_source(source, stanc_args)
    with _compile_lock(model_id(source, stanc_args)):
        stan_path = stan_file_for(source, stanc_args)
        expected = stan_path.parent / f"{stan_path.stem}_model.so"
        if expected.exists():  # re-check under lock: another thread may have built it
            return expected
        options: dict[str, Any] = {"stanc_args": source.manifest["stanc_args"]}
        if "make_args" in source.manifest:
            options["make_args"] = source.manifest["make_args"]
        built = bridgestan.compile_model(stan_path, **options)
        return Path(built)


def cmdstan_model(source: str, stanc_args: list[str] | None = None) -> CmdStanModel:
    toolchain.ensure_cmdstan()
    import cmdstanpy

    source = refresh_source(source, stanc_args)
    opts = {}
    for a in source.manifest["stanc_args"]:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v or True
    with _compile_lock(model_id(source, stanc_args)):
        stan_path = stan_file_for(source, stanc_args)
        return cmdstanpy.CmdStanModel(stan_file=str(stan_path), stanc_options=opts or None)


def compile_model(
    stan: str, stanc_args: list[str] | None = None, warn_pedantic: bool = False
) -> dict[str, Any]:
    try:
        source = refresh_source(resolve_source(stan, stanc_args))
        so = compiled_so(source, stanc_args)
        cm = cmdstan_model(source, stanc_args)
        prepared = _snapshot(source, stanc_args, refresh_build=True)
        if prepared.manifest["model_id"] != source.manifest["model_id"]:
            # First-use provisioning may have changed the effective toolchain.
            source = prepared
            so = compiled_so(source)
            cm = cmdstan_model(source)
        out: dict[str, Any] = {
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
