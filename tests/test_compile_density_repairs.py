import json
import multiprocessing
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import bridgestan
import cmdstanpy
import numpy as np
import pytest

from stan_mcp import compilation, density, errors, toolchain


def _compile_process(
    home: str, attempted: Any, release: Any, started: Any, finished: Any, results: Any, builds: Any
) -> None:
    os.environ["STAN_MCP_HOME"] = home
    toolchain.ensure_bridgestan = lambda: "unused"

    def build(path: Path, *, stanc_args: list[str]) -> Path:
        builds.put(os.getpid())
        output = path.with_name(path.stem + "_model.so")
        output.write_text("partial")
        started.set()
        if not release.wait(10):
            raise RuntimeError("parent did not release the compiler")
        output.write_text("complete artifact")
        return output

    bridgestan.compile_model = build
    attempted.set()
    try:
        path = compilation.compiled_so("parameters { real x; } model {}")
        results.put((str(path), path.read_text()))
    finally:
        finished.set()


def test_two_process_compilation_publishes_one_complete_artifact(tmp_path):
    context = multiprocessing.get_context("spawn")
    release, started = context.Event(), context.Event()
    attempts = [context.Event(), context.Event()]
    finished = [context.Event(), context.Event()]
    results, builds = context.Queue(), context.Queue()
    processes = [
        context.Process(
            target=_compile_process,
            args=(str(tmp_path), attempts[i], release, started, finished[i], results, builds),
        )
        for i in range(2)
    ]
    processes[0].start()
    try:
        assert started.wait(10)
        processes[1].start()
        assert attempts[1].wait(10)
        assert not finished[1].wait(0.1)
    finally:
        release.set()
        for process in processes:
            if process.pid is not None:
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
    assert all(process.exitcode == 0 for process in processes)
    outputs = [results.get(timeout=5), results.get(timeout=5)]
    assert outputs[0] == outputs[1]
    assert outputs[0][1] == "complete artifact"
    builds.get(timeout=5)
    with pytest.raises(queue.Empty):
        builds.get_nowait()


def test_density_file_snapshot_identifies_the_data_actually_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    data_file = tmp_path / "data.json"
    data_file.write_text('{"n": 1}')
    loaded: list[dict] = []

    def load(source, data, stanc_args=None, seed=1234):
        data_file.write_text('{"n": 2}')
        loaded.append(data)
        return object()

    monkeypatch.setattr(density, "_new_model", load)
    density.clear_cache()
    first = density.get_model("model {}", str(data_file))
    assert density.get_model("model {}", {"n": 1}) is first
    assert loaded == [{"n": 1}]
    assert density.get_model("model {}", str(data_file)) is not first
    assert loaded == [{"n": 1}, {"n": 2}]
    density.clear_cache()


def test_registered_file_data_is_resolved_once(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    data_file = tmp_path / "data.json"
    data_file.write_text('{"n": 1}')
    canonical_data = compilation.canonical_data

    def resolve(data):
        snapshot = canonical_data(data)
        data_file.write_text('{"n": 2}')
        return snapshot

    monkeypatch.setattr(compilation, "canonical_data", resolve)
    registered = compilation.register_data(str(data_file))
    assert registered == compilation.data_hash({"n": 1})
    assert canonical_data(registered) == {"n": 1}


def test_source_snapshot_retains_dependencies_origin_and_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    origin = tmp_path / "source"
    (origin / "nested").mkdir(parents=True)
    model_file = origin / "model.stan"
    model_file.write_text("parameters {\n#include nested/params.stan\n}\nmodel {}\n")
    params = origin / "nested" / "params.stan"
    params.write_text("#include nested/declaration.stan\n")
    declaration = origin / "nested" / "declaration.stan"
    declaration.write_text("real x;\n")
    source = compilation.resolve_source(str(model_file), ["--O1"])
    mid = compilation.model_id(source)
    path = compilation.stan_file_for(source)
    assert "#include" not in path.read_text()
    assert "real x;" in path.read_text()
    declaration.write_text("real y;\n")
    assert compilation.model_id(compilation.resolve_source(str(model_file), ["--O1"])) != mid
    model_file.unlink()
    params.unlink()
    declaration.unlink()
    restored = compilation.restore_source(str(source), compilation.model_manifest(source))
    assert compilation.model_id(restored) == mid

    calls: dict[str, Any] = {}

    def so(stan_file, *, stanc_args):
        calls["bridge"] = stanc_args
        return Path(stan_file).with_name(f"{mid}_model.so")

    def cmdstan(*, stan_file, stanc_options):
        calls["cmdstan"] = stanc_options
        return object()

    monkeypatch.setattr(toolchain, "ensure_bridgestan", lambda: "unused")
    monkeypatch.setattr(toolchain, "ensure_cmdstan", lambda: "unused")
    monkeypatch.setattr(bridgestan, "compile_model", so)
    monkeypatch.setattr(cmdstanpy, "CmdStanModel", cmdstan)
    # A handle lookup must use its already verified source/configuration.
    monkeypatch.setattr(compilation, "_identity", lambda *a: pytest.fail("rehash on handle"))
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("reread on handle"))
    handle = compilation.resolve_source(mid)
    compilation.compiled_so(handle)
    compilation.cmdstan_model(handle)
    assert calls == {"bridge": ["--O1"], "cmdstan": {"O1": True}}


def test_source_origin_and_build_settings_change_artifact_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    first = tmp_path / "one.stan"
    second = tmp_path / "two.stan"
    first.write_text("model {}")
    second.write_text("model {}")
    a = compilation.model_id(compilation.resolve_source(str(first)))
    assert compilation.model_id(compilation.resolve_source(str(second))) != a
    monkeypatch.setenv("CXXFLAGS", "-O0")
    assert compilation.model_id(compilation.resolve_source(str(first))) != a


def test_persisted_snapshot_refreshes_changed_build_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    source = compilation.resolve_source("parameters { real x; } model {}", ["--O1"])
    old_id = compilation.model_id(source)
    compilation.stan_file_for(source)
    monkeypatch.setenv("STAN_NO_RANGE_CHECKS", "true")
    refreshed = compilation.refresh_source(compilation.resolve_source(old_id))
    assert compilation.model_id(refreshed) != old_id
    assert compilation.model_manifest(refreshed)["stanc_args"] == ["--O1"]
    assert (
        compilation.model_manifest(refreshed)["build"]["environment"]["STAN_NO_RANGE_CHECKS"]
        == "true"
    )


def test_include_lexing_preserves_code_comments_strings_and_search_order(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    (alternate / "p space.stan").write_text("real y;")
    (tmp_path / "p space.stan").write_text("real x;")
    model = tmp_path / "model.stan"
    model.write_text(
        'parameters { /* comment\n*/ #include "p space.stan"\n}\nmodel { print("/*"); }'
    )
    source = compilation.resolve_source(str(model), [f"--include-paths={alternate}"])
    assert "parameters { /* comment\n*/ real y;\n\n}" in source
    assert 'print("/*")' in source
    model.write_text('parameters { #include "p space.stan"\n}\nmodel {}')
    assert "real y;" in compilation.resolve_source(str(model), [f"--include-paths={alternate}"])


def test_direct_native_calls_serialize_across_loaded_models(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()

    class Native:
        def __init__(self, so, data, seed):
            self.n = json.loads(data)["n"]

        def log_density(self, theta):
            if self.n == 1:
                first_entered.set()
                assert release_first.wait(5)
            else:
                second_entered.set()
            return -float(np.dot(theta, theta)) / 2

    monkeypatch.setattr(compilation, "compiled_so", lambda *a: "model.so")
    monkeypatch.setattr(bridgestan, "StanModel", Native)
    density.clear_cache()
    first = density.get_model("model {}", {"n": 1})
    second = density.get_model("model {}", {"n": 2})

    def second_call():
        second_attempted.set()
        return second.log_density([2.0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first.log_density, [1.0])
        assert first_entered.wait(5)
        b = pool.submit(second_call)
        assert second_attempted.wait(5)
        try:
            assert not second_entered.wait(0.05)
        finally:
            release_first.set()
        assert a.result() == -0.5
        assert b.result() == -2.0
    density.clear_cache()


def test_vector_matrix_transform_roundtrip_preserves_json_axes(monkeypatch):
    class Containers:
        def param_unc_num(self):
            return 6

        def param_names(self, **kwargs):
            return ["v.1", "v.2", "m.1.1", "m.2.1", "m.1.2", "m.2.2"]

        def param_constrain(self, theta, **kwargs):
            return theta

        def param_unconstrain_json(self, text):
            obj = json.loads(text)
            assert obj == {"v": [1.0, 2.0], "m": [[3.0, 5.0], [4.0, 6.0]]}
            return [*obj["v"], obj["m"][0][0], obj["m"][1][0], obj["m"][0][1], obj["m"][1][1]]

    monkeypatch.setattr(density, "get_model", lambda *a, **kw: Containers())
    theta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    constrained = density.param_constrain("model {}", None, theta)["constrained"]
    assert density.param_unconstrain("model {}", None, constrained)["theta_unc"] == theta
    flat = dict(zip(Containers().param_names(), theta, strict=True))
    assert density.param_unconstrain("model {}", None, flat)["theta_unc"] == theta


def test_complex_tuple_constraints_use_stan_json_shapes():
    names = ["z.real", "z.imag", "t:1", "t:2.1", "t:2.2"]
    actual = density.constrained_values(names, [1, 2, 3, 4, 5])
    assert json.loads(json.dumps(actual)) == {"z": [1, 2], "t": {"1": 3, "2": [4, 5]}}


def test_parameterless_constrain_retains_empty_json(monkeypatch):
    class Parameterless:
        def param_unc_num(self):
            return 0

        def param_names(self, **kwargs):
            return []

        def param_constrain(self, theta, **kwargs):
            return []

    monkeypatch.setattr(density, "get_model", lambda *a, **kw: Parameterless())
    assert density.param_constrain("model {}", None, []) == {"constrained": {}}


def test_bound_native_method_retains_serialized_owner_until_destruction(monkeypatch):
    ownership: list[bool] = []

    class Guard:
        depth = 0

        def __enter__(self):
            self.depth += 1

        def __exit__(self, *args):
            self.depth -= 1

    guard = Guard()
    monkeypatch.setattr(density, "_NATIVE_LOCK", guard)

    class Native:
        def operation(self):
            return 1

        def __del__(self):
            ownership.append(guard.depth > 0)

    proxy = density._SerializedModel(Native())
    bound = proxy.operation
    del proxy
    assert ownership == []
    assert bound() == 1
    del bound
    assert ownership == [True]


def test_complex_tuple_array_inverse_uses_declared_numeric_order(monkeypatch):
    names = [
        "a.1:1.real",
        "a.1:1.imag",
        "a.1:2.1",
        "a.1:2.2",
        "a.2:1.real",
        "a.2:1.imag",
        "a.2:2.1",
        "a.2:2.2",
    ]

    class Containers:
        def param_names(self):
            return names

        def param_unconstrain(self, values):
            np.testing.assert_array_equal(values, np.arange(8))
            return values

        def param_unconstrain_json(self, text):
            pytest.fail("complex tuple arrays must avoid the broken native JSON reader")

    monkeypatch.setattr(density, "get_model", lambda *a, **kw: Containers())
    constrained = density.constrained_values(names, range(8))
    assert density.param_unconstrain("model {}", None, constrained)["theta_unc"] == list(range(8))
    invalid = json.loads(json.dumps(constrained))
    invalid["a"][1]["2"].append(999)
    with pytest.raises(errors.ValidationError):
        density.param_unconstrain("model {}", None, invalid)


def test_hot_model_handle_evaluation_avoids_source_and_toolchain_io(tmp_path, monkeypatch):
    from tests.fakes import FakeModel

    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    source = compilation.resolve_source("parameters { real x; } model {}")
    compilation.stan_file_for(source)
    mid = compilation.model_id(source)
    monkeypatch.setattr(density, "_new_model", lambda *a, **kw: FakeModel())
    density.clear_cache()
    density.log_density(mid, None, [0.5])
    with monkeypatch.context() as hot:
        hot.setattr(
            compilation, "_build_identity", lambda: pytest.fail("toolchain scan on hot handle")
        )
        hot.setattr(compilation, "_identity", lambda *a: pytest.fail("source hash on hot handle"))
        hot.setattr(Path, "stat", lambda *a, **kw: pytest.fail("filesystem access on hot handle"))
        actual = density.log_density(mid, None, [0.5])
    assert actual == {"lp": -0.125}
    density.clear_cache()


def test_cache_listing_marks_legacy_models_for_recompilation(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "home"))
    current = compilation.stan_file_for("model {}")
    legacy = current.parent / ("0" * 16 + ".stan")
    legacy.write_text("model {}")
    models = {row["model_id"]: row for row in compilation.list_compiled()["models"]}
    assert models[current.stem]["recompile_required"] is False
    assert models[legacy.stem]["recompile_required"] is True


@pytest.mark.parametrize("fails", [False, True])
def test_installer_child_keeps_progress_off_protocol_stdout(tmp_path, monkeypatch, capsys, fails):
    package = tmp_path / "bridgestan"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "compile.py").write_text(
        'def get_bridgestan_path():\n    print("installer progress")\n'
        + ('    raise RuntimeError("download failed")\n' if fails else '    return "/installed"\n')
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("BRIDGESTAN", str(tmp_path / "not-installed"))
    if fails:
        with pytest.raises(errors.ToolchainError):
            toolchain.ensure_bridgestan()
    else:
        assert toolchain.ensure_bridgestan() == "/installed"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "installer progress" in captured.err
