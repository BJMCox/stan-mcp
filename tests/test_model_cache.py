"""Tests for model cache."""

import pytest

from stan_mcp import compilation as c
from stan_mcp import errors


def test_model_id_handle_resolves_to_source(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    src = "model {}"
    c.stan_file_for(src)  # writes compiled/<model_id>.stan
    assert c.resolve_source(c.model_id(src)) == src


def test_unknown_model_id_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    with pytest.raises(errors.ValidationError):
        c.resolve_source("0" * 16)


def test_missing_stan_file_raises():
    with pytest.raises(errors.ValidationError):
        c.resolve_source("does/not/exist.stan")


def test_register_data_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    did = c.register_data({"N": 3})
    assert c.canonical_data(did) == {"N": 3}


def test_unknown_data_id_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    with pytest.raises(errors.ValidationError):
        c.canonical_data("f" * 16)


def test_missing_json_file_raises():
    with pytest.raises(errors.ValidationError):
        c.canonical_data("does/not/exist.json")


def test_compiled_so_builds_once_under_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    import sys
    import threading
    import time
    from pathlib import Path

    monkeypatch.setattr(c.toolchain, "ensure_bridgestan", lambda: None)
    calls = []

    class _FakeBS:
        @staticmethod
        def compile_model(stan_path, stanc_args=None):
            calls.append(1)
            time.sleep(0.15)  # widen the window a naive impl would race in
            so = Path(str(stan_path).replace(".stan", "_model.so"))
            so.write_text("so")
            return str(so)

    monkeypatch.setitem(sys.modules, "bridgestan", _FakeBS)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(c.compiled_so("model {}"))) for _ in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1  # 5 concurrent callers, one build
    assert all(str(r).endswith("_model.so") for r in results)


def test_prune_compiled_keep(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    from stan_mcp import config

    d = config.compiled_dir()
    for mid in ("a" * 16, "b" * 16):
        (d / f"{mid}.stan").write_text("model {}")
        (d / f"{mid}_model.so").write_text("x")
    res = c.prune_compiled(keep=1)
    assert len(res["removed_models"]) == 1
    remaining = {p.stem for p in d.glob("*.stan")}
    assert len(remaining) == 1


def test_prune_compiled_requires_criterion(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    with pytest.raises(errors.ValidationError):
        c.prune_compiled()


def test_stan_file_touch_on_reuse_orders_prune_by_last_use(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    old, new = "model { real a; }", "model { real b; }"
    c.stan_file_for(old)
    c.stan_file_for(new)
    c.stan_file_for(old)  # reuse the older model -> it is now the freshest
    res = c.prune_compiled(keep=1)
    assert res["removed_models"] == [c.model_id(new)]


def test_stan_file_reuse_does_not_restamp_source(tmp_path, monkeypatch):
    """Last use must be recorded out of band, never by touching the .stan.

    cmdstanpy rebuilds whenever the source is newer than the exe, so restamping
    it on reuse relinks the binary on every call — and relinking one that a
    concurrent run is executing has the kernel kill its chains (retcode -1)."""
    import os

    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    src = "model { real a; }"
    path = c.stan_file_for(src)
    os.utime(path, (1_000_000, 1_000_000))
    assert c.stan_file_for(src).stat().st_mtime == 1_000_000  # source untouched
    assert c._last_used(path) > 1_000_000  # ...but last use still advanced


def test_list_compiled(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    c.stan_file_for("model {}")
    did = c.register_data({"N": 1})
    out = c.list_compiled()
    assert out["models"][0]["model_id"] == c.model_id("model {}")
    assert [d["data_id"] for d in out["data"]] == [did]
