import pytest

from stan_mcp import errors, toolchain


def test_ensure_bridgestan_returns_path(monkeypatch):
    monkeypatch.setattr(toolchain, "_bridgestan_path", lambda: "/x/.bridgestan/bridgestan-2.9.0")
    assert toolchain.ensure_bridgestan() == "/x/.bridgestan/bridgestan-2.9.0"


def test_ensure_bridgestan_wraps_failure(monkeypatch):
    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(toolchain, "_bridgestan_path", boom)
    with pytest.raises(errors.ToolchainError) as e:
        toolchain.ensure_bridgestan()
    assert "BridgeStan" in str(e.value)


def test_ensure_cmdstan_installs_when_missing(monkeypatch):
    calls = {"installed": False}

    def path():
        if not calls["installed"]:
            raise ValueError("no cmdstan")
        return "/x/.cmdstan/cmdstan-2.x"

    def install():
        calls["installed"] = True

    monkeypatch.setattr(toolchain, "_cmdstan_path", path)
    monkeypatch.setattr(toolchain, "_install_cmdstan", install)
    assert toolchain.ensure_cmdstan() == "/x/.cmdstan/cmdstan-2.x"
    assert calls["installed"] is True
