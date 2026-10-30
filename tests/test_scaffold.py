"""Scaffold smoke test: the package modules import cleanly.

Replaced/expanded by real unit tests during the build (see design "Testing").
"""

from stan_mcp import config


def test_home_default(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "stan-mcp"))
    assert config.home() == tmp_path / "stan-mcp"
