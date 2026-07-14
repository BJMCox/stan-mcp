import config


def test_dirs_created_under_home(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    assert config.compiled_dir() == tmp_path / "sm" / "compiled"
    assert config.runs_dir() == tmp_path / "sm" / "runs"
    assert config.compiled_dir().is_dir()
    assert config.runs_dir().is_dir()
