import runstore


def test_run_id_format():
    assert runstore.new_run_id("sample", 1) == "sample-0001"


def test_meta_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("sample-0001", {"method": "sample", "status": "done"})
    assert runstore.read_meta("sample-0001")["status"] == "done"


def test_list_runs_filters_and_orders(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("sample-0001", {"model_id": "A", "method": "sample", "seq": 1})
    runstore.write_meta("sample-0002", {"model_id": "B", "method": "sample", "seq": 2})
    rows = runstore.list_runs()
    assert [r["run_id"] for r in rows] == ["sample-0002", "sample-0001"]
    assert [r["run_id"] for r in runstore.list_runs(model_id="A")] == ["sample-0001"]


def test_estimate_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    runstore.write_meta("optimize-0001", {"method": "optimize"})
    runstore.persist_estimate("optimize-0001", {"theta": 0.6, "lp__": -3.2})
    assert runstore.load_estimate("optimize-0001")["theta"] == 0.6
