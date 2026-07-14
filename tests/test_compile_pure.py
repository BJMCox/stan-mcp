import json

import compile as c


def test_resolve_inline_source_passthrough():
    src = "parameters { real x; } model { x ~ normal(0,1); }"
    assert c.resolve_source(src) == src


def test_resolve_path_reads_file(tmp_path):
    f = tmp_path / "m.stan"
    f.write_text("model {}")
    assert c.resolve_source(str(f)) == "model {}"


def test_model_id_stable_and_args_sensitive():
    a = c.model_id("model {}")
    assert a == c.model_id("model {}")
    assert a != c.model_id("model {} ")  # whitespace differs
    assert a != c.model_id("model {}", ["--O1"])


def test_data_hash_canonical_dict_order_invariant():
    assert c.data_hash({"a": 1, "b": 2}) == c.data_hash({"b": 2, "a": 1})
    assert c.data_hash(None) == c.data_hash(None)
    assert c.data_hash({"a": 1}) != c.data_hash({"a": 2})


def test_data_hash_reads_json_path(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"N": 3}))
    assert c.data_hash(str(f)) == c.data_hash({"N": 3})


def test_stan_file_for_writes_under_compiled(tmp_path, monkeypatch):
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    p = c.stan_file_for("model {}")
    assert p.exists() and p.read_text() == "model {}"
    assert p.parent.name == "compiled"
