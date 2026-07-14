import density


def test_lru_evicts_beyond_capacity(monkeypatch):
    density.clear_cache()
    monkeypatch.setattr(density.config, "MODEL_LRU_SIZE", 2)
    made = []

    def fake_new_model(source, data, stanc_args=None, seed=1234):
        made.append(source)
        return object()

    monkeypatch.setattr(density, "_new_model", fake_new_model)
    a = density.get_model("A", None)
    b = density.get_model("B", None)
    assert density.get_model("A", None) is a  # cache hit, no rebuild
    density.get_model("C", None)  # evicts B (LRU)
    assert made.count("A") == 1
    density.get_model("B", None)  # rebuilt after eviction
    assert made.count("B") == 2
    assert b is not None
