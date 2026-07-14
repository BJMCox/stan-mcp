import density
from tests.fakes import FakeModel


def _use_fake(monkeypatch, model):
    monkeypatch.setattr(density, "get_model", lambda *a, **k: model)


def test_param_constrain_maps_names(monkeypatch):
    _use_fake(monkeypatch, FakeModel(unc_num=2, names=["a", "b"]))
    out = density.param_constrain("src", None, [1.0, 2.0])
    assert out["constrained"] == {"a": 1.0, "b": 2.0}


def test_param_unconstrain_roundtrips(monkeypatch):
    _use_fake(monkeypatch, FakeModel(unc_num=2, names=["a", "b"]))
    out = density.param_unconstrain("src", None, {"a": 3.0, "b": 4.0})
    assert out["theta_unc"] == [3.0, 4.0]


def test_check_gradient_agrees_for_quadratic(monkeypatch):
    # FakeModel lp = -0.5||theta||^2, analytic grad = -theta; finite diff matches.
    _use_fake(monkeypatch, FakeModel(unc_num=2))
    out = density.check_gradient("src", None, [0.3, -0.7])
    assert out["max_abs_err"] < 1e-4
    assert out["mismatches"] == []
