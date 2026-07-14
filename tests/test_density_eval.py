import pytest

import density
import errors
from tests.fakes import FakeModel


def _use_fake(monkeypatch, model):
    monkeypatch.setattr(density, "get_model", lambda *a, **k: model)


def test_log_density(monkeypatch):
    _use_fake(monkeypatch, FakeModel(unc_num=2))
    out = density.log_density("src", None, [1.0, 0.0])
    assert out["lp"] == pytest.approx(-0.5)


def test_gradient_shape(monkeypatch):
    _use_fake(monkeypatch, FakeModel(unc_num=2))
    out = density.log_density_gradient("src", None, [1.0, 2.0])
    assert out["lp"] == pytest.approx(-2.5)
    assert out["gradient"] == pytest.approx([-1.0, -2.0])


def test_hessian_shape(monkeypatch):
    _use_fake(monkeypatch, FakeModel(unc_num=2))
    out = density.log_density_hessian("src", None, [1.0, 2.0])
    assert out["hessian"] == [[-1.0, 0.0], [0.0, -1.0]]


def test_wrong_length_raises(monkeypatch):
    _use_fake(monkeypatch, FakeModel(unc_num=2))
    with pytest.raises(errors.ValidationError):
        density.log_density("src", None, [1.0])
