import pytest

from stan_mcp import errors, methods


def test_unknown_method_raises():
    with pytest.raises(errors.ValidationError):
        methods.build_kwargs("bogus", None, None, {})


def test_filters_and_injects_common():
    kw = methods.build_kwargs("sample", 42, None, {"chains": 4, "iter_sampling": 500})
    assert kw == {"chains": 4, "iter_sampling": 500, "seed": 42}


def test_unknown_arg_key_raises():
    with pytest.raises(errors.ValidationError):
        methods.build_kwargs("optimize", None, None, {"chains": 4})


def test_is_draws_method():
    assert methods.is_draws_method("sample") is True
    assert methods.is_draws_method("optimize") is False
