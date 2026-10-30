"""Tests for method arguments."""

import pytest

from stan_mcp import errors, methods


def test_inits_rejected_for_laplace():
    with pytest.raises(errors.ValidationError):
        methods.build_kwargs("laplace", None, {"theta": 0.0}, {})


def test_inits_rejected_for_generate_quantities():
    with pytest.raises(errors.ValidationError):
        methods.build_kwargs("generate_quantities", None, {"theta": 0.0}, {})


def test_inits_accepted_for_sample():
    kw = methods.build_kwargs("sample", None, {"theta": 0.0}, {})
    assert kw["inits"] == {"theta": 0.0}
