"""BridgeStan density tools (primary use).

Holds a bounded in-process LRU of loaded BridgeStan StanModel objects so
repeated density pokes on a hot model skip the reload. Provides log_density,
log_density_gradient, log_density_hessian, param_constrain, param_unconstrain,
param_info, and check_gradient (finite-diff vs autodiff). Every theta_unc is
length-validated against param_unc_num before the C++ call.

Implemented during the build (see design tool group B).
"""

import json
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

import compile as _compile
import config
import errors

if TYPE_CHECKING:
    import bridgestan

_CACHE: "OrderedDict[tuple, bridgestan.StanModel]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _new_model(source, data, stanc_args=None, seed=1234) -> "bridgestan.StanModel":
    import bridgestan

    so = _compile.compiled_so(source, stanc_args)
    # BridgeStan wants a JSON string (or path); resolve dicts / data_ids to JSON.
    canon = _compile.canonical_data(data)
    data_arg = json.dumps(canon) if canon is not None else ""
    return bridgestan.StanModel(str(so), data_arg, seed=seed)


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def get_model(source, data, stanc_args=None, seed=1234) -> "bridgestan.StanModel":
    key = (_compile.model_id(source, stanc_args), _compile.data_hash(data), seed)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    # Build outside the lock (compilation/load is slow); the redundant work of a
    # concurrent miss on the same key is acceptable and the last writer wins.
    model = _new_model(source, data, stanc_args, seed)
    with _CACHE_LOCK:
        _CACHE[key] = model
        _CACHE.move_to_end(key)
        while len(_CACHE) > config.MODEL_LRU_SIZE:
            _CACHE.popitem(last=False)
    return model


def param_info(stan, data) -> dict:
    source = _compile.resolve_source(stan)
    m = get_model(source, data)
    return {
        "param_names": m.param_names(),
        "param_unc_names": m.param_unc_names(),
        "param_num": m.param_num(),
        "param_unc_num": m.param_unc_num(),
    }


def validate_data(stan, data) -> dict:
    """Check that `data` satisfies the model's data block before a run. Building
    the BridgeStan model raises if a required variable is missing or malformed;
    on success return the parameter counts."""
    source = _compile.resolve_source(stan)
    m = get_model(source, data)
    return {"ok": True, "param_num": m.param_num(), "param_unc_num": m.param_unc_num()}


def _validate_theta(model, theta_unc) -> np.ndarray:
    if isinstance(theta_unc, str):
        theta_unc = _compile.read_json_file(theta_unc, "theta_unc")
        if not isinstance(theta_unc, list):
            raise errors.ValidationError("theta_unc JSON file must contain a numeric array")
    arr = np.asarray(theta_unc, dtype=float)
    n = model.param_unc_num()
    if arr.shape != (n,):
        raise errors.ValidationError(
            f"theta_unc has length {arr.size}, expected {n} unconstrained params"
        )
    return arr


def log_density(stan, data, theta_unc, propto=True, jacobian=True) -> dict:
    source = _compile.resolve_source(stan)
    m = get_model(source, data)
    theta = _validate_theta(m, theta_unc)
    return {"lp": float(m.log_density(theta, propto=propto, jacobian=jacobian))}


def log_density_gradient(stan, data, theta_unc, propto=True, jacobian=True) -> dict:
    source = _compile.resolve_source(stan)
    m = get_model(source, data)
    theta = _validate_theta(m, theta_unc)
    lp, grad = m.log_density_gradient(theta, propto=propto, jacobian=jacobian)
    return {"lp": float(lp), "gradient": np.asarray(grad).tolist()}


def log_density_hessian(stan, data, theta_unc, propto=True, jacobian=True) -> dict:
    source = _compile.resolve_source(stan)
    m = get_model(source, data)
    theta = _validate_theta(m, theta_unc)
    lp, grad, hess = m.log_density_hessian(theta, propto=propto, jacobian=jacobian)
    return {
        "lp": float(lp),
        "gradient": np.asarray(grad).tolist(),
        "hessian": np.asarray(hess).tolist(),
    }


def param_constrain(stan, data, theta_unc, include_tp=False, include_gq=False) -> dict:
    source = _compile.resolve_source(stan)
    m = get_model(source, data)
    theta = _validate_theta(m, theta_unc)
    # BridgeStan requires an rng when generated quantities are included.
    rng = m.new_rng(seed=1234) if include_gq else None
    vals = np.asarray(
        m.param_constrain(theta, include_tp=include_tp, include_gq=include_gq, rng=rng)
    ).tolist()
    names = m.param_names(include_tp=include_tp, include_gq=include_gq)
    return {"constrained": dict(zip(names, vals, strict=True))}


def param_unconstrain(stan, data, constrained: dict | str) -> dict:
    if isinstance(constrained, str):
        constrained = _compile.read_json_file(constrained, "constrained")
    if not isinstance(constrained, dict):
        raise errors.ValidationError("constrained must contain a named parameter object")
    source = _compile.resolve_source(stan)
    m = get_model(source, data)
    theta = m.param_unconstrain_json(json.dumps(constrained))
    return {"theta_unc": np.asarray(theta).tolist()}


def check_gradient(stan, data, theta_unc, eps=1e-6, tol=1e-4) -> dict:
    source = _compile.resolve_source(stan)
    m = get_model(source, data)
    theta = _validate_theta(m, theta_unc)
    _, ad_grad = m.log_density_gradient(theta)
    ad_grad = np.asarray(ad_grad, dtype=float)
    fd = np.zeros_like(theta)
    for i in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        fd[i] = (m.log_density(tp) - m.log_density(tm)) / (2 * eps)
    abs_err = np.abs(ad_grad - fd)
    denom = np.maximum(np.abs(ad_grad), 1e-12)
    rel_err = abs_err / denom
    mismatches = [
        {"index": int(i), "autodiff": float(ad_grad[i]), "finite_diff": float(fd[i])}
        for i in range(theta.size)
        if abs_err[i] > tol
    ]
    return {
        "max_abs_err": float(abs_err.max()) if theta.size else 0.0,
        "max_rel_err": float(rel_err.max()) if theta.size else 0.0,
        "mismatches": mismatches,
    }
