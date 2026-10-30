"""BridgeStan density tools (primary use).

Holds a bounded in-process LRU of loaded BridgeStan StanModel objects so
repeated density pokes on a hot model skip the reload. Provides log_density,
log_density_gradient, log_density_hessian, param_constrain, param_unconstrain,
param_info, and check_gradient (finite-diff vs autodiff). Every theta_unc is
length-validated against param_unc_num before the C++ call.

Implemented during the build (see design tool group B).
"""

import hashlib
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import numpy as np

from stan_mcp import compilation as _compile
from stan_mcp import config, errors

_CACHE: OrderedDict[tuple, _SerializedModel] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_NATIVE_LOCK = threading.RLock()


class _SerializedModel:
    """All BridgeStan objects share native autodiff state in non-threaded builds."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        with _NATIVE_LOCK:
            attribute = getattr(self._model, name)
        if not callable(attribute):
            return attribute

        def call(*args: Any, **kwargs: Any) -> Any:
            with _NATIVE_LOCK:
                return getattr(self._model, name)(*args, **kwargs)

        return call

    def __del__(self) -> None:
        with _NATIVE_LOCK:
            del self._model


def _new_model(
    source: str, data: Any, stanc_args: list[str] | None = None, seed: int = 1234
) -> _SerializedModel:
    import bridgestan

    so = _compile.compiled_so(source, stanc_args)
    # BridgeStan wants a JSON string (or path); resolve dicts / data_ids to JSON.
    canon = _compile.canonical_data(data)
    data_arg = json.dumps(canon) if canon is not None else ""
    with _NATIVE_LOCK:
        return _SerializedModel(bridgestan.StanModel(str(so), data_arg, seed=seed))


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def get_model(
    source: str, data: Any, stanc_args: list[str] | None = None, seed: int = 1234
) -> _SerializedModel:
    source = _compile.resolve_source(source, stanc_args)
    # Resolve mutable file inputs once; both cache identity and native loading
    # consume exactly this owned JSON snapshot.
    payload = json.dumps(_compile.canonical_data(data), sort_keys=True)
    key = (_compile.model_id(source), hashlib.sha256(payload.encode()).hexdigest()[:16], seed)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    source = _compile.refresh_source(source, stanc_args)
    key = (_compile.model_id(source), key[1], seed)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    # Build outside the lock (compilation/load is slow); the redundant work of a
    # concurrent miss on the same key is acceptable and the last writer wins.
    model = _new_model(source, json.loads(payload), stanc_args, seed)
    # Provisioning on the first load can change the toolchain identity.
    key = (_compile.refresh_source(source).manifest["model_id"], key[1], seed)
    with _CACHE_LOCK:
        _CACHE[key] = model
        _CACHE.move_to_end(key)
        while len(_CACHE) > config.MODEL_LRU_SIZE:
            _CACHE.popitem(last=False)
    return model


def param_info(stan: str, data: Any) -> dict[str, Any]:
    m = get_model(stan, data)
    return {
        "param_names": m.param_names(),
        "param_unc_names": m.param_unc_names(),
        "param_num": m.param_num(),
        "param_unc_num": m.param_unc_num(),
    }


def validate_data(stan: str, data: Any) -> dict[str, Any]:
    """Check that `data` satisfies the model's data block before a run. Building
    the BridgeStan model raises if a required variable is missing or malformed;
    on success return the parameter counts."""
    m = get_model(stan, data)
    return {"ok": True, "param_num": m.param_num(), "param_unc_num": m.param_unc_num()}


def _validate_theta(
    model: _SerializedModel, theta_unc: list[float] | np.ndarray | str
) -> np.ndarray:
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


def log_density(
    stan: str,
    data: Any,
    theta_unc: list[float] | np.ndarray | str,
    propto: bool = True,
    jacobian: bool = True,
) -> dict[str, Any]:
    m = get_model(stan, data)
    theta = _validate_theta(m, theta_unc)
    return {"lp": float(m.log_density(theta, propto=propto, jacobian=jacobian))}


def log_density_gradient(
    stan: str,
    data: Any,
    theta_unc: list[float] | np.ndarray | str,
    propto: bool = True,
    jacobian: bool = True,
) -> dict[str, Any]:
    m = get_model(stan, data)
    theta = _validate_theta(m, theta_unc)
    lp, grad = m.log_density_gradient(theta, propto=propto, jacobian=jacobian)
    return {"lp": float(lp), "gradient": np.asarray(grad).tolist()}


def log_density_hessian(
    stan: str,
    data: Any,
    theta_unc: list[float] | np.ndarray | str,
    propto: bool = True,
    jacobian: bool = True,
) -> dict[str, Any]:
    m = get_model(stan, data)
    theta = _validate_theta(m, theta_unc)
    lp, grad, hess = m.log_density_hessian(theta, propto=propto, jacobian=jacobian)
    return {
        "lp": float(lp),
        "gradient": np.asarray(grad).tolist(),
        "hessian": np.asarray(hess).tolist(),
    }


def param_constrain(
    stan: str,
    data: Any,
    theta_unc: list[float] | np.ndarray | str,
    include_tp: bool = False,
    include_gq: bool = False,
) -> dict[str, Any]:
    m = get_model(stan, data)
    theta = _validate_theta(m, theta_unc)
    # BridgeStan requires an rng when generated quantities are included.
    rng = m.new_rng(seed=1234) if include_gq else None
    vals = np.asarray(
        m.param_constrain(theta, include_tp=include_tp, include_gq=include_gq, rng=rng)
    ).tolist()
    names = m.param_names(include_tp=include_tp, include_gq=include_gq)
    return {"constrained": constrained_values(names, vals)}


def constrained_values(names: list[str], values: Iterable[float]) -> dict[str, Any]:
    """Convert BridgeStan's indexed scalar names to Stan's nested JSON arrays."""
    import stanio
    from stanio.json import process_dictionary

    header = ",".join(
        re.sub(r"\[([\d,]+)\]", lambda m: "." + m[1].replace(",", "."), name) for name in names
    )
    array = np.asarray(list(values), dtype=float)
    if len(names) != array.size:
        raise errors.ValidationError("constrained names and values have different lengths")
    if not names:
        return {}
    metadata = stanio.parse_header(header)
    return dict(
        process_dictionary(
            {name: variable.extract_reshape(array) for name, variable in metadata.items()}
        )
    )


def unconstrain_parameters(model: _SerializedModel, constrained: dict[str, Any]) -> np.ndarray:
    """Unconstrain nested Stan JSON, including complex fields inside tuple arrays."""
    names: list[str] = model.param_names()
    if not any(":" in name or name.endswith((".real", ".imag")) for name in names):
        return np.asarray(model.param_unconstrain_json(json.dumps(constrained)))

    # Stan 2.39's JSON var_context misreads complex fields in tuple arrays.
    # The numeric BridgeStan transform accepts their declared scalar ordering.
    template = constrained_values(names, np.zeros(len(names)))

    def validate_shape(actual: Any, expected: Any) -> Any:
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                raise errors.ValidationError("constrained tuple must contain named fields")
            result: dict[str, Any] = {}
            for key, item in expected.items():
                actual_key = key if key in actual else str(key)
                if actual_key not in actual:
                    raise errors.ValidationError(
                        f"missing constrained parameter or tuple field {key!r}"
                    )
                result[key] = validate_shape(actual[actual_key], item)
            return result
        if isinstance(expected, list):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                raise errors.ValidationError(
                    "constrained parameter has the wrong container dimensions"
                )
            return [validate_shape(a, e) for a, e in zip(actual, expected, strict=True)]
        return float(actual)

    canonical = validate_shape(constrained, template)
    values: list[float] = []
    for name in names:
        root = re.split(r"[.:]", name, maxsplit=1)[0]
        value = canonical[root]
        for separator, component in re.findall(r"([.:])([^.:]+)", name[len(root) :]):
            if separator == ":":
                value = value[int(component)]
            else:
                index = {"real": 0, "imag": 1}.get(component)
                value = value[int(component) - 1 if index is None else index]
        values.append(value)
    return np.asarray(model.param_unconstrain(np.asarray(values, dtype=float)))


def param_unconstrain(stan: str, data: Any, constrained: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(constrained, str):
        constrained = _compile.read_json_file(constrained, "constrained")
    if not isinstance(constrained, dict):
        raise errors.ValidationError("constrained must contain a named parameter object")
    m = get_model(stan, data)
    if any("." in name or "[" in name or ":" in name for name in constrained):
        constrained = constrained_values(list(constrained), list(constrained.values()))
    theta = unconstrain_parameters(m, constrained)
    return {"theta_unc": np.asarray(theta).tolist()}


def check_gradient(
    stan: str,
    data: Any,
    theta_unc: list[float] | np.ndarray | str,
    eps: float = 1e-6,
    tol: float = 1e-4,
) -> dict[str, Any]:
    m = get_model(stan, data)
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
