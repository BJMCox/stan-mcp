"""BridgeStan density tools (primary use).

Holds a bounded in-process LRU of loaded BridgeStan StanModel objects so
repeated density pokes on a hot model skip the reload. Provides log_density,
log_density_gradient, log_density_hessian, param_constrain, param_unconstrain,
param_info, and check_gradient (finite-diff vs autodiff). Every theta_unc is
length-validated against param_unc_num before the C++ call.

Implemented during the build (see design tool group B).
"""

from collections import OrderedDict
from typing import TYPE_CHECKING

import compile as _compile
import config

if TYPE_CHECKING:
    import bridgestan

_CACHE: "OrderedDict[tuple, bridgestan.StanModel]" = OrderedDict()


def _new_model(source, data, stanc_args=None, seed=1234) -> "bridgestan.StanModel":
    import bridgestan

    so = _compile.compiled_so(source, stanc_args)
    return bridgestan.StanModel(str(so), data if data is not None else "", seed=seed)


def clear_cache() -> None:
    _CACHE.clear()


def get_model(source, data, stanc_args=None, seed=1234) -> "bridgestan.StanModel":
    key = (_compile.model_id(source, stanc_args), _compile.data_hash(data), seed)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    model = _new_model(source, data, stanc_args, seed)
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
