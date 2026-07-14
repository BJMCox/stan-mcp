"""Duck-typed stand-ins for BridgeStan models, for toolchain-free unit tests."""

import numpy as np


class FakeModel:
    def __init__(self, unc_num=1, names=None):
        self._unc = unc_num
        self._names = names or ["x"]

    def param_unc_num(self):
        return self._unc

    def param_num(self, *, include_tp=False, include_gq=False):
        return len(self._names)

    def param_names(self, *, include_tp=False, include_gq=False):
        return list(self._names)

    def param_unc_names(self):
        return list(self._names)

    def log_density(self, theta, *, propto=True, jacobian=True):
        return -0.5 * float(np.dot(theta, theta))

    def log_density_gradient(self, theta, *, propto=True, jacobian=True):
        theta = np.asarray(theta, dtype=float)
        return -0.5 * float(np.dot(theta, theta)), -theta

    def log_density_hessian(self, theta, *, propto=True, jacobian=True):
        theta = np.asarray(theta, dtype=float)
        n = theta.size
        return -0.5 * float(np.dot(theta, theta)), -theta, -np.eye(n)

    def param_constrain(self, theta, *, include_tp=False, include_gq=False, rng=None):
        return np.asarray(theta, dtype=float)

    def param_unconstrain_json(self, json_str):
        import json

        return np.asarray(list(json.loads(json_str).values()), dtype=float)
