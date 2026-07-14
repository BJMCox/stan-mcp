"""BridgeStan density tools (primary use).

Holds a bounded in-process LRU of loaded BridgeStan StanModel objects so
repeated density pokes on a hot model skip the reload. Provides log_density,
log_density_gradient, log_density_hessian, param_constrain, param_unconstrain,
param_info, and check_gradient (finite-diff vs autodiff). Every theta_unc is
length-validated against param_unc_num before the C++ call.

Implemented during the build (see design tool group B).
"""
