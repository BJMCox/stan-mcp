"""Inference-method registry and kwargs builder for CmdStanPy."""

from errors import ValidationError

METHODS: dict[str, dict] = {
    "sample": {
        "cmd": "sample",
        "draws": True,
        "args": {"chains", "iter_warmup", "iter_sampling", "adapt_delta", "max_treedepth"},
    },
    "optimize": {"cmd": "optimize", "draws": False, "args": {"algorithm", "iter"}},
    "variational": {
        "cmd": "variational",
        "draws": True,
        "args": {"algorithm", "grad_samples", "elbo_samples"},
    },
    "pathfinder": {"cmd": "pathfinder", "draws": True, "args": {"num_paths", "draws"}},
    "laplace": {"cmd": "laplace_sample", "draws": True, "args": {"mode", "draws"}},
    "generate_quantities": {
        "cmd": "generate_quantities",
        "draws": True,
        "args": {"from_run_id"},
    },
}


def is_draws_method(method: str) -> bool:
    if method not in METHODS:
        raise ValidationError(f"unknown method {method!r}")
    return METHODS[method]["draws"]


def build_kwargs(method: str, seed, inits, method_args: dict) -> dict:
    if method not in METHODS:
        raise ValidationError(
            f"unknown method {method!r}; choose from {sorted(METHODS)}"
        )
    allowed = METHODS[method]["args"]
    ma = dict(method_args or {})
    bad = set(ma) - allowed
    if bad:
        raise ValidationError(
            f"method {method!r} does not accept {sorted(bad)}; allowed: {sorted(allowed)}"
        )
    kw = dict(ma)
    if seed is not None:
        kw["seed"] = seed
    if inits is not None:
        kw["inits"] = inits
    return kw
