"""Small statistical summaries shared by predictive checks and calibration."""

from math import sqrt
from typing import Any

import numpy as np


def proportion(successes: int, total: int) -> dict[str, Any]:
    """A Bernoulli rate with a pointwise 95% Wilson interval across independent cases."""
    if not total:
        return {"count": successes, "total": 0, "estimate": None, "wilson_95": None}
    p = successes / total
    z = 1.959963984540054
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return {
        "count": successes,
        "total": total,
        "estimate": p,
        "wilson_95": [max(0.0, center - half), min(1.0, center + half)],
    }


def mean_uncertainty(
    values: np.ndarray, method: str | None, independent_draws: bool = False
) -> dict[str, Any]:
    """Uncertainty of a mean, distinct from spread in the sampled distribution."""
    if values.size < 2:
        return {"mcse": None, "reason": "at least two draws are required"}
    if np.all(values == values.flat[0]):
        return {"mcse": None, "reason": "constant draws do not identify Monte Carlo uncertainty"}
    if method == "fixed_param" and independent_draws:
        return {
            "mcse": float(np.std(values, ddof=1) / np.sqrt(values.size)),
            "method": "user-declared independent RNG draws",
        }
    if method != "sample":
        return {"mcse": None, "reason": "independence is unverified for this sampling method"}
    if values.shape[1] < 4:
        return {"mcse": None, "reason": "too few draws per chain for autocorrelation estimation"}
    import arviz as az

    error = float(az.mcse(values, method="mean"))
    ess = float(az.ess(values, method="mean"))
    if not np.isfinite(error) or not np.isfinite(ess):
        return {"mcse": None, "reason": "autocorrelation estimate is undefined"}
    return {"mcse": error, "ess": ess, "method": "autocorrelation-aware mean MCSE"}
