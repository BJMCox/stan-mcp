"""Predictive discrepancies from explicit observed and replicated quantities."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from stan_mcp import compilation, draws, runstore
from stan_mcp.batch.store import atomic_json, read_json
from stan_mcp.errors import RunError, ValidationError
from stan_mcp.validation_stats import mean_uncertainty, proportion

STATISTICS = ("mean", "sd", "min", "max", "median", "q05", "q95", "zero_fraction")


def statistic(values: np.ndarray, name: str, event_axes: tuple[int, ...]) -> np.ndarray:
    """Reduce only event axes, leaving chain and draw axes untouched."""
    if name == "sd":
        size = int(np.prod([values.shape[axis] for axis in event_axes]))
        if size < 2:
            raise ValidationError("sd needs at least two observations")
        return np.std(values, axis=event_axes, ddof=1)
    if name in {"q05", "q95", "median"}:
        q = {"q05": 0.05, "q95": 0.95, "median": 0.5}[name]
        return np.quantile(values, q, axis=event_axes)
    if name == "zero_fraction":
        return np.mean(values == 0, axis=event_axes)
    return {"mean": np.mean, "min": np.min, "max": np.max}[name](values, axis=event_axes)


def _plot(rows: list[dict[str, Any]], distributions: list[np.ndarray], path: Path) -> None:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7, 2.6 * len(rows)), layout="constrained")
    for index, (row, values) in enumerate(zip(rows, distributions, strict=True), start=1):
        axis = figure.add_subplot(len(rows), 1, index)
        axis.hist(values.ravel(), bins=30)
        axis.axvline(row["observed_statistic"], color="red", label="observed")
        axis.set_title(f"{row['replicated']}: {row['statistic']}")
        axis.set_ylabel("Draw count")
        axis.legend()
    figure.savefig(path)


def check_predictive(request_file: str) -> dict[str, Any]:
    request = read_json(request_file)
    if set(request) - {"run_id", "observed", "checks", "kind", "independent_draws"}:
        raise ValidationError("unknown predictive request fields")
    run_id = request["run_id"]
    meta = runstore.read_meta(run_id)
    if meta.get("status") != "done":
        raise RunError("predictive checks require a completed run")
    kind = request.get("kind", "posterior")
    if kind not in {"prior", "posterior"}:
        raise ValidationError("kind must be prior or posterior")
    method = draws.sampling_method(run_id)
    independent = request.get("independent_draws", False)
    if type(independent) is not bool:
        raise ValidationError("independent_draws must be a boolean")
    if kind == "posterior" and method == "fixed_param":
        raise ValidationError("fixed_param simulation has no posterior sampling law")
    observed = compilation.canonical_data(request.get("observed"))
    if not isinstance(observed, dict):
        raise ValidationError("observed must resolve to a JSON object")
    checks = request.get("checks")
    if not isinstance(checks, list) or not 1 <= len(checks) <= 20:
        raise ValidationError("checks must contain 1 to 20 explicit variable/statistic pairs")
    d = draws.load(run_id)
    if d.truncated_to is not None:
        raise RunError("predictive checks require complete, equal-length chains")
    rows: list[dict[str, Any]] = []
    distributions: list[np.ndarray] = []
    for check in checks:
        if not isinstance(check, dict) or set(check) != {"observed", "replicated", "statistic"}:
            raise ValidationError("each check needs observed, replicated, and statistic")
        name = check["statistic"]
        if name not in STATISTICS:
            raise ValidationError(f"statistic must be one of {STATISTICS}")
        if check["observed"] not in observed or check["replicated"] not in d.params:
            raise ValidationError("observed or replicated variable is missing")
        actual = np.asarray(observed[check["observed"]], dtype=float)
        replicated = d.params[check["replicated"]]
        if replicated.shape[2:] != actual.shape or not actual.size:
            raise ValidationError("observed and replicated event shapes must match and be nonempty")
        if not np.isfinite(actual).all() or not np.isfinite(replicated).all():
            raise ValidationError("predictive quantities must be finite")
        value = float(statistic(actual, name, tuple(range(actual.ndim))))
        samples = statistic(replicated, name, tuple(range(2, replicated.ndim)))
        if not np.isfinite(value) or not np.isfinite(samples).all():
            raise ValidationError("predictive statistic overflowed")
        tail = samples >= value
        row: dict[str, Any] = {
            **check,
            "observed_statistic": value,
            "replicated_mean": float(np.mean(samples)),
            "replicated_quantiles": dict(
                zip(
                    ("q05", "q50", "q95"),
                    np.quantile(samples, [0.05, 0.5, 0.95]).tolist(),
                    strict=True,
                )
            ),
            "mean_uncertainty": mean_uncertainty(samples, method, independent),
            "tail_probability_ge": float(np.mean(tail)),
            "equality_fraction": float(np.mean(samples == value)),
            "tail_uncertainty": mean_uncertainty(tail.astype(float), method, independent),
        }
        if method == "fixed_param" and independent:
            row["tail_uncertainty"]["wilson_95"] = proportion(int(tail.sum()), tail.size)[
                "wilson_95"
            ]
        rows.append(row)
        distributions.append(samples)
    directory = runstore.run_dir(run_id) / "predictive" / uuid4().hex
    directory.mkdir(parents=True)
    report = {
        "schema": "stan-mcp.predictive.v1",
        "run_id": run_id,
        "kind": kind,
        "sampling_method": method,
        "independent_draws_declared": independent,
        "chains": d.n_chain,
        "draws_per_chain": d.n_draw,
        "seed": meta.get("seed"),
        "interpretation": "Predictive discrepancies are not calibrated goodness-of-fit p-values.",
        "uncertainty_note": "MCSE assumes the declared draw law and adequate MCMC convergence.",
        "checks": rows,
        "plot": str(directory / "checks.svg"),
    }
    if d.header_warning:
        report["header_warning"] = d.header_warning
    if method not in {"sample", "fixed_param"}:
        report["sampling_warning"] = (
            "Approximate or unverified draws; predictive results inherit that limitation."
        )
    atomic_json(directory / "request.json", {**request, "observed": observed})
    _plot(rows, distributions, directory / "checks.svg")
    atomic_json(directory / "report.json", report)
    return {**report, "report_file": str(directory / "report.json")}
