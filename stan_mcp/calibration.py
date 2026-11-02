"""Descriptive simulation-based calibration for explicit joint-simulation studies."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from safetensors.numpy import save_file

from stan_mcp import diagnostics, draws, runstore, studies
from stan_mcp.batch.store import atomic_json, file_hash, fingerprint, read_json
from stan_mcp.errors import RunError, ValidationError
from stan_mcp.validation_stats import proportion


def rank_interval(
    values: np.ndarray, truth: float, level: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Randomized exact-tie rank and finite-draw order-statistic interval."""
    n = values.size
    below = int(np.sum(values < truth))
    ties = int(np.sum(values == truth))
    rank = below + int(rng.integers(ties + 1))
    ordered = np.sort(values)
    k = max(1, int(np.floor((n + 1) * (1 - level) / 2)))
    ell = min(n, int(np.ceil((n + 1) * (1 + level) / 2)))
    lower, upper = float(ordered[k - 1]), float(ordered[ell - 1])
    return {
        "rank": rank,
        "below": below,
        "ties": ties,
        "lower": lower,
        "upper": upper,
        "covered": int(lower <= truth <= upper),
        "order_indices": [k, ell],
        "finite_draw_target": (ell - k) / (n + 1),
    }


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7, 2.6 * max(1, len(rows))), layout="constrained")
    for index, row in enumerate(rows, start=1):
        axis = figure.add_subplot(len(rows), 1, index)
        counts = row["rank_histogram"]["counts"]
        axis.bar(np.arange(len(counts)), counts, label="observed")
        expected = (
            np.asarray(row["rank_histogram"]["iid_reference_probabilities"]) * row["assessable"]
        )
        axis.plot(np.arange(len(counts)), expected, color="red", label="iid reference only")
        axis.set_title(f"{row['template_id']}: {row['quantity']}")
        axis.set_ylabel("Case count")
        axis.legend()
    if not rows:
        figure.text(0.1, 0.5, "No assessable calibration quantities")
    figure.savefig(path)


def check_sbc(request_file: str) -> dict[str, Any]:
    request = read_json(request_file)
    if set(request) - {"study_id", "draws", "thin", "bins", "interval", "seed"}:
        raise ValidationError("unsupported SBC request fields")
    n = request.get("draws", 100)
    thin = request.get("thin", 1)
    bins = request.get("bins", 10)
    level = request.get("interval", 0.9)
    seed = request.get("seed", 0)
    if type(n) is not int or not 1 <= n <= 100000:
        raise ValidationError("draws must be between 1 and 100000")
    if type(thin) is not int or thin < 1 or type(bins) is not int or not 2 <= bins <= 50:
        raise ValidationError("thin must be positive and bins must be between 2 and 50")
    if isinstance(level, bool) or not isinstance(level, (float, int)) or not 0 < level < 1:
        raise ValidationError("interval must be between zero and one")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValidationError("SBC seed must be an integer in [0, 2**32)")
    study_id = request["study_id"]
    directory = studies.study_dir(study_id)
    meta = runstore.read_meta(study_id)
    if meta["status"] not in {"done", "failed", "cancelled"}:
        raise RunError("wait for the study to finish or cancel before checking SBC")
    inventory = read_json(directory / "prepared.json")
    if fingerprint(inventory) != meta.get("prepared_hash"):
        raise RunError("prepared study snapshot changed")
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    outcomes = []
    records: list[dict[str, Any]] = []
    quantities: list[tuple[str, str]] = []
    for index, task in enumerate(inventory["tasks"]):
        targets = task.get("simulate", {}).get("truth_variables", {})
        if not targets:
            continue
        for name in targets:
            group = groups.setdefault((task["template_id"], name), {"planned": 0, "errors": []})
            group["planned"] += 1
        if len(groups) > 50:
            raise ValidationError("SBC supports at most 50 template/quantity groups")
        state = studies.task_state(directory, index)
        outcome: dict[str, Any] = {
            "task_id": task["id"],
            "status": state["status"],
            "attempted": bool(state["attempts"]),
            "assessed_quantities": 0,
            "requested_quantities": len(targets),
        }
        outcomes.append(outcome)
        if state["status"] != "done":
            outcome["error"] = state.get("error", "case did not complete")
            continue
        for stage in ("simulate", "fit"):
            studies.verify_artifacts(state[stage]["run_id"], state[stage]["hashes"])
        run_id = state["fit"]["run_id"]
        try:
            d = draws.load(run_id)
            if d.truncated_to is not None:
                raise RunError("ragged chains cannot enter SBC")
        except (RunError, ValueError) as exc:
            outcome.update(analysis_status="unassessable", error=str(exc)[:800])
            continue
        if d.header_warning:
            outcome["header_warning"] = d.header_warning
        outcome["sampling_method"] = draws.sampling_method(run_id)
        if outcome["sampling_method"] == "sample":
            try:
                outcome["diagnostics"] = diagnostics.run_diagnostics(run_id)
            except RunError as exc:
                outcome["diagnostics"] = {"verdict": "unknown", "reason": str(exc)[:300]}
        for name in targets:
            group = groups[(task["template_id"], name)]
            try:
                truth = np.asarray(state["truth"][name], dtype=float)
                values = d.params[name]
                if truth.shape != values.shape[2:]:
                    raise ValidationError("truth and fitted quantity shapes differ")
                shape = list(truth.shape)
                if group.setdefault("shape", shape) != shape:
                    raise ValidationError("quantity shape changed across study cases")
                retained = values[:, ::thin].swapaxes(0, 1).reshape(-1, *truth.shape)
                if len(retained) < n:
                    raise ValidationError(
                        "too few retained draws for the requested fixed SBC count"
                    )
                retained = retained[:n]
                if not np.isfinite(truth).all() or not np.isfinite(retained).all():
                    raise ValidationError("truth and posterior draws must be finite")
                flat = retained.reshape(n, -1)
                labels = list(draws.element_labels(name, truth.shape))
                if len(labels) + len(rows) > 50 and any(
                    (task["template_id"], label) not in rows for label, _ in labels
                ):
                    raise ValidationError("SBC reports support at most 50 scalar quantities")
                for position, (label, _) in enumerate(labels):
                    key = task["template_id"], label
                    if key not in rows:
                        rows[key] = []
                        quantities.append(key)
                    scalar = float(truth.reshape(-1)[position])
                    tie_seed = int(fingerprint([seed, task["id"], label]), 16)
                    result = rank_interval(
                        flat[:, position], scalar, level, np.random.default_rng(tie_seed)
                    )
                    record = {
                        **result,
                        "task_index": index,
                        "quantity_index": quantities.index(key),
                        "truth": scalar,
                    }
                    rows[key].append(record)
                    records.append(record)
                outcome["assessed_quantities"] += 1
            except (RunError, ValidationError, KeyError, ValueError) as exc:
                if len(group["errors"]) < 5:
                    group["errors"].append({"task_id": task["id"], "error": str(exc)[:300]})
    if not groups:
        raise ValidationError("study has no explicit simulated truth mappings for SBC")
    summaries = []
    probabilities = np.bincount(np.arange(n + 1) * bins // (n + 1), minlength=bins) / (n + 1)
    for (template, label), samples in rows.items():
        base = label.split("[")[0]
        group = groups[(template, base)]
        total = group["planned"]
        count = len(samples)
        coverage = proportion(sum(r["covered"] for r in samples), count)
        histogram = np.bincount([r["rank"] * bins // (n + 1) for r in samples], minlength=bins)
        summaries.append(
            {
                "template_id": template,
                "quantity": label,
                "planned": total,
                "assessable": count,
                "unresolved": total - count,
                "coverage": {
                    **coverage,
                    "requested_interval": level,
                    "order_indices": samples[0]["order_indices"],
                    "continuous_iid_finite_draw_target": samples[0]["finite_draw_target"],
                    "all_case_empirical_bounds": [
                        coverage["count"] / total,
                        (coverage["count"] + total - count) / total,
                    ],
                },
                "rank_histogram": {
                    "counts": histogram.tolist(),
                    "iid_reference_probabilities": probabilities.tolist(),
                    "pointwise_rate_intervals_95": [
                        proportion(int(c), count)["wilson_95"] for c in histogram
                    ],
                },
                "ties": sum(r["ties"] > 0 for r in samples),
                "errors": group["errors"],
            }
        )
    report_dir = directory / "sbc" / uuid4().hex
    report_dir.mkdir(parents=True)
    integer_fields = ("task_index", "quantity_index", "rank", "below", "ties", "covered")
    arrays: dict[str, np.ndarray] = {
        name: np.asarray([r[name] for r in records], dtype=np.int64) for name in integer_fields
    }
    arrays.update(
        {
            name: np.asarray([r[name] for r in records], dtype=np.float64)
            for name in ("truth", "lower", "upper")
        }
    )
    save_file(arrays, report_dir / "ranks.safetensors")
    report = {
        "schema": "stan-mcp.sbc.v1",
        "study_id": study_id,
        "study_seed": inventory["seed"],
        "seed": seed,
        "tie_seed_derivation": "SHA-256 of JSON [seed, task_id, quantity]",
        "numpy_version": np.__version__,
        "draws_per_case": n,
        "within_chain_thin": thin,
        "rank_ties": "Exact equality, with uniform integer tie-breaking on 0..ties inclusive.",
        "scope": "Selected user-supplied quantities under the declared joint simulator.",
        "validity": "Descriptive ranks. MCMC independence and uniformity remain unverified.",
        "coverage_note": "Wilson intervals use independent cases. Missing cases can bias results.",
        "reference_note": (
            "Rank references assume independent correct posterior draws. "
            "The coverage target also assumes a continuous quantity."
        ),
        "quantities": summaries,
        "attempted_cases": sum(o["attempted"] for o in outcomes),
        "planned_cases": len(outcomes),
        "completed_cases": sum(o["status"] == "done" for o in outcomes),
        "assessable_cases": sum(
            o["assessed_quantities"] == o["requested_quantities"] for o in outcomes
        ),
        "case_status_counts": {
            status: sum(o["status"] == status for o in outcomes)
            for status in {o["status"] for o in outcomes}
        },
        "unverified_header_cases": sum("header_warning" in o for o in outcomes),
        "sampling_method_counts": {
            method: sum(o.get("sampling_method", "unavailable") == method for o in outcomes)
            for method in sorted({o.get("sampling_method") or "unavailable" for o in outcomes})
        },
        "fit_diagnostic_verdicts": {
            verdict: sum(
                o.get("diagnostics", {}).get("verdict", "unavailable") == verdict for o in outcomes
            )
            for verdict in ("pass", "warn", "unknown", "unavailable")
        },
        "unassessable_quantities": [
            {"template_id": template, "quantity": name, **group}
            for (template, name), group in groups.items()
            if not any(
                t == template and (label == name or label.startswith(name + "["))
                for t, label in rows
            )
        ],
        "ranks_file": str(report_dir / "ranks.safetensors"),
        "ranks_sha256": file_hash(report_dir / "ranks.safetensors"),
        "ranks_axes": ["record"],
        "cases_file": str(report_dir / "cases.json"),
        "plot": str(report_dir / "ranks.svg"),
    }
    if any(o.get("sampling_method") in {"variational", "pathfinder", "laplace"} for o in outcomes):
        report["sampling_warning"] = (
            "Approximate fits are included; ranks can reflect approximation error."
        )
    atomic_json(report_dir / "request.json", request)
    atomic_json(
        report_dir / "cases.json",
        {
            "tasks": [t["id"] for t in inventory["tasks"]],
            "outcomes": outcomes,
            "quantity_axes": quantities,
        },
    )
    _plot(summaries, report_dir / "ranks.svg")
    atomic_json(report_dir / "report.json", report)
    # Full per-bin uncertainty and error lists stay in the durable report.
    bounded_rows = []
    for row in summaries[:10]:
        histogram = dict(row["rank_histogram"])
        histogram.pop("pointwise_rate_intervals_95")
        bounded_rows.append({**row, "rank_histogram": histogram, "errors": row["errors"][:1]})
    return {
        **report,
        "quantities": bounded_rows,
        "quantity_count": len(summaries),
        "omitted_quantities": max(0, len(summaries) - len(bounded_rows)),
        "unassessable_quantities": [
            {**group, "errors": group["errors"][:1]}
            for group in report["unassessable_quantities"][:10]
        ],
        "report_file": str(report_dir / "report.json"),
    }
