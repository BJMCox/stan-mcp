"""Inspect posterior conditioning in verified model parameter coordinates."""

from typing import Any

import numpy as np

from stan_mcp import diagnostics, draws, runstore
from stan_mcp.errors import RunError, ValidationError

# --- posterior geometry (linear degeneracy) ------------------------------
#
# Complements run_pathologies rather than overlapping it. Divergences, ESS, R-hat
# and E-BFMI between them catch non-linear pathology loudly (measured on Neal's
# funnel: ESS 18 of 4000, E-BFMI 0.06). What none of them catch is a posterior
# that is merely very ill-conditioned — collinear predictors burning gradient
# evaluations along a ridge while every one of those four looks healthy. That is
# what this is for, and it is why a funnel does NOT show up here: its dependence
# is not linear, so its max |r| is indistinguishable from noise (0.262 measured,
# against a 1/sqrt(ESS) noise scale of 0.236).
DEFAULT_MAX_DIMS = 500
_MIN_ABS_R = 0.5  # large enough to hurt sampling...
_MIN_FISHER_Z = 4.0  # ...and firmly established given the effective sample size
# The sample correlation matrix over p dims from n_eff effective draws is
# ill-conditioned BY CONSTRUCTION once p is a meaningful fraction of n_eff: the
# smallest eigenvalue collapses toward zero from noise alone. Eigen output is
# suppressed unless p is below n_eff / this.
#
# 4 (i.e. p/n_eff <= 0.25) rather than a stricter guess, from measuring what
# independent data actually yields: condition number 3.7 at p/n_eff = 0.10, 8.0
# at 0.25, 19.5 at 0.40, 219.6 at 0.80. Only 0.8 breaches the threshold at which
# a condition number is reported as an issue, so 0.25 keeps noise a comfortable
# 12x clear of it. A stricter ratio of 10 suppressed the eigen block on every
# real run tried — including a 151-dimension model at ESS 673, which would have
# needed near-perfect mixing to qualify — making the one statistic that catches
# multi-way degeneracy effectively unreachable.
_EIGEN_ESS_RATIO = 4
_EIGEN_GAP = 1.5  # min relative gap to the next eigenvalue before naming loadings


def _constrained_matrix(d: draws.Draws, bs_names: list[str]) -> np.ndarray:
    """Return model parameters in BridgeStan CSV order, excluding other outputs."""
    return draws.parameter_matrix(d, bs_names)


def _unconstrained_draws(
    run_id: str,
    d: draws.Draws,
    *,
    params: list[str] | None = None,
    max_dims: int = 0,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """(chain, draw, dim) unconstrained draws and their names.

    The sampler explores the UNCONSTRAINED space, so that is the geometry whose
    conditioning costs step size. Correlating constrained draws would report the
    mechanical dependence of simplex components and Cholesky factors as findings.
    Falls back to constrained draws, saying so, only if the model cannot be
    rebuilt at all."""
    from stan_mcp import compilation as _compile
    from stan_mcp import density

    inputs = runstore.load_inputs(run_id)
    source = inputs["source"]
    if source is not None and inputs.get("source_manifest"):
        source = _compile.restore_source(source, inputs["source_manifest"])
    n_chain, n_draw = d.n_chain, d.n_draw
    if source is None:
        raise RunError(f"run {run_id!r} has no persisted model.stan to unconstrain with")

    state = _compile.request_so(source)
    if state["status"] == "building":
        raise _GeometryPending(
            "building the BridgeStan artifact for this model (a run compiles only "
            "the CmdStan executable); retry shortly"
        )
    if state["status"] == "failed":
        raise RunError(
            f"parameter geometry unavailable: {state['error']}. "
            "The model parameter coordinates cannot be verified without BridgeStan."
        )

    m = density.get_model(source, inputs["data"])
    names = list(m.param_unc_names())
    selected = [
        i
        for i, name in enumerate(names)
        if not params or name in params or _block_of(name) in params
    ]
    if not selected:
        raise ValidationError(f"no parameters matched {params!r} (have {len(names)} elements)")
    if max_dims > 0 and len(selected) > max_dims:
        raise ValidationError(
            f"{len(selected)} unconstrained dimensions exceeds max_dims={max_dims}; "
            "scope with params or raise max_dims."
        )
    X = _constrained_matrix(d, m.param_names())
    unc = np.empty((X.shape[0], m.param_unc_num()))
    try:
        for i in range(X.shape[0]):
            unc[i] = m.param_unconstrain(X[i])
    except Exception as exc:  # noqa: BLE001
        # Stan's *_free functions VALIDATE the constraint they invert, and a draw
        # read back from a CSV only holds it to the written precision: at CmdStan's
        # default 6 sig figs a simplex sums to 1 to about 1e-8, which IS
        # simplex_free's tolerance, so ~1.5% of draws are rejected (measured:
        # 15/1000) — and one rejection aborts the whole loop. Runs sampled since
        # inference._OUTPUT_SIG_FIGS avoid it; a run recorded before that cannot be
        # unconstrained at all, so report constrained draws and say why instead of
        # failing the call. Same for any other constraint whose inverse validates
        # (unit_vector, corr_matrix, cholesky_factor_corr).
        cnames = list(m.param_names())
        keep = [
            i
            for i, name in enumerate(cnames)
            if not params or name in params or _block_of(name) in params
        ]
        if max_dims > 0 and len(keep) > max_dims:
            raise ValidationError(
                f"{len(keep)} constrained dimensions exceeds max_dims={max_dims}"
            ) from exc
        labels = {
            csv_name: label
            for name, spec in draws.schema(run_id)["params"].items()
            for label, csv_name in draws.spec_element_labels(name, spec)
        }
        return (
            X[:, keep].reshape(n_chain, n_draw, -1),
            [labels[cnames[i]] for i in keep],
            {
                "scale": "constrained",
                "scale_reason": (
                    f"could not unconstrain this run's draws ({exc}); reporting the "
                    "constrained scale, on which some dependence is mechanical. "
                    "Re-run the model to get the unconstrained geometry."
                ),
            },
        )
    return (
        unc[:, selected].reshape(n_chain, n_draw, -1),
        [names[i] for i in selected],
        {"scale": "unconstrained"},
    )


class _GeometryPending(RunError):
    """The BridgeStan artifact is still building; the caller should retry."""


def _fisher_z(r: float, n_eff: float) -> float:
    """atanh(r) has standard error 1/sqrt(n-3), which is how a correlation gets
    calibrated against the effective sample size rather than the draw count."""
    if n_eff <= 4 or abs(r) >= 1.0:
        return float("inf") if abs(r) >= 1.0 else 0.0
    return float(np.arctanh(r) * np.sqrt(n_eff - 3))


def _block_of(name: str) -> str:
    return name.split(".")[0].split("[")[0].split(":")[0]


def _mp_condition(p: int, n_eff: float) -> float:
    """Condition number independent parameters alone would give at this p/n_eff.

    Marchenko-Pastur: the sample eigenvalues of independent data spread to
    (1 -/+ sqrt(p/n))^2, so some conditioning is structural rather than a finding.
    Reported alongside the observed value so a moderate one is not over-read."""
    ratio = p / n_eff if n_eff > 0 else 1.0
    if ratio >= 1.0:
        return float("inf")
    s = np.sqrt(ratio)
    return float(((1 + s) / (1 - s)) ** 2)


def run_geometry(
    run_id: str,
    params: list[str] | None = None,
    max_dims: int = DEFAULT_MAX_DIMS,
    min_abs_r: float = _MIN_ABS_R,
    min_z: float = _MIN_FISHER_Z,
    top_k: int = 10,
) -> dict[str, Any]:
    """Posterior conditioning: correlated parameter pairs and the worst-
    conditioned direction, on the unconstrained scale the sampler works in."""
    import arviz as az

    meta = runstore.read_meta(run_id)
    if meta.get("method") != "sample":
        raise RunError(
            f"geometry needs MCMC draws with chain structure; run {run_id!r} "
            f"used {meta.get('method')!r}"
        )
    d = draws.load(run_id)
    try:
        u3, names, scale = _unconstrained_draws(run_id, d, params=params, max_dims=max_dims)
    except _GeometryPending as pending:
        # Not an error: the artifact is being built off the caller's thread.
        return {"run_id": run_id, "status": "preparing", "note": str(pending)}
    if d.header_warning:
        scale["header_warning"] = d.header_warning

    if params:
        want = set(params)
        sel = [i for i, nm in enumerate(names) if nm in want or _block_of(nm) in want]
        if not sel:
            raise ValidationError(f"no parameters matched {params!r} (have {len(names)} elements)")
        u3, names = u3[:, :, sel], [names[i] for i in sel]

    p = len(names)
    if max_dims > 0 and p > max_dims:
        raise ValidationError(
            f"{p} unconstrained dimensions exceeds max_dims={max_dims}; the "
            "correlation matrix is O(p^2) and its eigendecomposition O(p^3). "
            "Scope with `params`, or raise max_dims to accept the cost."
        )

    # Centre each chain before correlating. Pooling unmixed chains manufactures
    # correlation out of chain offsets: two chains sitting at different points
    # look collinear even when the within-chain geometry is spherical.
    resid = (u3 - u3.mean(axis=1, keepdims=True)).reshape(-1, p)
    sd = resid.std(0)
    keep = np.isfinite(sd) & (sd > 0)
    if keep.sum() < 2:
        return {
            "run_id": run_id,
            **scale,
            "note": "fewer than two varying parameters; no geometry to describe",
        }
    resid, names = resid[:, keep], [nm for nm, k in zip(names, keep, strict=True) if k]
    p = len(names)

    ess = np.asarray(
        az.ess(az.from_dict({"posterior": {"u": u3[:, :, keep]}}))["u"].values, dtype=float
    )
    # Left as NaN where arviz could not compute it, rather than clamped to a
    # fake value that would silently read as "calibrated against 1 draw".
    ess = np.where(np.isfinite(ess) & (ess > 0), ess, np.nan)
    finite_ess = ess[np.isfinite(ess)]

    # asarray: np.corrcoef is typed as possibly-scalar, which blocks 2-D indexing.
    R: np.ndarray = np.asarray(np.corrcoef(resid, rowvar=False), dtype=float)
    np.fill_diagonal(R, 0.0)
    out: dict[str, Any] = {
        "run_id": run_id,
        **scale,
        "n_dims": p,
        "min_ess": round(float(finite_ess.min()), 1) if finite_ess.size else None,
    }

    # Within-block pairs are collapsed: for theta[1..200] they are 19,900 of the
    # 20,000 pairs, all saying the same one thing, and would crowd out the single
    # cross-block pair that names the actual problem.
    blocks = np.array([_block_of(nm) for nm in names])
    iu, ju = np.triu_indices(p, k=1)
    same = blocks[iu] == blocks[ju]
    absr = np.abs(R[iu, ju])
    # abs: Fisher z carries the SIGN of r, and it is used here purely as strength
    # of evidence. Comparing the signed value against a positive threshold drops
    # every negatively correlated pair — which is the commonest form collinearity
    # takes, two coefficients trading off (measured: b1~b2 at r=-0.99995, z=-133,
    # silently discarded). Direction is already reported in `r`.
    # np.minimum, not min(): it propagates NaN, where Python's min() would pick
    # whichever operand came first and quietly launder a missing ESS into a real one.
    n_eff = np.minimum(ess[iu], ess[ju])
    # Below ~4 effective draws Fisher z is undefined, and arviz can hand back a
    # NaN ESS outright. Such a pair is NOT dropped: an inability to calibrate is
    # not evidence of absence, and a strong |r| whose ESS is too low to certify is
    # still the finding — often *because* the degeneracy destroyed the ESS.
    calibratable = np.isfinite(n_eff) & (n_eff > 4)
    raw_z = [
        _fisher_z(R[i, j], nz if np.isfinite(nz) else 0.0)
        for (i, j), nz in zip(zip(iu, ju, strict=True), n_eff, strict=True)
    ]
    zs = np.where(calibratable, np.abs(raw_z), np.nan)

    issues: list[str] = []
    pairs = []
    cross = ~same
    if cross.any():
        sel = np.flatnonzero(cross)
        # Both thresholds at 0 admits everything, per the 0 = unlimited convention.
        # An uncalibratable pair qualifies on |r| alone rather than vanishing.
        ok_z = (~calibratable[sel]) | (zs[sel] >= min_z)
        keepsel = sel[(absr[sel] >= min_abs_r) & ok_z]
        for k in keepsel[np.argsort(absr[keepsel])[::-1]][: max(top_k, 0) or None]:
            pair = {
                "a": names[iu[k]],
                "b": names[ju[k]],
                "r": round(float(R[iu[k], ju[k]]), 4),
                "z": round(float(zs[k]), 1) if calibratable[k] else None,
            }
            if not calibratable[k]:
                pair["z_note"] = f"effective sample size {n_eff[k]:.0f} too low to calibrate"
            pairs.append(pair)
    out["pairs"] = pairs
    out["max_abs_r_cross_block"] = round(float(absr[cross].max()), 4) if cross.any() else None
    if pairs:
        issues.append(
            f"{len(pairs)} correlated cross-block pair(s), worst |r|={abs(pairs[0]['r'])}"
        )

    within = []
    for b in dict.fromkeys(blocks):
        m = same & (blocks[iu] == b)
        if m.any():
            within.append(
                {
                    "block": b,
                    "n_pairs": int(m.sum()),
                    "max_abs_r": round(float(absr[m].max()), 4),
                    "median_abs_r": round(float(np.median(absr[m])), 4),
                }
            )
    if within:
        out["within_block"] = within
        for w in within:
            if w["max_abs_r"] >= min_abs_r:
                issues.append(
                    f"block {w['block']} internally correlated (max |r|={w['max_abs_r']})"
                )

    # Pairwise is structurally blind to multi-way degeneracy: if x3 ~ x1 + x2,
    # every pair can sit near 0.5 while the posterior is near-singular along
    # x1 + x2 - x3. The eigenvalues of the same matrix see it.
    np.fill_diagonal(R, 1.0)
    # A missing ESS cannot license the eigen block: treat it as the worst case,
    # since the guard exists precisely to stop reporting artifact.
    ess_min = float(finite_ess.min()) if finite_ess.size == p else 0.0
    if p * _EIGEN_ESS_RATIO > ess_min:
        out["eigen_note"] = (
            f"suppressed: {p} dimensions against min ESS {ess_min:.0f}. A sample "
            "correlation matrix is ill-conditioned by construction when p is not "
            f"small relative to n_eff, so the condition number would be artifact."
        )
    else:
        ev = np.linalg.eigvalsh(R)
        lo = max(float(ev[0]), 1e-12)
        eig: dict[str, Any] = {
            "condition_number": round(float(ev[-1]) / lo, 1),
            # Six significant figures, not five decimal places: a near-singular
            # direction's eigenvalue is exactly the small number worth reading,
            # and rounding places would report 1e-7 as a flat 0.0.
            "min_eigenvalue": float(f"{float(ev[0]):.6g}"),
            # What independent parameters alone would produce at this p/n_eff
            # (Marchenko-Pastur; matches measurement to ~0.4 and is conservative
            # beyond). Without it a condition number of 8 looks like mild
            # degeneracy when at p/n_eff = 0.25 it is exactly what noise gives.
            "noise_condition_number": round(_mp_condition(p, ess_min), 1),
        }
        if ev.size > 1 and ev[1] > 0 and ev[1] / lo >= _EIGEN_GAP:
            w = np.linalg.eigh(R)[1][:, 0]
            top = np.argsort(np.abs(w))[::-1][:5]
            eig["direction"] = [{"param": names[i], "loading": round(float(w[i]), 3)} for i in top]
        else:
            eig["direction_note"] = (
                "smallest eigenvalues are not separated, so the eigenvector is "
                "arbitrary within its subspace and loadings would be misleading"
            )
        out["eigen"] = eig
        if eig["condition_number"] > 100:
            issues.append(f"condition number {eig['condition_number']}")

    # Read convergence alone: the same R-hat, without spawning a
    # second stansummary subprocess and re-parsing every CSV for sampler stats to
    # reach one field. It also keeps this the only public entry point that is not
    # re-entered from inside another — which is what dragged the stansummary
    # binary into the offline geometry tests.
    rhat = diagnostics.convergence(run_id).get("max_rhat")
    if rhat is not None and rhat > diagnostics.RHAT_MAX:
        out["provisional"] = (
            f"max R-hat {rhat} > {diagnostics.RHAT_MAX}: the chains disagree, so there is not "
            "yet one posterior whose geometry this describes. Fix mixing first."
        )
    out.update({"verdict": "pass" if not issues else "warn", "issues": issues})
    return out
