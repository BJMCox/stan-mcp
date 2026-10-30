"""Shared model stand-ins and persisted-run builders for toolchain-free tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import cmdstanpy
import numpy as np
from cmdstanpy.stanfit.metadata import InferenceMetadata

from stan_mcp import diagnostics, runstore


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


def _typed_run(tmp_path, monkeypatch, method="sample", ragged=False):
    """A run with a sidecar: one scalar and one length-2 vector, 2 chains x 2 draws.

    ragged=True drops chain b's second draw -- 2 draws vs 1, the shape a
    malformed or partially-written CSV set leaves behind -- so
    `Draws.truncated_to` is set.
    """
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = f"{method}-0001"
    runstore.write_meta(rid, {"method": method, "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "a.csv").write_text("# c\nlp__,theta,z.1,z.2\n-1,0.1,10,20\n-2,0.2,11,21\n")
    b_body = "-3,0.3,12,22\n" if ragged else "-3,0.3,12,22\n-4,0.4,13,23\n"
    (csvdir / "b.csv").write_text(f"# c\nlp__,theta,z.1,z.2\n{b_body}")
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": ["a.csv", "b.csv"],
                "params": {"theta": {"dims": [], "cols": [1]}, "z": {"dims": [2], "cols": [2, 3]}},
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    return rid


def _typed_run_wide(tmp_path, monkeypatch, method="sample"):
    """2 chains x 10 draws, scalar theta + 3-vector beta -- big enough that the
    max_draws/thin math is non-trivial (the minimal 2x2 `_typed_run` above
    collapses every budget check to a no-op)."""
    monkeypatch.setenv("STAN_MCP_HOME", str(tmp_path / "sm"))
    rid = f"{method}-0001"
    runstore.write_meta(rid, {"method": method, "status": "done"})
    csvdir = runstore.run_dir(rid, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    header = "lp__,theta,beta.1,beta.2,beta.3"
    for chain, offset in enumerate((0, 10)):
        rows = "\n".join(
            f"-1,{offset + t},{3 * (offset + t)},{3 * (offset + t) + 1},{3 * (offset + t) + 2}"
            for t in range(10)
        )
        (csvdir / f"chain{chain}.csv").write_text(f"# c\n{header}\n{rows}\n")
    diagnostics.draws.schema_path(rid).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": ["chain0.csv", "chain1.csv"],
                "params": {
                    "theta": {"dims": [], "cols": [1]},
                    "beta": {"dims": [3], "cols": [2, 3, 4]},
                },
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )
    return rid


def _write_run_with_stats(run_id, *, diverging, tree_depth, max_td=10):
    """A typed sample run: one scalar param (theta) plus the raw CmdStan sampler
    stats columns (`divergent__`, `treedepth__`, `accept_stat__`, `energy__`)
    run_diagnostics now reads directly off the typed draws layer, no idata."""
    rng = np.random.default_rng(0)
    nchain, ndraw = 2, 1000
    runstore.write_meta(
        run_id, {"method": "sample", "status": "done", "method_args": {"max_treedepth": max_td}}
    )
    csvdir = runstore.run_dir(run_id, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    theta = rng.normal(size=(nchain, ndraw))
    energy = rng.normal(size=(nchain, ndraw))
    files = []
    for chain in range(nchain):
        rows = "\n".join(
            f"{theta[chain, i]},{int(diverging)},{tree_depth},0.9,{energy[chain, i]}"
            for i in range(ndraw)
        )
        name = f"chain{chain}.csv"
        (csvdir / name).write_text(
            f"# c\ntheta,divergent__,treedepth__,accept_stat__,energy__\n{rows}\n"
        )
        files.append(name)
    diagnostics.draws.schema_path(run_id).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": files,
                "params": {"theta": {"dims": [], "cols": [0]}},
                "sampler_stats": {
                    "divergent__": {"dims": [], "cols": [1]},
                    "treedepth__": {"dims": [], "cols": [2]},
                    "accept_stat__": {"dims": [], "cols": [3]},
                    "energy__": {"dims": [], "cols": [4]},
                },
            }
        )
    )


def _write_csv_run(run_id):
    runstore.write_meta(run_id, {"method": "variational", "status": "done"})
    csvdir = runstore.run_dir(run_id, create=True) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    (csvdir / "c.csv").write_text(
        "# comment\nlp__,theta,beta.1,beta.2\n0,0.5,1.0,2.0\n0,0.7,1.5,2.5\n"
    )


def _write_typed_csv_run(run_id):
    """`_write_csv_run`'s single-file layout, plus the sidecar `extract_draws`
    now requires. Kept separate from `_write_csv_run` itself -- get_run_progress
    still reads that fixture straight off the untyped CSV and must keep working
    unmigrated."""
    _write_csv_run(run_id)
    diagnostics.draws.schema_path(run_id).write_text(
        json.dumps(
            {
                "schema": 2,
                "files": ["c.csv"],
                "params": {
                    "theta": {"dims": [], "cols": [1]},
                    "beta": {"dims": [2], "cols": [2, 3]},
                },
                "sampler_stats": {"lp__": {"dims": [], "cols": [0]}},
            }
        )
    )


def _fit_csv(path, method, extra_rows=False, values=(1.0, 2.0)):
    config = {"model": "input", "method": method, "id": 1}
    header = "lp__,x\n"
    body = "".join(f"-1,{v}\n" for v in values)
    if method == "sample":
        config.update(
            algorithm="hmc",
            engine="nuts",
            metric="diag_e",
            max_depth=10,
            num_samples=2,
            num_warmup=1,
            save_warmup=int(extra_rows),
            thin=1,
        )
        header = "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,x\n"
        body = "-1,1,1,1,1,0,1,999\n" if extra_rows else ""
        body += (
            "# Adaptation terminated\n# Step size = 1\n"
            "# Diagonal elements of inverse mass matrix:\n# 1\n"
        )
        body += "".join(f"-1,1,1,1,1,0,1,{v}\n" for v in values)
        body += (
            "#  Elapsed Time: 0 seconds (Warm-up)\n#  0 seconds (Sampling)\n#  0 seconds (Total)\n"
        )
    elif method == "variational":
        config.update(
            algorithm="meanfield",
            iter=10,
            grad_samples=1,
            elbo_samples=1,
            eta=1,
            tol_rel_obj=0.01,
            eval_elbo=1,
            output_samples=2,
        )
        body = "# Stepsize adaptation complete.\n# eta = 1\n-1,999\n" + body
    else:
        config.update(algorithm="lbfgs", save_iterations=int(extra_rows))
        body = ("-1,999\n" if extra_rows else "") + f"-1,{values[-1]}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"# {k} = {v}\n" for k, v in config.items()) + header + body)
    return path


class _GeneratedModel:
    def generate_quantities(self, data, previous_fit, gq_output_dir, **kwargs):
        if isinstance(previous_fit, list):
            previous_fit = cmdstanpy.from_csv(previous_fit)
        assert isinstance(
            previous_fit, (cmdstanpy.CmdStanMCMC, cmdstanpy.CmdStanMLE, cmdstanpy.CmdStanVB)
        )
        outputs = []
        for i, path in enumerate(previous_fit.runset.csv_files):
            rows = [
                ln for ln in Path(path).read_text().splitlines() if ln and not ln.startswith("#")
            ]
            target = Path(gq_output_dir) / f"gq_{i + 1}.csv"
            target.write_text(
                "g\n" + "".join(f"{2 * float(ln.split(',')[-1])}\n" for ln in rows[1:])
            )
            outputs.append(str(target))
        return SimpleNamespace(
            previous_fit=previous_fit,
            metadata=InferenceMetadata({"raw_header": "g"}),
            runset=SimpleNamespace(csv_files=outputs),
        )
