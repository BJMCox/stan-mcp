"""CmdStan's own `stansummary`, as plain dicts.

MCMC moments, R-hat and ESS come from CmdStan rather than arviz because CmdStan is
the reference implementation and the two disagree: on `sample-0013`'s `theta`,
arviz's Dataset `method="tail"` gives 506.52 where `stansummary` and arviz's own
bare-array path at `prob=(0.05, 0.95)` both give 403.87.

We shell out and parse the `-c` CSV rather than calling cmdstanpy's `fit.summary()`,
whose only job is to load that same CSV into a pandas DataFrame.

Only ever call this for `method == "sample"`. `stansummary` does not refuse other
input: handed a variational run it reports `R_hat 0.999997` and `ESS_bulk 991.432`
for IID approximate draws — meaningless numbers in valid-looking output.

Every failure of this module's public surface (`summarize`, `parse`) is a
`RunError`. That is a contract, not an accident: `diagnostics._convergence`
degrades a failure here into a warning so `run_diagnostics` still answers, and a
third-party exception type leaking through instead used to escape both
`run_diagnostics` and `run_geometry` as an unhandled `ValueError`.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from errors import RunError

# stansummary's own default is 2 significant figures, which would silently round
# every number this tool reports. 18 is CmdStan's maximum and matches the output
# precision runs are now sampled at.
SIG_FIGS = 18
_TIMEOUT_S = 300


def _binary() -> Path:
    """Path to CmdStan's `stansummary`, or a `RunError` saying how to get one.

    `cmdstanpy.cmdstan_path()` raises a bare `ValueError` when CmdStan is absent
    ("No CmdStan installation found", "No CmdStan directory, path ... does not
    exist") — a third-party type no caller of this module catches, which is how
    it used to propagate out of `run_diagnostics` and `run_geometry` instead of
    degrading to a warning.

    Resolve-and-report only, deliberately NOT `toolchain.ensure_cmdstan()`: that
    *installs* CmdStan on demand, so routing a read-only diagnostic through it
    would kick off a multi-minute download as a side effect of asking for R-hat.
    The remediation wording is `toolchain`'s, so the hint is the same either way.
    """
    try:
        import cmdstanpy

        root = Path(cmdstanpy.cmdstan_path())
    except Exception as exc:  # noqa: BLE001 - surfaced as a remediation message
        raise RunError(
            "CmdStan is not available, so its stansummary cannot be run "
            f'({exc}). Run `python -c "import cmdstanpy; '
            'cmdstanpy.install_cmdstan()"` manually.'
        ) from exc
    p = root / "bin" / "stansummary"
    if not p.exists():
        raise RunError(f"CmdStan's stansummary binary is not present at {p}")
    return p


def _to_float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def parse(text: str) -> dict[str, dict[str, float]]:
    """`{cmdstan_name: {column: value}}` from stansummary's --csv_filename output."""
    # stansummary appends trailing `#`-prefixed commentary after the data (model
    # name, warmup/sampling timing, a glossary of the column meanings) — legitimate
    # non-data lines, not malformed rows. Stan identifiers can't start with `#`, so
    # this filter never hides a real parameter.
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise RunError(
            "stansummary produced an empty summary; it leaves a 0-byte CSV behind when it fails"
        )
    header = [h.strip() for h in lines[0].split(",")]
    out: dict[str, dict[str, float]] = {}
    for ln in lines[1:]:
        parts = ln.split(",")
        # Stan identifiers cannot contain a comma, so splitting on it is safe;
        # stansummary quotes the name column.
        name = parts[0].strip().strip('"')
        if len(parts) != len(header):
            # A column-count mismatch means the output is malformed. We must not
            # guess which parameter went missing by silently dropping the row —
            # that would return a summary with one parameter quietly absent and
            # zero signal that it happened.
            raise RunError(
                f"stansummary row {name!r} has {len(parts)} columns, expected "
                f"{len(header)} (matching the header); malformed output cannot "
                "be safely parsed"
            )
        out[name] = {k: _to_float(v) for k, v in zip(header[1:], parts[1:], strict=True)}
    if not out:
        # A header with no data rows is the same "invisible damage" as the
        # 0-byte-CSV case above, reached through a second path: don't let it
        # parse as a model with zero parameters.
        raise RunError("stansummary produced a header but no parameter rows")
    return out


def summarize(
    csv_paths: list[str],
    percentiles: tuple[int, ...] = (5, 50, 95),
    sig_figs: int = SIG_FIGS,
) -> dict[str, dict[str, float]]:
    if not csv_paths:
        raise RunError("stansummary needs at least one draws CSV")
    with tempfile.TemporaryDirectory() as td:
        # --csv_filename requires a path that does not yet exist.
        out = Path(td) / "summary.csv"
        cmd = [
            str(_binary()),
            "--sig_figs",
            str(sig_figs),
            "--percentiles",
            ",".join(str(p) for p in percentiles),
            "--csv_filename",
            str(out),
            # Input files LAST: --include_param and friends are greedy and will
            # consume a positional path if any option precedes it, failing with
            # "Unrecognized parameter(s): '..._1[csv]'".
            *csv_paths,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            # TimeoutExpired is not a RunError and would escape this module's
            # contract exactly as cmdstan_path()'s ValueError did.
            raise RunError(
                f"stansummary did not finish within {_TIMEOUT_S}s over "
                f"{len(csv_paths)} draws CSV(s)"
            ) from exc
        except OSError as exc:
            raise RunError(f"could not execute stansummary at {cmd[0]}: {exc}") from exc
        if proc.returncode != 0:
            # Check the EXIT CODE, never the file: a failure still creates the
            # output path, empty, which would parse as a model with no parameters.
            detail = (proc.stderr or proc.stdout or "").strip()[:400]
            raise RunError(f"stansummary failed (exit {proc.returncode}): {detail}")
        if not out.exists():
            raise RunError("stansummary reported success but wrote no summary")
        return parse(out.read_text())
