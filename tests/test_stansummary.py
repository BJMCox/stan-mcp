"""Tests for the stansummary wrapper.

Parsing is tested without the binary; the binary's own behaviour is covered by the
live smoke tests. The failure paths matter most: stansummary leaves a 0-byte CSV
behind when it fails, so a wrapper that trusts the file's existence parses an empty
summary as a model with no parameters.
"""

import pytest

import errors
import stansummary

SAMPLE = """name,Mean,MCSE,StdDev,MAD,5%,50%,95%,ESS_bulk,ESS_tail,ESS_bulk/s,R_hat
"lp__",-79.9936610583333589,0.606804122372384,8.70491691166983,7.877681681,-95.73361925,-79.3718975,-65.7694671,209.617931654,324.534706969,6352.058535,1.00083007
"theta",0.575955,0.004735,0.179269,0.201338,0.270112,0.584963,0.860466,1496.0727,403.871,45335.5,1.00139
"z.1",0.011508,0.033099,0.958229,0.925627,-1.5265,-0.032575,1.61126,906.343,481.432,27464.9,1.00834
"""


def test_parse_keys_by_cmdstan_name_and_coerces_floats():
    got = stansummary.parse(SAMPLE)
    assert set(got) == {"lp__", "theta", "z.1"}
    assert got["theta"]["ESS_bulk"] == pytest.approx(1496.0727)
    assert got["theta"]["ESS_tail"] == pytest.approx(403.871)
    assert got["theta"]["R_hat"] == pytest.approx(1.00139)
    assert got["z.1"]["Mean"] == pytest.approx(0.011508)
    # quotes stripped from the name, values are floats not strings
    assert isinstance(got["theta"]["Mean"], float)


def test_parse_handles_nan_and_inf():
    text = 'name,Mean,ESS_bulk,R_hat\n"lp__",0,nan,nan\n"theta",1.5,inf,1.0\n'
    got = stansummary.parse(text)
    assert got["lp__"]["ESS_bulk"] != got["lp__"]["ESS_bulk"]  # nan
    assert got["theta"]["ESS_bulk"] == float("inf")


def test_parse_rejects_empty_output():
    """A 0-byte CSV is what a failed stansummary leaves behind."""
    with pytest.raises(errors.RunError, match="empty"):
        stansummary.parse("")


def test_parse_rejects_header_only_output():
    """A header with zero data rows is the same invisible-damage failure as the
    0-byte-CSV case, reached through a second, unguarded path: it must not
    silently parse as a model with no parameters."""
    with pytest.raises(errors.RunError, match="no parameter rows"):
        stansummary.parse("name,Mean,ESS_bulk,R_hat\n")


def test_parse_rejects_row_with_too_few_columns():
    """A dropped column must raise, not vanish silently — the message names
    the offending row."""
    text = 'name,Mean,ESS_bulk,R_hat\n"theta",1.5,10\n'
    with pytest.raises(errors.RunError, match="theta"):
        stansummary.parse(text)


def test_parse_rejects_row_with_too_many_columns():
    text = 'name,Mean,ESS_bulk,R_hat\n"theta",1.5,10,1.0,999\n'
    with pytest.raises(errors.RunError, match="theta"):
        stansummary.parse(text)


def test_parse_ignores_trailing_comment_lines():
    """The real binary appends `#`-prefixed commentary (model name, timing,
    a column-meaning glossary) after the data rows. These are not malformed
    rows and must not raise — only genuine column-count mismatches should."""
    text = (
        "name,Mean,ESS_bulk,R_hat\n"
        '"theta",1.5,10,1.0\n'
        "# Inference for Stan model: _9531cf59c37e1977_model\n"
        "# 2 chains: each with iter=300; warmup=300; thin=1; 300 iterations saved.\n"
        "# \n"
        "# Samples were drawn using hmc with nuts.\n"
    )
    got = stansummary.parse(text)
    assert got == {"theta": {"Mean": 1.5, "ESS_bulk": 10.0, "R_hat": 1.0}}


def test_summarize_requires_csvs():
    with pytest.raises(errors.RunError, match="at least one"):
        stansummary.summarize([])


def test_summarize_raises_on_nonzero_exit(monkeypatch, tmp_path):
    """Detect failure by exit code, never by output-file existence."""
    import subprocess

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "--include_param: Unrecognized parameter(s): 'x[csv]'"

    monkeypatch.setattr(stansummary, "_binary", lambda: tmp_path / "stansummary")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(errors.RunError, match="Unrecognized parameter"):
        stansummary.summarize(["a.csv"])


def test_summarize_passes_sig_figs_18_and_percentiles(monkeypatch, tmp_path):
    """--sig_figs defaults to 2; unset, every reported number rounds to 2 s.f."""
    import subprocess

    seen = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        # stansummary writes its CSV to the --csv_filename path
        out = cmd[cmd.index("--csv_filename") + 1]
        with open(out, "w") as fh:
            fh.write(SAMPLE)
        return _Proc()

    monkeypatch.setattr(stansummary, "_binary", lambda: tmp_path / "stansummary")
    monkeypatch.setattr(subprocess, "run", fake_run)
    got = stansummary.summarize(["a.csv", "b.csv"])
    assert got["theta"]["ESS_tail"] == pytest.approx(403.871)
    cmd = seen["cmd"]
    assert cmd[cmd.index("--sig_figs") + 1] == "18"
    assert cmd[cmd.index("--percentiles") + 1] == "5,50,95"
    # the input files come last, so --include_param style options cannot swallow one
    assert cmd[-2:] == ["a.csv", "b.csv"]
