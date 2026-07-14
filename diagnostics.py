"""Method-aware summary and diagnostics over stored runs (arviz-backed).

get_run_summary: draws-producing methods -> arviz summary (mean, sd, HDI,
bulk/tail ESS, R-hat); optimize -> point estimate + lp__.
get_run_diagnostics: sample -> divergences, tree-depth hits, E-BFMI, R-hat;
variational -> ELBO convergence; pathfinder -> path diagnostics; optimize ->
convergence / lp__.

Implemented during the build (see design tool group C).
"""
