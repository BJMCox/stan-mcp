# Model validation

Use explicit Stan simulation code and saved runs to check predictions and
inference. Model-specific simulation and test quantities remain under your
control. Inputs, reports and plot paths refer to the server host, including
when connected over SSH.

Generate runnable request files:

```sh
pixi run python examples/model_validation.py /absolute/demo
```

The example uses a Beta–Binomial model. Its simulator draws the parameter,
observations and a data-dependent joint log density. Keep simulations explicit:
checking parameter marginals alone can miss an inference method that ignores data.

## Predictive checks

Call `start_run` with the arguments in `demo/prior.json`, then `await_run`.
`method="fixed_param"` runs RNG-only simulation with zero warmup. It remains a
separate method from posterior MCMC and receives no R-hat, ESS or ordinary LOO
claims. Allowed arguments are chains, iter_sampling, parallel_chains and sig_figs.

Write a predictive request, replacing the ID with the completed run:

```json
{
  "run_id": "fixed_param-0001",
  "kind": "prior",
  "independent_draws": true,
  "observed": "/absolute/demo/observed.json",
  "checks": [
    {"observed": "y", "replicated": "y_sim", "statistic": "mean"}
  ]
}
```

Call `check_predictive(request_file="/absolute/demo/predictive.json")`.
The example script writes this file when passed `--prior-run RUN_ID`.
For a posterior run, use `kind="posterior"` and the model's `y_rep` variable.
`observed` accepts a JSON file, registered data handle or inline object.

Supported statistics are mean, sample SD, min, max, median, q05, q95 and
zero_fraction. Observed and replicated event shapes must match exactly. Each
statistic reduces observation dimensions within a draw, preserving chain and
draw order. Sample SD requires at least two observations.

Reports contain the observed statistic, replicated-statistic mean and predictive
quantiles, the inclusive tail fraction `P(T(y_rep) >= T(y))`, equality fraction,
mean/tail Monte Carlo uncertainty, and an SVG histogram. Predictive quantiles
describe variation in replicated data; they are not Monte Carlo error bars.
There is no universal goodness-of-fit verdict or classical p-value claim.

MCMC uncertainty uses each actual statistic/indicator series and requires
adequate convergence. Constant series or undefined uncertainty remain explicit.
Approximate draws remain labelled and have no assumed iid uncertainty.
For fixed-parameter simulation, iid MCSE and Wilson tail intervals require
`independent_draws=true`: every stochastic ancestor must be redrawn each iteration.
A latent drawn once in transformed data or initialization breaks that assumption.
The flag is a user declaration, not an automatic independence check.

Reports, resolved observed inputs and plots live in an immutable per-call
`runs/<run-id>/predictive/<id>/` directory. At most 20 checks enter one request.

## Resumable studies

Call `start_study(request_file="/absolute/demo/study.json")` with this structure:

```json
{
  "schema": "stan-mcp.study.v1",
  "seed": 49,
  "replicates": 20,
  "max_parallel": 2,
  "tasks": [{
    "id": "beta-binomial",
    "stan": "/absolute/demo/model.stan",
    "data": "/absolute/demo/data.json",
    "method": "sample",
    "method_args": {"chains": 2, "iter_warmup": 300, "iter_sampling": 300},
    "simulate": {
      "stan": "/absolute/demo/simulate.stan",
      "data": "/absolute/demo/data.json",
      "data_variables": {"y": "y_sim"},
      "integer_data": ["y"],
      "truth_variables": {"theta": "theta_sim", "joint": "joint_sim"}
    }
  }]
}
```

Ordinary studies omit `simulate`. Each task supports the existing inference
methods, arguments and initialization values. Relative input paths resolve
against the manifest directory. Absolute paths are preferred. `replicates`
repeats each task with stable case IDs and distinct persisted per-stage seeds.
A study expands to at most 10,000 cases and permits 1–32 active stages.
The shared inference pool still bounds execution across studies and ordinary runs.

A simulation stage runs one fixed-parameter draw in one chain. Map generated
quantities to fitted data through `data_variables`. Generated data overrides
matching base-data fields. CSVs do not retain integer declarations, so name
integer data targets explicitly in `integer_data`; conversion requires exact
integers within Stan's integer range. Map fitted test quantities to their
simulated truths through `truth_variables`. Shapes must match. Data-dependent
quantities such as the example's `joint` work through this same mapping.

Use `await_run`, `get_study_result`, `get_run_status`, `get_run_progress` and
`cancel_run` with the returned study ID. Completion means all scheduled cases
finished, including failures. Reports retain every planned case and bounded
failure excerpts. Full case checkpoints preserve attempts, seeds, child run
references, generated data and truths.

`resume_study(study_id=...)` retries unfinished or failed cases with the same
inputs and seeds. It reuses completed simulations and leaves successful cases
unchanged. Resume verifies source/include snapshots, settings, build identity,
input CSVs and completed artifact hashes, including child metadata. Changed or
pruned artifacts require a new study. Keep child runs when pruning runs.
Preparation compiles each distinct model before freezing the build identity.
A missing committed prepared manifest cannot be reconstructed silently.

Cancellation records intent during compilation, then drains owned child work.
Server restarts do not adopt computations. Resume refuses surviving orphan
processes or unverifiable process ownership. Only one coordinator owns a study.
This is a local coordinator over existing runs, with no detached remote daemon.

## Simulation-based calibration

After the study finishes, call `check_sbc` with a request file:

```json
{
  "study_id": "study-0002",
  "draws": 100,
  "thin": 5,
  "bins": 10,
  "interval": 0.9,
  "seed": 17
}
```

The example script writes this file when passed `--study-id STUDY_ID`.
The explicit simulator must draw each truth from the declared prior and then
simulate observations conditional on that truth. Fixed truths do not support
ordinary prior-averaged SBC claims.

Within each chain, retain every `thin`th draw. Interleave chains and retain the
same requested count for every assessable case. Insufficient or nonfinite draws
remain unresolved. Exact ties use seeded uniform integer rank randomization.
Histogram references account for the discrete support `0..draws`, including
unequal bin probabilities when the support does not divide evenly.

Coverage uses order-statistic endpoints. Its finite-draw continuous-iid target
is reported separately from the requested interval probability. Wilson intervals
measure uncertainty across independent simulation cases. Missing-case bounds
retain all planned cases; successful-only results can be selected.

Ranks remain descriptive. Thinning does not establish posterior independence,
MCMC convergence is a separate check, and approximate fits retain their limits.
No calibrated rank-uniformity verdict or simultaneous histogram bands are given.
Reports include available fit diagnostic verdicts and missing-header warnings.

Raw per-case ranks, ties, truths and interval endpoints use Safetensors with
I64 index/rank columns and F64 numerical columns. JSON files define task and
quantity axes, record counts and content hashes. Reports and SVG plots live in
`runs/<study-id>/study/sbc/<id>/`. Summaries support at most 50 scalar quantities.

See Stan's [predictive checks](https://mc-stan.org/docs/stan-users-guide/ppcs.html)
and [SBC guide](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html)
for the statistical assumptions behind these workflows.
