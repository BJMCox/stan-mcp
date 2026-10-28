# Batched evaluation and numerical reference checks

Use absolute paths on the MCP server. JSON requests and manifests describe the
computation. Safetensors files hold numerical arrays. Inline arrays are not tool
arguments. The format uses row-major F32/F64 tensors and I64 IDs; it works with
FlatPPL's Rust and JavaScript Safetensors readers.

## Evaluate a batch

Write a JSON request:

```json
{
  "stan": "/absolute/model.stan",
  "data": "/absolute/data.json",
  "points": "/absolute/points/manifest.json",
  "quantities": ["log_density", "gradient", "hessian", "hvp"],
  "target": {"propto": false, "jacobian": false},
  "seed": 1234,
  "workers": 1,
  "chunk_rows": 1024
}
```

Call `evaluate_batch(request_file="/absolute/evaluate.json")`. It returns
`job_id` and `run_id`, which identify the same persisted job. Use
`get_run_status`, `get_run_progress`, `await_run`, and `cancel_run` with `run_id`.
Call `get_batch_result(job_id=...)` for the output manifest and bounded counts.
Use `list_runs(method="evaluate_batch")` to recover jobs.

One batch uses one model/data context. Compose several batches for several
contexts. Model and data handles remain supported. Files are preferred. Inputs
are snapshotted before submission returns. Structural validation and compilation
continue in the background. A structural error fails the job. A point exception
sets that row's status and continues the batch.

`quantities` defaults to `["log_density"]`. Both target flags default to false.
Gradients, Hessians and HVPs always use **Stan unconstrained coordinates**, even
when input points are constrained or `jacobian=false`. The result records the
coordinate names and order. Curvature uses a separate BridgeStan build with
`BRIDGESTAN_AD_HESSIAN=true`; the worker checks the loaded model's build flags.
The evaluator does not silently substitute finite differences.

## Artifact schema

A points manifest contains:

```json
{
  "schema": "stan-mcp.batch.v1",
  "kind": "points",
  "point_space": "constrained",
  "parameter_names": ["x", "v.1", "v.2"],
  "count": 2,
  "shards": [{
    "path": "00000000.safetensors",
    "sha256": "<SHA-256 of file bytes>",
    "rows": 2
  }]
}
```

Shard paths resolve from the manifest directory. Absolute shard paths also work.
`count` is optional on inputs. IDs must be nonnegative and strictly increasing
across all shards; IDs need not be consecutive. Inputs accept F32 and F64 points.
Results use F64. Names use BridgeStan's flattened element names. Constrained
input columns may be reordered by name. Unconstrained columns require native
order. Outputs always contain canonical constrained points in native order.
F32 inputs widen exactly for the F64 native ABI. Rounded points on strict
manifolds can fail Stan's support checks; use F64 source points for those cases.

| Tensor | Dtype | Axes |
|---|---|---|
| `row_id` | I64 | `[point]` |
| `points` | F32/F64 | `[point, constrained_parameter]`, or unconstrained parameters on input |
| `vectors` | F32/F64 | `[point, direction, derivative_parameter]`; native inputs use Stan unconstrained coordinates |
| `status` | I64 | `[point]`; results require 0 for success or 1 for point failure |
| `log_density` | F32/F64 | `[point]` |
| `gradient` | F32/F64 | `[point, derivative_parameter]` |
| `hessian` | F32/F64 | `[point, derivative_parameter, derivative_parameter]` |
| `hvp` | F32/F64 | `[point, direction, derivative_parameter]` |

Result manifests have `kind: "results"`, `point_space: "constrained"`, explicit
`quantities` and `target`, and this declaration for native derivatives:

```json
"derivative_coordinates": {
  "kind": "stan_unconstrained",
  "names": ["x", "v.1", "v.2"],
  "model_id": "<native coordinate identity>"
}
```

External results must declare their actual coordinates. Matching declarations
are a caller claim, not proof that two transforms agree. Copy this declaration
only when the external implementation uses that same transform and ordering.
Different transforms can still compare scalar log densities at common
constrained points. Arbitrary derivative coordinate maps are outside v1.

The evaluator writes NaN numerical placeholders and an error sidecar for failed
points. The shard descriptor names `errors` and `errors_sha256`. The JSON sidecar
maps row ID strings to error records. Messages are bounded and mark truncation.
External failed rows may omit error details; their numerical placeholders never
count as successful comparisons. Nonfinite numerical results also never pass.

Each Safetensors shard and error sidecar is limited to 64 MiB. Evaluation and
comparison split owned inputs as needed, at most 4096 output rows per shard.
A single row that exceeds the byte limit fails before evaluation. Metadata
scans retain shard descriptors; numerical processing holds bounded shards.

## Compare supplied results

Write a comparison request:

```json
{
  "left": "/absolute/stan-results/manifest.json",
  "right": "/absolute/external-results/manifest.json",
  "quantities": ["log_density", "gradient"],
  "tolerances": {
    "log_density": {"atol": 1e-10, "rtol": 1e-10},
    "gradient": {"atol": 1e-9, "rtol": 1e-9}
  },
  "additive_constant": false
}
```

Call `compare_batches(request_file="/absolute/compare.json")`. The comparator
reads files; it does not execute another runtime. Both result manifests must
declare `propto=false` and `jacobian=false`. Canonical point names, ordering,
IDs and finite point values must agree. Shard boundaries may differ. Missing
rows produce incomplete coverage. Extra left rows are allowed for explicit
finite-difference perturbations. Failed rows with nonfinite point placeholders
remain failures; finite point mismatches are structural errors.

Native derivative comparison requires identical coordinate declarations and
HVP direction tensors. Tolerances are explicit per quantity. Omitting a quantity's
tolerance yields error reporting without a pass/fail claim. The component scale
is `atol + rtol * max(abs(left), abs(right))`. All components must meet that scale.
Zero tolerance requires exact equality. Missing, failed or inconclusive coverage
yields `unknown`. Complete comparisons yield `pass`, `fail` or `report_only`.

`additive_constant=true` anchors the left-minus-right log-density difference at
the first valid paired row, then checks residuals across every paired row.
The report records the anchor ID and offset. This checks a constant difference;
it does not fit a point-dependent correction. Resume keeps the original anchor.

The output manifest has `kind: "comparison"`. Shards contain `row_id`, `points`,
`status`, `abs_error_<quantity>`, `scaled_error_<quantity>`, and
`comparison_status_<quantity>`. Per-quantity statuses are 0 for comparable,
1 for failure, and 2 for inconclusive. Comparable does not mean within tolerance.
The bounded summary records coverage, maxima, verdicts, and up to eight worst or
failed row IDs per quantity. Job status `done` means the computation finished;
inspect the numerical verdict separately.

## Explicit finite differences in other coordinates

When transforms differ, the caller can supply constrained perturbation points
generated in its own coordinates. Evaluate their scalar Stan log densities in
the left batch. Keep the external base-point results in the right batch.
Add this object to the comparison request:

```json
"finite_difference": {
  "steps": [0.001, 0.0005, 0.00025],
  "stencils": "/absolute/stencils/manifest.json",
  "quantities": ["gradient", "hessian", "hvp"],
  "derivative_coordinates": {
    "kind": "external",
    "names": ["u.1", "u.2"],
    "model_id": "<external coordinate identity>"
  }
}
```

The declaration must equal the right result declaration. Steps must be positive
and strictly descending, with at least two values. The stencil artifact is a
constrained `points` artifact containing the right base IDs and points, plus
I64 tensors whose entries reference left result row IDs:

| Tensor | Shape | Referenced perturbations, in order |
|---|---|---|
| `fd_gradient_rows` | `[base, step, D, 2]` | `u-h*e_i`, `u+h*e_i` |
| `fd_hessian_rows` | `[base, step, D, D, 4]` | `u+h*e_i+h*e_j`, `u+h*e_i-h*e_j`, `u-h*e_i+h*e_j`, `u-h*e_i-h*e_j` |
| `fd_hvp_rows` | `[base, step, K, D, 4]` | `u+h*e_i+h*v_k`, `u+h*e_i-h*v_k`, `u-h*e_i+h*v_k`, `u-h*e_i-h*v_k` |

For a diagonal Hessian entry, use `u+2h*e_i`, `u`, `u`, `u-2h*e_i`.
Gradient weights are `[-1,+1]/(2h)`. Curvature weights are `[+1,-1,-1,+1]/(4h²)`.
Generate all points through the caller's transform. The comparator cannot verify
that geometry from the declared IDs alone. User-supplied weights are unsupported.

The finest-step estimate is the reference. `stability_<quantity>` records the
maximum adjacent-step absolute difference. `roundoff_<quantity>` estimates
floating roundoff from stencil value magnitudes and their source precision.
The pass budget includes absolute discrepancy, empirical stability and estimated
roundoff together. A discrepancy must exceed the tolerance plus estimated
uncertainty to yield a mismatch. Overlapping intervals, unstable estimates and
insufficient coverage stay inconclusive.
These estimates are empirical checks, not rigorous error bounds. Without an
explicit tolerance, finite-difference checks remain inconclusive.

## Workers, cancellation and resume

Each evaluation worker owns one native model and reuses it across shards.
The default is one process per job. `workers` enables explicit parallelism up
to `STAN_MCP_MAX_WORKERS` (default 2). A shared cap bounds active batch workers
and comparisons across this server. The inference pool uses its existing cap
separately. Comparison runs sequentially and consumes one batch slot.
Compilation and structural preparation run in coordinator threads before
numerical worker slots are acquired.

Batch jobs persist under `runs/<job_id>/batch/`. They store the request, owned
inputs, plan, identity hashes, checkpoints and result shards. A lease prevents
two servers from resuming the same job. There is no detached daemon or scheduler.

Cancellation stops active numerical workers. A compile in progress finishes
before cancellation takes effect. Completed shards remain immutable.
`resume_batch(job_id=...)` verifies owned inputs, request/settings, seed, build
identity, native binary and completed output hashes, then schedules unfinished
shards. Interrupted preparation cannot resume. Changed computation needs a new
job. Resume drains verified abandoned workers before clearing unpublished files.

After a server restart, abandoned jobs appear failed. Resume them explicitly.
Keep `STAN_MCP_HOME` stable across restarts. SSH transport uses the same paths
and persistence on the remote server; it does not copy local input files.

## Runnable example

```sh
pixi run python examples/batch_gaussian.py /absolute/demo
```

Submit the printed `evaluate.json` path to `evaluate_batch`. Obtain the completed
manifest with `get_batch_result`, then write an independent analytic reference:

```sh
pixi run python examples/batch_gaussian.py /absolute/demo --stan-result /absolute/result/manifest.json
```

Submit the printed `compare.json` path to `compare_batches`. The example uses an
unconstrained Gaussian, so its analytic derivatives share Stan's identity transform.
