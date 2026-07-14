"""Run store: directory layout and persistence under STAN_MCP_HOME/runs/<run-id>/.

Persists model.stan, data.json, the CmdStan CSVs, inference.nc (arviz
InferenceData, for draws-producing methods) or estimate.json (optimize),
meta.json (model_id, method, method_args, status, timestamps, stderr on
failure), and a cached summary.json. list_runs reads meta.json across run dirs;
failed runs are persisted with status="failed" and captured stderr.

Implemented during the build (see design "Run store").
"""
