"""Source hashing and disk-cached compilation.

Every tool takes `stan` (a .stan path or inline source) and hashes
(source, stanc_args) to a model_id. On a cache miss the model is compiled once;
both the BridgeStan .so and the CmdStan executable are cached under
STAN_MCP_HOME/compiled/. The agent never tracks a handle.

Implemented during the build (see design "Model lifecycle" and tool A).
"""
