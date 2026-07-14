"""Toolchain provisioning: ensure the C++ toolchain and the BridgeStan / CmdStan
source trees are present.

pixi pins the compiler + make; the BridgeStan and CmdStan source trees are
auto-downloaded to STAN_MCP_HOME on first compile. Any gap (missing compiler,
failed download) fails loud with an exact remediation message — never silent.
Downloads are large; log clearly so a first compile does not look hung.

Implemented during the build (see design "Toolchain provisioning").
"""
