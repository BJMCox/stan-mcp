"""stan-mcp configuration: environment variables, resolved paths, caps.

A single home root holds everything (override with STAN_MCP_HOME):

    ~/stan-mcp/
      bridgestan/     auto-downloaded BridgeStan source tree
      cmdstan/        auto-downloaded CmdStan (CmdStanPy-managed)
      compiled/       .so + executables, cached by source hash
      runs/<run-id>/  persisted inference runs

Path/cap logic is implemented during the build (see design "Home root" and
"Model lifecycle").
"""

import os
from pathlib import Path


def home() -> Path:
    """Resolve STAN_MCP_HOME (default ~/stan-mcp), creating it on first use."""
    raw = os.environ.get("STAN_MCP_HOME") or str(Path.home() / "stan-mcp")
    return Path(raw)


# Bounded in-process LRU of loaded BridgeStan StanModel objects.
MODEL_LRU_SIZE = int(os.environ.get("STAN_MCP_LRU_SIZE") or 8)
