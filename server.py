"""stan-mcp — MCP server for compiling, inspecting, and running Stan models.

Primary use: in-memory log-density / gradient interaction via BridgeStan.
Secondary: full inference runs (all six CmdStanPy methods) via CmdStanPy.

See docs/2026-07-14-stan-mcp-design.md for the full design. Tools are registered
here per the design's tool surface:

- A. compile / metadata      -> compile.py
- B. BridgeStan density      -> density.py   (primary)
- C. runs (all methods)      -> sample.py + runstore.py + diagnostics.py

Modules keep the logic; this file stays thin (registration + arg validation +
offload of blocking work via asyncio.to_thread).
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stan-mcp")

# Tools are registered during implementation (see design sections A-C).


if __name__ == "__main__":
    mcp.run()
