"""Structured error types for stan-mcp tools.

Every tool converts a caught StanMcpError into a plain dict via as_error_dict
so FastMCP returns structured JSON, never a raw traceback.
"""

from typing import Any


class StanMcpError(Exception):
    """Base class for all stan-mcp tool errors."""


class ToolchainError(StanMcpError):
    """C++ toolchain or Stan source tree missing/unprovisionable."""


class CompileError(StanMcpError):
    """A Stan model failed to compile."""


class ValidationError(StanMcpError):
    """A tool argument failed validation (e.g. wrong theta_unc length)."""


class RunError(StanMcpError):
    """A sampling/inference run failed or was referenced incorrectly."""


def as_error_dict(exc: Exception) -> dict[str, Any]:
    return {"error": str(exc), "error_type": type(exc).__name__}
