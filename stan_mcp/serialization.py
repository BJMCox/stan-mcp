"""Normalize numeric results for strict JSON serialization."""

from typing import Any

import numpy as np


def json_safe(record: Any) -> Any:
    """Replace non-finite floats (NaN/inf) with None so the result is valid JSON."""
    import math

    if isinstance(record, dict):
        return {k: json_safe(v) for k, v in record.items()}
    if isinstance(record, (list, tuple)):
        return [json_safe(v) for v in record]
    if isinstance(record, np.ndarray):
        return json_safe(record.tolist())
    if isinstance(record, np.generic):
        return json_safe(record.item())
    if isinstance(record, float) and not math.isfinite(record):
        return None
    return record
