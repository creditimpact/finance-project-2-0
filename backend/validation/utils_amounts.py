"""Utilities for applying tolerance checks to numeric amounts."""
from __future__ import annotations

from math import isfinite, isnan
from typing import Iterable, Optional, Tuple


DEFAULT_ABSOLUTE_TOLERANCE = 50.0
DEFAULT_RATIO_TOLERANCE = 0.01


def _to_float(value: object) -> Optional[float]:
    """Attempt to coerce ``value`` to a float, returning ``None`` on failure."""
    if value is None:
        return None

    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None

    if isnan(coerced) or not isfinite(coerced):
        return None

    return coerced


def are_amounts_within_tolerance(
    values: Iterable[object],
    abs_tol: float,
    ratio_tol: float,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Determine whether the numeric values provided stay within the configured tolerance.

    Parameters
    ----------
    values:
        Iterable of raw values which may be numeric or coercible to float.
    abs_tol:
        Absolute dollar tolerance. Differences below this value are ignored.
    ratio_tol:
        Ratio tolerance expressed as a decimal (e.g., ``0.01`` for 1%).

    Returns
    -------
    Tuple[bool, Optional[float], Optional[float]]
        ``True``/``False`` for whether the values are within tolerance, the
        computed absolute difference if at least two numeric values were
        provided, and the maximum numeric value encountered.
    """

    try:
        effective_abs_tol = float(abs_tol)
    except (TypeError, ValueError):
        effective_abs_tol = DEFAULT_ABSOLUTE_TOLERANCE
    else:
        if effective_abs_tol < 0:
            effective_abs_tol = DEFAULT_ABSOLUTE_TOLERANCE

    try:
        effective_ratio_tol = float(ratio_tol)
    except (TypeError, ValueError):
        effective_ratio_tol = DEFAULT_RATIO_TOLERANCE
    else:
        if effective_ratio_tol < 0:
            effective_ratio_tol = DEFAULT_RATIO_TOLERANCE

    numeric_values = [value for value in (_to_float(v) for v in values) if value is not None]

    if len(numeric_values) < 2:
        return True, None, None

    maximum = max(numeric_values)
    minimum = min(numeric_values)
    diff = maximum - minimum

    max_magnitude = max(abs(maximum), abs(minimum))
    ratio_cap = max_magnitude * effective_ratio_tol
    threshold = max(effective_abs_tol, ratio_cap)

    return diff <= threshold, diff, maximum
