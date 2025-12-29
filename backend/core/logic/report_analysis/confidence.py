"""Confidence calibration helpers for report analysis."""

from __future__ import annotations


def combine_confidence(
    heading_coverage: float,
    *,
    schema_valid: bool,
    extractor_agreement: float,
) -> float:
    """Combine individual quality metrics into a single confidence score.

    Each metric should be in the range ``[0.0, 1.0]``. ``schema_valid`` is treated
    as ``1.0`` when valid and ``0.0`` otherwise. The returned score is the
    arithmetic mean clamped to ``[0.0, 1.0]``.
    """

    schema_score = 1.0 if schema_valid else 0.0
    parts = [heading_coverage, schema_score, extractor_agreement]
    score = sum(parts) / len(parts)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
