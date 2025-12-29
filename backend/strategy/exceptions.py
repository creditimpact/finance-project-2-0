"""Custom exceptions for the strategy planner stage."""

from __future__ import annotations


class StrategyPlannerError(Exception):
    """Base error for strategy planner failures."""


class UnsupportedDurationUnitError(StrategyPlannerError):
    """Raised when validation findings contain unsupported duration units."""


class PlannerConfigurationError(StrategyPlannerError):
    """Raised when planner configuration is invalid or inconsistent."""
