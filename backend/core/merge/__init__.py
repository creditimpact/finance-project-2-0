"""Core helpers for merge candidate generation."""

from .acctnum import (
    AccountNumberMatch,
    NormalizedAccountNumber,
    best_account_number_match,
    match_level,
    normalize_display,
)

__all__ = [
    "AccountNumberMatch",
    "NormalizedAccountNumber",
    "best_account_number_match",
    "match_level",
    "normalize_display",
]
