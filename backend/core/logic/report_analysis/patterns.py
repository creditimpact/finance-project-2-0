from __future__ import annotations

import re

# Canonical account-line pattern: reuse extractor token to avoid drift
try:
    from .extractors.tokens import ACCOUNT_RE as ACCOUNT_LINE  # type: ignore
except Exception:
    # Fallback: permissive account-number pattern
    ACCOUNT_LINE = re.compile(
        r"ac(?:count|ct)\s*(?:#|number)?[:\s]*([A-Za-z0-9]+)", re.IGNORECASE
    )


# Initial SmartCredit heading patterns. These are intentionally broad and will
# be refined using captured text dumps and canonical schema review.
LATE_HEADERS = re.compile(
    r"(Late\s*Payments?|Payment\s*History|Delinquen(?:cy|t)|Past\s*Due)",
    re.IGNORECASE,
)

