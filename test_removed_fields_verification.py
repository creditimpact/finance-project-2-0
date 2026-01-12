#!/usr/bin/env python3
"""Verify that the 4 removed fields are absent from bureaus.json output."""

import json
from backend.core.logic.report_analysis.problem_case_builder import _build_bureaus_payload_from_stagea

# Create a test account with all possible legacy fields
test_account = {
    "heading": "TEST CARD",
    "triad_fields": {
        "transunion": {"account_number": "1234", "balance": "100"},
        "experian": {"account_number": "1234", "balance": "100"},
        "equifax": {"account_number": "1234", "balance": "100"},
    },
    "two_year_payment_history": {
        "transunion": ["OK", "OK", "30"],
        "experian": ["OK", "OK", "OK"],
        "equifax": ["OK", "OK", "OK"],
    },
    "seven_year_history": {
        "transunion": {"late30": 1, "late60": 0, "late90": 0},
        "experian": {"late30": 0, "late60": 0, "late90": 0},
        "equifax": {"late30": 0, "late60": 0, "late90": 0},
    },
    # These should be removed from bureaus.json output
    "two_year_payment_history_monthly": {
        "transunion": [{"month": "Jan", "value": "OK"}],
        "experian": [{"month": "Jan", "value": "OK"}],
        "equifax": [],
    },
    "two_year_payment_history_months": ["Jan", "Feb", "Mar"],
    "two_year_payment_history_months_by_bureau": {
        "transunion": ["Jan", "Feb"],
        "experian": ["Jan", "Feb"],
        "equifax": ["Jan"],
    },
    "two_year_payment_history_months_tsv_v2": {
        "transunion": ["Jan", "Feb"],
        "experian": ["Jan", "Feb"],
        "equifax": ["Jan"],
    },
    # This should STAY in bureaus.json
    "two_year_payment_history_monthly_tsv_v2": {
        "transunion": [{"month": "Jan", "value": "OK"}],
        "experian": [{"month": "Jan", "value": "OK"}],
        "equifax": [{"month": "Jan", "value": "90"}],
    },
}

# Build the bureaus payload
payload = _build_bureaus_payload_from_stagea(test_account)

# Check which keys are present
print("\n=== BUREAUS.JSON PAYLOAD KEYS ===")
print("\nLegacy fields (should be present):")
print(f"  ✓ two_year_payment_history: {'PRESENT' if 'two_year_payment_history' in payload else 'MISSING'}")
print(f"  ✓ seven_year_history: {'PRESENT' if 'seven_year_history' in payload else 'MISSING'}")

print("\nRemoved fields (should be ABSENT):")
removed_fields = [
    "two_year_payment_history_monthly",
    "two_year_payment_history_months",
    "two_year_payment_history_months_by_bureau",
    "two_year_payment_history_months_tsv_v2",
]
for field in removed_fields:
    status = "✗ PRESENT (SHOULD BE REMOVED!)" if field in payload else "✓ ABSENT (CORRECT)"
    print(f"  {field}: {status}")

print("\nNew field (should be present):")
print(f"  ✓ two_year_payment_history_monthly_tsv_v2: {'PRESENT' if 'two_year_payment_history_monthly_tsv_v2' in payload else 'MISSING'}")

# Show the v2_monthly content if present
if "two_year_payment_history_monthly_tsv_v2" in payload:
    print(f"\n    Content: {json.dumps(payload['two_year_payment_history_monthly_tsv_v2'], indent=2)}")

# Summary
print("\n=== SUMMARY ===")
all_removed_absent = all(field not in payload for field in removed_fields)
v2_monthly_present = "two_year_payment_history_monthly_tsv_v2" in payload

if all_removed_absent and v2_monthly_present:
    print("✅ ALL ACCEPTANCE CRITERIA MET")
    exit(0)
else:
    print("❌ ACCEPTANCE CRITERIA FAILED")
    if not all_removed_absent:
        print(f"   - Some removed fields are still present")
    if not v2_monthly_present:
        print(f"   - two_year_payment_history_monthly_tsv_v2 is missing")
    exit(1)
