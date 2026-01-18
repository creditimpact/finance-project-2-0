"""Quick integration test for F3.B01 on real SID data."""
import json
from pathlib import Path
from backend.tradeline_check.f3_b01_post_closure_monthly_ok_detection import evaluate_f3_b01

# Test on SID 06fa74b1-d515-4e84-931f-1ffc14c7d9ee, Account 11
sid = "06fa74b1-d515-4e84-931f-1ffc14c7d9ee"
account = "11"
bureau = "transunion"  # State 3, closed 1/1/2025

# Load data
runs_root = Path("runs")
account_dir = runs_root / sid / "cases" / "accounts" / account

bureaus_path = account_dir / "bureaus.json"
tcheck_path = account_dir / "tradeline_check" / f"{bureau}.json"

bureaus_data = json.loads(bureaus_path.read_text(encoding="utf-8"))
tcheck_data = json.loads(tcheck_path.read_text(encoding="utf-8"))

bureau_obj = bureaus_data[bureau]

# Build payload
payload = {
    "routing": tcheck_data.get("routing", {}),
    "root_checks": tcheck_data.get("root_checks", {}),
}

placeholders = {"--", "n/a", "unknown"}

# Run F3.B01
print(f"\n{'='*80}")
print(f"F3.B01 Integration Test: SID {sid}, Account {account}, Bureau {bureau}")
print(f"{'='*80}\n")

print(f"R1 State: {payload['routing'].get('R1', {}).get('state_num')}")
print(f"Q1: {payload['root_checks'].get('Q1', {}).get('declared_state')}")
print(f"root_checks contains keys: {list(payload['root_checks'].keys())}")
print(f"closed_date: {bureau_obj.get('closed_date')}")
print()

result = evaluate_f3_b01(bureau_obj, bureaus_data, bureau, payload, placeholders)

print(f"F3.B01 Result:")
print(f"  Status: {result['status']}")
print(f"  Eligible: {result['eligible']}")
print(f"  Executed: {result['executed']}")
print(f"  Fired: {result['fired']}")
print(f"  Explanation: {result['explanation']}")

if "evidence" in result:
    print(f"\nEvidence:")
    evidence = result["evidence"]
    if "closed_date_parsed" in evidence:
        print(f"  closed_date_parsed: {evidence['closed_date_parsed']}")
    if "first_forbidden_month" in evidence:
        print(f"  first_forbidden_month: {evidence['first_forbidden_month']}")
    if "post_closure_ok_months" in evidence:
        print(f"  post_closure_ok_months: {len(evidence['post_closure_ok_months'])} month(s)")
        for month in evidence.get("post_closure_ok_months", []):
            print(f"    - {month['month_year_key']}: {month['status']}")

print(f"\n{'='*80}")
print("Test Complete")
print(f"{'='*80}\n")
