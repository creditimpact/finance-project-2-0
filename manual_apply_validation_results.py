"""Manually apply validation results to summaries for testing."""

import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manual_apply_validation_results.py <sid>")
        sys.exit(1)
    
    sid = sys.argv[1]
    runs_root = Path("runs").resolve()
    
    print(f"Applying validation results for SID: {sid}")
    print(f"Runs root: {runs_root}\n")
    
    from backend.validation.apply_results_v2 import apply_validation_results_for_sid
    
    stats = apply_validation_results_for_sid(sid, runs_root)
    
    print("\nResults:")
    print(f"  Accounts total: {stats.get('accounts_total')}")
    print(f"  Accounts updated: {stats.get('accounts_updated')}")
    print(f"  Results total: {stats.get('results_total')}")
    print(f"  Results applied: {stats.get('results_applied')}")
    print(f"  Results unmatched: {stats.get('results_unmatched')}")
    
    if stats.get("error"):
        print(f"\n❌ Error: {stats.get('error')}")
    elif stats.get("results_applied", 0) > 0:
        print("\n✅ Results successfully applied!")
    else:
        print("\n⚠️  No results were applied")
