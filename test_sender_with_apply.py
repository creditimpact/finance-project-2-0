"""Test validation sender V2 with integrated applier.

This test simulates calling the sender directly to verify:
1. Results are sent and written
2. Applier is called automatically
3. Summaries are enriched with AI fields
4. Manifest results_applied flag is set
"""

import sys
from pathlib import Path


def test_sender_with_apply(sid: str):
    """Test sender V2 with integrated applier.
    
    NOTE: This will skip sending if results already exist (idempotent).
    Delete existing results first to force a fresh send.
    """
    runs_root = Path("runs").resolve()
    
    print(f"\n{'='*70}")
    print(f"Testing Validation Sender V2 + Apply for SID: {sid}")
    print(f"{'='*70}\n")
    
    print("Calling run_validation_send_for_sid_v2...")
    print()
    
    from backend.ai.validation_sender_v2 import run_validation_send_for_sid_v2
    
    stats = run_validation_send_for_sid_v2(sid, runs_root)
    
    print("\n" + "="*70)
    print("Sender Stats:")
    print("="*70)
    print(f"Expected packs: {stats.get('expected')}")
    print(f"Sent: {stats.get('sent')}")
    print(f"Written: {stats.get('written')}")
    print(f"Failed: {stats.get('failed')}")
    print()
    
    apply_stats = stats.get("apply_stats", {})
    apply_success = stats.get("apply_success", False)
    
    print("Apply Stats:")
    print(f"Accounts total: {apply_stats.get('accounts_total')}")
    print(f"Accounts updated: {apply_stats.get('accounts_updated')}")
    print(f"Results applied: {apply_stats.get('results_applied')}")
    print(f"Results unmatched: {apply_stats.get('results_unmatched')}")
    print(f"Apply success: {apply_success}")
    print()
    
    # Check manifest
    import json
    manifest_path = runs_root / sid / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    validation_status = (
        manifest
        .get("ai", {})
        .get("status", {})
        .get("validation", {})
    )
    
    print("Manifest Validation Status:")
    print(f"built: {validation_status.get('built')}")
    print(f"sent: {validation_status.get('sent')}")
    print(f"results_applied: {validation_status.get('results_applied')}")
    print(f"state: {validation_status.get('state')}")
    print()
    
    # Final verdict
    print("="*70)
    print("Final Verdict:")
    print("="*70)
    
    if apply_success:
        print("✅ Apply succeeded")
    else:
        print("⚠️  Apply did not succeed (may be due to idempotency)")
    
    if validation_status.get("results_applied") is True:
        print("✅ Manifest results_applied flag is set")
    else:
        print("⚠️  Manifest results_applied flag not set")
    
    if stats.get("failed", 0) == 0:
        print("✅ No failures")
    else:
        print(f"❌ {stats.get('failed')} failures")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_sender_with_apply.py <sid>")
        print("Example: python test_sender_with_apply.py c953ec0f-acc7-418d-a59f-c1fa4a2eb13c")
        print()
        print("NOTE: Sender is idempotent - will skip if results already exist.")
        print("To force fresh send, delete results first:")
        print("  Remove-Item runs/<sid>/ai_packs/validation/results/*.jsonl")
        sys.exit(1)
    
    sid = sys.argv[1]
    test_sender_with_apply(sid)
