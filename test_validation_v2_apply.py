"""Test validation sender V2 + results applier integration.

This test verifies:
1. Validation packs are built and autosend triggers
2. Results are sent to AI and written to disk
3. Results are applied to summary.json requirement blocks
4. Runflow only marks validation.status=success after summaries are updated
"""

import json
import sys
from pathlib import Path


def test_validation_v2_with_apply(sid: str):
    """Test full validation V2 flow: build → send → apply → runflow.
    
    Args:
        sid: Run session ID to test (e.g., c953ec0f-acc7-418d-a59f-c1fa4a2eb13c)
    """
    runs_root = Path("runs").resolve()
    run_dir = runs_root / sid
    
    print(f"\n{'='*70}")
    print(f"Testing Validation V2 + Apply for SID: {sid}")
    print(f"{'='*70}\n")
    
    # Step 1: Check manifest paths
    print("Step 1: Checking manifest...")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    validation_status = (
        manifest
        .get("ai", {})
        .get("status", {})
        .get("validation", {})
    )
    
    print(f"Validation status before:")
    print(f"  built: {validation_status.get('built')}")
    print(f"  sent: {validation_status.get('sent')}")
    print(f"  results_applied: {validation_status.get('results_applied')}")
    print(f"  state: {validation_status.get('state')}")
    print()
    
    # Step 2: Check validation index
    print("Step 2: Checking validation index...")
    index_path = run_dir / "ai_packs" / "validation" / "index.json"
    if not index_path.exists():
        print(f"❌ Index not found: {index_path}")
        return
    
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    
    packs = index_data.get("packs", [])
    print(f"Index packs: {len(packs)}")
    
    expected_accounts = set()
    for pack in packs:
        account_id = pack.get("account_id")
        if account_id is not None:
            expected_accounts.add(str(account_id))
    
    print(f"Expected accounts: {sorted(expected_accounts)}")
    print()
    
    # Step 3: Check results exist
    print("Step 3: Checking validation results...")
    results_dir = run_dir / "ai_packs" / "validation" / "results"
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    result_files = list(results_dir.glob("acc_*.result.jsonl"))
    print(f"Result files found: {len(result_files)}")
    
    results_by_account = {}
    for result_file in result_files:
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    account_id = str(result.get("account_id"))
                    if account_id not in results_by_account:
                        results_by_account[account_id] = []
                    results_by_account[account_id].append(result)
    
    print(f"Results for accounts: {sorted(results_by_account.keys())}")
    print()
    
    # Step 4: Check summaries have AI fields
    print("Step 4: Checking AI fields in summary.json files...")
    cases_dir = run_dir / "cases" / "accounts"
    if not cases_dir.exists():
        print(f"❌ Cases directory not found: {cases_dir}")
        return
    
    summaries_updated = 0
    summaries_missing_ai = 0
    
    for account_id in sorted(expected_accounts):
        account_dir = cases_dir / account_id
        summary_path = account_dir / "summary.json"
        
        if not summary_path.exists():
            print(f"  ❌ Summary not found for account {account_id}")
            continue
        
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        validation_block = summary.get("validation_requirements", {})
        findings = validation_block.get("findings", [])
        
        has_ai_fields = False
        ai_fields_count = 0
        
        for finding in findings:
            if "ai_validation_decision" in finding:
                has_ai_fields = True
                ai_fields_count += 1
        
        if has_ai_fields:
            print(f"  ✅ Account {account_id}: {ai_fields_count} requirement(s) have AI fields")
            summaries_updated += 1
            
            # Show example
            for finding in findings:
                if "ai_validation_decision" in finding:
                    print(f"     Field: {finding.get('field')}")
                    print(f"     Decision: {finding.get('ai_validation_decision')}")
                    print(f"     Reason code: {finding.get('reason_code')}")
                    break
        else:
            print(f"  ❌ Account {account_id}: No AI fields found")
            summaries_missing_ai += 1
    
    print()
    print(f"Summaries with AI fields: {summaries_updated}/{len(expected_accounts)}")
    print()
    
    # Step 5: Check runflow
    print("Step 5: Checking runflow...")
    runflow_path = run_dir / "runflow.json"
    if not runflow_path.exists():
        print(f"❌ Runflow not found: {runflow_path}")
        return
    
    with open(runflow_path, "r", encoding="utf-8") as f:
        runflow = json.load(f)
    
    validation_stage = runflow.get("stages", {}).get("validation", {})
    print(f"Validation stage status: {validation_stage.get('status')}")
    print(f"Validation metrics:")
    
    metrics = validation_stage.get("metrics", {})
    print(f"  results_total: {metrics.get('results_total')}")
    print(f"  results_received: {metrics.get('results_received')}")
    print(f"  missing_results: {metrics.get('missing_results')}")
    print()
    
    # Step 6: Re-check manifest after apply
    print("Step 6: Checking manifest after apply...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    validation_status = (
        manifest
        .get("ai", {})
        .get("status", {})
        .get("validation", {})
    )
    
    print(f"Validation status after:")
    print(f"  built: {validation_status.get('built')}")
    print(f"  sent: {validation_status.get('sent')}")
    print(f"  results_applied: {validation_status.get('results_applied')}")
    print(f"  state: {validation_status.get('state')}")
    print()
    
    # Final verdict
    print(f"\n{'='*70}")
    print("Final Verdict:")
    print(f"{'='*70}")
    
    all_good = True
    
    if validation_status.get("results_applied") is True:
        print("✅ Manifest shows results_applied=true")
    else:
        print("❌ Manifest results_applied is not true")
        all_good = False
    
    if summaries_updated == len(expected_accounts):
        print(f"✅ All {len(expected_accounts)} account summaries have AI fields")
    else:
        print(f"❌ Only {summaries_updated}/{len(expected_accounts)} summaries have AI fields")
        all_good = False
    
    if validation_stage.get("status") == "success":
        print("✅ Runflow validation status is 'success'")
    else:
        print(f"❌ Runflow validation status is '{validation_stage.get('status')}'")
        all_good = False
    
    if metrics.get("missing_results", 0) == 0:
        print("✅ No missing results")
    else:
        print(f"❌ Missing results: {metrics.get('missing_results')}")
        all_good = False
    
    print()
    if all_good:
        print("🎉 ALL CHECKS PASSED!")
    else:
        print("⚠️  Some checks failed")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_validation_v2_apply.py <sid>")
        print("Example: python test_validation_v2_apply.py c953ec0f-acc7-418d-a59f-c1fa4a2eb13c")
        sys.exit(1)
    
    sid = sys.argv[1]
    test_validation_v2_with_apply(sid)
