"""
Test script to validate the new validation_sender_v2 implementation.

Usage:
  python test_validation_sender_v2.py <SID>

This will:
  1. Check that validation packs exist
  2. Run the new validation_sender_v2
  3. Verify results are written
  4. Check runflow validation stage
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

def main(sid: str) -> int:
    from backend.ai.validation_sender_v2 import run_validation_send_for_sid_v2
    from backend.pipeline.runs import load_manifest_from_disk
    from backend.validation.index_schema import load_validation_index
    from backend.runflow.decider import get_runflow_snapshot
    
    runs_root = Path("runs")
    
    print(f"\n=== Testing Validation Sender V2 for SID: {sid} ===\n")
    
    # Step 1: Check manifest paths
    print("Step 1: Loading manifest...")
    try:
        manifest = load_manifest_from_disk(runs_root, sid)
        data = manifest.data
        validation_section = data.get("ai", {}).get("packs", {}).get("validation", {})
        
        packs_dir = validation_section.get("packs_dir")
        results_dir = validation_section.get("results_dir")
        index_path = validation_section.get("index")
        
        print(f"  Packs dir: {packs_dir}")
        print(f"  Results dir: {results_dir}")
        print(f"  Index path: {index_path}")
        
        if not packs_dir or not results_dir or not index_path:
            print("  ERROR: Validation paths missing in manifest!")
            return 1
    except Exception as exc:
        print(f"  ERROR loading manifest: {exc}")
        return 1
    
    # Step 2: Check validation index
    print("\nStep 2: Loading validation index...")
    try:
        index = load_validation_index(Path(index_path))
        packs_count = len(index.packs)
        print(f"  Packs in index: {packs_count}")
        
        if packs_count == 0:
            print("  No packs to send!")
            return 0
        
        for record in index.packs[:3]:  # Show first 3
            print(f"    - Account {record.account_id}: {record.weak_fields}")
    except Exception as exc:
        print(f"  ERROR loading index: {exc}")
        return 1
    
    # Step 3: Check runflow before send
    print("\nStep 3: Checking runflow BEFORE send...")
    try:
        snapshot = get_runflow_snapshot(sid, runs_root=runs_root)
        validation_stage = snapshot.get("stages", {}).get("validation", {})
        results_info = validation_stage.get("results", {})
        
        print(f"  Status: {validation_stage.get('status')}")
        print(f"  Results total: {results_info.get('results_total', 0)}")
        print(f"  Results received: {results_info.get('results_received', 0)}")
        print(f"  Missing results: {results_info.get('missing_results', 0)}")
        
        errors = validation_stage.get("errors", [])
        if errors:
            print(f"  Errors: {[e.get('type') for e in errors]}")
    except Exception as exc:
        print(f"  WARNING: Could not read runflow: {exc}")
    
    # Step 4: Run the new sender
    print("\nStep 4: Running validation_sender_v2...")
    try:
        stats = run_validation_send_for_sid_v2(sid, runs_root)
        
        print(f"  Expected: {stats['expected']}")
        print(f"  Sent: {stats['sent']}")
        print(f"  Written: {stats['written']}")
        print(f"  Failed: {stats['failed']}")
        
        if stats['errors']:
            print(f"  Errors:")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"    - Account {error.get('account_id')}: {error.get('error')}")
        
        if stats['written'] == stats['expected'] and stats['failed'] == 0:
            print("\n  ✅ All packs sent successfully!")
        else:
            print(f"\n  ⚠️  Partial success: {stats['written']}/{stats['expected']} written")
    except Exception as exc:
        print(f"  ERROR during send: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 5: Check runflow after send
    print("\nStep 5: Checking runflow AFTER send...")
    try:
        snapshot = get_runflow_snapshot(sid, runs_root=runs_root)
        validation_stage = snapshot.get("stages", {}).get("validation", {})
        results_info = validation_stage.get("results", {})
        
        print(f"  Status: {validation_stage.get('status')}")
        print(f"  Results total: {results_info.get('results_total', 0)}")
        print(f"  Results received: {results_info.get('results_received', 0)}")
        print(f"  Missing results: {results_info.get('missing_results', 0)}")
        
        errors = validation_stage.get("errors", [])
        if errors:
            print(f"  Errors: {[e.get('type') for e in errors]}")
        else:
            print(f"  ✅ No errors!")
        
        if results_info.get('missing_results', 0) == 0:
            print("\n  ✅ Runflow shows all results received!")
        else:
            print(f"\n  ⚠️  Still missing {results_info.get('missing_results')} results")
    except Exception as exc:
        print(f"  WARNING: Could not read runflow: {exc}")
    
    # Step 6: Verify result files on disk
    print("\nStep 6: Verifying result files on disk...")
    try:
        results_dir_path = Path(results_dir)
        if results_dir_path.exists():
            result_files = list(results_dir_path.glob("*.json*"))
            print(f"  Found {len(result_files)} result files")
            
            for result_file in result_files[:3]:  # Show first 3
                size = result_file.stat().st_size
                print(f"    - {result_file.name}: {size} bytes")
        else:
            print(f"  Results dir does not exist: {results_dir_path}")
    except Exception as exc:
        print(f"  WARNING: Could not check result files: {exc}")
    
    print("\n=== Test Complete ===\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_validation_sender_v2.py <SID>")
        sys.exit(1)
    
    sid = sys.argv[1]
    sys.exit(main(sid))
