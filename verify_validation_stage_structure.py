"""Verify validation stage structure is clean and properly organized."""
import json
from pathlib import Path

def verify_validation_stage(sid: str, runs_root: Path):
    """Check that validation stage has clean structure without merge pollution."""
    runflow_path = runs_root / sid / "runflow.json"
    if not runflow_path.exists():
        print(f"❌ No runflow.json for {sid}")
        return False
    
    data = json.loads(runflow_path.read_text(encoding="utf-8"))
    validation = data.get("stages", {}).get("validation", {})
    
    if not validation:
        print(f"❌ No validation stage for {sid}")
        return False
    
    print(f"\n✓ Validation stage for {sid}:")
    print(json.dumps(validation, indent=2))
    
    # Check for merge pollution at top level
    merge_keys = ["merge_zero_packs", "skip_counts", "skip_reason_top"]
    polluted_top = [k for k in merge_keys if k in validation]
    if polluted_top:
        print(f"❌ Top-level merge pollution: {polluted_top}")
        return False
    
    # Check metrics has no merge pollution
    metrics = validation.get("metrics", {})
    polluted_metrics = [k for k in merge_keys if k in metrics]
    if polluted_metrics:
        print(f"❌ Metrics merge pollution: {polluted_metrics}")
        return False
    
    # Check summary structure
    summary = validation.get("summary", {})
    
    # Check for required V2 flags
    required_flags = ["validation_ai_required", "validation_ai_completed", "validation_ai_applied"]
    missing_summary = [f for f in required_flags if f not in summary]
    if missing_summary:
        print(f"⚠ Missing summary flags: {missing_summary}")
    
    missing_metrics = [f for f in required_flags if f not in metrics]
    if missing_metrics:
        print(f"⚠ Missing metrics flags: {missing_metrics}")
    
    # Check merge context is properly nested
    if any(k in summary for k in merge_keys):
        polluted_summary_top = [k for k in merge_keys if k in summary]
        print(f"❌ Summary top-level merge pollution: {polluted_summary_top}")
        return False
    
    # Check if merge_context exists and is clean
    merge_context = summary.get("merge_context")
    if merge_context:
        print(f"✓ Merge context properly nested: {list(merge_context.keys())}")
    
    # Check nested summary.metrics doesn't have merge pollution
    summary_metrics = summary.get("metrics", {})
    if isinstance(summary_metrics, dict):
        polluted_summary_metrics = [k for k in merge_keys if k in summary_metrics]
        if polluted_summary_metrics:
            print(f"❌ Summary.metrics merge pollution: {polluted_summary_metrics}")
            return False
    
    print(f"✓ Validation stage structure is clean")
    print(f"✓ validation_ai_required: {summary.get('validation_ai_required')}")
    print(f"✓ validation_ai_completed: {summary.get('validation_ai_completed')}")
    print(f"✓ validation_ai_applied: {summary.get('validation_ai_applied')}")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python verify_validation_stage_structure.py <SID>")
        print("Example: python verify_validation_stage_structure.py d10a9dd4-aea7-4588-99e2-f71660d10727")
        sys.exit(1)
    
    sid = sys.argv[1]
    runs_root = Path("runs")
    
    success = verify_validation_stage(sid, runs_root)
    sys.exit(0 if success else 1)
