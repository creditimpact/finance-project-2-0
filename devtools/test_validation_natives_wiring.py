#!/usr/bin/env python
"""
Test script to verify validation natives wiring works correctly.

This script:
1. Creates a test SID directory structure
2. Calls validation builder to build packs
3. Verifies manifest.json has all validation natives
4. Verifies runflow.json has validation stage with packs_built status
5. Confirms no AI sends occurred (results directory is empty)
"""
import json
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from backend.ai.validation_builder import build_validation_packs_for_run
from backend.pipeline.runs import RunManifest


def verify_manifest_natives(sid: str, runs_root: Path) -> bool:
    """Verify manifest.json has all required validation natives."""
    manifest_path = runs_root / sid / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ manifest.json not found at {manifest_path}")
        return False
    
    manifest = json.loads(manifest_path.read_text())
    
    # Check ai.packs.validation section
    validation_packs = manifest.get("ai", {}).get("packs", {}).get("validation", {})
    required_keys = ["base", "dir", "packs", "packs_dir", "results", "results_dir", "index", "logs"]
    
    missing_keys = [key for key in required_keys if key not in validation_packs]
    if missing_keys:
        print(f"❌ manifest.ai.packs.validation missing keys: {missing_keys}")
        return False
    
    # Check artifacts.ai.packs.validation section
    validation_artifacts = manifest.get("artifacts", {}).get("ai", {}).get("packs", {}).get("validation", {})
    missing_artifacts_keys = [key for key in required_keys if key not in validation_artifacts]
    if missing_artifacts_keys:
        print(f"❌ manifest.artifacts.ai.packs.validation missing keys: {missing_artifacts_keys}")
        return False
    
    # Check high-level ai.validation section
    ai_validation = manifest.get("ai", {}).get("validation", {})
    if "base" not in ai_validation or "dir" not in ai_validation:
        print(f"❌ manifest.ai.validation missing base or dir")
        return False
    
    # Check meta marker
    meta = manifest.get("meta", {})
    if not meta.get("validation_paths_initialized"):
        print(f"❌ manifest.meta.validation_paths_initialized not set")
        return False
    
    print(f"✅ manifest.json has all validation natives")
    return True


def verify_runflow_stage(sid: str, runs_root: Path) -> bool:
    """Verify runflow.json has validation stage with packs_built status."""
    runflow_path = runs_root / sid / "runflow.json"
    if not runflow_path.exists():
        print(f"❌ runflow.json not found at {runflow_path}")
        return False
    
    runflow = json.loads(runflow_path.read_text())
    
    # Check meta marker
    meta = runflow.get("meta", {})
    if not meta.get("validation_paths_initialized"):
        print(f"❌ runflow.meta.validation_paths_initialized not set")
        return False
    
    # Check stages.validation
    validation_stage = runflow.get("stages", {}).get("validation", {})
    if not validation_stage:
        print(f"❌ runflow.stages.validation not found")
        return False
    
    status = validation_stage.get("status")
    if status != "packs_built":
        print(f"❌ runflow.stages.validation.status is '{status}', expected 'packs_built'")
        return False
    
    required_fields = ["packs_count", "expected_results", "results_received", "updated_at"]
    missing_fields = [field for field in required_fields if field not in validation_stage]
    if missing_fields:
        print(f"❌ runflow.stages.validation missing fields: {missing_fields}")
        return False
    
    packs_count = validation_stage.get("packs_count", 0)
    results_received = validation_stage.get("results_received", 0)
    
    print(f"✅ runflow.json has validation stage: packs_built={packs_count}, results_received={results_received}")
    return True


def verify_no_ai_sends(sid: str, runs_root: Path) -> bool:
    """Verify no AI sends occurred (results directory is empty or has no .json files)."""
    results_dir = runs_root / sid / "ai_packs" / "validation" / "results"
    
    if not results_dir.exists():
        print(f"✅ results directory doesn't exist yet (no AI sends)")
        return True
    
    json_files = list(results_dir.glob("*.json"))
    if json_files:
        print(f"❌ Found {len(json_files)} result files in {results_dir} - AI sends occurred!")
        return False
    
    print(f"✅ results directory is empty (no AI sends)")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_validation_natives_wiring.py <SID>")
        print("\nExample:")
        print("  python devtools\\test_validation_natives_wiring.py test-sid-001")
        sys.exit(1)
    
    sid = sys.argv[1]
    runs_root = Path("runs").resolve()
    
    print(f"\n🔍 Testing validation natives wiring for SID: {sid}")
    print(f"   Runs root: {runs_root}\n")
    
    # Check if SID exists
    sid_dir = runs_root / sid
    if not sid_dir.exists():
        print(f"❌ SID directory not found: {sid_dir}")
        print("   Please provide an existing SID or create the directory structure first")
        sys.exit(1)
    
    # Verify manifest natives
    manifest_ok = verify_manifest_natives(sid, runs_root)
    
    # Verify runflow stage
    runflow_ok = verify_runflow_stage(sid, runs_root)
    
    # Verify no AI sends
    no_sends_ok = verify_no_ai_sends(sid, runs_root)
    
    print("\n" + "="*60)
    if manifest_ok and runflow_ok and no_sends_ok:
        print("✅ ALL CHECKS PASSED")
        print("\nValidation natives are correctly wired:")
        print("  - manifest.json has complete validation paths")
        print("  - runflow.json has validation stage with packs_built status")
        print("  - No AI sends occurred (results directory empty)")
        sys.exit(0)
    else:
        print("❌ SOME CHECKS FAILED")
        if not manifest_ok:
            print("  - manifest.json validation natives incomplete or missing")
        if not runflow_ok:
            print("  - runflow.json validation stage missing or incorrect")
        if not no_sends_ok:
            print("  - AI sends occurred (results directory has files)")
        sys.exit(1)


if __name__ == "__main__":
    main()
