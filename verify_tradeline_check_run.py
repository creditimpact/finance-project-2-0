#!/usr/bin/env python3
"""Re-run validation pipeline for specific SID to verify tradeline_check integration.

Usage:
  python verify_tradeline_check_run.py <SID>
If no SID provided, uses the default hardcoded value.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from backend.validation.pipeline import run_validation_summary_pipeline

DEFAULT_SID = "62e31d47-0ad6-4e49-967c-6cb74d624373"
RUNS_ROOT = Path("runs")


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SID
    print(f"🔄 Re-running validation pipeline for SID: {sid}")
    
    run_dir = RUNS_ROOT / sid
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)
    
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        sys.exit(1)
    
    print(f"✓ Found run directory: {run_dir}")
    print(f"✓ Found manifest: {manifest_path}")
    
    try:
        print("\n▶ Starting validation pipeline...")
        result = run_validation_summary_pipeline(manifest_path)
        print(f"\n✅ Pipeline completed successfully!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
