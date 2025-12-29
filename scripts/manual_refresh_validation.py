"""Manual script to refresh validation stage from index for a SID."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.runflow.decider import refresh_validation_stage_from_index, reconcile_umbrella_barriers


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/manual_refresh_validation.py <SID> [--runs-root PATH]")
        sys.exit(1)
    
    sid = sys.argv[1]
    runs_root = None
    
    if "--runs-root" in sys.argv:
        idx = sys.argv.index("--runs-root")
        if idx + 1 < len(sys.argv):
            runs_root = Path(sys.argv[idx + 1])
    
    if runs_root is None:
        runs_root = PROJECT_ROOT / "runs"
    
    print(f"Refreshing validation stage for SID: {sid}")
    print(f"Runs root: {runs_root}")
    print()
    
    try:
        print("Step 1: Calling refresh_validation_stage_from_index...")
        refresh_validation_stage_from_index(sid, runs_root=runs_root)
        print("✓ Refresh completed")
        print()
        
        print("Step 2: Calling reconcile_umbrella_barriers...")
        barriers = reconcile_umbrella_barriers(sid, runs_root=runs_root)
        print("✓ Barriers reconciled")
        print()
        
        print("Barriers state:")
        for key, value in barriers.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
