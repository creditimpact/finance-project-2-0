from pathlib import Path
import json
import sys
import os

root = Path(r"c:\dev\credit-analyzer")
# Ensure backend package is importable
sys.path.insert(0, str(root))

from backend.runflow.decider import get_runflow_snapshot, reconcile_umbrella_barriers
sid = sys.argv[1] if len(sys.argv) > 1 else "160884fe-b510-493b-888d-dd2ec09b4bb5"

print("TESTING FIXES ON SID:", sid)

before = get_runflow_snapshot(sid, runs_root=root)
val_stage_before = (before.get("stages") or {}).get("validation") or {}
umb_before = before.get("umbrella_barriers") or {}
print("BEFORE status=", val_stage_before.get("status"), "ready_latched=", val_stage_before.get("ready_latched"), "reconciliation_idle=", umb_before.get("reconciliation_idle"))

statuses = reconcile_umbrella_barriers(sid, runs_root=root)
print("STATUSES:")
print(json.dumps(statuses, indent=2, sort_keys=True))

after = get_runflow_snapshot(sid, runs_root=root)
val_stage_after = (after.get("stages") or {}).get("validation") or {}
umb_after = after.get("umbrella_barriers") or {}
print("AFTER status=", val_stage_after.get("status"), "ready_latched=", val_stage_after.get("ready_latched"), "reconciliation_idle=", umb_after.get("reconciliation_idle"))

# Test Fix 7: mark idle and ensure reconcile returns existing flags without recompute
runflow_path = root / sid / "runflow.json"
try:
	import json as _json
	payload = _json.loads(runflow_path.read_text(encoding="utf-8"))
	umb = payload.get("umbrella_barriers") or {}
	umb["reconciliation_idle"] = True
	# Flip merge_ready to False to see early-return effect
	umb["merge_ready"] = False
	payload["umbrella_barriers"] = umb
	runflow_path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
	statuses2 = reconcile_umbrella_barriers(sid, runs_root=root)
	print("AFTER-IDLE-RECONCILE statuses:", _json.dumps(statuses2, indent=2, sort_keys=True))
except Exception as e:
	print("IDLE test skip due to:", repr(e))
