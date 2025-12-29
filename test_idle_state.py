import sys
sys.path.insert(0, ".")
from backend.runflow.decider import _load_runflow, _runflow_path
from pathlib import Path

sid = "160884fe-b510-493b-888d-dd2ec09b4bb5"
runflow_path = _runflow_path(sid, Path("runs"))
data = _load_runflow(runflow_path, sid)
existing = data.get("umbrella_barriers")
if isinstance(existing, dict):
    prev_idle = bool(existing.get("reconciliation_idle"))
    print(f"Previous idle state: {prev_idle}")
    print(f"Type: {type(prev_idle)}")
    print(f"Raw value: {existing.get('reconciliation_idle')}")
    print(f"NOT previous_idle: {not prev_idle}")
else:
    print("No umbrella_barriers")
