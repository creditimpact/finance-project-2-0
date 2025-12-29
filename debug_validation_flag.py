import json
from pathlib import Path

RUNS_ROOT = r"C:\dev\credit-analyzer\runs"
SID = "4e85b354-0b4c-47d9-b788-0060a0193fc7"

run_dir = Path(RUNS_ROOT) / SID

manifest_path = run_dir / "manifest.json"
runflow_path = run_dir / "runflow.json"

print("MANIFEST PATH:", manifest_path)
print("RUNFLOW  PATH:", runflow_path)

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

with open(runflow_path, "r", encoding="utf-8") as f:
    runflow = json.load(f)

manifest_validation = (
    manifest.get("ai", {})
            .get("status", {})
            .get("validation", {})
)

runflow_validation = (
    runflow.get("stages", {})
           .get("validation", {})
)

print("\n=== MANIFEST validation ===")
print(json.dumps(manifest_validation, indent=2))

print("\n=== RUNFLOW validation ===")
print(json.dumps(runflow_validation, indent=2))

print("\nVALUES:")
print("manifest.validation_ai_applied =", manifest_validation.get("validation_ai_applied"))
print("runflow.validation_ai_applied  =", runflow_validation.get("validation_ai_applied"))
