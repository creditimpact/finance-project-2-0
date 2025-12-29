"""
End-to-end test showing validation natives remain stable across all manifest writers.
This demonstrates that the disk-first manifest API prevents clobbering.
"""
from backend.pipeline.runs import persist_manifest, RunManifest
from pathlib import Path
import json

sid = "d22820d3-30d9-49cd-a8ec-37b106298948"

print("=" * 80)
print("Testing validation natives stability across persist_manifest")
print("=" * 80)

# Check initial state
manifest_json = Path(f"runs/{sid}/manifest.json")
data_before = json.loads(manifest_json.read_text())

print("\nBEFORE persist_manifest:")
print(f"  ai.packs.validation present: {bool(data_before.get('ai', {}).get('packs', {}).get('validation'))}")
print(f"  ai.validation present: {bool(data_before.get('ai', {}).get('validation'))}")
print(f"  artifacts.ai.packs.validation present: {bool(data_before.get('artifacts', {}).get('ai', {}).get('packs', {}).get('validation'))}")
print(f"  meta.validation_paths_initialized: {data_before.get('meta', {}).get('validation_paths_initialized')}")

# Call persist_manifest with artifacts (simulating normal usage)
manifest = RunManifest.for_sid(sid, allow_create=False)
test_path = Path(f"runs/{sid}/final_test.txt")
test_path.write_text("final test")

result = persist_manifest(
    manifest, 
    artifacts={"final_test": {"file": str(test_path)}}
)

# Check after state
data_after = json.loads(manifest_json.read_text())

print("\nAFTER persist_manifest:")
print(f"  ai.packs.validation present: {bool(data_after.get('ai', {}).get('packs', {}).get('validation'))}")
print(f"  ai.validation present: {bool(data_after.get('ai', {}).get('validation'))}")
print(f"  artifacts.ai.packs.validation present: {bool(data_after.get('artifacts', {}).get('ai', {}).get('packs', {}).get('validation'))}")
print(f"  meta.validation_paths_initialized: {data_after.get('meta', {}).get('validation_paths_initialized')}")
print(f"  final_test artifact added: {'final_test' in data_after.get('artifacts', {})}")

# Verify no clobbering
validation_stable = (
    bool(data_after.get('ai', {}).get('packs', {}).get('validation')) and
    bool(data_after.get('ai', {}).get('validation')) and
    bool(data_after.get('artifacts', {}).get('ai', {}).get('packs', {}).get('validation')) and
    data_after.get('meta', {}).get('validation_paths_initialized')
)

print("\n" + "=" * 80)
if validation_stable:
    print("✓ SUCCESS: Validation natives remain stable after persist_manifest")
else:
    print("✗ FAILURE: Validation natives were clobbered")
print("=" * 80)
