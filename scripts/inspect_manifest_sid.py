import json, pprint, os, sys
sid = os.environ.get("INSPECT_SID", "275cd0e2-7dd8-41fe-a8d9-4420bddb5bef")
path = fr"c:\dev\credit-analyzer\runs\{sid}\manifest.json"
print("Manifest path:", path)
if not os.path.exists(path):
    print("Manifest missing")
    sys.exit(0)
with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)
print("=== ai.packs.validation ===")
pprint.pprint(d.get("ai", {}).get("packs", {}).get("validation"))
print("=== artifacts.ai.packs.validation ===")
pprint.pprint(d.get("artifacts", {}).get("ai", {}).get("packs", {}).get("validation"))
print("=== meta.validation_paths_initialized ===")
print(d.get("meta", {}).get("validation_paths_initialized"))
