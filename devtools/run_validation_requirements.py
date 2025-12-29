import os, sys, glob, json
ROOT = os.getcwd()
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from backend.core.logic.validation_requirements import build_validation_requirements_for_account

sid = sys.argv[1]
acc_root = os.path.join("runs", sid, "cases", "accounts")

# מצא רק תיקיות מספריות (התעלם מקבצים כמו index.json)
def is_acc_dir(p):
    name = os.path.basename(p)
    return os.path.isdir(p) and name.isdigit()

acc_dirs = sorted([p for p in glob.glob(os.path.join(acc_root, "*")) if is_acc_dir(p)],
                  key=lambda p: int(os.path.basename(p)))

out = []
for acc_dir in acc_dirs:
    if os.path.isfile(os.path.join(acc_dir, "bureaus.json")):
        res = build_validation_requirements_for_account(acc_dir)
        out.append({"idx": os.path.basename(acc_dir), "status": res.get("status"), "count": res.get("count")})
    else:
        out.append({"idx": os.path.basename(acc_dir), "status": "skip_no_bureaus"})
print(json.dumps(out, ensure_ascii=False, indent=2))
