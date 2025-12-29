import os
import sys
import json
import glob

ROOT = os.getcwd()
sys.path.insert(0, ROOT)

from backend.core.logic.validation_requirements import build_validation_requirements_for_account


sid = sys.argv[1]
acc_root = os.path.join("runs", sid, "cases", "accounts")
for p in sorted(
    [d for d in glob.glob(os.path.join(acc_root, "*")) if os.path.isdir(d)],
    key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else 10**9,
):
    s = os.path.join(p, "summary.json")
    if not os.path.isfile(s):
        continue
    res = build_validation_requirements_for_account(p)
    print(os.path.basename(p), json.dumps(res.get("validation_requirements", {}), ensure_ascii=False))
