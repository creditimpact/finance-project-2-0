import os, sys, glob, json
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core.logic.validation_requirements import build_validation_requirements_for_account

def main(sid: str):
    acc_root = os.path.join("runs", sid, "cases", "accounts")
    # numeric account folders only
    acc_dirs = []
    for p in glob.glob(os.path.join(acc_root, "*")):
        name = os.path.basename(p)
        if os.path.isdir(p) and name.isdigit():
            acc_dirs.append((int(name), p))
    acc_dirs.sort(key=lambda t: t[0])

    out = []
    for idx, acc_dir in acc_dirs:
        if not os.path.isfile(os.path.join(acc_dir, "bureaus.json")):
            continue
        res = build_validation_requirements_for_account(acc_dir) or {}
        findings = res.get("findings") or []
        if not isinstance(findings, list):
            findings = []
        out.append({
            "account": idx,
            "count": res.get("count"),
            "fields": [r.get("field") for r in findings if isinstance(r, dict)]
        })
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: run_validation_once.py <SID>")
        sys.exit(2)
    main(sys.argv[1])
