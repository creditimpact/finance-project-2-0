import os, sys, json, glob

ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core.logic.intra_polarity import analyze_account_polarity

sid = sys.argv[1] if len(sys.argv) > 1 else None
if not sid:
    print("usage: run_polarity_once.py <SID>")
    sys.exit(2)

acc_root = os.path.join("runs", sid, "cases", "accounts")
entries = glob.glob(os.path.join(acc_root, "*"))

# קח רק תיקיות ששמן מספרי (למשל "7", "10") – התעלם מקבצים כמו index.json
acc_dirs = []
for p in entries:
    name = os.path.basename(p)
    if os.path.isdir(p) and name.isdigit():
        acc_dirs.append((int(name), p))
acc_dirs.sort(key=lambda t: t[0])

out = []
for idx, acc_dir in acc_dirs:
    bureaus = os.path.join(acc_dir, "bureaus.json")
    if not os.path.isfile(bureaus):
        out.append({"idx": idx, "status": "skip_no_bureaus"})
        continue
    try:
        res = analyze_account_polarity(sid, acc_dir)
        # מצפה שהפונקציה גם כתבה ל-summary.json וגם מחזירה dict עם המפתח polarity_check
        fields = sorted(list((res or {}).get("polarity_check", {}).keys()))
        out.append({"idx": idx, "status": "ok", "fields": fields})
    except Exception as e:
        out.append({"idx": idx, "status": "error", "error": str(e)})

print(json.dumps(out, ensure_ascii=False, indent=2))
