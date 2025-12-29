import os, sys, json, inspect, traceback

ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RUNS_ROOT = os.environ.get("RUNS_ROOT", os.path.join(ROOT, "runs"))

def main():
    if len(sys.argv) < 3:
        print("usage: call_score_subset.py <SID> <comma_separated_indices>")
        sys.exit(2)
    sid = sys.argv[1]
    idx_list = sorted({int(x) for x in sys.argv[2].split(",") if x.strip()})

    from backend.core.logic.report_analysis import account_merge as AM
    fn = getattr(AM, "score_all_pairs_0_100", None)
    if fn is None or not callable(fn):
        print(json.dumps({"error": "score_all_pairs_0_100 not found"}))
        sys.exit(3)

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    kwargs = {}
    # נכניס רק מה שקיים באמת בחתימה
    if "runs_root" in params: kwargs["runs_root"] = RUNS_ROOT
    if "sid"       in params: kwargs["sid"]       = sid
    if "idx_list"  in params: kwargs["idx_list"]  = idx_list
    if "write_packs" in params: kwargs["write_packs"] = True
    if "log_candidates" in params: kwargs["log_candidates"] = True
    if "force" in params: kwargs["force"] = True

    # אם "idx_list" לא קיים בשם הזה, ננסה שמות חלופיים נפוצים:
    alt_idx_names = [n for n in ("indices", "subset", "nodes", "indexes") if n in params and "idx_list" not in params]
    if not kwargs.get("idx_list") and alt_idx_names:
        kwargs[alt_idx_names[0]] = idx_list

    # עכשיו נריץ ב־kwargs בלבד כדי למנוע התנגשות positional/keyword
    try:
        print(f"[wrapper] calling score_all_pairs_0_100 with kwargs={kwargs}")
        res = fn(**kwargs)
        print(json.dumps({"ok": True, "called_with": kwargs, "result_type": type(res).__name__}))
    except Exception as e:
        traceback.print_exc()
        print(json.dumps({"ok": False, "called_with": kwargs, "error": str(e)}))
        sys.exit(4)

if __name__ == "__main__":
    main()
