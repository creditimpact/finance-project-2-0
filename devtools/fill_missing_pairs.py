import os, sys, json, inspect, traceback

ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)               # <<< פותר import backend
RUNS_ROOT = os.environ.get("RUNS_ROOT", os.path.join(ROOT, "runs"))

def load_module():
    from backend.core.logic.report_analysis import account_merge as AM
    return AM

def pack_path(sid, a, b):
    lo, hi = sorted([a, b])
    return os.path.join(RUNS_ROOT, sid, "ai_packs", f"pair_{lo:03d}_{hi:03d}.jsonl")

def exists_pack(sid, a, b):
    return os.path.isfile(pack_path(sid, a, b))

def try_build_pair(am, sid, a, b):
    fn = getattr(am, "build_ai_pack_for_pair", None)
    if fn is None or not inspect.isfunction(fn):
        return False, "build_ai_pack_for_pair not found"

    attempts = [
        ((RUNS_ROOT, sid, a, b), {"write": True}),
        ((RUNS_ROOT, sid, a, b), {"write_packs": True}),
        ((RUNS_ROOT, sid, a, b), {}),
        ((RUNS_ROOT, sid, a, b), {"force": True, "write": True}),
        ((RUNS_ROOT, sid, a, b), {"force": True, "write_packs": True}),
        ((RUNS_ROOT, sid, a, b), {"log_candidates": True, "write": True}),
    ]
    for args, kwargs in attempts:
        try:
            print(f"[fill] calling build_ai_pack_for_pair{args} {kwargs}")
            fn(*args, **kwargs)
            return True, "ok"
        except TypeError as te:
            print(f"[fill] TypeError: {te}")
            continue
        except Exception as e:
            print(f"[fill] raised: {e}")
            traceback.print_exc()
            if exists_pack(sid, a, b):
                return True, "ok_after_exc"
            return False, str(e)
    return False, "no_signature_match"

def try_score_subset(am, sid, subset):
    fn = getattr(am, "score_all_pairs_0_100", None)
    if fn is None or not inspect.isfunction(fn):
        return False, "score_all_pairs_0_100 not found"

    attempts = [
        ((RUNS_ROOT, sid), {"idx_list": subset, "write_packs": True}),
        ((RUNS_ROOT, sid), {"idx_list": subset, "force": True, "write_packs": True}),
        ((RUNS_ROOT, sid), {"idx_list": subset}),
    ]
    for args, kwargs in attempts:
        try:
            print(f"[fill] calling score_all_pairs_0_100{args} {kwargs}")
            fn(*args, **kwargs)
            return True, "ok"
        except TypeError as te:
            print(f"[fill] TypeError: {te}")
            continue
        except Exception as e:
            print(f"[fill] raised: {e}")
            traceback.print_exc()
            return False, str(e)
    return False, "no_signature_match"

def main():
    if len(sys.argv) < 2:
        print("usage: fill_missing_pairs.py <SID>")
        sys.exit(2)
    sid = sys.argv[1]
    target_pairs = [(14,29), (15,29)]  # הזוגות שחסרים

    am = load_module()

    results = []
    for a,b in target_pairs:
        if exists_pack(sid, a, b):
            results.append({"pair": (a,b), "status": "already_exists"})
            continue
        ok, msg = try_build_pair(am, sid, a, b)
        results.append({"pair": (a,b), "status": "built" if ok else f"try_build_failed:{msg}"})

    still_missing = [(a,b) for (a,b) in target_pairs if not exists_pack(sid,a,b)]
    if still_missing:
        subset = sorted({i for pair in still_missing for i in pair})
        ok, msg = try_score_subset(am, sid, subset)
        results.append({"subset_attempt": subset, "status": "ok" if ok else f"subset_failed:{msg}"})

    print(json.dumps({"sid": sid, "results": results}, ensure_ascii=False))

if __name__ == "__main__":
    main()
