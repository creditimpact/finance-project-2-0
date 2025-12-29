import os, sys, inspect, json, traceback

# בסיס סביבה
ROOT = os.path.abspath(os.getcwd())
os.environ.setdefault("PYTHONPATH", ROOT)
os.environ.setdefault("RUNS_ROOT", os.path.join(ROOT, "runs"))
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# דגלים סבירים לבנייה (לא מזיק)
os.environ.setdefault("ENABLE_AUTO_AI_PIPELINE", "1")
os.environ.setdefault("MERGE_V2_ONLY", "1")
os.environ.setdefault("WRITE_TAGS_ENABLED", "1")
os.environ.setdefault("TAGS_MINIMAL_ONLY", "1")
os.environ.setdefault("SUMMARY_INCLUDE_TAG_EXPLANATIONS", "1")
os.environ.setdefault("LOG_LEVEL", "INFO")

def pick_latest_sid(runs_root):
    try:
        entries = [d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
        if not entries:
            return None
        entries.sort(key=lambda d: os.path.getmtime(os.path.join(runs_root, d)), reverse=True)
        return entries[0]
    except Exception:
        return None

def try_call(fn, sid):
    # ננסה מגוון חתימות נפוצות
    tried = []
    for args, kwargs in [
        ((sid,), {}),
        ((), {"sid": sid}),
        ((sid, True), {}),
        ((sid,), {"write_packs": True}),
        ((sid,), {"force": True}),
        ((sid,), {"write_packs": True, "force": True}),
        ((sid,), {"log_candidates": True, "write_packs": True, "force": True}),
    ]:
        try:
            sig = inspect.signature(fn)
            sig.bind(*args, **kwargs)  # יוודא שהחתימה מתאימה
            print(f"[runner] calling {fn.__name__}{args} {kwargs}")
            res = fn(*args, **kwargs)
            print(f"[runner] {fn.__name__} returned: {type(res).__name__}")
            return True
        except TypeError as te:
            tried.append(f"{fn.__name__}{args} {kwargs} -> {te}")
        except Exception as e:
            print(f"[runner] {fn.__name__} raised: {e}")
            traceback.print_exc()
            return False
    print("[runner] Tried signatures:\n  " + "\n  ".join(tried))
    return False

def main():
    runs_root = os.environ.get("RUNS_ROOT", os.path.join(ROOT, "runs"))
    sid = os.environ.get("SID") or (sys.argv[1] if len(sys.argv) > 1 else pick_latest_sid(runs_root))
    if not sid:
        print("[runner] No SID found. Make sure runs/ contains at least one session.")
        sys.exit(2)
    print(f"[runner] Using SID: {sid}")

    # ננקה ai_packs ישן (אם קיים) כדי לראות בנייה נקייה
    pack_dir = os.path.join(runs_root, sid, "ai_packs")
    if os.path.isdir(pack_dir):
        import shutil
        print(f"[runner] Removing old pack dir: {pack_dir}")
        shutil.rmtree(pack_dir, ignore_errors=True)

    # מייבאים את מודול ה-build
    try:
        from backend.core.logic.report_analysis import account_merge as AM
    except Exception as e:
        print("[runner] Failed to import account_merge:", e)
        traceback.print_exc()
        sys.exit(3)

    # מאתרים פונקציה מתאימה
    candidates = []
    for name, obj in inspect.getmembers(AM, inspect.isfunction):
        low = name.lower()
        if ("score_all_pairs" in low) or ("build_ai" in low and "pack" in low) or ("build_packs" in low):
            candidates.append((name, obj))
    print("[runner] Candidate builders:", [n for n,_ in candidates])

    if not candidates:
        print("[runner] No builder-like function found. Open account_merge.py and look for the pack writer; expose it.")
        sys.exit(4)

    ok = False
    for name, fn in candidates:
        if try_call(fn, sid):
            ok = True
            break

    # סיכום
    exists = os.path.isdir(pack_dir)
    count = 0
    if exists:
        count = len([p for p in os.listdir(pack_dir) if p.startswith("pair_") and p.endswith(".jsonl")])
    print(json.dumps({"sid": sid, "ai_packs_dir_exists": exists, "pair_files": count}))
    sys.exit(0 if ok and exists and count >= 1 else 5)

if __name__ == "__main__":
    main()
