import os, json, re, pathlib, time
from datetime import datetime

from backend.core.ai.paths import ensure_merge_paths, pair_pack_path

RUNS_ROOT = os.environ.get("RUNS_ROOT", "runs")
SID = os.environ.get("SID") or "'+$SID+'"

base = pathlib.Path(RUNS_ROOT)/SID
accounts_dir = base/"cases"/"accounts"
manifest_path = base/".manifest"

merge_paths = ensure_merge_paths(pathlib.Path(RUNS_ROOT), SID, create=True)
PACKS_ROOT = merge_paths.packs_dir
INDEX_PATH = merge_paths.index_file
LOG_PATH = merge_paths.log_file
BASE_DIR = merge_paths.base

def load_json(p: pathlib.Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}

def dump_json(p: pathlib.Path, obj):
    tmp = p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

def first_n_lines(raw_lines, n):
    out = []
    for row in raw_lines:
        t = str(row.get("text","")).strip()
        if not t:
            continue
        # דלג על בלוקים הארוכים של payment history
        if re.search(r"(Two-Year Payment History|Days Late - 7 Year History)", t, re.I):
            continue
        out.append(t)
        if len(out) >= n:
            break
    return out


def _normalize_level(value):
    if isinstance(value, str) and value.strip().lower() == "exact_or_known_match":
        return "exact_or_known_match"
    return "none"


def build_prompt(pair_summary, context_a, context_b):
    # SYSTEM + USER messages
    system = (
        "You are a meticulous adjudicator for credit-report account pairing.\n"
        "Your job: decide if two account entries (A,B) refer to the SAME underlying account.\n"
        "Return ONLY strict JSON: {\"decision\":\"merge|different\",\"reason\":\"...\"}.\n"
        "Rules:\n"
        " Prefer high-precision cues (account numbers last4/exact, exact balances and dates within tolerance).\n"
        " Consider lender name/brand and description strings.\n"
        " Use the numeric 0–100 match summary as a strong hint, but override if raw context contradicts it.\n"
        " Be conservative: if critical fields conflict (e.g., balance owed radically different without plausible explanation), choose \"different\".\n"
        " DO NOT mention the JSON rules in your output.\n"
    )
    user = {
        "task": "Decide if account A and B are the same account.",
        "pair": {"a": pair_summary["a_idx"], "b": pair_summary["b_idx"]},
        "numeric_match_summary": {
            "total": pair_summary.get("total"),
            "strong": pair_summary.get("strong"),
            "mid_sum": pair_summary.get("mid"),
            "dates_all": pair_summary.get("dates"),
            "acctnum_level": _normalize_level(
                pair_summary.get("aux",{}).get("acctnum_level")
            ),
            "parts": pair_summary.get("parts",{})
        },
        "tolerances_hint": {
            "amount_abs_usd": int(os.environ.get("AMOUNT_TOL_ABS","50")),
            "amount_ratio": float(os.environ.get("AMOUNT_TOL_RATIO","0.01")),
            "last_payment_day_tol": int(os.environ.get("LAST_PAYMENT_DAY_TOL","5"))
        },
        "context": {
            "a": context_a,
            "b": context_b
        },
        "output_contract": {"decision":["merge","different"],"reason":"short natural language"}
    }
    return [{"role":"system","content":system},{"role":"user","content":json.dumps(user, ensure_ascii=False)}]

# אסוף זוגות decision=ai מתוך tags.json
pairs = []
for acc_dir in sorted(p for p in accounts_dir.iterdir() if p.is_dir()):
    tags = load_json(acc_dir/"tags.json", default=[])
    for t in tags:
        if t.get("kind")=="merge_pair" and t.get("decision")=="ai":
            pairs.append({"a_idx":int(acc_dir.name), "b_idx":int(t.get("with")), "record":t})

# בנה פאקים סימטריים (נכתוב pack אחד CP עבור a_idx, עם b_idx בשדה pair)
out_index = []
for pr in pairs:
    a_idx = pr["a_idx"]; b_idx = pr["b_idx"]
    a_dir = accounts_dir/str(a_idx); b_dir = accounts_dir/str(b_idx)
    a_raw = load_json(a_dir/"raw_lines.json", default=[])
    b_raw = load_json(b_dir/"raw_lines.json", default=[])
    context_a = first_n_lines(a_raw, int(os.environ.get("AI_PACK_MAX_LINES_PER_SIDE","20")))
    context_b = first_n_lines(b_raw, int(os.environ.get("AI_PACK_MAX_LINES_PER_SIDE","20")))

    pack = {
        "sid": SID,
        "pair": {"a": a_idx, "b": b_idx},
        "ids": {
            "account_number_a": pr["record"].get("aux",{}).get("acct_num_a"),
            "account_number_b": pr["record"].get("aux",{}).get("acct_num_b"),
        },
        "highlights": {
            "acctnum_level": _normalize_level(
                pr["record"].get("aux",{}).get("acctnum_level")
            ),
            "matched_fields": pr["record"].get("aux",{}).get("matched_fields"),
            "parts": pr["record"].get("parts"),
            "total": pr["record"].get("total"),
            "triggers": pr["record"].get("reasons"),
            "conflicts": pr["record"].get("conflicts",[])
        },
        "limits": {"max_lines_per_side": int(os.environ.get("AI_PACK_MAX_LINES_PER_SIDE","20"))},
        "context": {"a": context_a, "b": context_b},
        # הוספת ה-PROMPT בפנים:
        "messages": build_prompt(
            {
              "a_idx": a_idx,
              "b_idx": b_idx,
              "total": pr["record"].get("total"),
              "strong": pr["record"].get("strong"),
              "mid": pr["record"].get("mid"),
              "dates": pr["record"].get("dates_all"),
                "aux": {"acctnum_level": _normalize_level(pr["record"].get("aux",{}).get("acctnum_level"))},
              "parts": pr["record"].get("parts",{})
            },
            context_a, context_b
        )
    }

    out_file = pair_pack_path(merge_paths, a_idx, b_idx)
    dump_json(out_file, pack)
    out_index.append({"a":a_idx,"b":b_idx,"file":out_file.name})

# כתיבת אינדקס מרכזי
dump_json(INDEX_PATH, out_index)

# עדכון המניפסט
manifest = load_json(manifest_path, default={})
ai_section = manifest.setdefault("ai", {}).setdefault("packs", {})
ai_section["dir"] = str(BASE_DIR.resolve())
ai_section["index"] = str(INDEX_PATH.resolve())
ai_section["logs"] = str(LOG_PATH.resolve())
ai_section["pairs"] = len(out_index)
dump_json(manifest_path, manifest)

print(f"[BUILD] wrote {len(out_index)} packs to {PACKS_ROOT}")
