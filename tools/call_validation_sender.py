import os, json, inspect
from pathlib import Path

# לוודא שה-KEY קיים
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("ERROR: OPENAI_API_KEY missing")

sid = os.environ.get("SID") or "41495514-931e-4100-90ca-4928464dcda8"

# וידוא index.json
idx_path = Path(f"runs/41495514-931e-4100-90ca-4928464dcda8/ai_packs/validation/index.json")
if not idx_path.exists():
    raise SystemExit(f"INDEX_MISSING: {idx_path}")

# ייבוא הפונקציה
from backend.validation.send_packs import send_validation_packs

# הצגת חתימה אמיתית (נדפיס כדי לראות בדיוק אילו פרמטרים קיימים)
sig = inspect.signature(send_validation_packs)
print("send_validation_packs signature:", sig)

# בונים מועמדים לפי ה-ENV; נעביר רק מה שקיים בחתימה בפועל
def _get(name, default=None):
    v = os.getenv(name)
    return v if v is not None else default

candidates = {
    "sid": sid,
    "use_manifest_paths": True,
    "index_path": f"runs/41495514-931e-4100-90ca-4928464dcda8/ai_packs/validation/index.json",
    "packs_glob": "val_acc_*.jsonl",
    "results_dir": f"runs/41495514-931e-4100-90ca-4928464dcda8/ai_packs/validation/results",

    # פרמטרי מודל/פורמט
    "model": _get("VALIDATION_MODEL") or _get("AI_MODEL") or "gpt-4o-mini",
    "response_format": _get("AI_RESPONSE_FORMAT", "json_object"),
    "temperature": float(_get("AI_TEMPERATURE", "0") or 0),
    "top_p": float(_get("AI_TOP_P", "1") or 1),
    "max_tokens": int(_get("AI_MAX_TOKENS", "400") or 400),

    # רשת/רייט-לימיט/ריטריי
    "connect_timeout": int(_get("AI_HTTP_CONNECT_TIMEOUT", "10") or 10),
    "read_timeout": int(_get("AI_HTTP_READ_TIMEOUT", "40") or 40),
    "concurrency": int(_get("SENDER_CONCURRENCY", "2") or 2),
    "rate_limit_rps": float(_get("RATE_LIMIT_RPS", "2") or 2),
    "max_inflight": int(_get("RATE_LIMIT_MAX_INFLIGHT", "4") or 4),
    "retry_max": int(_get("SENDER_RETRY_MAX", "3") or 3),
    "retry_backoff_ms": _get("SENDER_RETRY_BACKOFF_MS", "100,250,500"),

    # כתיבה
    "write_jsonl": (_get("VALIDATION_WRITE_JSONL", "1") in ("1","true","True")),
    "write_json":  (_get("VALIDATION_WRITE_JSON",  "1") in ("1","true","True")),
}

allowed = set(sig.parameters.keys())
kwargs = {k: v for k, v in candidates.items() if k in allowed}

print("KWARGS used:", json.dumps(kwargs, indent=2, ensure_ascii=False))

res = send_validation_packs(**kwargs)
print("RESULT:", res if res is not None else "OK (no explicit return)")
