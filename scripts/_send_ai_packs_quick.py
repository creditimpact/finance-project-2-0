import json
import os
import pathlib
import time
import urllib.error
import urllib.request

try:  # pragma: no cover - convenience bootstrap for direct execution
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback when bootstrap is unavailable
    pass

from backend.core.ai import build_openai_headers

RUNS_ROOT = os.environ.get("RUNS_ROOT", "runs")
SID = os.environ.get("SID")
MODEL = os.environ.get("AI_MODEL", "gpt-4o-mini")
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
KEY = os.environ.get("OPENAI_API_KEY")
TIMEOUT = int(os.environ.get("AI_REQUEST_TIMEOUT", "30"))

base = pathlib.Path(RUNS_ROOT) / SID
manifest = json.loads((base / ".manifest").read_text(encoding="utf-8"))
pairs = manifest.get("artifacts", {}).get("ai", {}).get("pairs", [])


def read(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def write_tags(acc_dir: pathlib.Path, new_tag: dict):
    tags_p = acc_dir / "tags.json"
    try:
        tags = json.loads(tags_p.read_text(encoding="utf-8"))
    except Exception:
        tags = []
    tags = [t for t in tags if not (t.get("kind") == "ai_decision" and t.get("with") == new_tag.get("with"))]
    tags.append(new_tag)
    tmp = tags_p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tags, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(tags_p)


def chat(messages):
    headers = build_openai_headers(api_key=KEY)
    req = urllib.request.Request(
        url=f"{BASE}/chat/completions",
        headers=headers,
        data=json.dumps({"model": MODEL, "messages": messages, "temperature": 0}).encode("utf-8"),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_decision(text):
    try:
        obj = json.loads(text)
        d = obj.get("decision", "").lower()
        if d in ("merge", "different"):
            return d, obj.get("reason", "")
    except Exception:
        pass
    # לא JSON תקין  נפיל כ-different שמרני
    return "different", "fallback_parse_error"


for item in pairs:
    pack = read(item["file"])
    msgs = pack["messages"]
    try:
        resp = chat(msgs)
        content = resp["choices"][0]["message"]["content"]
        decision, reason = parse_decision(content)
    except Exception as e:
        decision, reason = "different", f"transport_error:{e.__class__.__name__}"

    a = int(pack["pair"]["a"])
    b = int(pack["pair"]["b"])
    # כתוב ai_decision בכל אחד משני החשבונות (סימטרי)
    for idx, with_idx in [(a, b), (b, a)]:
        acc_dir = base / "cases" / "accounts" / str(idx)
        tag = {
            "kind": "ai_decision",
            "tag": "ai_decision",
            "source": "ai_adjudicator",
            "with": with_idx,
            "decision": decision,
            "reason": reason,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_tags(acc_dir, tag)

print(f"[AI] adjudicated {len(pairs)} pairs")
