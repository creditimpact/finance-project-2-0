import json
from pathlib import Path


def compact_tags_for_sid(sid: str, runs_root: Path | None = None):
    base = Path(runs_root or "runs") / sid / "cases" / "accounts"
    if not base.exists():
        return
    for acc in base.iterdir():
        if not acc.is_dir():
            continue
        tags_p = acc / "tags.json"
        if not tags_p.exists():
            continue
        try:
            tags = json.loads(tags_p.read_text(encoding="utf-8"))
        except Exception:
            continue

        minimal = []
        merge_expl = []
        ai_expl = []

        for t in tags or []:
            k = t.get("kind")
            if k == "issue":
                minimal.append({"kind": "issue", "type": t.get("type")})
            elif k == "merge_best":
                minimal.append(
                    {
                        "kind": "merge_best",
                        "with": t.get("with"),
                        "decision": t.get("decision"),
                    }
                )
                merge_expl.append(t)
            elif k == "ai_decision":
                minimal.append(
                    {
                        "kind": "ai_decision",
                        "with": t.get("with"),
                        "decision": t.get("decision"),
                        "at": t.get("at"),
                    }
                )
                ai_expl.append(t)
            elif k == "same_debt_pair":
                minimal.append(
                    {
                        "kind": "same_debt_pair",
                        "with": t.get("with"),
                        "at": t.get("at"),
                    }
                )
                ai_expl.append(t)
            else:
                # Keep unknown tags in a minimal way
                m = {"kind": k}
                for f in ("with", "decision", "type", "at", "tag"):
                    if f in t:
                        m[f] = t[f]
                minimal.append(m)

        # write minimal tags
        tags_p.write_text(
            json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # move explanations to summary.json
        summ_p = acc / "summary.json"
        try:
            summary = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else {}
        except Exception:
            summary = {}
        summary["merge_explanations"] = merge_expl
        summary["ai_explanations"] = ai_expl
        summ_p.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
