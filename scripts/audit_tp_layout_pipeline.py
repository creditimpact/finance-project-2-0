from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict


def main(argv: list[str]) -> int:
    # Resolve PDF path from env or argument
    pdf_arg = os.getenv("AUDIT_PDF") or (argv[1] if len(argv) > 1 else None)
    if not pdf_arg:
        print("Usage: AUDIT_PDF=<path> python scripts/audit_tp_layout_pipeline.py [pdf_path]")
        return 2
    pdf_path = Path(pdf_arg)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return 2

    # Lazy imports to avoid heavy deps at import time
from backend.core.text.env_guard import ensure_env_and_paths
from backend.core.logic.report_analysis.text_provider import extract_and_cache_text
    from backend.core.logic.report_analysis.block_exporter import export_account_blocks
    try:
        from backend.core.text.text_provider import load_text_with_layout  # type: ignore
    except Exception:  # pragma: no cover - optional
        load_text_with_layout = None  # type: ignore

    ensure_env_and_paths()
    sid = uuid.uuid4().hex[:12]

    t0 = time.perf_counter()
    extract_meta: Dict[str, Any] = {}
    try:
        meta = extract_and_cache_text(sid, str(pdf_path))
        extract_meta = dict(meta.get("meta") or {})
    except Exception as e:
        print(f"extract_and_cache_text failed: {e}")
        return 1
    t1 = time.perf_counter()

    blocks: list[dict] = []
    try:
        blocks = export_account_blocks(session_id=sid, pdf_path=str(pdf_path))
    except Exception as e:
        print(f"export_account_blocks failed: {e}")
        return 1
    t2 = time.perf_counter()

    pages_count = 0
    tokens_total = 0
    if load_text_with_layout:
        try:
            layout = load_text_with_layout(str(pdf_path))
            pages = list(layout.get("pages") or [])
            pages_count = len(pages)
            tokens_total = sum(len(p.get("tokens") or []) for p in pages)
        except Exception:
            pages_count = 0
            tokens_total = 0

    # Build summary
    summary = {
        "session_id": sid,
        "pdf_path": str(pdf_path),
        "pages_count": pages_count,
        "tokens_total": tokens_total,
        "blocks_total": len(blocks),
        "extract_ms": int((t1 - t0) * 1000),
        "export_ms": int((t2 - t1) * 1000),
        "blocks": [],
    }
    for b in blocks:
        meta = b.get("meta", {}) if isinstance(b, dict) else {}
        presence = meta.get("bureau_presence") or {}
        summary["blocks"].append(
            {
                "heading": b.get("heading"),
                "bureau_presence": presence,
                "has_layout_tokens": bool(b.get("layout_tokens")),
            }
        )

    out_dir = Path("traces") / "blocks" / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            k: summary[k]
            for k in [
                "session_id",
                "pages_count",
                "tokens_total",
                "blocks_total",
                "extract_ms",
                "export_ms",
            ]
        },
        indent=2,
    ))
    print(f"Artifacts: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
