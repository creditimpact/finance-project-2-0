#!/usr/bin/env python
"""
Investigation helper: dump runflow events for a SID.
Usage (PowerShell):
  python scripts\dump_runflow_events_for_sid.py 126426ed-1a2d-4bc6-84ba-b0a9613042e1
Options:
  --stage <name>     Filter by stage (e.g., validation, merge, frontend)
  --event <name>     Filter by event (e.g., start, end, decide, barriers_reconciled)
  --contains <text>  Substring filter on the JSON line
If no SID is provided, defaults to the last current run in runs/current.txt if present.
"""

import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

def _read_current_sid() -> str | None:
    try:
        text = (RUNS / "current.txt").read_text(encoding="utf-8").strip()
        return text or None
    except OSError:
        return None


def main(argv: list[str]) -> int:
    sid: str | None = None
    stage_filter: str | None = None
    event_filter: str | None = None
    contains: str | None = None

    args = list(argv)
    if args and not args[0].startswith("-"):
        sid = args.pop(0)
    while args:
        tok = args.pop(0)
        if tok == "--stage" and args:
            stage_filter = args.pop(0)
        elif tok == "--event" and args:
            event_filter = args.pop(0)
        elif tok == "--contains" and args:
            contains = args.pop(0)
        else:
            print(f"Unknown option: {tok}", file=sys.stderr)
            return 2

    if sid is None:
        sid = _read_current_sid()
    if not sid:
        print("SID is required (provide as arg or ensure runs/current.txt exists)", file=sys.stderr)
        return 2

    events_path = RUNS / sid / "runflow_events.jsonl"
    if not events_path.exists():
        print(f"No events file found: {events_path}", file=sys.stderr)
        return 1

    try:
        raw = events_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Failed to read events: {exc}", file=sys.stderr)
        return 1

    printed = 0
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        if contains and contains not in text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            print(text)
            printed += 1
            continue
        if stage_filter and str(row.get("stage") or "").strip() != stage_filter:
            continue
        if event_filter and str(row.get("event") or "").strip() != event_filter:
            continue
        print(json.dumps(row, ensure_ascii=False))
        printed += 1

    if printed == 0:
        print("(no matching events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
