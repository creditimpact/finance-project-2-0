#!/usr/bin/env python3
"""Recompute outcome events from raw reports.

Usage:
    python tools/replay_outcomes.py report1.json [report2.json ...]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List

from backend.api import session_manager
from services.outcome_ingestion.ingest_report import ingest_report


def replay(paths: List[str]) -> None:
    if not paths:
        print("Provide one or more JSON report files")
        return
    os.environ.setdefault("SESSION_ID", "replay")
    tmp = tempfile.NamedTemporaryFile(delete=False)
    session_manager.SESSION_FILE = tmp.name
    try:
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                report = json.load(f)
            events = ingest_report(None, report)
            for e in events:
                print(json.dumps(asdict(e)))
    finally:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    replay(sys.argv[1:])
