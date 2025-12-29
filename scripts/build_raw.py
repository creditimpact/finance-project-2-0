from __future__ import annotations

"""
NOTE: Keep __future__ import at the very top. Then bootstrap sys.path.
"""

# --- bootstrap sys.path to project root ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------------

import sys
from pathlib import Path

from backend.core.logic.report_analysis.raw_builder import build_raw_from_windows


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_raw.py <SID>")
        sys.exit(2)
    sid = sys.argv[1]
    build_raw_from_windows(sid, Path.cwd())


if __name__ == "__main__":
    main()
