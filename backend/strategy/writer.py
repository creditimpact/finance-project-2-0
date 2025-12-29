"""Writers for persisting planner outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def write_master_and_weekdays(out_dir: Path, plan: Dict, prefix: str, master_name: str) -> None:
    """Write the master plan and weekday-specific variants atomically."""

    out_dir.mkdir(parents=True, exist_ok=True)

    master_tmp = out_dir / f"{master_name}.tmp"
    master_tmp.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    master_tmp.replace(out_dir / master_name)

    by_weekday = plan.get("by_weekday", {}) if isinstance(plan, dict) else {}
    for key, payload in by_weekday.items():
        weekday = int(key) if isinstance(key, str) and key.isdigit() else key
        filename = f"{prefix}{weekday}.json"
        tmp_path = out_dir / f"{filename}.tmp"
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(out_dir / filename)
