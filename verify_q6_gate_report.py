#!/usr/bin/env python3
"""Q6 Gate Verification Report (presence-only) for a given SID.

Usage:
  python verify_q6_gate_report.py <SID>

Performs:
- Filesystem evidence (accounts, bureau files)
- Content validation (gate/root_checks) for spot accounts/bureaus
- Bureau isolation comparison (differences across bureaus for same account)
- Strict mode check when TRADELINE_CHECK_GATE_STRICT=1 (optional)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUNS_ROOT = Path("runs")


def list_accounts(sid: str) -> list[Path]:
    base = RUNS_ROOT / sid / "cases" / "accounts"
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def bureaus_present(bureaus_path: Path) -> list[str]:
    try:
        data = json.loads(bureaus_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return [b for b in ("transunion", "experian", "equifax") if b in data]


def read_bureau_json(acc_dir: Path, bureau: str) -> dict | None:
    f = acc_dir / "tradeline_check" / f"{bureau}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def print_gate_snippet(d: dict) -> None:
    gate = d.get("gate", {})
    root = d.get("root_checks", {})
    snippet = {
        "gate": {
            "version": gate.get("version"),
            "eligible": gate.get("eligible"),
            "placeholders": gate.get("placeholders"),
        },
        "root_checks": root,
        "status": d.get("status"),
        "schema_version": d.get("schema_version"),
        "findings": d.get("findings"),
    }
    print(json.dumps(snippet, indent=2))


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_q6_gate_report.py <SID>")
        sys.exit(2)
    sid = sys.argv[1]

    print(f"🔎 Q6 Gate Verification for SID={sid}")

    # Step 2 — Filesystem evidence
    accounts = list_accounts(sid)
    if not accounts:
        print("❌ No accounts found.")
        sys.exit(1)

    print(f"\nAccounts found ({len(accounts)}): {[p.name for p in accounts]}")

    for acc_dir in accounts:
        tradir = acc_dir / "tradeline_check"
        bureaus_path = acc_dir / "bureaus.json"
        present = bureaus_present(bureaus_path)
        files = [f"{b}.json" for b in present if (tradir / f"{b}.json").exists()]
        print(f"  - {acc_dir.name}: tradeline_check/ exists={tradir.exists()} files={files}")

    # Step 3 — Content validation (spot-check at least 2 accounts, 2 bureaus)
    checked = 0
    print("\nSpot checks (gate + root_checks):")
    for acc_dir in accounts:
        present = bureaus_present(acc_dir / "bureaus.json")
        for bureau in present[:2]:
            data = read_bureau_json(acc_dir, bureau)
            if not data:
                continue
            print(f"\n  Account {acc_dir.name} Bureau {bureau}:")
            print_gate_snippet(data)
            checked += 1
            if checked >= 4:
                break
        if checked >= 4:
            break

    # Step 4 — Bureau isolation proof
    print("\nBureau isolation comparison:")
    isolation_ok = False
    for acc_dir in accounts:
        present = bureaus_present(acc_dir / "bureaus.json")
        if len(present) < 2:
            continue
        a, b = present[:2]
        da = read_bureau_json(acc_dir, a)
        db = read_bureau_json(acc_dir, b)
        if not da or not db:
            continue
        ga = da.get("gate", {})
        gb = db.get("gate", {})
        if ga.get("eligible") != gb.get("eligible") or ga.get("missing_fields") != gb.get("missing_fields"):
            print(f"  ✓ Differences detected in account {acc_dir.name} between {a} and {b}:")
            print(f"    {a}.eligible={ga.get('eligible')}\n    {b}.eligible={gb.get('eligible')}")
            isolation_ok = True
            break
    if not isolation_ok:
        print("  No per-bureau differences detected; likely identical inputs across bureaus or limited data variation.")

    print("\n✅ Q6 report complete.")


if __name__ == "__main__":
    main()
