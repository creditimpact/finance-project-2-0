#!/usr/bin/env python3
"""Scan Stage-A triad tokens against bureaus.json for alignment regressions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - convenience bootstrap
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - best effort path setup
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from backend.core.logic.report_analysis.normalize_fields import clean_value

UNICODE_COLONS = (":", "：", "﹕", "︓")
SPACE_RE = re.compile(r"\s+")
TOLERANCE = 0.5


@dataclass
class TriadLayout:
    tu_x0: float
    xp_x0: float
    eq_x0: float
    eq_caps: Dict[Tuple[int, int], float]


def _looks_like_label_token(text: str | None) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.endswith(("#",) + UNICODE_COLONS)


def _safe_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except Exception:
        return None


def _safe_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _clean_joined_text(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    joined = " ".join(t.replace("\u00ae", "") for t in tokens)
    joined = SPACE_RE.sub(" ", joined).strip()
    if not joined:
        return ""
    return clean_value(joined)


def _load_layout(debug_path: Path) -> TriadLayout:
    header_by_line: Dict[Tuple[int, int], Dict[str, float]] = {}
    tokens_by_line: Dict[Tuple[int, int], List[Tuple[float, str]]] = defaultdict(list)

    with debug_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            page = _safe_int(row.get("page"))
            line = _safe_int(row.get("line"))
            x0 = _safe_float(row.get("x0"))
            text = row.get("text") or ""
            if page is None or line is None or x0 is None:
                continue
            key = (page, line)
            tokens_by_line[key].append((x0, text))
            normalized = text.replace("\u00ae", "").strip().lower()
            if normalized in {"transunion", "experian", "equifax"}:
                header_by_line.setdefault(key, {})[normalized] = x0

    layout_positions: Optional[Dict[str, float]] = None
    for key in sorted(header_by_line.keys()):
        positions = header_by_line[key]
        if len(positions) == 3:
            layout_positions = positions
            break
    if not layout_positions:
        raise ValueError("unable to locate triad header tokens in debug TSV")

    tu_x0 = float(layout_positions["transunion"])
    xp_x0 = float(layout_positions["experian"])
    eq_x0 = float(layout_positions["equifax"])

    eq_caps: Dict[Tuple[int, int], float] = {}
    for key, tokens in tokens_by_line.items():
        tokens.sort(key=lambda item: item[0])
        cap: Optional[float] = None
        for x0, text in tokens:
            if not isinstance(text, str):
                continue
            if not _looks_like_label_token(text):
                continue
            if x0 <= eq_x0:
                continue
            if cap is None or x0 < cap:
                cap = x0
        if cap is not None:
            if cap < eq_x0:
                cap = eq_x0
            eq_caps[key] = cap

    return TriadLayout(tu_x0=tu_x0, xp_x0=xp_x0, eq_x0=eq_x0, eq_caps=eq_caps)


def _classify_token(
    x0: float,
    text: str,
    layout: TriadLayout,
    eq_cap: Optional[float],
) -> Optional[str]:
    if _looks_like_label_token(text):
        return None

    if x0 < layout.tu_x0 - TOLERANCE:
        return None
    if x0 < layout.xp_x0 - TOLERANCE:
        return "transunion"
    if x0 < layout.eq_x0 - TOLERANCE:
        return "experian"

    cap = eq_cap if eq_cap is not None else float("inf")
    if cap < layout.eq_x0:
        cap = layout.eq_x0
    if x0 < cap - TOLERANCE:
        return "equifax"
    return None


def _load_bureaus(bureaus_path: Path) -> Dict[str, Dict[str, str]]:
    data = json.loads(bureaus_path.read_text(encoding="utf-8"))
    bureaus: Dict[str, Dict[str, str]] = {}
    for bureau in ("transunion", "experian", "equifax"):
        value = data.get(bureau)
        if isinstance(value, Mapping):
            bureaus[bureau] = {str(k): clean_value(v) for k, v in value.items()}
        else:
            bureaus[bureau] = {}
    return bureaus


def _gather_row_tokens(
    trace_path: Path,
    layout: TriadLayout,
) -> Dict[str, Dict[str, List[Tuple[int, int, float, str]]]]:
    row_tokens: Dict[str, Dict[str, List[Tuple[int, int, float, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[Tuple[int, int, str, str, str]] = set()

    with trace_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            phase = (row.get("phase") or "").strip().lower()
            if phase not in {"labeled", "cont"}:
                continue
            label_key = (row.get("label_key") or "").strip()
            if not label_key:
                continue
            page = _safe_int(row.get("page"))
            line = _safe_int(row.get("line"))
            if page is None or line is None:
                continue
            x0 = _safe_float(row.get("x0"))
            if x0 is None:
                continue
            token_idx = (row.get("token") or "").strip()
            text_raw = row.get("text") or ""
            dedup_key = (page, line, token_idx, label_key, text_raw.strip())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            eq_cap = layout.eq_caps.get((page, line))
            bureau = _classify_token(x0, text_raw, layout, eq_cap)
            if bureau is None:
                continue
            row_tokens[label_key][bureau].append((page, line, x0, text_raw))

    return row_tokens


def _prepare_expected_values(
    row_tokens: Mapping[str, Mapping[str, List[Tuple[int, int, float, str]]]]
) -> Dict[str, Dict[str, Tuple[str, int]]]:
    expected: Dict[str, Dict[str, Tuple[str, int]]] = {}
    for label_key, bureau_map in row_tokens.items():
        expected[label_key] = {}
        for bureau, tokens in bureau_map.items():
            tokens_sorted = sorted(tokens, key=lambda item: (item[0], item[1], item[2]))
            texts = [text for _, _, _, text in tokens_sorted]
            cleaned = _clean_joined_text(texts)
            expected[label_key][bureau] = (cleaned, len(tokens_sorted))
    return expected


def scan_account(
    sid: str,
    account_index: int,
    trace_path: Path,
    debug_path: Path,
    bureaus_path: Path,
) -> Tuple[int, List[str]]:
    layout = _load_layout(debug_path)
    row_tokens = _gather_row_tokens(trace_path, layout)
    expected = _prepare_expected_values(row_tokens)
    bureaus = _load_bureaus(bureaus_path)

    mismatches: List[str] = []
    mismatch_count = 0

    for label_key, bureau_map in expected.items():
        for bureau, (value, token_count) in bureau_map.items():
            if token_count <= 0:
                continue
            actual = bureaus.get(bureau, {}).get(label_key, "")
            if actual != value:
                mismatch_count += 1
                mismatches.append(
                    f"    - {label_key}.{bureau}: expected {value!r} but found {actual!r}"
                )

    return mismatch_count, mismatches


def _load_stagea_accounts(accounts_dir: Path) -> List[Mapping[str, object]]:
    accounts: List[Mapping[str, object]] = []
    if accounts_dir.is_dir():
        for json_path in sorted(accounts_dir.glob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, Mapping):
                if "accounts" in payload and isinstance(payload.get("accounts"), list):
                    for item in payload["accounts"]:  # type: ignore[index]
                        if isinstance(item, Mapping):
                            accounts.append(item)
                else:
                    accounts.append(payload)
    return accounts


def _load_accounts_for_run(run_dir: Path) -> List[Mapping[str, object]]:
    traces_dir = run_dir / "traces"
    per_account_dir = traces_dir / "accounts_from_full"
    if per_account_dir.exists():
        accounts = _load_stagea_accounts(per_account_dir)
        if accounts:
            return accounts
    accounts_json = traces_dir / "accounts_table" / "accounts_from_full.json"
    if accounts_json.exists():
        try:
            data = json.loads(accounts_json.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(data, Mapping):
            items = data.get("accounts")
            if isinstance(items, list):
                return [item for item in items if isinstance(item, Mapping)]
    return []


def scan_run(run_dir: Path) -> Tuple[int, int]:
    sid = run_dir.name
    accounts = _load_accounts_for_run(run_dir)
    if not accounts:
        print(f"[{sid}] no Stage-A accounts found; skipping")
        return 0, 0

    traces_dir = run_dir / "traces" / "accounts_table" / "per_account_tsv"
    cases_dir = run_dir / "cases" / "accounts"

    total_accounts = 0
    total_mismatches = 0

    for account in accounts:
        account_index = account.get("account_index")
        try:
            idx = int(account_index) if account_index is not None else None
        except Exception:
            idx = None
        if idx is None:
            continue
        trace_path = traces_dir / f"_trace_account_{idx}.csv"
        debug_path = traces_dir / f"_debug_account_{idx}.tsv"
        bureaus_path = cases_dir / str(idx) / "bureaus.json"

        if not trace_path.exists() or not debug_path.exists() or not bureaus_path.exists():
            missing: List[str] = []
            if not trace_path.exists():
                missing.append("trace")
            if not debug_path.exists():
                missing.append("debug")
            if not bureaus_path.exists():
                missing.append("bureaus")
            missing_str = ", ".join(missing)
            print(f"[{sid}] account {idx}: skipped (missing {missing_str})")
            continue

        total_accounts += 1
        mismatch_count, details = scan_account(
            sid, idx, trace_path, debug_path, bureaus_path
        )
        total_mismatches += mismatch_count
        if mismatch_count:
            print(f"[{sid}] account {idx}: {mismatch_count} mismatches")
            for line in details:
                print(line)
        else:
            print(f"[{sid}] account {idx}: 0 mismatches")

    return total_accounts, total_mismatches


def iter_run_dirs(runs_root: Path, sid_filter: Optional[str]) -> Iterable[Path]:
    for path in sorted(runs_root.iterdir()):
        if not path.is_dir():
            continue
        if sid_filter and path.name != sid_filter:
            continue
        yield path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan bureaus.json alignment for runs")
    ap.add_argument("--runs-root", default="runs", help="Root directory containing runs")
    ap.add_argument("--sid", help="Optional SID filter")
    args = ap.parse_args(argv)

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"runs root {runs_root} not found", file=sys.stderr)
        return 2

    grand_total_accounts = 0
    grand_total_mismatches = 0

    for run_dir in iter_run_dirs(runs_root, args.sid):
        accounts_processed, mismatches = scan_run(run_dir)
        grand_total_accounts += accounts_processed
        grand_total_mismatches += mismatches

    if grand_total_accounts == 0:
        print("0 mismatches")
        return 0

    if grand_total_mismatches:
        print(f"TOTAL mismatches: {grand_total_mismatches}")
        return 1

    print("0 mismatches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
