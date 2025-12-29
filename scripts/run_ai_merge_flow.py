"""Run the end-to-end merge V2 AI flow for a single SID."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

try:  # pragma: no cover - convenience bootstrap when executed directly
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback to ensure repo modules are importable
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from backend.core.ai.paths import (
    MergePaths,
    ensure_merge_paths,
    merge_paths_from_any,
    pair_pack_filename,
    pair_pack_path,
)
from backend.core.io.tags import read_tags
from backend.core.logic.report_analysis import ai_sender
from backend.core.logic.report_analysis.ai_packs import build_merge_ai_packs
from backend.pipeline.runs import RunManifest

from scripts.score_bureau_pairs import (
    AUTO_DECISIONS,
    ScoreComputationResult,
    score_accounts,
)


@dataclass(frozen=True)
class PackArtifact:
    """Represents an AI pack payload and where it is stored."""

    a_idx: int
    b_idx: int
    filename: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class PackBuildResult:
    """Container for the pack build stage."""

    sid: str
    runs_root: Path
    directory: Path
    index_path: Path
    log_path: Path
    items: List[PackArtifact]


@dataclass(frozen=True)
class SendStats:
    total: int
    successes: int
    failures: int


@dataclass(frozen=True)
class AiPartnerOutcome:
    partner: int
    kind: str
    decision: str
    reason: str


def _write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(serialized + "\n", encoding="utf-8")


def _resolve_merge_paths(
    runs_root: Path,
    sid: str,
    override: Optional[str],
) -> MergePaths:
    canonical = ensure_merge_paths(runs_root, sid, create=True)

    if not override:
        return canonical

    return merge_paths_from_any(Path(override), create=True)


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _log_factory(path: Path, sid: str, pair: Mapping[str, int], file_name: str):
    def _log(event: str, payload: Mapping[str, object] | None = None) -> None:
        extras: Dict[str, object] = {
            "sid": sid,
            "pair": {"a": pair["a"], "b": pair["b"]},
            "file": file_name,
        }
        if payload:
            extras.update(payload)
        serialized = json.dumps(extras, ensure_ascii=False, sort_keys=True)
        line = f"{ai_sender.isoformat_timestamp()} AI_ADJUDICATOR_{event} {serialized}\n"
        _append_log(path, line)

    return _log


def build_packs_for_sid(
    sid: str,
    runs_root: Path | str,
    *,
    out_dir: Optional[str] = None,
    only_merge_best: bool = True,
    max_lines_per_side: int = 20,
) -> PackBuildResult:
    """Build packs, persist them to disk, and update the manifest."""

    sid_str = str(sid)
    runs_root_path = Path(runs_root)
    merge_paths = _resolve_merge_paths(runs_root_path, sid_str, out_dir)
    packs_dir = merge_paths.packs_dir
    base_dir = merge_paths.base
    index_path = merge_paths.index_file
    logs_path = merge_paths.log_file

    packs = build_merge_ai_packs(
        sid_str,
        runs_root_path,
        only_merge_best=only_merge_best,
        max_lines_per_side=max_lines_per_side,
    )

    items: List[PackArtifact] = []
    for pack in packs:
        pair = dict(pack.get("pair") or {})
        if "a" not in pair or "b" not in pair:
            raise ValueError(f"Pack missing pair indices: {pack}")
        try:
            a_idx = int(pair["a"])
            b_idx = int(pair["b"])
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("Pack indices must be integers") from exc
        filename = pair_pack_filename(a_idx, b_idx)
        _write_json_file(pair_pack_path(merge_paths, a_idx, b_idx), pack)
        items.append(PackArtifact(a_idx=a_idx, b_idx=b_idx, filename=filename, payload=pack))

    index_payload = [
        {"a": item.a_idx, "b": item.b_idx, "file": item.filename} for item in items
    ]
    _write_json_file(index_path, index_payload)

    manifest = RunManifest.for_sid(sid_str)
    manifest.update_ai_packs(
        dir=base_dir,
        index=index_path,
        logs=logs_path,
        pairs=len(items),
    )

    return PackBuildResult(
        sid=sid_str,
        runs_root=runs_root_path,
        directory=packs_dir,
        index_path=index_path,
        log_path=logs_path,
        items=items,
    )


def send_packs(
    sid: str,
    build_result: PackBuildResult,
    config: ai_sender.AISenderConfig,
) -> SendStats:
    total = 0
    successes = 0
    failures = 0

    for item in build_result.items:
        total += 1
        pair = {"a": item.a_idx, "b": item.b_idx}
        log = _log_factory(build_result.log_path, sid, pair, item.filename)
        log("PACK_START", {})

        outcome = ai_sender.process_pack(item.payload, config, log=log)
        timestamp = ai_sender.isoformat_timestamp()

        if outcome.success and outcome.decision and outcome.reason:
            ai_sender.write_decision_tags(
                build_result.runs_root,
                sid,
                item.a_idx,
                item.b_idx,
                outcome.decision,
                outcome.reason,
                timestamp,
                outcome.flags,
            )
            log(
                "PACK_SUCCESS",
                {"decision": outcome.decision, "reason": outcome.reason, "flags": outcome.flags},
            )
            successes += 1
        else:
            ai_sender.write_error_tags(
                build_result.runs_root,
                sid,
                item.a_idx,
                item.b_idx,
                outcome.error_kind or "Error",
                outcome.error_message or "",
                timestamp,
            )
            log(
                "PACK_FAILURE",
                {"error_kind": outcome.error_kind or "Error"},
            )
            failures += 1

    return SendStats(total=total, successes=successes, failures=failures)


def _load_ai_outcomes(
    sid: str,
    runs_root: Path,
) -> Dict[int, Dict[int, AiPartnerOutcome]]:
    base = runs_root / sid / "cases" / "accounts"
    outcomes: Dict[int, Dict[int, AiPartnerOutcome]] = {}

    if not base.exists():
        return outcomes

    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        try:
            account_idx = int(entry.name)
        except ValueError:
            continue

        per_partner: Dict[int, AiPartnerOutcome] = {}
        for tag in read_tags(entry / "tags.json"):
            if tag.get("source") != "ai_adjudicator":
                continue
            try:
                partner_idx = int(tag.get("with"))
            except (TypeError, ValueError):
                continue
            kind = str(tag.get("kind"))
            if kind == "ai_decision":
                decision = str(tag.get("decision", ""))
                reason = str(tag.get("reason", ""))
                per_partner[partner_idx] = AiPartnerOutcome(
                    partner=partner_idx,
                    kind="decision",
                    decision=decision,
                    reason=reason,
                )
            elif kind == "ai_error" and partner_idx not in per_partner:
                error_kind = str(tag.get("error_kind", "Error"))
                message = str(tag.get("message", ""))
                per_partner[partner_idx] = AiPartnerOutcome(
                    partner=partner_idx,
                    kind="error",
                    decision=error_kind,
                    reason=message,
                )

        if per_partner:
            outcomes[account_idx] = per_partner

    return outcomes


def _find_pair_result(
    scores_by_idx: Mapping[int, Mapping[int, Mapping[str, object]]],
    left: int,
    right: int,
) -> Optional[Mapping[str, object]]:
    left_map = scores_by_idx.get(left, {})
    result = left_map.get(right)
    if result:
        return result
    return scores_by_idx.get(right, {}).get(left)


def _extract_best_entry(
    computation: ScoreComputationResult,
    idx: int,
) -> Optional[Dict[str, object]]:
    best_info = computation.best_by_idx.get(idx)
    if not isinstance(best_info, Mapping):
        return None

    partner = best_info.get("partner_index")
    if not isinstance(partner, int):
        return None

    pair_result = _find_pair_result(computation.scores_by_idx, idx, partner)
    total = 0
    decision = "different"

    if isinstance(pair_result, Mapping):
        try:
            total = int(pair_result.get("total", 0) or 0)
        except (TypeError, ValueError):
            total = 0
        decision = str(pair_result.get("decision", "different"))
    else:
        result_payload = best_info.get("result")
        if isinstance(result_payload, Mapping):
            try:
                total = int(result_payload.get("total", 0) or 0)
            except (TypeError, ValueError):
                total = 0
            decision = str(result_payload.get("decision", "different"))

    return {"with": partner, "total": total, "decision": decision}


def _select_ai_outcome(
    idx: int,
    best_entry: Optional[Mapping[str, object]],
    outcomes: Mapping[int, Mapping[int, AiPartnerOutcome]],
) -> Optional[AiPartnerOutcome]:
    per_partner = outcomes.get(idx)
    if not per_partner:
        return None

    priority: List[int] = []
    if best_entry and isinstance(best_entry.get("with"), int):
        priority.append(int(best_entry["with"]))
    priority.extend(sorted(partner for partner in per_partner.keys() if partner not in priority))

    for partner in priority:
        outcome = per_partner.get(partner)
        if outcome:
            return outcome
    return None


def prepare_summary_rows(
    computation: ScoreComputationResult,
    outcomes: Mapping[int, Mapping[int, AiPartnerOutcome]],
) -> List[Dict[str, object]]:
    include_indices = set()

    for idx in computation.indices:
        best_entry = _extract_best_entry(computation, idx)
        if not best_entry:
            continue
        decision = str(best_entry.get("decision", "")).lower()
        if decision in AUTO_DECISIONS:
            include_indices.add(idx)

    include_indices.update(outcomes.keys())

    rows: List[Dict[str, object]] = []
    for idx in sorted(include_indices):
        best_entry = _extract_best_entry(computation, idx)
        ai_outcome = _select_ai_outcome(idx, best_entry, outcomes)

        best_with = best_entry.get("with") if best_entry else None
        pre_total = best_entry.get("total") if best_entry else None
        pre_decision = best_entry.get("decision") if best_entry else None

        ai_with: Optional[int]
        ai_decision: Optional[str]
        ai_reason: Optional[str]

        if ai_outcome:
            ai_with = ai_outcome.partner
            if ai_outcome.kind == "error":
                ai_decision = f"error:{ai_outcome.decision}"
            else:
                ai_decision = ai_outcome.decision
            ai_reason = ai_outcome.reason
        else:
            ai_with = None
            ai_decision = None
            ai_reason = None

        rows.append(
            {
                "idx": idx,
                "best_with": best_with,
                "pre_total": pre_total,
                "pre_decision": pre_decision,
                "ai_with": ai_with,
                "ai_decision": ai_decision,
                "ai_reason": ai_reason,
            }
        )

    return rows


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value)
    return text if text else "-"


def print_summary(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        print("[SUMMARY] no accounts to display")
        return

    header = (
        "idx",
        "best_with",
        "pre_total",
        "pre_decision",
        "ai_with",
        "ai_decision",
        "ai_reason",
    )
    print(" | ".join(header))
    print("-" * 96)
    for row in rows:
        print(
            " | ".join(
                [
                    _fmt(row.get("idx")),
                    _fmt(row.get("best_with")),
                    _fmt(row.get("pre_total")),
                    _fmt(row.get("pre_decision")),
                    _fmt(row.get("ai_with")),
                    _fmt(row.get("ai_decision")),
                    _fmt(row.get("ai_reason")),
                ]
            )
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True, help="Session identifier")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory containing runs/<sid> outputs",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for the AI packs directory",
    )
    parser.add_argument(
        "--max-lines-per-side",
        type=int,
        default=20,
        help="Maximum number of context lines per account",
    )
    parser.add_argument(
        "--only-merge-best",
        dest="only_merge_best",
        action="store_true",
        help="Include only merge_best pairs when building packs (default)",
    )
    parser.add_argument(
        "--include-all-pairs",
        dest="only_merge_best",
        action="store_false",
        help="Include all AI candidate pairs regardless of merge_best",
    )
    parser.set_defaults(only_merge_best=True)

    args = parser.parse_args(argv)

    sid = str(args.sid)
    runs_root = Path(args.runs_root)

    computation = score_accounts(
        sid,
        runs_root=runs_root,
        only_ai_rows=False,
        write_tags=True,
    )

    if not computation.indices:
        print(f"[SCORE] no accounts found for SID {sid!r} under {runs_root}")
        return

    ai_pairs = [
        row for row in computation.rows if str(row.get("decision", "")).lower() in AUTO_DECISIONS
    ]
    print(
        f"[SCORE] evaluated {len(computation.rows)} pairs; {len(ai_pairs)} flagged for AI/auto decisions"
    )

    build_result = build_packs_for_sid(
        sid,
        runs_root,
        out_dir=args.out_dir,
        only_merge_best=bool(args.only_merge_best),
        max_lines_per_side=int(args.max_lines_per_side),
    )

    print(
        f"[PACK] wrote {len(build_result.items)} packs to {build_result.directory}"
    )

    if build_result.items and ai_sender.is_enabled():
        config = ai_sender.load_config_from_env()
        stats = send_packs(sid, build_result, config)
        print(
            "[AI] adjudicated {total} packs ({successes} success, {failures} errors)".format(
                total=stats.total, successes=stats.successes, failures=stats.failures
            )
        )
    elif build_result.items:
        print("[AI] adjudicator disabled; skipping send step")
    else:
        print("[AI] no packs to adjudicate")

    outcomes = _load_ai_outcomes(sid, runs_root)
    summary_rows = prepare_summary_rows(computation, outcomes)
    print_summary(summary_rows)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

