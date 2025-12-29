try:  # pragma: no cover - import shim
    import scripts._bootstrap  # KEEP FIRST
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parent.parent
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    import scripts._bootstrap  # type: ignore  # KEEP FIRST

"""Helper CLI to inspect account tags for a run."""

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from backend.core.io.tags import read_tags


def _iter_account_indices(accounts_root: Path) -> Iterable[Tuple[int, Path]]:
    for entry in accounts_root.iterdir():
        if not entry.is_dir():
            continue
        try:
            idx = int(entry.name)
        except ValueError:
            continue
        yield idx, entry / "tags.json"


def _format_issue(tags: List[dict]) -> str:
    for entry in tags:
        if entry.get("kind") == "issue":
            value = entry.get("type")
            if isinstance(value, str) and value:
                return value
    return "-"


def _format_best(tags: List[dict]) -> str:
    for entry in tags:
        if entry.get("kind") != "merge_best":
            continue
        partner = entry.get("with")
        decision = entry.get("decision")
        score = entry.get("score_total")
        if score is None:
            score = entry.get("total")
        return f"{_format_field(partner)}:{_format_field(decision)}:{_format_field(score)}"
    return "-"


def _format_merge_pairs(tags: List[dict]) -> str:
    pairs: List[str] = []
    for entry in tags:
        if entry.get("kind") != "merge_pair":
            continue
        partner = entry.get("with")
        decision = entry.get("decision")
        total = entry.get("total")
        pairs.append(
            f"{_format_field(partner)}:{_format_field(decision)}:{_format_field(total)}"
        )
    return ",".join(pairs)


def _format_field(value: object) -> str:
    if value is None:
        return "-"
    return str(value)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Show tags for all accounts in a run")
    parser.add_argument("--sid", required=True, help="Run SID to inspect")

    args = parser.parse_args(argv)

    runs_root = Path("runs")
    accounts_root = runs_root / args.sid / "cases" / "accounts"
    if not accounts_root.exists():
        raise SystemExit(f"No accounts directory found for SID: {args.sid}")

    rows: List[Tuple[int, str, str, str]] = []
    for idx, tag_path in sorted(_iter_account_indices(accounts_root), key=lambda item: item[0]):
        tags = read_tags(tag_path)
        issue = _format_issue(tags)
        best = _format_best(tags)
        merge_pairs = _format_merge_pairs(tags)
        rows.append((idx, issue, best, merge_pairs))

    if not rows:
        print(f"No tags found for SID {args.sid}")
        return

    print(f"Tags for SID {args.sid}:")
    for idx, issue, best, merge_pairs in rows:
        if merge_pairs:
            pairs_repr = merge_pairs
        else:
            pairs_repr = ""
        print(
            f"[{idx}] issue={issue} best={best} merge_pairs=[{pairs_repr}]"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
