try:  # pragma: no cover - import shim
    import scripts._bootstrap  # KEEP FIRST
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parent.parent
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    import scripts._bootstrap  # type: ignore  # KEEP FIRST

import argparse
import shutil
from pathlib import Path

from backend.pipeline.runs import RunManifest


def migrate_one(sid: str, move: bool = False) -> None:
    m = RunManifest.for_sid(sid)
    # legacy location:
    legacy = Path("traces") / "blocks" / sid / "accounts_table"
    if not legacy.exists():
        print(f"[SKIP] no legacy traces for {sid}")
        return

    # canonical locations:
    traces_dir = m.ensure_run_subdir("traces_dir", "traces")
    acct_dir = (traces_dir / "accounts_table").resolve()
    acct_dir.mkdir(parents=True, exist_ok=True)

    # files/dirs we care about:
    files = [
        "accounts_from_full.json",
        "general_info_from_full.json",
        "_debug_full.tsv",
    ]
    for name in files:
        src = legacy / name
        if src.exists():
            dst = acct_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.move if move else shutil.copy2)(src, dst)

    per_acc_src = legacy / "per_account_tsv"
    if per_acc_src.exists():
        per_acc_dst = acct_dir / "per_account_tsv"
        if move:
            shutil.move(str(per_acc_src), str(per_acc_dst))
        else:
            shutil.copytree(per_acc_src, per_acc_dst, dirs_exist_ok=True)

    # register in manifest:
    m.set_artifact(
        "traces.accounts_table", "accounts_json", acct_dir / "accounts_from_full.json"
    )
    m.set_artifact(
        "traces.accounts_table",
        "general_json",
        acct_dir / "general_info_from_full.json",
    )
    m.set_artifact(
        "traces.accounts_table", "debug_full_tsv", acct_dir / "_debug_full.tsv"
    )
    m.set_artifact(
        "traces.accounts_table", "per_account_tsv_dir", acct_dir / "per_account_tsv"
    )

    print(f"[OK] migrated traces for {sid} -> {acct_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Migrate legacy traces/blocks/<SID>/accounts_table to runs/<SID>/traces/accounts_table",
    )
    ap.add_argument("--sid", required=True, help="Run SID to migrate")
    ap.add_argument("--move", action="store_true", help="Move instead of copy")
    args = ap.parse_args()
    migrate_one(args.sid, move=bool(args.move))


if __name__ == "__main__":
    main()

