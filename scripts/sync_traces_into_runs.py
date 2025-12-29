import scripts._bootstrap  # KEEP FIRST

import shutil
from pathlib import Path
from backend.pipeline.runs import RunManifest


def sync_one(sid: str, move: bool = False) -> None:
    m = RunManifest.for_sid(sid)
    legacy = Path("traces") / "blocks" / sid / "accounts_table"
    if not legacy.exists():
        print(f"[SKIP] no legacy traces for {sid}")
        return

    traces_dir = m.ensure_run_subdir("traces_dir", "traces")
    acct_dir = (traces_dir / "accounts_table").resolve()
    acct_dir.mkdir(parents=True, exist_ok=True)

    # files to pull
    for name in [
        "accounts_from_full.json",
        "general_info_from_full.json",
        "_debug_full.tsv",
    ]:
        src = legacy / name
        if src.exists():
            dst = acct_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.move if move else shutil.copy2)(src, dst)

    per_src = legacy / "per_account_tsv"
    if per_src.exists():
        per_dst = acct_dir / "per_account_tsv"
        if move:
            shutil.move(str(per_src), str(per_dst))
        else:
            shutil.copytree(per_src, per_dst, dirs_exist_ok=True)

    # register
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

    print(f"[OK] synced traces for {sid} -> {acct_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True)
    ap.add_argument("--move", action="store_true")
    args = ap.parse_args()
    sync_one(args.sid, move=args.move)

