try:  # pragma: no cover - import shim
    import scripts._bootstrap  # KEEP FIRST
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parent.parent
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    import scripts._bootstrap  # type: ignore  # KEEP FIRST

#!/usr/bin/env python3
"""CLI helpers for interacting with run manifests.

Allows ops to set or get artifact paths in the global run registry.
"""

import argparse
from pathlib import Path

from backend.pipeline.runs import RunManifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run manifest helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("get", help="Get an artifact path")
    g.add_argument("sid")
    g.add_argument("group")
    g.add_argument("key")

    sa = sub.add_parser("set-artifact", help="Set an artifact path")
    sa.add_argument("sid")
    sa.add_argument("group")
    sa.add_argument("key")
    sa.add_argument("value")

    sb = sub.add_parser("set-base-dir", help="Set a base directory")
    sb.add_argument("sid")
    sb.add_argument("label")
    sb.add_argument("path")

    args = ap.parse_args(argv)

    if args.cmd == "get":
        m = RunManifest.for_sid(args.sid)
        print(m.get(args.group, args.key))
    elif args.cmd == "set-artifact":
        m = RunManifest.for_sid(args.sid)
        m.set_artifact(args.group, args.key, Path(args.value))
    elif args.cmd == "set-base-dir":
        m = RunManifest.for_sid(args.sid)
        m.set_base_dir(args.label, Path(args.path))


if __name__ == "__main__":  # pragma: no cover
    main()
