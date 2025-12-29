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
"""Debug CLI for interacting with run manifests.

This utility mirrors ``scripts/run_manifest.py`` but under a more explicit name
and with an extra ``print`` command that dumps the manifest JSON.  It operates
on the global run registry located under ``runs/<SID>/manifest.json``.
"""

import argparse
import json
from pathlib import Path

from backend.pipeline.runs import RunManifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run registry helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("get", help="Get an artifact path")
    # Support both positional and flagged usage for convenience
    g.add_argument("sid", nargs="?")
    g.add_argument("group", nargs="?")
    g.add_argument("key", nargs="?")
    g.add_argument("--sid", dest="sid_flag")
    g.add_argument("--group", dest="group_flag")
    g.add_argument("--key", dest="key_flag")

    sa = sub.add_parser("set-artifact", help="Set an artifact path")
    sa.add_argument("sid")
    sa.add_argument("group")
    sa.add_argument("key")
    sa.add_argument("value")

    sb = sub.add_parser("set-base-dir", help="Set a base directory")
    sb.add_argument("sid")
    sb.add_argument("label")
    sb.add_argument("path")

    pr = sub.add_parser("print", help="Print the full manifest as JSON")
    pr.add_argument("sid")

    args = ap.parse_args(argv)

    if args.cmd == "get":
        sid = args.sid_flag or args.sid
        group = args.group_flag or args.group
        key = args.key_flag or args.key
        if not (sid and group and key):
            raise SystemExit("Usage: runs_cli.py get <sid> <group> <key> or --sid --group --key")
        m = RunManifest.for_sid(sid)
        print(m.get(group, key))
    elif args.cmd == "set-artifact":
        m = RunManifest.for_sid(args.sid)
        m.set_artifact(args.group, args.key, Path(args.value))
    elif args.cmd == "set-base-dir":
        m = RunManifest.for_sid(args.sid)
        m.set_base_dir(args.label, Path(args.path))
    elif args.cmd == "print":
        m = RunManifest.for_sid(args.sid)
        print(json.dumps(m.data, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
