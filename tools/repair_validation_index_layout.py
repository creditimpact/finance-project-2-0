import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

backend_module = sys.modules.get("backend")
backend_path = str(REPO_ROOT / "backend")
if backend_module is None:
    backend_module = ModuleType("backend")
    backend_module.__path__ = [backend_path]
    sys.modules["backend"] = backend_module
else:
    existing_backend_path = list(getattr(backend_module, "__path__", []))
    if backend_path not in existing_backend_path:
        existing_backend_path.append(backend_path)
    backend_module.__path__ = existing_backend_path

validation_pkg = sys.modules.get("backend.validation")
validation_path = str(REPO_ROOT / "backend" / "validation")
if validation_pkg is None:
    validation_pkg = ModuleType("backend.validation")
    validation_pkg.__path__ = [validation_path]
    sys.modules["backend.validation"] = validation_pkg
else:
    existing_validation_path = list(getattr(validation_pkg, "__path__", []))
    if validation_path not in existing_validation_path:
        existing_validation_path.append(validation_path)
    validation_pkg.__path__ = existing_validation_path

manifest_module = importlib.import_module("backend.validation.manifest")
paths_module = importlib.import_module("backend.core.ai.paths")

rewrite_index_to_canonical_layout = manifest_module.rewrite_index_to_canonical_layout
validation_index_path = paths_module.validation_index_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite validation index.json files so they reference the ai_packs/"
            "validation directory layout"
        )
    )
    parser.add_argument(
        "--sid",
        help="Rewrite only the manifest for the provided SID",
    )
    parser.add_argument(
        "--root",
        help="Runs root directory to scan (defaults to RUNS_ROOT env or ./runs)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _iter_index_paths(root: Path) -> Iterable[Path]:
    pattern = ("ai_packs", "validation", "index.json")
    for candidate in root.rglob("index.json"):
        parts = candidate.parts
        if len(parts) >= 3 and parts[-3:] == pattern:
            yield candidate


def _resolve_sid_index_path(sid: str, runs_root: Path | None) -> Path:
    index_path = validation_index_path(sid, runs_root=runs_root, create=False)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Validation index for SID {sid!r} not found at {index_path}"
        )
    return index_path


def migrate(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.sid:
        runs_root_path = Path(args.root).resolve() if args.root else None
        try:
            index_paths = [_resolve_sid_index_path(args.sid, runs_root=runs_root_path)]
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 2
    else:
        runs_root_path = Path(args.root).resolve() if args.root else Path("runs")
        if not runs_root_path.exists():
            print(f"Runs root {runs_root_path} does not exist", file=sys.stderr)
            return 2
        index_paths = sorted(_iter_index_paths(runs_root_path))

    if not index_paths:
        print("No validation index files found.")
        return 0

    rewritten = 0
    skipped = 0

    for index_path in index_paths:
        print(f"Processing {index_path}")
        try:
            _, changed = rewrite_index_to_canonical_layout(
                index_path,
                runs_root=runs_root_path,
                stream=sys.stdout,
            )
        except FileNotFoundError:
            print("  Skipped: index file missing", file=sys.stderr)
            skipped += 1
            continue
        except Exception as exc:
            print(f"  Failed to rewrite manifest: {exc}", file=sys.stderr)
            skipped += 1
            continue

        if changed:
            rewritten += 1
            print("  Manifest rewritten.")
        else:
            print("  Already canonical.")

    total = len(index_paths)
    unchanged = total - rewritten - skipped
    print(
        f"rewritten: {rewritten}, unchanged: {unchanged}, skipped: {skipped}"
    )

    return 0 if skipped == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(migrate())
