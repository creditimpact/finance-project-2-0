import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(p: Path) -> Dict[str, Any] | List[Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _token_key(tok: Dict[str, Any], page: int) -> Tuple[int, float, float]:
    try:
        line = float(tok.get("line"))
    except Exception:
        line = float(tok.get("y0", 0.0))
    try:
        x0 = float(tok.get("x0", 0.0))
    except Exception:
        x0 = 0.0
    return page, line, x0


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump full report tokens to TSV")
    ap.add_argument("--sid", required=True, help="Session ID under traces/blocks")
    ap.add_argument(
        "--root", default=str(Path.cwd()), help="Project root (defaults to CWD)"
    )
    args = ap.parse_args()

    root = Path(args.root)
    base = root / "traces" / "blocks" / args.sid
    layout_path = base / "layout_snapshot.json"
    out_dir = base / "accounts_table"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "_debug_full.tsv"

    layout = _load_json(layout_path) or {}
    pages = list(layout.get("pages") or [])

    rows: List[Tuple[int, Any, float, float, float, float, str]] = []
    for idx, page in enumerate(pages, start=1):
        tokens = list(page.get("tokens") or [])
        for tok in tokens:
            rows.append(
                (
                    idx,
                    tok.get("line"),
                    float(tok.get("y0", 0.0)),
                    float(tok.get("y1", 0.0)),
                    float(tok.get("x0", 0.0)),
                    float(tok.get("x1", 0.0)),
                    (tok.get("text") or "").replace("\t", " "),
                )
            )

    rows.sort(key=lambda r: _token_key({"line": r[1], "x0": r[4], "y0": r[2]}, r[0]))

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("page\tline\ty0\ty1\tx0\tx1\ttext\n")
        for pg, ln, y0, y1, x0, x1, text in rows:
            ln_str = "" if ln is None else str(ln)
            fh.write(f"{pg}\t{ln_str}\t{y0}\t{y1}\t{x0}\t{x1}\t{text}\n")

    print(f"Wrote {len(rows)} tokens to {out_path}")


if __name__ == "__main__":
    main()
