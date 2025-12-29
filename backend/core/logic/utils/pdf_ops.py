# logic/utils/pdf_ops.py

import os
from pathlib import Path
from typing import Any, Iterable, List, Tuple




def convert_txts_to_pdfs(folder: Path):
    """
    Converts .txt files in the given folder to styled PDFs with Unicode support.

    When DISABLE_PDF_RENDER is set to true/1/yes, this function becomes a no-op
    so tests/integration won’t attempt to use PDF libraries.
    """
    # --- Guardrail for test mode / CI ---
    if os.getenv("DISABLE_PDF_RENDER", "").lower() in ("1", "true", "yes"):
        print(
            "[INFO] PDF rendering disabled via DISABLE_PDF_RENDER – skipping conversion."
        )
        return

    # Import locally so tests that skip PDF don't even load the lib
    from fpdf import FPDF

    txt_files = list(Path(folder).glob("*.txt"))
    output_folder = Path(folder) / "converted"
    output_folder.mkdir(exist_ok=True)

    for txt_path in txt_files:
        pdf = FPDF()
        pdf.add_page()
        # If you have a Unicode TTF, you can register it here. For now, use core font.
        # pdf.add_font("DejaVu", "", fonts_path("DejaVuSans.ttf"), uni=True)
        # pdf.set_font("DejaVu", size=12)
        pdf.set_font("Helvetica", size=12)

        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        pdf.multi_cell(0, 8, content)

        out_path = output_folder / (txt_path.stem + ".pdf")
        pdf.output(str(out_path))
        print(f"[INFO] Created PDF: {out_path}")


# --- Supporting docs utilities (used by goodwill_prompting / gpt_prompting etc.) ---


def _iter_candidate_paths(sources: Any) -> Iterable[Path]:
    """
    Normalize various input shapes into an iterator of Path objects.
    Empty/whitespace strings should yield no paths.
    """
    if sources is None:
        return []

    # Empty or whitespace-only string -> no paths
    if isinstance(sources, str) and not sources.strip():
        return []

    # Single Path or non-empty string
    if isinstance(sources, (str, Path)):
        p = Path(sources)
        # Avoid treating "" as "."
        if str(p).strip() == "":
            return []
        return [p]

    # Dict-like
    if hasattr(sources, "items"):
        vals = []
        for _, v in sources.items():
            if isinstance(v, str) and not v.strip():
                continue
            if v:
                vals.append(Path(v))
        return vals

    # Dataclass / object with __dict__
    if hasattr(sources, "__dict__"):
        try:
            d = {k: getattr(sources, k) for k in dir(sources) if not k.startswith("_")}
            if not d and hasattr(sources, "__dict__"):
                d = dict(sources.__dict__)
            out = []
            for v in d.values():
                if isinstance(v, str) and not v.strip():
                    continue
                if isinstance(v, (str, Path)):
                    out.append(Path(v))
            return out
        except Exception:
            pass

    # Iterable
    if isinstance(sources, Iterable):
        out: List[Path] = []
        for item in sources:
            if isinstance(item, str) and not item.strip():
                continue
            if item:
                out.append(Path(item))
        return out

    return []


def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        # Last resort
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            return f"[WARN] Unable to read file: {p.name} ({e})"


def gather_supporting_docs(sources: Any) -> Tuple[List[str], List[str], List[str]]:
    """
    Return (texts, names, paths) collected from supporting documents.

    - For .txt files: read content.
    - For folders: read *.txt within.
    - For PDFs/other types during tests: return a safe placeholder line.
    - For missing inputs: return ([], [], []).

    This keeps tests stable even when PDF parsing/rendering is disabled.
    """
    texts: List[str] = []
    names: List[str] = []
    paths_out: List[str] = []

    paths = list(_iter_candidate_paths(sources))

    expanded: List[Path] = []
    for p in paths:
        if not p:
            continue
        p = Path(p)
        if p.is_dir():
            expanded.extend(p.glob("*.txt"))
        else:
            expanded.append(p)

    for p in expanded:
        suffix = p.suffix.lower()
        if suffix == ".txt" and p.exists():
            texts.append(_read_text_file(p))
            names.append(p.name)
            paths_out.append(str(p))
        elif suffix == ".pdf" and p.exists():
            # We don't parse PDFs here (tests run with DISABLE_PDF_RENDER)
            texts.append(f"[PDF attached: {p.name}]")
            names.append(p.name)
            paths_out.append(str(p))
        else:
            if p.exists():
                texts.append(f"[Attachment: {p.name}]")
                names.append(p.name)
                paths_out.append(str(p))
            else:
                # Missing file — record a placeholder
                texts.append(f"[Missing attachment: {p}]")
                names.append(p.name if p.name else str(p))
                paths_out.append(str(p))

    return texts, names, paths_out


def gather_supporting_docs_text(sources: Any, max_chars: int = 4000) -> str:
    """
    Join supporting docs into a single text blob, truncated to max_chars.
    Returns empty string if there are no usable sources.
    """
    texts, _, _ = gather_supporting_docs(sources)
    if not texts:
        return ""
    blob = "\n\n---\n\n".join(texts).strip()
    if len(blob) > max_chars:
        blob = blob[:max_chars] + "\n\n[TRUNCATED]"
    return blob
