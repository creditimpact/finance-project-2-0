from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from backend.config import (
    OCR_ENABLED,
    OCR_LANGS,
    OCR_PROVIDER,
    OCR_TIMEOUT_MS,
    PDF_TEXT_MIN_CHARS_PER_PAGE,
)

from .ocr_provider import get_ocr_provider

# Import heavy PDF dependency lazily to avoid import-time crashes in
# environments without PyMuPDF.
def _extract_text_per_page(pdf_path: str) -> list[str]:
    import fitz  # type: ignore

    texts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            texts.append(page.get_text("text") or "")
    return texts


def _merge_text_with_ocr(page_texts: list[str], ocr_texts: Mapping[int, str]) -> list[str]:
    from .pdf_io import merge_text_with_ocr as _impl

    return _impl(page_texts, ocr_texts)

BASE_DIR = Path("traces/texts")


def _session_dir(session_id: str) -> Path:
    return BASE_DIR / session_id


def extract_and_cache_text(
    session_id: str,
    pdf_path: str | Path,
    *,
    ocr_enabled: bool | None = None,
) -> dict[str, Any]:
    """Extract text from ``pdf_path`` and cache to ``traces/texts/<session_id>``.

    Returns a mapping with ``pages``, ``full`` and ``meta`` keys. If no text is
    extracted a ``ValueError`` is raised.
    """

    pdf_path = str(pdf_path)
    ocr_enabled = OCR_ENABLED if ocr_enabled is None else ocr_enabled

    start = time.perf_counter()
    page_texts = _extract_text_per_page(pdf_path)
    extract_text_ms = int((time.perf_counter() - start) * 1000)

    counts = [len(t) for t in page_texts]
    pages_ocr = 0
    ocr_latency_ms_total = 0
    ocr_errors = 0

    if ocr_enabled:
        provider = get_ocr_provider(OCR_PROVIDER)
        ocr_texts: dict[int, str] = {}
        for idx, count in enumerate(counts):
            if count < PDF_TEXT_MIN_CHARS_PER_PAGE:
                pages_ocr += 1
                res = provider.ocr_page(
                    pdf_path,
                    idx,
                    timeout_ms=OCR_TIMEOUT_MS,
                    langs=OCR_LANGS,
                )
                ocr_latency_ms_total += res.duration_ms
                if res.text:
                    ocr_texts[idx] = res.text
                else:
                    ocr_errors += 1
        if ocr_texts:
            page_texts = _merge_text_with_ocr(page_texts, ocr_texts)

    full_text = "\n".join(page_texts)
    if not full_text.strip():
        raise ValueError("no_text_extracted")

    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    for i, txt in enumerate(page_texts, start=1):
        (session_dir / f"page_{i:03d}.txt").write_text(txt, encoding="utf-8")
    (session_dir / "full.txt").write_text(full_text, encoding="utf-8")

    meta = {
        "pages_total": len(page_texts),
        "extract_text_ms": extract_text_ms,
        "pages_ocr": pages_ocr,
        "ocr_latency_ms_total": ocr_latency_ms_total,
        "ocr_errors": ocr_errors,
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    return {"pages": page_texts, "full": full_text, "meta": meta}


def load_cached_text(session_id: str) -> Mapping[str, Any] | None:
    """Load cached text for ``session_id``.

    Returns ``None`` if cache files are missing.
    """

    session_dir = _session_dir(session_id)
    meta_path = session_dir / "meta.json"
    full_path = session_dir / "full.txt"
    if not meta_path.exists() or not full_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    pages = []
    for path in sorted(session_dir.glob("page_*.txt")):
        pages.append(path.read_text(encoding="utf-8"))

    full_text = full_path.read_text(encoding="utf-8")
    return {"pages": pages, "full": full_text, "full_text": full_text, "meta": meta}
