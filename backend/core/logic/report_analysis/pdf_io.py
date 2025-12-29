from typing import List, Mapping


def char_count(s: str) -> int:
    """Return the length of ``s`` treating ``None`` as empty string."""

    return len(s or "")


def merge_text_with_ocr(
    page_texts: List[str], ocr_texts: Mapping[int, str]
) -> List[str]:
    """Merge per-page OCR results into ``page_texts``.

    ``ocr_texts`` maps a 0-indexed page number to its OCR extracted text. Any
    non-empty OCR result replaces the corresponding entry in ``page_texts``.
    Pages without an OCR result remain unchanged.
    """

    merged = list(page_texts)
    for idx, txt in ocr_texts.items():
        if 0 <= idx < len(merged) and txt:
            merged[idx] = txt
    return merged
