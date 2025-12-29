from __future__ import annotations


def extract_text(pdf_path: str, *, prefer_mupdf: bool = True) -> str:
    """Legacy wrapper preserved for compatibility.

    Direct PDF parsing has moved to :mod:`backend.core.logic.report_analysis`
    text provider. This function now raises ``NotImplementedError`` to steer
    callers to the new API.
    """

    raise NotImplementedError("use text_provider.extract_and_cache_text")
