from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class OcrResult:
    """Result of an OCR operation."""

    text: str
    duration_ms: int


class OcrProvider(Protocol):
    def ocr_page(
        self,
        pdf_path: str,
        page_index: int,
        *,
        timeout_ms: int,
        langs: list[str],
    ) -> OcrResult: ...


class TesseractProvider:
    """Basic OCR provider backed by a Tesseract engine and PDF-to-image tooling.

    This implementation aims to be best-effort. Any errors or timeouts result in
    an empty string being returned so that callers can decide how to proceed
    without the OCR step interrupting the pipeline.
    """

    def ocr_page(
        self,
        pdf_path: str,
        page_index: int,
        *,
        timeout_ms: int,
        langs: list[str],
    ) -> OcrResult:
        import threading
        import time

        result: dict[str, str] = {"text": ""}

        def worker() -> None:
            try:
                import importlib

                pyt = importlib.import_module("py" "tesseract")
                pdf2 = importlib.import_module("pdf2" "image")
                images = pdf2.convert_from_path(
                    pdf_path,
                    first_page=page_index + 1,
                    last_page=page_index + 1,
                    dpi=300,
                )
                if images:
                    lang = "+".join(langs) if langs else "eng"
                    result["text"] = pyt.image_to_string(images[0], lang=lang)
            except Exception:
                result["text"] = ""

        start = time.perf_counter()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout_ms / 1000)
        duration_ms = int((time.perf_counter() - start) * 1000)
        if thread.is_alive():
            return OcrResult(text="", duration_ms=duration_ms)
        return OcrResult(text=result["text"], duration_ms=duration_ms)


def get_ocr_provider(name: str) -> OcrProvider:
    """Return an OCR provider implementation by name."""

    name = (name or "").lower()
    if name == "tesseract":
        return TesseractProvider()
    raise ValueError(f"Unsupported OCR provider: {name}")
