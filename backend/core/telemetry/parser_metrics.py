"""Telemetry helpers for parser audit instrumentation."""

from backend.core.case_store.telemetry import emit


def emit_parser_audit(
    *,
    session_id: str,
    pages_total: int,
    pages_with_text: int,
    pages_empty_text: int,
    extract_text_ms: int,
    fields_written: int | None,
    errors: str | None,
    parser_pdf_pages_ocr: int = 0,
    parser_ocr_latency_ms_total: int = 0,
    parser_ocr_errors: int = 0,
    normalize_dates_converted: int = 0,
    normalize_amounts_converted: int = 0,
    normalize_bidi_stripped: int = 0,
    normalize_space_reduced_chars: int = 0,
    extract_sections_ms: int | None = None,
    extract_accounts_ms: int | None = None,
    extract_report_meta_ms: int | None = None,
    extract_summary_ms: int | None = None,
    extractor_field_coverage_total: int | None = None,
    extractor_accounts_total: dict[str, int] | None = None,
) -> None:
    """Emit a ``parser_audit`` telemetry event."""

    emit(
        "parser_audit",
        session_id=session_id,
        pages_total=pages_total,
        pages_with_text=pages_with_text,
        pages_empty_text=pages_empty_text,
        extract_text_ms=extract_text_ms,
        fields_written=fields_written,
        errors=errors,
        parser_pdf_pages_ocr=parser_pdf_pages_ocr,
        parser_ocr_latency_ms_total=parser_ocr_latency_ms_total,
        parser_ocr_errors=parser_ocr_errors,
        normalize_dates_converted=normalize_dates_converted,
        normalize_amounts_converted=normalize_amounts_converted,
        normalize_bidi_stripped=normalize_bidi_stripped,
        normalize_space_reduced_chars=normalize_space_reduced_chars,
        extract_sections_ms=extract_sections_ms,
        extract_accounts_ms=extract_accounts_ms,
        extract_report_meta_ms=extract_report_meta_ms,
        extract_summary_ms=extract_summary_ms,
        extractor_field_coverage_total=extractor_field_coverage_total,
        extractor_accounts_total=extractor_accounts_total,
    )
