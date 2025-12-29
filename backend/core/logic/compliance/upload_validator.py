from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PyPDF2 import PdfReader

from backend.pipeline.runs import RunManifest, persist_manifest

MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".pdf"}


def is_valid_filename(file_path: Path) -> bool:
    return file_path.name.replace(" ", "").isalnum() or file_path.name.endswith(".pdf")


def contains_suspicious_pdf_elements(file_path: Path) -> bool:
    try:
        with open(file_path, "rb") as f:
            content = f.read().lower()
            suspicious_keywords = [
                b"/js",
                b"/javascript",
                b"/launch",
                b"/aa",
                b"/openaction",
            ]
            return any(keyword in content for keyword in suspicious_keywords)
    except Exception as e:
        print(f"[⚠️] Failed to scan for PDF threats: {e}")
        return True


def is_safe_pdf(file_path: Path) -> bool:
    print(f"[INFO] Checking PDF: {file_path.name}")

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        print(f"[✗] Blocked: Unsupported file extension {file_path.suffix}")
        return False

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        print(f"[✗] Blocked: File size {size_mb:.2f} MB exceeds {MAX_UPLOAD_SIZE_MB} MB")
        return False

    suspicious = contains_suspicious_pdf_elements(file_path)
    print(f"[INFO] Suspicious markers found: {suspicious}")
    if suspicious:
        print("[⚠️] Suspicious PDF markers detected but not blocking")

    try:
        reader = PdfReader(str(file_path))
        page_count = len(reader.pages)
        print(f"[INFO] Pages found: {page_count}")
    except Exception as e:
        print(f"[✗] Failed to open PDF: {e}")
        return False

    print("[✅] PDF passed all checks.")
    return True


def move_uploaded_file(
    uploaded_path: Path,
    session_id: str,
    *,
    allow_create: bool,
) -> Path:
    manifest = RunManifest.for_sid(session_id, allow_create=allow_create)
    uploads_dir = manifest.ensure_run_subdir("uploads_dir", "uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    resolved_src = Path(uploaded_path).resolve()
    canonical_path = (uploads_dir / "smartcredit_report.pdf").resolve()

    runs_root = uploads_dir.parent
    try:
        relative = resolved_src.relative_to(runs_root)
    except ValueError:
        relative = None

    if not allow_create and relative is not None:
        source_sid = relative.parts[0] if relative.parts else ""
        if source_sid and source_sid != session_id:
            raise RuntimeError(
                "SID mismatch: uploaded file belongs to a different run"
            )

    if resolved_src != canonical_path:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_src.replace(canonical_path)

    manifest.set_artifact("uploads", "smartcredit_report", canonical_path)
    persist_manifest(manifest, inputs={"report_pdf": canonical_path})
    return canonical_path
