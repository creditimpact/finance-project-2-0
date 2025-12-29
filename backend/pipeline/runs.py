from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import inspect
from datetime import datetime, timezone
from typing import Any, Mapping
import json, os, glob, shutil, logging
import tempfile
import time
from backend.core.ai.paths import (
    MergePaths,
    ensure_merge_paths,
    merge_paths_from_any,
    normalize_note_style_account_id,
    ensure_validation_paths,
)

RUNS_ROOT_ENV = "RUNS_ROOT"                 # optional override
MANIFEST_ENV  = "REPORT_MANIFEST_PATH"      # explicit manifest path


logger = logging.getLogger(__name__)

def _runs_root() -> Path:
    rr = os.getenv(RUNS_ROOT_ENV)
    return Path(rr) if rr else Path("runs")

RUNS_ROOT = _runs_root()


def _resolve_caller() -> str:
    try:
        stack = inspect.stack()[2:]
    except Exception:
        return "unknown"
    for frame in stack:
        module = frame.frame.f_globals.get("__name__", "")
        func = frame.function
        if module.startswith("backend"):
            return f"{module}.{func}"
    if stack:
        return stack[0].function
    return "unknown"


def _env_flag_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"", "0", "false", "no", "off"}:
        return False
    return True


def _new_sid_upload_only_enabled() -> bool:
    return _env_flag_enabled("RUNFLOW_NEW_SID_ON_UPLOAD_ONLY", False)


def _note_style_stage_snapshot(
    sid: str, runs_root: Path | str | None = None
):
    from backend.ai.note_style.io import note_style_snapshot

    return note_style_snapshot(sid, runs_root=runs_root)


def get_runs_root() -> Path:
    """Return the configured runs/ root directory."""

    return _runs_root()

def _utc_now():
    # timezone-aware UTC to avoid deprecation; normalize suffix to 'Z'
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_manifest_from_disk(runs_root: Path, sid: str) -> RunManifest:
    """Load the latest manifest.json from disk for `sid`.
    
    INVARIANT: This is the ONLY way to get a fresh manifest for mutation.
    Never cache or reuse old RunManifest instances across phases.
    """
    manifest_path = runs_root / sid / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = RunManifest(manifest_path)
            manifest.load()
            return manifest
        except Exception as exc:
            logger.warning(
                "MANIFEST_LOAD_FROM_DISK_FAILED sid=%s path=%s error=%s",
                sid,
                manifest_path,
                exc,
                exc_info=True,
            )
            raise
    
    # Create minimal structure if missing
    manifest = RunManifest.load_or_create(manifest_path, sid, allow_create=True)
    return manifest


def save_manifest_to_disk(
    runs_root: Path,
    sid: str,
    mutate_fn: callable,
    *,
    caller: str | None = None,
) -> RunManifest:
    """Atomically update manifest.json with validation natives invariant.
    
    1. Load the latest manifest.json from disk.
    2. Call mutate_fn(manifest) to apply changes IN PLACE.
    3. Enforce validation natives are sticky (never removed).
    4. Save back to disk.
    
    Parameters
    ----------
    runs_root:
        Runs root directory.
    sid:
        Session ID.
    mutate_fn:
        Callable that accepts a RunManifest and mutates it in place.
    caller:
        Optional caller name for diagnostics.
    
    Returns
    -------
    RunManifest:
        The saved manifest instance.
    """
    manifest = load_manifest_from_disk(runs_root, sid)
    
    # Snapshot validation natives BEFORE mutation
    def _get_section(data: dict, path: tuple[str, ...]) -> Any:
        cur = data
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur
    
    pre_ai_packs_val = _get_section(manifest.data, ("ai", "packs", "validation"))
    pre_ai_validation = _get_section(manifest.data, ("ai", "validation"))
    pre_artifacts_val = _get_section(manifest.data, ("artifacts", "ai", "packs", "validation"))
    pre_meta_flag = bool(_get_section(manifest.data, ("meta", "validation_paths_initialized")))
    
    # Apply caller's mutations
    mutate_fn(manifest)
    
    # Restore validation natives if they were dropped
    def _set_section(data: dict, path: tuple[str, ...], value: Any) -> None:
        cur = data
        for key in path[:-1]:
            nxt = cur.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[key] = nxt
            cur = nxt
        cur[path[-1]] = value
    
    post_ai_packs_val = _get_section(manifest.data, ("ai", "packs", "validation"))
    post_ai_validation = _get_section(manifest.data, ("ai", "validation"))
    post_artifacts_val = _get_section(manifest.data, ("artifacts", "ai", "packs", "validation"))
    post_meta_flag = bool(_get_section(manifest.data, ("meta", "validation_paths_initialized")))
    
    preserved_fields: list[str] = []
    if isinstance(pre_ai_packs_val, dict) and not isinstance(post_ai_packs_val, dict):
        _set_section(manifest.data, ("ai", "packs", "validation"), dict(pre_ai_packs_val))
        preserved_fields.append("ai.packs.validation")
    if isinstance(pre_ai_validation, dict) and not isinstance(post_ai_validation, dict):
        _set_section(manifest.data, ("ai", "validation"), dict(pre_ai_validation))
        preserved_fields.append("ai.validation")
    if isinstance(pre_artifacts_val, dict) and not isinstance(post_artifacts_val, dict):
        _set_section(manifest.data, ("artifacts", "ai", "packs", "validation"), dict(pre_artifacts_val))
        preserved_fields.append("artifacts.ai.packs.validation")
    if pre_meta_flag and not post_meta_flag:
        meta = manifest.data.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["validation_paths_initialized"] = True
            preserved_fields.append("meta.validation_paths_initialized")
    
    if preserved_fields:
        logger.info(
            "MANIFEST_VALIDATION_NATIVE_PRESERVE sid=%s caller=%s fields=%s",
            sid,
            caller or "unknown",
            ",".join(preserved_fields),
        )
    
    # Save to disk
    manifest.save()
    return manifest


def safe_replace(dst_path: str, data: str, *, attempts: int = 5, delay: float = 0.1) -> None:
    dst_dir = os.path.dirname(dst_path) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dst_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # fsync while handle is open

        last_error: PermissionError | None = None
        for attempt in range(1, max(attempts, 1) + 1):
            try:
                os.replace(tmp_path, dst_path)  # atomic on Windows & POSIX
                return
            except PermissionError as exc:  # transient Windows lock
                last_error = exc
                logger.warning(
                    "SAFE_REPLACE_PERMISSION_RETRY dst=%s tmp=%s attempt=%d/%d err=%r",
                    dst_path,
                    tmp_path,
                    attempt,
                    attempts,
                    exc,
                )
                if attempt >= attempts:
                    break
                time.sleep(delay * attempt)

        if last_error is not None:
            logger.error(
                "SAFE_REPLACE_PERMISSION_FAILED dst=%s tmp=%s attempts=%d err=%r",
                dst_path,
                tmp_path,
                attempts,
                last_error,
            )
            raise last_error
        logger.error(
            "SAFE_REPLACE_PERMISSION_FAILED dst=%s tmp=%s attempts=%d err=%s",
            dst_path,
            tmp_path,
            attempts,
            "unknown",
        )
        raise PermissionError(f"os.replace failed for {dst_path} without explicit error")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _resolve_optional_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        return RUNS_ROOT
    if isinstance(runs_root, Path):
        return runs_root
    try:
        text = str(runs_root)
    except Exception:
        return RUNS_ROOT
    normalized = text.strip()
    if not normalized:
        return RUNS_ROOT
    return Path(normalized)

@dataclass
class _RunManifestInputs:
    """Proxy object exposing manifest input paths via attribute access."""

    def __init__(self, manifest: "RunManifest") -> None:
        self._manifest = manifest

    def _section(self) -> dict[str, Any]:
        return self._manifest._ensure_inputs_section()

    @property
    def report_pdf(self) -> Path | None:
        value = self._section().get("report_pdf")
        if not value:
            return None
        try:
            return Path(value)
        except TypeError:
            return None

    @report_pdf.setter
    def report_pdf(self, value: Path | str | None) -> None:
        inputs = self._section()
        if value is None:
            inputs.pop("report_pdf", None)
            return
        path = Path(value).resolve()
        inputs["report_pdf"] = str(path)


class RunManifest:
    path: Path
    data: dict = field(default_factory=dict)
    _inputs_accessor: _RunManifestInputs | None = field(default=None, init=False, repr=False)

    # --- Back-compat shim: allow old-style initialization and materialize fields ---
    def __init__(self, path: Path | str | None = None, **kwargs) -> None:
        """Back-compat for legacy call sites that do RunManifest(path=...)."""
        from pathlib import Path as _Path
        import json as _json

        for k, v in kwargs.items():
            setattr(self, k, v)

        existing_data = getattr(self, "data", None)
        if not isinstance(existing_data, dict):
            self.data = {}

        self.path = None
        if path is not None:
            p = _Path(path)
            self.path = p
            if p.exists():
                try:
                    data = _json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
                if isinstance(data, dict):
                    self.data = data
                if not hasattr(self, "sid"):
                    try:
                        self.sid = p.parent.name
                    except Exception:
                        pass

        # Reset accessor so the property materializes a fresh helper
        self._inputs_accessor = None
        try:
            inputs_section = self._ensure_inputs_section()
        except Exception:
            inputs_section = None
        if isinstance(inputs_section, dict):
            value = inputs_section.get("report_pdf")
            if value:
                try:
                    inputs_section["report_pdf"] = str(_Path(value))
                except Exception:
                    pass

    @property
    def manifest_path(self) -> Path | None:
        """Legacy-friendly attribute some call sites expect."""
        from pathlib import Path as _Path

        path_value = getattr(self, "path", None)
        if path_value is not None:
            return path_value if isinstance(path_value, _Path) else _Path(path_value)
        if hasattr(self, "manifest_file"):
            return _Path(getattr(self, "manifest_file"))
        return None
    # --- end shim ---

    # -------- creation / loading ----------
    @classmethod
    def for_sid(
        cls,
        sid: str,
        *,
        allow_create: bool = True,
        runs_root: Path | str | None = None,
    ) -> "RunManifest":
        base_root: Path
        if runs_root is None:
            base_root = _runs_root()
        else:
            base_root = Path(runs_root)

        base = base_root / sid
        manifest_path = base / "manifest.json"
        exists = manifest_path.exists()

        logger.info(
            "RUNMANIFEST_FOR_SID sid=%s allow_create=%d path=%s exists=%d",
            sid,
            1 if allow_create else 0,
            str(manifest_path),
            1 if exists else 0,
        )

        if exists:
            manifest = cls(manifest_path)
            return manifest._load_or_create(sid)

        if not allow_create:
            caller = _resolve_caller()
            logger.warning("RUN_DIR_CREATE_BLOCKED sid=%s caller=%s", sid, caller)
            raise FileNotFoundError(f"run manifest missing for {sid}")

        base.mkdir(parents=True, exist_ok=True)
        manifest = cls(manifest_path)
        return manifest._load_or_create(sid)

    @staticmethod
    def from_env_or_latest() -> "RunManifest":
        p = os.getenv(MANIFEST_ENV)
        if p:
            return RunManifest(Path(p)).load()
        cands = glob.glob(str(_runs_root() / "*/manifest.json"))
        if not cands:
            raise FileNotFoundError("No manifests found under runs/*/manifest.json")
        newest = max(map(Path, cands), key=lambda x: x.stat().st_mtime)
        return RunManifest(newest).load()

    @staticmethod
    def load_or_create(
        path: Path, sid: str | None = None, *, allow_create: bool | None = None
    ) -> "RunManifest":
        """Load an existing manifest at ``path`` or create a new one."""

        manifest = RunManifest(path)
        if path.exists():
            return manifest.load()

        effective_sid = sid or path.parent.name
        if not effective_sid:
            raise ValueError("sid is required to create a new manifest")

        if allow_create is None:
            allow_create = not _new_sid_upload_only_enabled()

        if not allow_create:
            raise FileNotFoundError(
                f"run manifest missing for {effective_sid}; creation disabled by env"
            )

        manifest.path.parent.mkdir(parents=True, exist_ok=True)
        return manifest._load_or_create(effective_sid)

    def _load_or_create(self, sid: str) -> "RunManifest":
        if self.path.exists():
            return self.load()
        self.data = {
            "sid": sid,
            "created_at": _utc_now(),
            "status": "in_progress",
            "base_dirs": {
                "uploads_dir": None,
                "traces_dir": None,
                "cases_dir": None,
                "exports_dir": None,
                "logs_dir": None,
            },
            "ai": {
                "packs": {
                    "base": None,
                    "dir": None,
                    "packs": None,
                    "packs_dir": None,
                    "results": None,
                    "results_dir": None,
                    "index": None,
                    "pairs": 0,
                    "last_built_at": None,
                    "logs": None,
                    "validation": {
                        "base": None,
                        "dir": None,
                        "packs": None,
                        "packs_dir": None,
                        "results": None,
                        "results_dir": None,
                        "index": None,
                        "last_built_at": None,
                        "logs": None,
                    },
                    "note_style": {
                        "base": None,
                        "dir": None,
                        "packs": None,
                        "packs_dir": None,
                        "results": None,
                        "results_dir": None,
                        "index": None,
                        "last_built_at": None,
                        "logs": None,
                        "status": {"built": False, "completed_at": None},
                    },
                },
                "validation": {
                    "base": None,
                    "dir": None,
                    "accounts": None,
                    "accounts_dir": None,
                    "last_prepared_at": None,
                },
                "status": {
                    "enqueued": False,
                    "built": False,
                    "sent": False,
                    "compacted": False,
                    "skipped_reason": None,
                    "merge": {
                        "built": False,
                        "sent": False,
                        "completed_at": None,
                    },
                    "validation": {
                        "built": False,
                        "sent": False,
                        "completed_at": None,
                    },
                    "note_style": {
                        "built": False,
                        "completed_at": None,
                    },
                },
            },
            "artifacts": {
                "uploads": {},
                "traces": {"accounts_table": {}},
                "cases": {},
                "exports": {},
                "logs": {},
            },
            "env_snapshot": {},
            "inputs": {
                "report_pdf": None,
            },
            "frontend": {
                "base": None,
                "dir": None,
                "packs": None,
                "packs_dir": None,
                "results": None,
                "results_dir": None,
                "index": None,
                "built": False,
                "packs_count": 0,
                "counts": {"packs": 0, "responses": 0},
                "last_built_at": None,
                "last_responses_at": None,
            },
        }
        self._update_index(sid)
        (self.path.parent.parent / "current.txt").write_text(sid, encoding="utf-8")
        return self.save()

    def load(self) -> "RunManifest":
        with self.path.open("r", encoding="utf-8") as fh:
            self.data = json.load(fh)
        self._ensure_inputs_section()
        return self

    def save(self) -> "RunManifest":
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Invariant: Preserve validation natives across all writers.
        # If current in-memory data is missing validation natives but on-disk manifest has them,
        # re-inject them before saving. Also log callsite and current values for diagnostics.
        caller = _resolve_caller()
        try:
            on_disk: dict[str, Any] | None = None
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as fh:
                    on_disk = json.load(fh)
        except Exception:
            on_disk = None

        def _get_section(root: Mapping[str, Any] | None, path: tuple[str, ...]) -> Any:
            cur: Any = root if isinstance(root, Mapping) else {}
            for key in path:
                if not isinstance(cur, Mapping):
                    return None
                cur = cur.get(key)
            return cur

        def _set_section(root: dict, path: tuple[str, ...], value: Any) -> None:
            cur: dict = root
            for key in path[:-1]:
                nxt = cur.get(key)
                if not isinstance(nxt, dict):
                    nxt = {}
                    cur[key] = nxt
                cur = nxt
            cur[path[-1]] = value

        # Resolve current and prior validation-native sections
        current_ai_packs_val = _get_section(self.data, ("ai", "packs", "validation"))
        current_ai_validation = _get_section(self.data, ("ai", "validation"))
        current_artifacts_val = _get_section(self.data, ("artifacts", "ai", "packs", "validation"))
        current_meta_flag = bool(_get_section(self.data, ("meta", "validation_paths_initialized")))

        prior_ai_packs_val = _get_section(on_disk, ("ai", "packs", "validation")) if on_disk else None
        prior_ai_validation = _get_section(on_disk, ("ai", "validation")) if on_disk else None
        prior_artifacts_val = _get_section(on_disk, ("artifacts", "ai", "packs", "validation")) if on_disk else None
        prior_meta_flag = bool(_get_section(on_disk, ("meta", "validation_paths_initialized"))) if on_disk else False

        # Log before-save snapshot for investigation
        try:
            logger.info(
                "MANIFEST_SAVE_BEFORE sid=%s caller=%s has_ai_packs_validation=%s has_ai_validation=%s has_artifacts_validation=%s meta_initialized=%s",
                getattr(self, "sid", self.path.parent.name),
                caller,
                isinstance(current_ai_packs_val, Mapping),
                isinstance(current_ai_validation, Mapping),
                isinstance(current_artifacts_val, Mapping),
                current_meta_flag,
            )
        except Exception:
            pass

        # Preservation: if prior sections exist but current are missing/non-mapping, restore them
        preserved = False
        if isinstance(prior_ai_packs_val, Mapping) and not isinstance(current_ai_packs_val, Mapping):
            _set_section(self.data, ("ai", "packs", "validation"), dict(prior_ai_packs_val))
            preserved = True
        if isinstance(prior_ai_validation, Mapping) and not isinstance(current_ai_validation, Mapping):
            _set_section(self.data, ("ai", "validation"), dict(prior_ai_validation))
            preserved = True
        if isinstance(prior_artifacts_val, Mapping) and not isinstance(current_artifacts_val, Mapping):
            _set_section(self.data, ("artifacts", "ai", "packs", "validation"), dict(prior_artifacts_val))
            preserved = True
        if prior_meta_flag and not current_meta_flag:
            meta = self.data.setdefault("meta", {}) if isinstance(self.data, dict) else {}
            if isinstance(meta, dict):
                meta["validation_paths_initialized"] = True
                preserved = True

        if preserved:
            try:
                logger.info(
                    "MANIFEST_VALIDATION_NATIVE_PRESERVE sid=%s caller=%s",
                    getattr(self, "sid", self.path.parent.name),
                    caller,
                )
            except Exception:
                pass

        # Defensive backfill: if meta.validation_paths_initialized is True but
        # ai.packs.validation primary path fields are still null/empty, recover
        # them from artifacts.ai.packs.validation or recompute canonically.
        try:
            meta_section = self.data.get("meta") if isinstance(self.data, Mapping) else None
            initialized = bool(meta_section and meta_section.get("validation_paths_initialized"))
            if initialized:
                ai_section = self.data.get("ai") if isinstance(self.data, Mapping) else None
                packs_section = ai_section.get("packs") if isinstance(ai_section, Mapping) else None
                validation_packs = packs_section.get("validation") if isinstance(packs_section, Mapping) else None
                artifacts_validation = self.data.get("artifacts", {}).get("ai", {}).get("packs", {}).get("validation")
                if isinstance(validation_packs, Mapping):
                    keys = ["base","dir","packs","packs_dir","results","results_dir","index","logs"]
                    missing_all = all(
                        not (isinstance(validation_packs.get(k), str) and validation_packs.get(k).strip())
                        for k in keys
                    )
                    if missing_all:
                        backfilled = False
                        if isinstance(artifacts_validation, Mapping):
                            for k in keys:
                                v = artifacts_validation.get(k)
                                if isinstance(v, str) and v.strip():
                                    validation_packs[k] = v
                            backfilled = True
                        else:
                            try:
                                runs_root = self.path.parent.parent
                                canon = ensure_validation_paths(runs_root, self.sid, create=False)
                                validation_packs.update({
                                    "base": str(canon.base),
                                    "dir": str(canon.base),
                                    "packs": str(canon.packs_dir),
                                    "packs_dir": str(canon.packs_dir),
                                    "results": str(canon.results_dir),
                                    "results_dir": str(canon.results_dir),
                                    "index": str(canon.index_file),
                                    "logs": str(canon.log_file),
                                })
                                backfilled = True
                            except Exception:
                                pass
                        if backfilled:
                            try:
                                logger.info(
                                    "MANIFEST_VALIDATION_NATIVE_BACKFILL sid=%s caller=%s", getattr(self, "sid", self.path.parent.name), caller
                                )
                            except Exception:
                                pass
        except Exception:
            pass

        # Proceed with normal upgrade/mirroring and save
        self._upgrade_ai_packs_structure()
        self._mirror_ai_to_legacy_artifacts()
        self._ensure_inputs_section()
        
        data = json.dumps(self.data, ensure_ascii=False, indent=2)
        safe_replace(str(self.path), data)
        return self

    def _upgrade_ai_packs_structure(self) -> None:
        ai_section = self.data.get("ai")
        if not isinstance(ai_section, dict):
            return

        packs_section = ai_section.get("packs")
        if not isinstance(packs_section, dict):
            return

        base_value = packs_section.get("base")
        packs_value = packs_section.get("packs")
        results_value = packs_section.get("results")
        dir_value = packs_section.get("dir")

        needs_upgrade = not (base_value and packs_value and results_value)

        if not needs_upgrade and isinstance(dir_value, str):
            normalized = dir_value.rstrip("/\\")
            if normalized.endswith("ai_packs"):
                needs_upgrade = True

        if not needs_upgrade:
            return

        run_root = self.path.parent
        runs_root = run_root.parent
        canonical_paths = ensure_merge_paths(runs_root, self.sid, create=False)

        merge_paths: MergePaths | None = None
        candidates = [
            packs_section.get("base"),
            packs_section.get("packs"),
            packs_section.get("results"),
            packs_section.get("dir"),
            packs_section.get("packs_dir"),
            packs_section.get("results_dir"),
        ]

        for candidate in candidates:
            if not candidate:
                continue
            try:
                merge_paths = merge_paths_from_any(Path(candidate), create=False)
                break
            except ValueError:
                continue

        if merge_paths is None:
            merge_paths = canonical_paths

        existing_logs = packs_section.get("logs")
        prefer_existing_index = bool(packs_section.get("index"))
        self._apply_merge_paths_to_packs(
            packs_section,
            merge_paths,
            prefer_existing_index=prefer_existing_index,
        )

        if existing_logs:
            packs_section["logs"] = existing_logs

    def _mirror_ai_to_legacy_artifacts(self) -> None:
        ai_section = self.data.get("ai")
        if not isinstance(ai_section, dict):
            return

        artifacts = self.data.setdefault("artifacts", {})
        artifacts.pop("ai_packs", None)

        legacy_ai = artifacts.setdefault("ai", {})

        packs = ai_section.get("packs")
        if isinstance(packs, dict):
            legacy_packs = legacy_ai.setdefault("packs", {})
            for key in (
                "base",
                "dir",
                "packs",
                "packs_dir",
                "results",
                "results_dir",
                "index",
                "pairs",
                "last_built_at",
                "logs",
            ):
                if key in packs:
                    legacy_packs[key] = packs[key]

        status = ai_section.get("status")
        if isinstance(status, dict):
            legacy_status = legacy_ai.setdefault("status", {})
            for key, value in status.items():
                if key == "validation":  # do not duplicate validation block
                    continue
                legacy_status[key] = value
            # Explicitly remove any previously mirrored validation block
            legacy_status.pop("validation", None)

    def _update_index(self, sid: str) -> None:
        idx = _runs_root() / "index.json"
        rec = {"sid": sid, "created_at": _utc_now()}
        if idx.exists():
            try:
                obj = json.loads(idx.read_text(encoding="utf-8"))
            except Exception:
                obj = {"runs": []}
        else:
            obj = {"runs": []}
        obj.setdefault("runs", []).append(rec)
        tmp = idx.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        tmp.replace(idx)

    # -------- API ----------
    def snapshot_env(self, keys: list[str]) -> "RunManifest":
        for k in keys:
            v = os.getenv(k)
            if v is not None:
                self.data["env_snapshot"][k] = v
        return self.save()

    def set_base_dir(self, label: str, path: Path) -> "RunManifest":
        resolved = Path(path).resolve()
        self.data.setdefault("base_dirs", {})[label] = str(resolved)
        return self.save()

    def _ensure_inputs_section(self) -> dict[str, Any]:
        inputs = self.data.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
            self.data["inputs"] = inputs
        if "report_pdf" not in inputs:
            inputs.setdefault("report_pdf", None)
        return inputs

    @property
    def inputs(self) -> _RunManifestInputs:
        if self._inputs_accessor is None:
            self._inputs_accessor = _RunManifestInputs(self)
        return self._inputs_accessor

    def set_artifact(self, group: str, name: str, path: str | Path) -> "RunManifest":
        """Record an artifact path under ``artifacts.<group>.<name>``.

        Parameters
        ----------
        group:
            Dotted path indicating the artifact grouping (for example
            ``"traces.accounts_table"``).  Missing intermediate dictionaries are
            created automatically.
        name:
            Artifact identifier within the group.
        path:
            Filesystem location to store.  The path is normalized to an absolute
            string representation.
        """

        resolved_path = str(Path(path).resolve())
        cursor: dict[str, object] = self.data.setdefault("artifacts", {})
        for part in str(group).split("."):
            if not part:
                continue
            next_cursor = cursor.setdefault(part, {})
            if not isinstance(next_cursor, dict):
                raise TypeError(
                    f"Cannot assign artifact into non-mapping at group '{group}'"
                )
            cursor = next_cursor
        cursor[str(name)] = resolved_path
        return self.save()

    @staticmethod
    def _stage_status_defaults(stage_key: str) -> dict[str, object]:
        key = stage_key.strip().lower()
        defaults: dict[str, object] = {
            "built": False,
            "sent": False,
            "failed": False,
            "completed_at": None,
            "state": None,
        }
        return defaults

    def _ensure_ai_section(self) -> tuple[dict[str, object], dict[str, object]]:
        ai = self.data.setdefault("ai", {})
        packs = ai.setdefault(
            "packs",
            {
                "base": None,
                "dir": None,
                "packs": None,
                "packs_dir": None,
                "results": None,
                "results_dir": None,
                "index": None,
                "pairs": 0,
                "last_built_at": None,
                "logs": None,
            },
        )
        validation_section = packs.get("validation")
        if not isinstance(validation_section, dict):
            validation_section = {
                "base": None,
                "dir": None,
                "packs": None,
                "packs_dir": None,
                "results": None,
                "results_dir": None,
                "index": None,
                "last_built_at": None,
                "logs": None,
            }
            packs["validation"] = validation_section
        else:
            for key in (
                "base",
                "dir",
                "packs",
                "packs_dir",
                "results",
                "results_dir",
                "index",
                "last_built_at",
                "logs",
            ):
                validation_section.setdefault(key, None)
        note_style_section = packs.get("note_style")
        if not isinstance(note_style_section, dict):
            note_style_section = {
                "base": None,
                "dir": None,
                "packs": None,
                "packs_dir": None,
                "results": None,
                "results_dir": None,
                "index": None,
                "last_built_at": None,
                "logs": None,
                "status": {"built": False, "completed_at": None},
            }
            packs["note_style"] = note_style_section
        else:
            for key in (
                "base",
                "dir",
                "packs",
                "packs_dir",
                "results",
                "results_dir",
                "index",
                "last_built_at",
                "logs",
            ):
                note_style_section.setdefault(key, None)
        status_payload = note_style_section.get("status")
        if not isinstance(status_payload, dict):
            status_payload = {
                "built": False,
                "sent": False,
                "failed": False,
                "completed_at": None,
            }
            note_style_section["status"] = status_payload
        else:
            status_payload.setdefault("built", False)
            status_payload.setdefault("sent", False)
            status_payload.setdefault("failed", False)
            status_payload.setdefault("completed_at", None)
        ai.setdefault(
            "validation",
            {
                "base": None,
                "dir": None,
                "accounts": None,
                "accounts_dir": None,
                "last_prepared_at": None,
            },
        )
        status = ai.setdefault(
            "status",
            {
                "enqueued": False,
                "built": False,
                "sent": False,
                "compacted": False,
                "skipped_reason": None,
            },
        )
        for stage_key in ("merge", "validation", "note_style"):
            stage_defaults = self._stage_status_defaults(stage_key)
            stage_status = status.get(stage_key)
            if not isinstance(stage_status, dict):
                status[stage_key] = dict(stage_defaults)
                continue

            for key, value in stage_defaults.items():
                stage_status.setdefault(key, value)

        return packs, status

    def ensure_ai_stage_status(self, stage: str) -> dict[str, object]:
        """Ensure and return the mutable status mapping for ``stage``."""

        stage_key = str(stage).strip().lower()
        if not stage_key:
            raise ValueError("stage is required")
        _, status = self._ensure_ai_section()
        stage_defaults = self._stage_status_defaults(stage_key)
        stage_status = status.get(stage_key)
        if not isinstance(stage_status, dict):
            stage_status = dict(stage_defaults)
            status[stage_key] = stage_status
        else:
            for key, value in stage_defaults.items():
                stage_status.setdefault(key, value)
        return stage_status

    def get_ai_stage_status(self, stage: str) -> dict[str, object]:
        """Return a copy of the status mapping for ``stage``."""

        stage_key = str(stage).strip().lower()
        if not stage_key:
            raise ValueError("stage is required")
        ai = self.data.get("ai")
        stage_defaults = self._stage_status_defaults(stage_key)
        if not isinstance(ai, Mapping):
            return dict(stage_defaults)
        status = ai.get("status")
        fallback = dict(stage_defaults)
        if not isinstance(status, Mapping):
            return fallback
        stage_status = status.get(stage_key)
        if not isinstance(stage_status, Mapping):
            return fallback
        payload: dict[str, object] = {}
        for key in stage_defaults:
            if key in {"built", "sent", "failed"}:
                payload[key] = bool(stage_status.get(key))
            else:
                payload[key] = stage_status.get(key)
        if (
            stage_key == "validation"
            and isinstance(stage_status.get("merge_results"), Mapping)
        ):
            payload["merge_results"] = dict(stage_status["merge_results"])
        return payload

    def _ensure_ai_validation_pack_section(self) -> dict[str, object]:
        packs, _ = self._ensure_ai_section()
        validation = packs.setdefault(
            "validation",
            {
                "base": None,
                "dir": None,
                "packs": None,
                "packs_dir": None,
                "results": None,
                "results_dir": None,
                "index": None,
                "last_built_at": None,
                "logs": None,
            },
        )
        if not isinstance(validation, dict):
            raise TypeError("ai.packs.validation must be a mapping")
        return validation

    def _apply_merge_paths_to_packs(
        self,
        packs: dict[str, object],
        merge_paths: MergePaths,
        *,
        prefer_existing_index: bool = False,
    ) -> None:
        base_str = str(merge_paths.base)
        packs_str = str(merge_paths.packs_dir)
        results_str = str(merge_paths.results_dir)

        packs["base"] = base_str
        packs["dir"] = base_str
        packs["packs"] = packs_str
        packs["packs_dir"] = packs_str
        packs["results"] = results_str
        packs["results_dir"] = results_str
        packs["logs"] = str(merge_paths.log_file)

        if not (prefer_existing_index and packs.get("index")):
            packs["index"] = str(merge_paths.index_file)

    def _ensure_validation_section(self) -> dict[str, object]:
        ai = self.data.setdefault("ai", {})
        validation = ai.setdefault(
            "validation",
            {
                "base": None,
                "dir": None,
                "accounts": None,
                "accounts_dir": None,
                "last_prepared_at": None,
            },
        )
        return validation

    def upsert_validation_packs_dir(
        self,
        base_dir: Path,
        *,
        packs_dir: Path | None = None,
        results_dir: Path | None = None,
        index_file: Path | None = None,
        log_file: Path | None = None,
        account_dir: Path | None = None,
    ) -> "RunManifest":
        validation = self._ensure_validation_section()
        resolved = Path(base_dir).resolve()
        resolved_str = str(resolved)
        timestamp = _utc_now()

        # PHASE 2: After T0 injection, path fields are immutable.
        # This function now ONLY updates timestamps and status, not paths.
        # See: VALIDATION_NATIVES_T0_INVESTIGATION.md Section 6 (Phase 2)
        
        # Update timestamps only (paths should already exist from T0)
        validation = self._ensure_validation_section()
        validation["last_prepared_at"] = timestamp
        
        packs_validation = self._ensure_ai_validation_pack_section()
        packs_validation["last_built_at"] = timestamp
        
        # Log if paths are missing (should not happen after T0 injection)
        if not packs_validation.get("packs_dir"):
            logger.warning(
                "VALIDATION_PATHS_MISSING_AT_UPSERT sid=%s - paths should have been injected at T0",
                getattr(self, "sid", "unknown")
            )

        validation_stage_status = self.ensure_ai_stage_status("validation")
        validation_stage_status["built"] = True
        validation_stage_status["sent"] = False
        validation_stage_status["completed_at"] = None

        return self.save()

    def _ensure_ai_note_style_pack_section(self) -> dict[str, object]:
        packs, _ = self._ensure_ai_section()
        note_style = packs.get("note_style")
        if not isinstance(note_style, dict):
            note_style = {
                "base": None,
                "dir": None,
                "packs": None,
                "packs_dir": None,
                "results": None,
                "results_dir": None,
                "index": None,
                "last_built_at": None,
                "logs": None,
                "status": {"built": False, "completed_at": None},
            }
            packs["note_style"] = note_style
        status_payload = note_style.setdefault(
            "status", {"built": False, "completed_at": None}
        )
        if not isinstance(status_payload, dict):
            status_payload = {"built": False, "completed_at": None}
            note_style["status"] = status_payload
        else:
            status_payload.setdefault("built", False)
            status_payload.setdefault("completed_at", None)
            status_payload.pop("sent", None)
        return note_style

    def upsert_note_style_packs_dir(
        self,
        base_dir: Path,
        *,
        packs_dir: Path | None = None,
        results_dir: Path | None = None,
        index_file: Path | None = None,
        log_file: Path | None = None,
        last_built_at: str | None = None,
    ) -> "RunManifest":
        note_style = self._ensure_ai_note_style_pack_section()
        resolved_base = Path(base_dir).resolve()
        resolved_str = str(resolved_base)
        timestamp = last_built_at or _utc_now()

        note_style["base"] = resolved_str
        note_style["dir"] = resolved_str

        packs_path = (
            Path(packs_dir).resolve()
            if packs_dir is not None
            else (resolved_base / "packs").resolve()
        )
        note_style["packs"] = str(packs_path)
        note_style["packs_dir"] = str(packs_path)

        results_path = (
            Path(results_dir).resolve()
            if results_dir is not None
            else (resolved_base / "results").resolve()
        )
        note_style["results"] = str(results_path)
        note_style["results_dir"] = str(results_path)

        index_path = (
            Path(index_file).resolve(strict=False)
            if index_file is not None
            else (resolved_base / "index.json").resolve(strict=False)
        )
        note_style["index"] = str(index_path)

        log_path = (
            Path(log_file).resolve(strict=False)
            if log_file is not None
            else (resolved_base / "logs.txt").resolve(strict=False)
        )
        note_style["logs"] = str(log_path)
        note_style["last_built_at"] = timestamp

        status_payload = note_style.setdefault(
            "status", {"built": False, "sent": False, "completed_at": None}
        )
        if not isinstance(status_payload, dict):
            status_payload = {}
            note_style["status"] = status_payload
        status_payload["built"] = True
        status_payload["sent"] = False
        status_payload["completed_at"] = None

        stage_status = self.ensure_ai_stage_status("note_style")
        stage_status["built"] = True
        stage_status["sent"] = False
        stage_status["completed_at"] = None

        return self.save()

    def upsert_ai_packs_dir(self, packs_dir: Path) -> "RunManifest":
        packs, _ = self._ensure_ai_section()
        merge_paths = merge_paths_from_any(packs_dir, create=False)
        self._apply_merge_paths_to_packs(packs, merge_paths, prefer_existing_index=True)
        return self

    def set_ai_enqueued(self) -> "RunManifest":
        _, status = self._ensure_ai_section()
        status["enqueued"] = True
        status["skipped_reason"] = None
        for stage_key in ("merge", "validation"):
            stage_status = self.ensure_ai_stage_status(stage_key)
            stage_status["built"] = False
            stage_status["sent"] = False
            stage_status["completed_at"] = None
        return self.save()

    def set_ai_built(self, packs_dir: Path, pairs: int) -> "RunManifest":
        packs, status = self._ensure_ai_section()
        merge_paths = merge_paths_from_any(packs_dir, create=False)
        self._apply_merge_paths_to_packs(packs, merge_paths)
        packs["pairs"] = int(pairs)
        packs["last_built_at"] = _utc_now()
        status["built"] = True
        status["skipped_reason"] = None
        merge_status = self.ensure_ai_stage_status("merge")
        merge_status["built"] = True
        merge_status["sent"] = False
        merge_status["completed_at"] = None
        return self.save()

    def set_ai_sent(self) -> "RunManifest":
        _, status = self._ensure_ai_section()
        status["sent"] = True
        merge_status = self.ensure_ai_stage_status("merge")
        merge_status["sent"] = True
        return self.save()

    def set_ai_compacted(self) -> "RunManifest":
        _, status = self._ensure_ai_section()
        status["compacted"] = True
        merge_status = self.ensure_ai_stage_status("merge")
        merge_status["completed_at"] = _utc_now()
        return self.save()

    def set_ai_skipped(self, reason: str) -> "RunManifest":
        _, status = self._ensure_ai_section()
        status["skipped_reason"] = str(reason)
        status["built"] = False
        status["sent"] = False
        status["compacted"] = False
        for stage_key in ("merge", "validation"):
            stage_status = self.ensure_ai_stage_status(stage_key)
            stage_status["built"] = False
            stage_status["sent"] = False
            stage_status["completed_at"] = None
        return self.save()

    def update_ai_packs(
        self,
        *,
        dir: Path | str | None = None,
        index: Path | str | None = None,
        logs: Path | str | None = None,
        pairs: int | None = None,
        last_built_at: str | None = None,
    ) -> "RunManifest":
        packs, _ = self._ensure_ai_section()

        merge_paths: MergePaths | None = None
        if dir is not None:
            dir_path = Path(dir).resolve()
            try:
                merge_paths = merge_paths_from_any(dir_path, create=False)
            except ValueError:
                packs["legacy_dir"] = str(dir_path)
                packs["legacy_packs_dir"] = str(dir_path)
            else:
                self._apply_merge_paths_to_packs(
                    packs,
                    merge_paths,
                    prefer_existing_index=index is None,
                )
                packs.pop("legacy_dir", None)
                packs.pop("legacy_packs_dir", None)

        if index is not None:
            packs["index"] = str(Path(index).resolve())

        if logs is not None:
            packs["logs"] = str(Path(logs).resolve())
        elif merge_paths is not None:
            packs["logs"] = str(merge_paths.log_file)

        if pairs is not None:
            packs["pairs"] = int(pairs)

        if last_built_at is not None:
            packs["last_built_at"] = str(last_built_at)

        return self.save()

    def mark_validation_merge_applied(
        self,
        *,
        applied: bool = True,
        applied_at: str | None = None,
        source: str | None = None,
    ) -> "RunManifest":
        """Record that validation AI merge results were applied."""

        # NEW BEHAVIOR: For validation stage we no longer write legacy
        # merge_results_applied* or merge_results payload. We keep minimal
        # monotonic completion semantics (sent/completed/state) only.

        stage_status = self.ensure_ai_stage_status("validation")
        applied_flag = bool(applied)
        timestamp_value: str | None = None
        if applied_at is not None:
            normalized_timestamp = str(applied_at).strip()
            timestamp_value = normalized_timestamp or None
        if applied_flag and timestamp_value is None:
            timestamp_value = _utc_now()

        changed = False
        if applied_flag:
            current_state = stage_status.get("state")
            if current_state not in ("success", "error"):
                stage_status["state"] = "success"
                changed = True
            if stage_status.get("sent") is not True:
                stage_status["sent"] = True
                changed = True
            completed_value = stage_status.get("completed_at")
            if not isinstance(completed_value, str) or not completed_value.strip():
                stage_status["completed_at"] = timestamp_value or _utc_now()
                changed = True
            if bool(stage_status.get("failed")):
                stage_status["failed"] = False
                changed = True

        # Ensure packs.validation.status reflects sent/completed for UI consistency
        packs, _ = self._ensure_ai_section()
        validation_section = packs.get("validation")
        if not isinstance(validation_section, dict):
            validation_section = {}
            packs["validation"] = validation_section
        pack_status = validation_section.setdefault("status", {})
        if not isinstance(pack_status, dict):
            pack_status = {}
            validation_section["status"] = pack_status
        if applied_flag:
            if pack_status.get("sent") is not True:
                pack_status["sent"] = True
                changed = True
            pack_completed = pack_status.get("completed_at")
            if not isinstance(pack_completed, str) or not pack_completed.strip():
                pack_status["completed_at"] = timestamp_value or _utc_now()
                changed = True

        if changed:
            return self.save()
        return self

    def mark_strategy_started(self) -> "RunManifest":
        """Initialize ai.status.strategy when strategy stage begins.
        
        Mirrors the behavior of merge/validation stage initialization,
        setting up the status dict with appropriate defaults.
        
        Monotonic guarantee: Never revert terminal states (success/error) back to in_progress.
        """
        stage_status = self.ensure_ai_stage_status("strategy")
        
        # Monotonic completion: Don't revert success/error to in_progress
        current_state = stage_status.get("state")
        if current_state in ("success", "error"):
            # Already in terminal state, don't modify
            return self
        
        # Set initial state for strategy execution
        stage_status["built"] = True
        stage_status.setdefault("sent", False)
        stage_status["failed"] = False
        stage_status["state"] = current_state or "in_progress"
        stage_status["started_at"] = stage_status.get("started_at") or _utc_now()
        
        return self.save()

    def mark_strategy_completed(
        self,
        stats: dict[str, object] | None = None,
        *,
        failed: bool = False,
        state: str | None = None,
    ) -> "RunManifest":
        """Mark strategy stage as completed with timestamps and failure info.
        
        Parameters
        ----------
        stats:
            Optional dictionary with strategy execution statistics.
            Typically includes: plans_written, planner_errors, accounts_seen,
            accounts_with_openers.
        failed:
            Whether the strategy execution failed.
        state:
            Optional explicit state string. If not provided, defaults to
            "error" if failed=True, otherwise "success".
        """
        stage_status = self.ensure_ai_stage_status("strategy")
        
        # Record completion timestamp
        stage_status["completed_at"] = _utc_now()
        stage_status["failed"] = bool(failed)
        
        # Determine final state
        if state is not None:
            stage_status["state"] = str(state)
        else:
            stage_status["state"] = "error" if failed else "success"
        
        # Optionally store stats
        if isinstance(stats, dict) and stats:
            stage_status["stats"] = dict(stats)
        
        return self.save()

    def register_strategy_artifacts_for_account(
        self,
        account_id: str,
        *,
        runs_root: Path | str | None = None,
    ) -> "RunManifest":
        """Register strategy planner artifacts for an account into the manifest.
        
        Scans the account's strategy directory for bureau-specific outputs
        (plan.json, plan_wd*.json, logs.txt) and records their paths under
        artifacts.cases.accounts.<account_id>.strategy.
        
        Parameters
        ----------
        account_id:
            Account identifier (e.g., "idx-007" or "7")
        runs_root:
            Optional runs root path. If not provided, uses the manifest's
            parent directory structure to infer the run directory.
        """
        account_key = str(account_id).strip()
        if not account_key:
            raise ValueError("account_id is required")
        
        # Resolve runs_root and run_dir
        if runs_root is None:
            run_dir = self.path.parent
        else:
            runs_root_path = Path(runs_root) if isinstance(runs_root, str) else runs_root
            run_dir = runs_root_path / self.sid
        
        # Locate strategy base directory
        strategy_base = run_dir / "cases" / "accounts" / account_key / "strategy"
        
        if not strategy_base.exists():
            # No strategy artifacts yet - nothing to register
            return self
        
        # Ensure artifacts structure exists
        artifacts = self.data.setdefault("artifacts", {})
        cases = artifacts.setdefault("cases", {})
        accounts = cases.setdefault("accounts", {})
        account_block = accounts.setdefault(account_key, {})
        
        # Set account dir
        account_dir = strategy_base.parent.resolve()
        account_block.setdefault("dir", str(account_dir))
        
        # Initialize strategy block
        strategy_block = account_block.setdefault("strategy", {})
        strategy_block["dir"] = str(strategy_base.resolve())
        
        # Scan for bureau-specific directories
        for bureau_dir in strategy_base.iterdir():
            if not bureau_dir.is_dir():
                continue
            
            bureau_name = bureau_dir.name
            bureau_payload: dict[str, str] = {}
            bureau_payload["dir"] = str(bureau_dir.resolve())
            
            # Register standard artifacts
            plan_file = bureau_dir / "plan.json"
            if plan_file.exists():
                bureau_payload["plan"] = str(plan_file.resolve())
            
            log_file = bureau_dir / "logs.txt"
            if log_file.exists():
                bureau_payload["log"] = str(log_file.resolve())
            
            # Register weekday-specific plans
            for wd_file in sorted(bureau_dir.glob("plan_wd*.json")):
                key = wd_file.stem  # e.g., "plan_wd0" -> "plan_wd0"
                bureau_payload[key] = str(wd_file.resolve())
            
            strategy_block[bureau_name] = bureau_payload
        
        return self

    def get_ai_merge_paths(self) -> dict[str, Path | None]:
        """Return the resolved merge AI pack locations for this manifest.

        The returned mapping always includes canonical ``merge`` paths rooted at
        ``runs/<sid>/ai_packs/merge``.  When the manifest still references the
        legacy flat ``ai_packs`` directory, ``legacy_dir`` points to that base so
        callers can perform read-only operations without migrating data.  The
        ``index_file`` and ``log_file`` entries prefer existing files regardless
        of layout.
        """

        run_root = self.path.parent
        runs_root = run_root.parent
        canonical_paths = ensure_merge_paths(runs_root, self.sid, create=False)

        ai_section = self.data.get("ai")
        packs_section: dict[str, object] = {}
        if isinstance(ai_section, dict):
            packs_value = ai_section.get("packs")
            if isinstance(packs_value, dict):
                packs_section = packs_value

        base_value = packs_section.get("base")
        packs_value = packs_section.get("packs")
        results_value = packs_section.get("results")
        dir_value = packs_section.get("dir")
        packs_dir_value = packs_section.get("packs_dir")
        results_dir_value = packs_section.get("results_dir")
        index_value = packs_section.get("index")
        logs_value = packs_section.get("logs")
        legacy_dir_value = packs_section.get("legacy_dir")
        legacy_packs_dir_value = packs_section.get("legacy_packs_dir")

        merge_paths = canonical_paths
        legacy_dir: Path | None = None
        legacy_packs_dir: Path | None = None

        def _apply_candidate(value: object) -> None:
            nonlocal merge_paths, legacy_dir, legacy_packs_dir
            if not value:
                return
            try:
                candidate_path = Path(value)
            except (TypeError, ValueError):
                return
            try:
                merge_paths = merge_paths_from_any(candidate_path, create=False)
            except ValueError:
                resolved = candidate_path.resolve()
                if legacy_dir is None:
                    legacy_dir = resolved
                    legacy_packs_dir = resolved

        for candidate in (
            base_value,
            packs_value,
            results_value,
            dir_value,
            packs_dir_value,
            results_dir_value,
        ):
            _apply_candidate(candidate)
            if merge_paths is not canonical_paths:
                break

        base_path = merge_paths.base
        packs_dir = merge_paths.packs_dir
        results_dir = merge_paths.results_dir

        if legacy_dir is None and legacy_dir_value:
            try:
                legacy_dir = Path(legacy_dir_value).resolve()
            except (TypeError, ValueError):
                legacy_dir = None
        if legacy_packs_dir is None and legacy_packs_dir_value:
            try:
                legacy_packs_dir = Path(legacy_packs_dir_value).resolve()
            except (TypeError, ValueError):
                legacy_packs_dir = None

        index_candidates: list[Path] = []
        if index_value:
            try:
                index_candidates.append(Path(index_value).resolve())
            except (TypeError, ValueError):
                pass
        index_candidates.append(merge_paths.index_file)
        if canonical_paths.index_file not in index_candidates:
            index_candidates.append(canonical_paths.index_file)
        if legacy_dir is not None:
            index_candidates.append((legacy_dir / "index.json").resolve())

        index_file: Path | None = None
        for candidate in index_candidates:
            if candidate.exists():
                index_file = candidate
                break
        if index_file is None and index_candidates:
            index_file = index_candidates[0]

        log_candidates: list[Path] = []
        if logs_value:
            try:
                log_candidates.append(Path(logs_value).resolve())
            except (TypeError, ValueError):
                pass
        log_candidates.append(merge_paths.log_file)
        if canonical_paths.log_file not in log_candidates:
            log_candidates.append(canonical_paths.log_file)
        if legacy_dir is not None:
            log_candidates.append((legacy_dir / "logs.txt").resolve())

        log_file: Path | None = None
        for candidate in log_candidates:
            if candidate.exists():
                log_file = candidate
                break
        if log_file is None and log_candidates:
            log_file = log_candidates[0]

        paths: dict[str, Path | None] = {
            "base": base_path,
            "packs_dir": packs_dir.resolve(),
            "packs": packs_dir.resolve(),
            "results_dir": results_dir.resolve(),
            "results": results_dir.resolve(),
            "index_file": index_file,
            "log_file": log_file,
        }

        if legacy_dir is not None:
            paths["legacy_dir"] = legacy_dir.resolve()
            if legacy_packs_dir is not None:
                paths["legacy_packs_dir"] = legacy_packs_dir.resolve()

        return paths

    def get_ai_packs_dir(self) -> Path | None:
        paths = self.get_ai_merge_paths()
        packs_dir = paths.get("packs_dir")
        return packs_dir if isinstance(packs_dir, Path) else None

    def get_ai_index_path(self) -> Path | None:
        paths = self.get_ai_merge_paths()
        index_path = paths.get("index_file")
        return index_path if isinstance(index_path, Path) else None

    def ensure_run_subdir(self, label: str, rel: str) -> Path:
        """
        Ensure runs/<SID>/<rel> exists, register it as ``base_dirs[label]``,
        and return the absolute :class:`~pathlib.Path`.
        """

        base = (self.path.parent / rel).resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.set_base_dir(label, base)
        return base

    def get(self, group: str, key: str) -> str:
        cursor = self.data.get("artifacts", {})
        for part in group.split("."):
            cursor = cursor.get(part, {})
        if key not in cursor:
            raise KeyError(f"Missing {group}.{key} in manifest")
        return cursor[key]

    @property
    def sid(self) -> str:
        return str(self.data.get("sid"))

# -------- breadcrumbs --------
def write_breadcrumb(target_manifest: Path, breadcrumb_file: Path) -> None:
    breadcrumb_file.write_text(str(target_manifest.resolve()), encoding="utf-8")


def persist_manifest(
    manifest: RunManifest,
    *,
    artifacts: Mapping[str, Mapping[str, Path | str | int]] | None = None,
    inputs: Mapping[str, Path | str] | None = None,
) -> RunManifest:
    """Persist ``manifest`` after applying artifact path updates.

    CRITICAL: This function now reloads manifest from disk before applying changes
    to prevent clobbering validation natives or other concurrent updates.

    Parameters
    ----------
    manifest:
        The manifest instance (used only to derive sid/path). WILL BE RELOADED FROM DISK.
    artifacts:
        Optional mapping of artifact groups to update before saving.  Values are
        converted to absolute paths.
    inputs:
        Optional mapping of input paths to update.
    """
    
    # Derive sid and runs_root from the stale manifest
    sid = getattr(manifest, "sid", None)
    if not sid or sid == "None":
        sid = manifest.path.parent.name
    runs_root = manifest.path.parent.parent
    
    # Define mutation function for save_manifest_to_disk
    def _apply_updates(fresh_manifest: RunManifest) -> None:
        if inputs:
            inputs_section = fresh_manifest._ensure_inputs_section()
            for key, value in inputs.items():
                try:
                    resolved = Path(value).resolve()
                except Exception:
                    resolved = Path(str(value)).resolve()
                inputs_section[str(key)] = str(resolved)

        if not artifacts:
            return
        
        manifest_artifacts = fresh_manifest.data.setdefault("artifacts", {})
        for group, entries in artifacts.items():
            if group == "ai_packs":
                packs_updates = dict(entries)
                packs, _ = fresh_manifest._ensure_ai_section()

                base_value = packs_updates.pop("base", None)
                dir_value = packs_updates.pop("dir", None)
                packs_value = packs_updates.pop("packs", None)
                packs_dir_value = packs_updates.pop("packs_dir", None)
                results_value = packs_updates.pop("results", None)
                results_dir_value = packs_updates.pop("results_dir", None)

                index_value_raw = packs_updates.pop("index", None)
                logs_value_raw = packs_updates.pop("logs", None)
                pairs_value = packs_updates.pop("pairs", None)
                pairs_count_value = packs_updates.pop("pairs_count", None)
                last_built_value = packs_updates.pop("last_built_at", None)

                merge_paths: MergePaths | None = None
                candidate_pairs = [
                    ("base", base_value),
                    ("packs", packs_value),
                    ("results", results_value),
                    ("dir", dir_value),
                    ("packs_dir", packs_dir_value),
                    ("results_dir", results_dir_value),
                ]

                prefer_existing_index = True

                legacy_dir_resolved: Path | None = None
                for key, candidate in candidate_pairs:
                    if candidate is None or str(candidate) == "":
                        continue
                    try:
                        merge_paths = merge_paths_from_any(Path(candidate), create=False)
                    except ValueError:
                        if key in {"base", "dir", "packs", "packs_dir"} and legacy_dir_resolved is None:
                            try:
                                legacy_dir_resolved = Path(candidate).resolve()
                            except (TypeError, ValueError):
                                legacy_dir_resolved = None
                        continue

                    prefer_existing_index = index_value_raw is None or str(index_value_raw) == ""
                    manifest._apply_merge_paths_to_packs(
                        packs,
                        merge_paths,
                        prefer_existing_index=prefer_existing_index,
                    )
                    packs.pop("legacy_dir", None)
                    packs.pop("legacy_packs_dir", None)
                    break

                if legacy_dir_resolved is not None:
                    packs["legacy_dir"] = str(legacy_dir_resolved)
                    packs["legacy_packs_dir"] = str(legacy_dir_resolved)

                index_value: Path | None = None
                if index_value_raw is not None and str(index_value_raw) != "":
                    index_value = Path(index_value_raw).resolve()

                if index_value is not None:
                    packs["index"] = str(index_value)
                elif packs.get("dir") and not packs.get("index"):
                    packs["index"] = str((Path(packs["dir"]) / "index.json").resolve())

                if logs_value_raw is not None and str(logs_value_raw) != "":
                    packs["logs"] = str(Path(logs_value_raw).resolve())
                elif merge_paths is not None:
                    packs["logs"] = str(merge_paths.log_file)

                if pairs_value is not None:
                    packs["pairs"] = int(pairs_value)

                if pairs_count_value is not None:
                    packs["pairs"] = int(pairs_count_value)

                if last_built_value is not None:
                    packs["last_built_at"] = str(last_built_value)

                for extra_key, extra_value in packs_updates.items():
                    packs[str(extra_key)] = extra_value

                continue

        cursor = manifest_artifacts
        for part in str(group).split("."):
            cursor = cursor.setdefault(part, {})
        for key, value in entries.items():
            cursor[str(key)] = str(Path(value).resolve())
    
    # Use the new disk-first API
    return save_manifest_to_disk(
        runs_root,
        sid,
        _apply_updates,
        caller="backend.pipeline.runs.persist_manifest",
    )
def require_pdf_for_sid(sid: str) -> Path:
    """Return the manifest-managed report PDF path for ``sid``.

    Raises ``FileNotFoundError`` with a descriptive message when the run lacks
    a recorded input PDF or when the file is missing or empty. Logs a
    diagnostic line that downstream tests assert against.
    """

    manifest = RunManifest.for_sid(str(sid), allow_create=False)

    try:
        inputs_accessor = manifest.inputs
    except AttributeError:
        inputs_accessor = None

    try:
        pdf_path = inputs_accessor.report_pdf if inputs_accessor else None
    except AttributeError:
        pdf_path = None

    if not pdf_path:
        run_dir = _runs_root() / str(sid)
        uploads_dir = run_dir / "uploads"
        candidates = sorted(uploads_dir.glob("*.pdf"))
        if not candidates:
            logger.error(
                "RUN_PDF_MISSING sid=%s reason=recover_fail uploads_dir=%s",
                sid,
                str(uploads_dir),
            )
            raise FileNotFoundError(f"No PDF found under {uploads_dir}")
        pdf_candidate = candidates[0]
        try:
            manifest.inputs.report_pdf = pdf_candidate
        except AttributeError:
            # If accessor still unusable, fall back to manifest data dict
            manifest._ensure_inputs_section()["report_pdf"] = str(pdf_candidate)
        pdf_path = Path(pdf_candidate)

    if pdf_path is None:
        logger.error("RUN_PDF_MISSING sid=%s reason=no_manifest_input", sid)
        raise FileNotFoundError(
            f"run {sid} is missing inputs.report_pdf in the manifest"
        )

    resolved = Path(pdf_path).resolve()
    try:
        size = resolved.stat().st_size
    except FileNotFoundError as exc:
        logger.error(
            "RUN_PDF_MISSING sid=%s reason=file_missing path=%s",
            sid,
            str(resolved),
        )
        raise FileNotFoundError(
            f"report_pdf for run {sid} not found at {resolved}"
        ) from exc

    if size <= 0:
        logger.error(
            "RUN_PDF_INVALID sid=%s reason=empty path=%s",
            sid,
            str(resolved),
        )
        raise FileNotFoundError(
            f"report_pdf for run {sid} is empty at {resolved}"
        )

    logger.info(
        "STAGE_A_INPUT sid=%s pdf=%s exists=True size=%d",
        sid,
        str(resolved),
        size,
    )
    return resolved


def _run_dir_for_sid(sid: str, runs_root: Path | str | None = None) -> Path:
    base = Path(runs_root) if runs_root is not None else get_runs_root()
    return Path(base).resolve() / str(sid)


def _load_runflow_payload(
    sid: str, *, runs_root: Path | str | None = None
) -> Mapping[str, Any] | None:
    run_dir = _run_dir_for_sid(sid, runs_root)
    runflow_path = run_dir / "runflow.json"
    try:
        raw = runflow_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning(
            "RUNS_RUNFLOW_LOAD_FAILED sid=%s path=%s", sid, runflow_path, exc_info=True
        )
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "RUNS_RUNFLOW_INVALID sid=%s path=%s", sid, runflow_path, exc_info=True
        )
        return None

    if isinstance(payload, Mapping):
        return payload
    return None


def get_stage_status(
    sid: str,
    *,
    stage: str,
    runs_root: Path | str | None = None,
) -> str | None:
    """Return the lower-cased runflow status for ``stage`` if available."""

    payload = _load_runflow_payload(sid, runs_root=runs_root)
    if not isinstance(payload, Mapping):
        return None

    stages = payload.get("stages")
    if not isinstance(stages, Mapping):
        return None

    stage_payload = stages.get(str(stage))
    if not isinstance(stage_payload, Mapping):
        return None

    status = stage_payload.get("status")
    if isinstance(status, str):
        normalized = status.strip().lower()
        return normalized or None
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            text = str(value).strip()
        except Exception:
            return None
        if not text:
            return None
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return None


def _extract_stage_counts(mapping: Mapping[str, Any] | None) -> tuple[int | None, int, int]:
    if not isinstance(mapping, Mapping):
        return (None, 0, 0)
    total = (
        _coerce_int(mapping.get("results_total"))
        or _coerce_int(mapping.get("packs_total"))
        or _coerce_int(mapping.get("total"))
    )
    completed = _coerce_int(mapping.get("completed")) or 0
    failed = _coerce_int(mapping.get("failed")) or 0
    return (total, completed, failed)


def account_result_ready(
    sid: str,
    acc_id: str,
    *,
    runs_root: Path | str | None = None,
) -> bool:
    """Return ``True`` when the note_style result for ``acc_id`` is readable JSON."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        return False

    normalized_account = normalize_note_style_account_id(acc_id)
    if not normalized_account:
        return False
    snapshot = _note_style_stage_snapshot(sid_text, runs_root=runs_root)
    terminal_accounts = snapshot.packs_completed | snapshot.packs_failed
    return normalized_account in terminal_accounts


def all_note_style_results_terminal(
    sid: str, *, runs_root: Path | str | None = None
) -> bool:
    """Return ``True`` when all expected note_style results are present."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        return False
    snapshot = _note_style_stage_snapshot(sid_text, runs_root=runs_root)
    if not snapshot.packs_expected:
        return True

    terminal_accounts = snapshot.packs_completed | snapshot.packs_failed
    return snapshot.packs_expected.issubset(terminal_accounts)

