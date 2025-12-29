from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Mapping, Sequence


def _fsync_directory(directory: Path) -> None:
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        try:
            os.close(dir_fd)
        except OSError:
            pass


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.unlink()
    except FileNotFoundError:
        pass

    with tmp_path.open("w", encoding=encoding, newline="") as handle:
        handle.write(text)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass

    os.replace(tmp_path, path)
    _fsync_directory(path.parent)


def write_jsonl(path: Path, objs: Sequence[Mapping]) -> None:
    if not objs:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return

    lines = [json.dumps(obj, ensure_ascii=False) for obj in objs]
    payload = "\n".join(lines) + "\n"
    _atomic_write_text(path, payload)


def write_json(path: Path, obj: Mapping) -> None:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n"
    _atomic_write_text(path, payload)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
