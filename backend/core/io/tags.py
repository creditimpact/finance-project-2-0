"""Helper functions for managing account tags."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple


_TAG_FILENAME = "tags.json"


def _resolve_tags_path(account_dir: os.PathLike | str) -> Path:
    """Return the canonical path to ``tags.json`` for ``account_dir``."""

    path = Path(account_dir)
    if path.suffix:  # already a file path such as ``.../tags.json``
        return path
    return path / _TAG_FILENAME


def _ensure_mapping(tag: object, *, location: Path) -> dict[str, object]:
    if not isinstance(tag, MappingABC):
        raise ValueError(f"Expected mapping tag entry in {location}")
    return dict(tag)


def read_tags(account_dir: os.PathLike | str) -> List[dict[str, object]]:
    """Read tags for an account directory.

    Returns an empty list when the file does not exist.
    """

    tag_path = _resolve_tags_path(account_dir)
    if not tag_path.exists():
        return []

    with tag_path.open("r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid JSON in tags file: {tag_path}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected list in tags file: {tag_path}")

    return [
        _ensure_mapping(entry, location=tag_path) for entry in data
    ]


def write_tags_atomic(account_dir: os.PathLike | str, tags: Iterable[Mapping[str, object]]) -> None:
    """Write ``tags`` for ``account_dir`` using an atomic replace."""

    tag_path = _resolve_tags_path(account_dir)
    tag_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = [dict(tag) for tag in tags]

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(tag_path.parent), delete=False
        ) as temp_file:
            json.dump(
                serializable,
                temp_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)

        if temp_path is None:  # pragma: no cover - defensive
            raise RuntimeError("Temporary tags path was not created")

        os.replace(temp_path, tag_path)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        raise


def _build_key(tag: Mapping[str, object], unique_keys: Sequence[str]) -> Tuple[object, ...]:
    return tuple(tag.get(field) for field in unique_keys)


def upsert_tag(
    account_dir: os.PathLike | str,
    new_tag: Mapping[str, object],
    unique_keys: Sequence[str] = ("kind", "with", "source"),
) -> None:
    """Upsert ``new_tag`` keyed by ``unique_keys`` without introducing duplicates."""

    if not isinstance(new_tag, MappingABC):
        raise TypeError("new_tag must be a mapping")

    tags = read_tags(account_dir)
    tag_path = _resolve_tags_path(account_dir)

    unique: list[dict[str, object]] = []
    index_by_key: dict[Tuple[object, ...], int] = {}

    for entry in tags:
        mapping = _ensure_mapping(entry, location=tag_path)
        key = _build_key(mapping, unique_keys)
        if key in index_by_key:
            existing = unique[index_by_key[key]].copy()
            existing.update(mapping)
            unique[index_by_key[key]] = existing
        else:
            index_by_key[key] = len(unique)
            unique.append(dict(mapping))

    new_key = _build_key(new_tag, unique_keys)
    if new_key in index_by_key:
        idx = index_by_key[new_key]
        merged = unique[idx].copy()
        merged.update(new_tag)
        unique[idx] = merged
    else:
        index_by_key[new_key] = len(unique)
        unique.append(dict(new_tag))

    write_tags_atomic(account_dir, unique)
