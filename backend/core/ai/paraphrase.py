from __future__ import annotations

from typing import Iterable, Optional


def paraphrase(text: str, banned_terms: Iterable[str] | None = None) -> Optional[str]:
    """Return a paraphrased version of ``text``.

    This is a lightweight stub used for tests. In production this would call
    an AI service with low temperature and enforce a similarity check. If any
    banned term appears in ``text`` the paraphrase is considered failed and
    ``None`` is returned.
    """

    banned = {t.lower() for t in (banned_terms or [])}
    lowered = text.lower()
    if any(term in lowered for term in banned):
        return None
    return text


__all__ = ["paraphrase"]
