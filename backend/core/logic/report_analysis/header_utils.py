import re


def normalize_bureau_header(text: str) -> str:
    """Normalize a header token text to a canonical bureau name.

    The input may include registered trademark symbols (®), punctuation, or
    spacing variations. This function lowercases the text, strips common
    trademark symbols and punctuation, and returns only the alpha characters.
    If the result matches one of the three bureau names it will do so in
    canonical form.
    """
    s = (text or "").lower()
    # Remove common trademark symbols
    s = s.replace("\u00ae", "").replace("\u2122", "")
    # Drop any remaining non-letter characters
    s = re.sub(r"[^a-z]", "", s)
    return s
