from pathlib import Path

# ×ž×™×¤×•×™ ×ª×•×•×™× "×'×¢×™×™×ª×™×™×" -> ×ª×•×•×™ ASCII ×¤×©×•×˜×™×
REPL = {
    # ×ž×¨×›××•×ª ×-×›×ž×•×ª
    "\u2018": "'",  # â€˜
    "\u2019": "'",  # â€™
    "\u201C": '"',  # â€œ
    "\u201D": '"',  # â€
    # ×ž×§×¤×™×
    "\u2013": "-",  # â€"
    "\u2014": "-",  # â€"
    # ××œ×™×¤×¡×™×¡
    "\u2026": "...",
    # ×¨×•×•×-×™×/×'×§×¨×•×ª × ×"×™×¨×•×ª â€" ×ž×¡×™×¨×™×/×ž×ž×™×¨×
    "\u00A0": " ",  # nbsp
    "\u200B": "",  # zero-width space
    "\u0081": "",
    "\u008D": "",
    "\u008F": "",
    "\u0090": "",
    "\u009D": "",
}

BAD_BYTES = (0x81, 0x8D, 0x8F, 0x90, 0x9D)


def cleanse_text(s: str) -> str:
    for k, v in REPL.items():
        s = s.replace(k, v)
    return s


def decode_best_effort(b: bytes) -> str:
    # × ×¤×˜×¨×™× ×ž×'×™×™×˜×™× ×'×¢×™×™×ª×™×™× ×ž×¨××©
    for bad in BAD_BYTES:
        b = b.replace(bytes([bad]), b"")
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("latin-1", errors="ignore")


def process_file(p: Path) -> bool:
    raw = p.read_bytes()
    text = decode_best_effort(raw)
    cleaned = cleanse_text(text)
    # ×ª×ž×™×" ×›×•×ª×'×™× ×-×-×¨×" ×'-UTF-8 ×¢× ×©×•×¨×•×ª \n ×›×"×™ ×œ×™×™×©×¨ ××ª ×"×§×™×"×•×" ×¡×•×¤×™×ª
    p.write_text(cleaned, encoding="utf-8", newline="\n")
    return True


def main():
    changed = 0
    scanned = 0
    for p in Path(".").rglob("*.py"):
        parts = {x.lower() for x in p.parts}
        if "venv" in parts or ".venv" in parts:
            continue
        scanned += 1
        try:
            if process_file(p):
                print(f"Fixed: {p}")
                changed += 1
        except Exception as e:
            print(f"ERROR {p}: {e}")
    print(f"\nDone. Scanned {scanned} files, fixed {changed} (re-encoded UTF-8).")


if __name__ == "__main__":
    main()
