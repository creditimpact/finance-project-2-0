import re
from pathlib import Path

HEX_ONLY = re.compile(r"^[0-9a-f]{7,40}\s*$", re.IGNORECASE)

hits = []
for p in Path(".").rglob("*.py"):
    try:
        s = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    if HEX_ONLY.match(s.strip()):
        hits.append(str(p))

if hits:
    print("Hash-only .py files:")
    for h in hits:
        print(h)
else:
    print("No hash-only .py files found")
