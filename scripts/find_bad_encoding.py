from pathlib import Path

bad = []
for p in Path(".").rglob("*.py"):
    try:
        # ×›×ž×• ×"×˜×¡×˜: ×œ×œ× ×¦×™×•×Ÿ encoding
        _ = p.read_text()
    except UnicodeDecodeError as e:
        bad.append((str(p), str(e)))
if not bad:
    print("OK: no undecodable .py files")
else:
    for path, err in bad:
        print(path, "->", err)
