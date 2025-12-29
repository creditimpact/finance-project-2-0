import json, os, sys, pathlib, time

ROOT = pathlib.Path(r'C:\dev\credit-analyzer')
sid  = '41495514-931e-4100-90ca-4928464dcda8'
idx  = ROOT / 'runs' / sid / 'ai_packs' / 'validation' / 'index.json'
if not idx.exists():
    print('INDEX_MISSING', idx)
    sys.exit(1)

with idx.open('r', encoding='utf-8') as f:
    j = json.load(f)

packs = j.get('packs', [])
print(f'INDEX_LOADED count={len(packs)}')

for p in packs:
    print('PACK', p['account_id'], 'status=', p.get('status'), 'pack=', p.get('pack_path'))

# === כאן היית קורא לשולח האמיתי אם יש מודול אצלכם ===
# from backend.validation import sender
# sender.send_from_index(sid)  # או sender.run(sid=sid) או מה שקיים

print('DRY_RUN_OK')
