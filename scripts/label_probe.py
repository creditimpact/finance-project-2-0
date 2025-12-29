import os, json, pathlib, re, sys, argparse

def mid(a,b): return (float(a)+float(b))/2.0

p = argparse.ArgumentParser()
p.add_argument("--sid", required=True)
p.add_argument("--block", type=int, required=True)
args = p.parse_args()

root = pathlib.Path("traces/blocks")/args.sid
idx  = json.loads((root/"accounts_table"/"_table_index.json").read_text(encoding="utf-8"))
entry = next((b for b in idx.get("blocks",[]) if b.get("block_id")==args.block), None)
if not entry:
    print(f"[ERR] block {args.block} not found in _table_index.json"); sys.exit(2)
tbl_path = pathlib.Path(entry["table_path"])
tbl = json.loads(tbl_path.read_text(encoding="utf-8"))
meta = tbl.get("meta") or {}
lmax = float(meta.get("label_max_x", 128.0))
eff_x_min = float(meta.get("eff_x_min", 0.0))
eff_x_max = float(meta.get("eff_x_max", 9e9))

wins = json.loads((root/"block_windows.json").read_text(encoding="utf-8"))
layout = json.loads((root/"layout_snapshot.json").read_text(encoding="utf-8"))
win_rec = next((b for b in wins.get("blocks",[]) if b.get("block_id")==args.block), None)
if not win_rec or not win_rec.get("window"):
    print(f"[ERR] missing window for block {args.block}"); sys.exit(2)
w = win_rec["window"]
page = next((p for p in layout.get("pages",[]) if int(p.get("number",0))==int(w.get("page",0))), None)
if not page:
    print(f"[ERR] page {w.get('page')} missing"); sys.exit(2)

y0, y1 = float(w["y_top"]), float(w["y_bottom"])
x0, x1 = eff_x_min, eff_x_max

cands = []
tokens = page.get("tokens") or []
for t in tokens:
    mx = mid(t["x0"], t["x1"]); my = mid(t["y0"], t["y1"])
    if not (x0 <= mx <= x1 and y0 <= my <= y1): 
        continue
    txt = (t.get("text") or "").strip()
    if txt.endswith(":"):
        cands.append((my, mx, txt))

cands.sort()
print("=== LABEL PROBE ===")
print(f"label_max_x={lmax:.1f}  eff_x=[{x0:.1f},{x1:.1f}]  y=[{y0:.1f},{y1:.1f}]")
print(f"found {len(cands)} label-like tokens ending with ':' in window")
for my, mx, txt in cands:
    side = "LEFT" if mx <= lmax else "RIGHT"
    print(f"y={my:.1f} x={mx:.1f}  {txt!r}  -> {side}-of-label_max_x")
