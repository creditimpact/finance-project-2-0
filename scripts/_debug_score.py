import os, json
from backend.core.logic.report_analysis.account_merge import score_all_pairs_0_100
sid = os.environ.get('SID') or 'ab6ed8a2-2abf-4a43-8935-dedfee069a69'
runs_root = os.environ.get('RUNS_ROOT', 'runs')
res = score_all_pairs_0_100(sid=sid, runs_root=runs_root)
out = os.path.join(runs_root, sid, 'merge_pairs_0_100.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print('WROTE', out)
