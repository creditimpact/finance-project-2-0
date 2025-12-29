import os, sys, json, traceback
from pathlib import Path

root = Path(r"c:\dev\credit-analyzer")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

os.environ.setdefault("ENABLE_AUTO_AI_PIPELINE", "1")
os.environ.setdefault("AUTO_AI_QUEUE_ON_NO_CANDIDATES", "1")

sid = "cf3140e9-4515-4461-b93e-ec024a6c94c0"
runs_root = Path(r"c:\dev\credit-analyzer\runs")

try:
    from backend.pipeline.auto_ai import maybe_queue_auto_ai_pipeline
    result = maybe_queue_auto_ai_pipeline(
        sid,
        runs_root=runs_root,
        flag_env=os.environ,
        force=True,
        inflight_ttl_seconds=0,
    )
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
except Exception as e:
    print(json.dumps({
        "ok": False,
        "error": str(e),
        "type": e.__class__.__name__,
        "trace": traceback.format_exc(limit=3)
    }, ensure_ascii=False))
