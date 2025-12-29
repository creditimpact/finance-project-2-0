from flask import Blueprint, jsonify, request

from backend.api.pipeline import run_full_pipeline

bp = Blueprint("smoke", __name__)


@bp.route("/run", methods=["POST"])
def smoke_run():
    """Trigger the canonical pipeline for a given SID.

    This endpoint expects JSON with a ``sid`` field and enqueues the
    Stage-A → Cleanup → Problematic pipeline for that session.
    """
    sid = request.json.get("sid") if request.is_json else None
    if not sid:
        return jsonify({"error": "sid required"}), 400
    run_full_pipeline(sid)
    return jsonify({"sid": sid, "queued": True})
