import io
import zipfile
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from telegram_alert import send_admin_login_alert

from backend.api.config import get_app_config

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def is_authenticated():
    return session.get("admin_authenticated") is True


@admin_bp.before_request
def require_login():
    if request.endpoint == "admin.login":
        return
    if not is_authenticated():
        return redirect(url_for("admin.login"))


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password", "")
        admin_password = get_app_config().admin_password
        if admin_password and password == admin_password:
            session["admin_authenticated"] = True
            send_admin_login_alert(request.remote_addr)
            return redirect(url_for("admin.index"))
        return render_template("admin_login.html", error="Invalid password")
    return render_template("admin_login.html")


@admin_bp.route("/logout")
def logout():
    session.pop("admin_authenticated", None)
    return redirect(url_for("admin.login"))


@admin_bp.route("/")
def index():
    clients = []
    clients_base = Path("Clients").resolve()
    if clients_base.exists():
        for path in clients_base.rglob("*"):
            if path.is_dir() and path != clients_base:
                clients.append(str(path.relative_to(clients_base)))
    analytics_dir = Path("analytics_data")
    analytics = (
        [f.name for f in analytics_dir.glob("*.json")] if analytics_dir.exists() else []
    )
    return render_template(
        "admin_index.html", clients=sorted(clients), analytics=sorted(analytics)
    )


@admin_bp.route("/download/client/<path:folder>")
def download_client(folder):
    base_dir = Path("Clients").resolve()
    target = (base_dir / folder).resolve()
    if not target.is_dir() or base_dir not in target.parents and target != base_dir:
        abort(404)
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in target.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=str(file_path.relative_to(target)))
    memory.seek(0)
    return send_file(memory, as_attachment=True, download_name=f"{target.name}.zip")


@admin_bp.route("/download/analytics/<path:filename>")
def download_analytics(filename):
    analytics_dir = Path("analytics_data").resolve()
    target = (analytics_dir / filename).resolve()
    if not target.is_file() or target.parent != analytics_dir:
        abort(404)
    return send_file(target, as_attachment=True)
