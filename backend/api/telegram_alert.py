from datetime import datetime


def send_admin_login_alert(ip: str | None = None) -> None:
    """Log admin login events locally instead of sending network requests."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"âš ï¸ Admin panel login detected! Time: {timestamp}"
    if ip:
        msg += f" | IP: {ip}"
    print(msg)
