import os

AUTO_PURGE_AFTER_EXPORT = os.getenv("AUTO_PURGE_AFTER_EXPORT", "1") not in {"0", "false", "False"}
