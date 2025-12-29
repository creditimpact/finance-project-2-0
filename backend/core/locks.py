from __future__ import annotations

"""Simple per-account locking utilities.

This module provides a lightweight in-memory locking mechanism keyed by
account id.  In a production system this would likely be backed by a
shared resource such as Redis using ``SETNX`` or a database row lock.
For our purposes an in-process ``threading.Lock`` is sufficient to
prevent concurrent planner operations for the same account.
"""

from contextlib import contextmanager
import threading
from typing import Dict

# Global map of account_id -> Lock
_locks: Dict[str, threading.Lock] = {}
_master_lock = threading.Lock()


@contextmanager
def account_lock(account_id: str):
    """Context manager acquiring a lock for the given account id."""
    with _master_lock:
        lock = _locks.setdefault(account_id, threading.Lock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
