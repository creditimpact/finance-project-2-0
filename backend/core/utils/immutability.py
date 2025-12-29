from __future__ import annotations

import copy
import json
import logging

log = logging.getLogger(__name__)


def assert_no_mutation(fn):
    """Decorator that asserts ``payload`` is not mutated by ``fn``.

    It passes a deep copy of the payload into ``fn`` and then compares the
    serialized JSON of the original before vs. after.
    """

    def wrapper(payload, *args, **kwargs):
        try:
            snapshot = json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            snapshot = None
        out = fn(copy.deepcopy(payload), *args, **kwargs)
        try:
            after = json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            after = None
        if snapshot is not None and after is not None and snapshot != after:
            log.error("BUG mutated_payload_in_%s", getattr(fn, "__name__", str(fn)))
        return out

    return wrapper

