from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from backend.telemetry.metrics import emit_counter


def _chunked(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield lists of up to ``size`` items from ``iterable``."""

    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def _write_dlq(dlq_dir: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``dlq_dir`` for later replay."""

    dlq_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{int(time.time()*1000)}_{uuid.uuid4().hex}.json"
    with open(dlq_dir / filename, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def export_accounts(
    accounts: Iterable[Mapping[str, Any]],
    worker: Callable[[Mapping[str, Any]], Any],
    *,
    chunk_size: int = 100,
    max_workers: int | None = None,
    output_queue_max: int = 100,
    dlq_dir: Path | str = Path("batch_dlq"),
) -> list[Any]:
    """Process ``accounts`` with ``worker`` using a thread pool.

    The function submits accounts in chunks to a thread pool, applying
    backpressure when the number of pending outputs exceeds ``output_queue_max``.
    Results are returned as a list in completion order.
    """

    dlq_path = Path(dlq_dir)
    start = time.perf_counter()
    jobs_total = 0
    failures_total = 0
    records_exported = 0
    results: list[Any] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map: dict[Future[Any], Mapping[str, Any]] = {}
        for chunk in _chunked(accounts, chunk_size):
            for payload in chunk:
                fut = ex.submit(worker, payload)
                future_map[fut] = payload
                jobs_total += 1
                if len(future_map) >= output_queue_max:
                    done, _ = wait(list(future_map), return_when=FIRST_COMPLETED)
                    for d in done:
                        payload_d = future_map.pop(d)
                        try:
                            results.append(d.result())
                            records_exported += 1
                        except Exception:
                            failures_total += 1
                            _write_dlq(dlq_path, payload_d)
            # Drain any completed futures before submitting next chunk
            done_now = [f for f in list(future_map) if f.done()]
            for d in done_now:
                payload_d = future_map.pop(d)
                try:
                    results.append(d.result())
                    records_exported += 1
                except Exception:
                    failures_total += 1
                    _write_dlq(dlq_path, payload_d)
        # Final drain
        for fut, payload in future_map.items():
            try:
                results.append(fut.result())
                records_exported += 1
            except Exception:
                failures_total += 1
                _write_dlq(dlq_path, payload)

    duration_ms = (time.perf_counter() - start) * 1000
    emit_counter("batch.jobs_total", jobs_total)
    emit_counter("batch.failures_total", failures_total)
    emit_counter("batch.duration_ms", duration_ms)
    emit_counter("batch.records_exported", records_exported)
    return results


__all__ = ["export_accounts"]
