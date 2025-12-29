# Router Benchmark

The `scripts/benchmark_router.py` utility compares sequential routing using
`select_template` against the new parallel `route_accounts` helper.

```
$ PYTHONPATH=. LETTERS_ROUTER_PHASED=1 python scripts/benchmark_router.py 1000
sequential_ms=5.54
parallel_ms=12.90
```

On synthetic data with 1,000 accounts the parallel version is slower due to the
thread‑pool overhead. For workloads where template selection performs heavier
I/O, the batch API allows the router to take advantage of multiple CPU cores and
can provide noticeable speedups.
