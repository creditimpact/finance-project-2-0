# Finalize Benchmark

`backend/analytics/batch_runner.py` provides a `benchmark_finalize` helper to
measure finalize routing throughput using the `route_accounts` thread pool.

```bash
$ PYTHONPATH=. LETTERS_ROUTER_PHASED=1 python -c "from backend.analytics.batch_runner import benchmark_finalize; benchmark_finalize(1000)"
throughput_lps=3169.90
```

On synthetic data with 1,000 accounts this repo's environment produced the
above throughput. Actual numbers will vary by hardware and configuration.
