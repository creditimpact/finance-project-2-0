import time

from backend.core.letters.router import route_accounts, select_template


def _build_accounts(n: int):
    base_ctx = {
        "bureau": "experian",
        "creditor_name": "Acme",
        "account_number_masked": "1234",
        "legal_safe_summary": "ok",
        "is_identity_theft": True,
    }
    return [("fraud_dispute", dict(base_ctx), f"sess-{i}") for i in range(n)]


def main(num: int = 100) -> None:
    accounts = _build_accounts(num)
    start = time.perf_counter()
    for tag, ctx, sid in accounts:
        select_template(tag, ctx, "candidate", sid)
    sequential = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    route_accounts(accounts, phase="candidate")
    parallel = (time.perf_counter() - start) * 1000

    print(f"sequential_ms={sequential:.2f}")
    print(f"parallel_ms={parallel:.2f}")


if __name__ == "__main__":
    main()
