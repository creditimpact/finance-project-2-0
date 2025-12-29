"""Command line entry point for the credit repair workflow."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the credit repair process using CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the credit repair workflow")
    parser.add_argument("report", help="Path to the SmartCredit report PDF")
    parser.add_argument("email", help="Client email address")
    parser.add_argument("--goal", default="Not specified", help="Client goal")
    parser.add_argument(
        "--identity-theft",
        action="store_true",
        help="Flag the run as an identity theft case",
    )
    args = parser.parse_args()

    from backend.core.models import ClientInfo, ProofDocuments
    from backend.core.orchestrators import run_credit_repair_process

    client = ClientInfo.from_dict(
        {
            "name": "Unknown",
            "address": "Unknown",
            "email": args.email,
            "goal": args.goal,
            "session_id": "cli",
        }
    )
    proofs = ProofDocuments.from_dict({"smartcredit_report": args.report})
    run_credit_repair_process(client, proofs, args.identity_theft)


if __name__ == "__main__":  # pragma: no cover - CLI usage only
    main()
