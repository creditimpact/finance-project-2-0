#!/usr/bin/env python3
"""Test TSV v2 monthly extractor on the 13672c5d SID."""

import sys
import json
import logging
from pathlib import Path

# Setup paths
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)

def test_tsv_v2_monthly():
    """Test the TSV v2 monthly extractor."""
    from backend.core.logic.report_analysis.tsv_v2_monthly_extractor import extract_tsv_v2_monthly
    
    sid = "13672c5d-fd7d-4cbc-a4a8-44422c7da029"
    run_dir = repo_root / "runs" / sid
    
    # Check if accounts_from_full.json exists
    accounts_file = run_dir / "traces" / "accounts_table" / "accounts_from_full.json"
    if not accounts_file.exists():
        logger.error(f"File not found: {accounts_file}")
        return False
    
    logger.info(f"Loading accounts from {accounts_file}")
    with open(accounts_file) as f:
        accounts_data = json.load(f)
    
    accounts = accounts_data.get("accounts", [])
    logger.info(f"Found {len(accounts)} accounts")
    
    # Check if any account has the new monthly field
    has_monthly = False
    for idx, acc in enumerate(accounts):
        history = acc.get("history_out", {})
        monthly_v2 = history.get("two_year_payment_history_monthly_tsv_v2")
        
        if monthly_v2:
            has_monthly = True
            logger.info(f"Account {idx} has monthly v2: {json.dumps(monthly_v2, ensure_ascii=False, indent=2)}")
    
    if not has_monthly:
        logger.warning("No accounts found with monthly v2 field")
        logger.info("Field may not have been generated yet. Check if Stage-A needs to be re-run.")
    
    return True

if __name__ == "__main__":
    test_tsv_v2_monthly()
