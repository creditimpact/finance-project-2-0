#!/usr/bin/env python3
"""Direct test of TSV v2 monthly extractor on smartcredit_report.pdf"""

import sys
import json
from pathlib import Path
from pdfplumber import open as pdf_open

# Setup paths
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress pdfplumber logs
logging.getLogger('pdfplumber').setLevel(logging.WARNING)

from backend.core.logic.report_analysis.tsv_v2_monthly_extractor import extract_tsv_v2_monthly

def main():
    """Test the extractor on the PDF."""
    pdf_path = repo_root / "runs" / "13672c5d-fd7d-4cbc-a4a8-44422c7da029" / "uploads" / "smartcredit_report.pdf"
    
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)
    
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Extract tokens from PDF (same way as split_accounts_from_tsv does)
    tokens_by_line = {}
    lines = []
    line_counter = 0
    
    try:
        with pdf_open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                logger.info(f"Processing page {page_num + 1}")
                
                # Extract text
                text = page.extract_text() or ""
                page_lines = text.split("\n")
                lines.extend(page_lines)
                
                # Extract tokens (simplified - just character positions)
                tokens = page.extract_words()
                
                for line_text in page_lines:
                    if line_text.strip():
                        # Create tokens for this line
                        line_tokens = []
                        for token in tokens:
                            # Check if token is on this line (rough approximation)
                            if token.get("text", "") in line_text:
                                line_tokens.append(token)
                        
                        if line_tokens:
                            tokens_by_line[(page_num, line_counter)] = line_tokens
                    
                    line_counter += 1
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info(f"Extracted {len(tokens_by_line)} lines with tokens")
    
    # Test on each account
    for idx, heading in enumerate(["JPMCB CARD", "STUDENT LOAN", "TRANSUNION"]):
        logger.info(f"\n=== Testing account {idx} with heading '{heading}' ===")
        
        try:
            result = extract_tsv_v2_monthly(
                session_id="13672c5d-fd7d-4cbc-a4a8-44422c7da029",
                heading=heading,
                idx=idx,
                tokens_by_line=tokens_by_line,
                lines=lines,
            )
            
            if result:
                print(f"\nAccount {idx} ({heading}) - Results:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"\nAccount {idx} ({heading}) - No results")
        
        except Exception as e:
            logger.error(f"Error processing account {idx}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
