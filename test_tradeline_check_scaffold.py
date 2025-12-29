#!/usr/bin/env python3
"""Manual test script for tradeline_check scaffold.

Usage:
    python test_tradeline_check_scaffold.py [--enable] [--debug]

The script creates a minimal test run and validates outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def setup_test_env(enable: bool = True, debug: bool = False) -> Path:
    """Create a minimal test directory structure and set env flags.

    Returns
    -------
    Path
        Path to temporary test runs directory
    """
    temp_runs_root = Path(tempfile.mkdtemp(prefix="tradeline_test_"))
    log.info("Test runs root: %s", temp_runs_root)

    # Set environment flags
    os.environ["RUNS_ROOT"] = str(temp_runs_root)
    os.environ["TRADELINE_CHECK_ENABLED"] = "1" if enable else "0"
    os.environ["TRADELINE_CHECK_WRITE_DEBUG"] = "1" if debug else "0"

    return temp_runs_root


def create_test_account(
    runs_root: Path,
    sid: str = "test-sid-001",
    account_key: str = "idx-007",
    bureaus: list[str] | None = None,
) -> tuple[Path, Path]:
    """Create minimal account directory structure for testing.

    Returns
    -------
    tuple[Path, Path]
        (account_dir, bureaus_path)
    """
    if bureaus is None:
        bureaus = ["equifax", "experian", "transunion"]

    accounts_dir = runs_root / sid / "cases" / "accounts"
    account_dir = accounts_dir / account_key
    account_dir.mkdir(parents=True, exist_ok=True)

    # Create meta.json
    meta_path = account_dir / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "account_id": account_key,
                "account_index": 7,
                "heading_guess": "ACME Bank Account",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Create summary.json (minimal)
    summary_path = account_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "findings": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Create bureaus.json with all requested bureaus
    bureaus_path = account_dir / "bureaus.json"
    bureaus_data = {bureau: {"accounts": []} for bureau in bureaus}
    bureaus_path.write_text(
        json.dumps(bureaus_data, indent=2),
        encoding="utf-8",
    )

    log.info("Created test account: %s", account_dir)
    return account_dir, bureaus_path


def test_tradeline_check_disabled(runs_root: Path) -> bool:
    """Test that no tradeline_check folder is created when disabled.

    Returns
    -------
    bool
        True if test passes
    """
    log.info("TEST: tradeline_check disabled")

    # Disable the module
    os.environ["TRADELINE_CHECK_ENABLED"] = "0"

    account_dir, _ = create_test_account(
        runs_root,
        sid="test-disabled",
        account_key="idx-001",
    )

    # Import and run
    from backend.validation.pipeline import AccountContext
    from backend.tradeline_check import run_for_account

    acc_ctx = AccountContext(
        sid="test-disabled",
        runs_root=runs_root,
        index=1,
        account_key="idx-001",
        account_id="idx-001",
        account_dir=account_dir,
        summary_path=account_dir / "summary.json",
        bureaus_path=account_dir / "bureaus.json",
    )

    result = run_for_account(acc_ctx)

    tradeline_dir = account_dir / "tradeline_check"
    if tradeline_dir.exists():
        log.error("FAIL: tradeline_check folder created when disabled")
        return False

    if result.get("status") != "disabled":
        log.error("FAIL: result status should be 'disabled', got %s", result.get("status"))
        return False

    log.info("PASS: tradeline_check disabled correctly")
    return True


def test_tradeline_check_enabled(runs_root: Path) -> bool:
    """Test that tradeline_check folder and files are created when enabled.

    Returns
    -------
    bool
        True if test passes
    """
    log.info("TEST: tradeline_check enabled")

    # Enable the module
    os.environ["TRADELINE_CHECK_ENABLED"] = "1"

    account_dir, bureaus_path = create_test_account(
        runs_root,
        sid="test-enabled",
        account_key="idx-007",
        bureaus=["equifax", "experian"],
    )

    # Import and run
    from backend.validation.pipeline import AccountContext
    from backend.tradeline_check import run_for_account

    acc_ctx = AccountContext(
        sid="test-enabled",
        runs_root=runs_root,
        index=7,
        account_key="idx-007",
        account_id="idx-007",
        account_dir=account_dir,
        summary_path=account_dir / "summary.json",
        bureaus_path=bureaus_path,
    )

    result = run_for_account(acc_ctx)

    # Validate result
    if result.get("status") != "ok":
        log.error("FAIL: result status should be 'ok', got %s", result.get("status"))
        return False

    if result.get("wrote_files") != 2:
        log.error("FAIL: should write 2 files, wrote %d", result.get("wrote_files"))
        return False

    # Validate directory structure
    tradeline_dir = account_dir / "tradeline_check"
    if not tradeline_dir.exists():
        log.error("FAIL: tradeline_check folder not created")
        return False

    # Validate individual bureau files
    expected_files = ["equifax.json", "experian.json"]
    for filename in expected_files:
        filepath = tradeline_dir / filename
        if not filepath.exists():
            log.error("FAIL: %s not created", filepath)
            return False

        try:
            payload = json.loads(filepath.read_text(encoding="utf-8"))
        except Exception as exc:
            log.error("FAIL: %s is not valid JSON: %s", filepath, exc)
            return False

        # Validate payload structure
        required_keys = {"schema_version", "generated_at", "account_key", "bureau", "status", "findings", "notes"}
        missing_keys = required_keys - set(payload.keys())
        if missing_keys:
            log.error("FAIL: %s missing keys: %s", filepath, missing_keys)
            return False

        # Validate payload values
        if payload.get("schema_version") != 1:
            log.error("FAIL: schema_version should be 1, got %s", payload.get("schema_version"))
            return False

        if payload.get("account_key") != "idx-007":
            log.error("FAIL: account_key mismatch")
            return False

        expected_bureau = filename.replace(".json", "")
        if payload.get("bureau") != expected_bureau:
            log.error("FAIL: bureau mismatch in %s", filepath)
            return False

        if payload.get("status") != "ok":
            log.error("FAIL: status should be 'ok', got %s", payload.get("status"))
            return False

        log.info("PASS: %s valid", filename)

    # Validate that unwanted files were NOT created
    unwanted_file = tradeline_dir / "transunion.json"
    if unwanted_file.exists():
        log.error("FAIL: transunion.json should not be created (not in bureaus.json)")
        return False

    log.info("PASS: tradeline_check enabled correctly")
    return True


def test_no_modification_of_existing_files(runs_root: Path) -> bool:
    """Test that tradeline_check does not modify summary.json, bureaus.json, meta.json, or tags.json.

    Returns
    -------
    bool
        True if test passes
    """
    log.info("TEST: no modification of existing files")

    os.environ["TRADELINE_CHECK_ENABLED"] = "1"

    account_dir, bureaus_path = create_test_account(
        runs_root,
        sid="test-nomod",
        account_key="idx-999",
    )

    # Create and snapshot existing files
    summary_path = account_dir / "summary.json"
    summary_original = summary_path.read_text(encoding="utf-8")

    bureaus_original = bureaus_path.read_text(encoding="utf-8")

    meta_path = account_dir / "meta.json"
    meta_original = meta_path.read_text(encoding="utf-8")

    tags_path = account_dir / "tags.json"
    tags_path.write_text(json.dumps([], indent=2), encoding="utf-8")
    tags_original = tags_path.read_text(encoding="utf-8")

    # Run tradeline_check
    from backend.validation.pipeline import AccountContext
    from backend.tradeline_check import run_for_account

    acc_ctx = AccountContext(
        sid="test-nomod",
        runs_root=runs_root,
        index=999,
        account_key="idx-999",
        account_id="idx-999",
        account_dir=account_dir,
        summary_path=summary_path,
        bureaus_path=bureaus_path,
    )

    run_for_account(acc_ctx)

    # Verify files unchanged
    files_to_check = {
        "summary.json": (summary_path, summary_original),
        "bureaus.json": (bureaus_path, bureaus_original),
        "meta.json": (meta_path, meta_original),
        "tags.json": (tags_path, tags_original),
    }

    all_pass = True
    for name, (path, original) in files_to_check.items():
        current = path.read_text(encoding="utf-8")
        if current != original:
            log.error("FAIL: %s was modified", name)
            all_pass = False
        else:
            log.info("PASS: %s unchanged", name)

    return all_pass


def test_empty_bureaus(runs_root: Path) -> bool:
    """Test behavior when bureaus.json is empty or missing.

    Returns
    -------
    bool
        True if test passes
    """
    log.info("TEST: empty bureaus list")

    os.environ["TRADELINE_CHECK_ENABLED"] = "1"

    account_dir, _ = create_test_account(
        runs_root,
        sid="test-empty",
        account_key="idx-888",
        bureaus=[],  # No bureaus
    )

    from backend.validation.pipeline import AccountContext
    from backend.tradeline_check import run_for_account

    acc_ctx = AccountContext(
        sid="test-empty",
        runs_root=runs_root,
        index=888,
        account_key="idx-888",
        account_id="idx-888",
        account_dir=account_dir,
        summary_path=account_dir / "summary.json",
        bureaus_path=account_dir / "bureaus.json",
    )

    result = run_for_account(acc_ctx)

    # Should complete OK but write no files
    if result.get("status") != "ok":
        log.error("FAIL: result status should be 'ok', got %s", result.get("status"))
        return False

    if result.get("wrote_files") != 0:
        log.error("FAIL: should write 0 files when no bureaus, wrote %d", result.get("wrote_files"))
        return False

    tradeline_dir = account_dir / "tradeline_check"
    if tradeline_dir.exists() and list(tradeline_dir.glob("*.json")):
        log.error("FAIL: no JSON files should exist in tradeline_check")
        return False

    log.info("PASS: empty bureaus handled correctly")
    return True


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test tradeline_check scaffold")
    parser.add_argument("--enable", action="store_true", help="Enable tradeline_check for all tests")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Setup test environment
    runs_root = setup_test_env(enable=args.enable, debug=args.debug)

    try:
        all_tests = [
            test_tradeline_check_disabled,
            test_tradeline_check_enabled,
            test_no_modification_of_existing_files,
            test_empty_bureaus,
        ]

        results = {}
        for test_fn in all_tests:
            try:
                results[test_fn.__name__] = test_fn(runs_root)
            except Exception as exc:
                log.exception("Test %s raised exception", test_fn.__name__)
                results[test_fn.__name__] = False

        # Summary
        log.info("\n" + "=" * 60)
        log.info("TEST SUMMARY")
        log.info("=" * 60)
        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for name, passed_flag in results.items():
            status = "✓ PASS" if passed_flag else "✗ FAIL"
            log.info("%s: %s", status, name)

        log.info("=" * 60)
        log.info("Total: %d/%d tests passed", passed, total)

        sys.exit(0 if passed == total else 1)

    finally:
        # Cleanup
        shutil.rmtree(runs_root, ignore_errors=True)
        log.info("Cleaned up test directory: %s", runs_root)


if __name__ == "__main__":
    main()
