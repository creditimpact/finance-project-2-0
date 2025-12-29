#!/usr/bin/env python3
"""Quick validation that tradeline_check module imports correctly."""

from __future__ import annotations

import sys

try:
    from backend.tradeline_check import run_for_account
    from backend.tradeline_check.config import TradlineCheckConfig
    from backend.tradeline_check.schema import bureau_output_template, SCHEMA_VERSION
    from backend.tradeline_check.writer import write_bureau_findings
    from backend.tradeline_check.runner import run_for_account as runner_entry
    
    print("✓ All imports successful")
    print(f"✓ Schema version: {SCHEMA_VERSION}")
    print(f"✓ Config class: {TradlineCheckConfig.__name__}")
    print(f"✓ Entry point: {run_for_account.__name__}")
    
    # Test config parsing
    cfg = TradlineCheckConfig.from_env()
    print(f"✓ Config loaded: enabled={cfg.enabled}, debug={cfg.write_debug}")
    
    # Test schema
    template = bureau_output_template("idx-007", "equifax", "2024-12-24T00:00:00Z")
    print(f"✓ Schema template keys: {list(template.keys())}")
    
    print("\n✅ All validation checks passed!")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
