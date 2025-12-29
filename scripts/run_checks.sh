#!/usr/bin/env bash
set -euo pipefail

ruff check tests/test_architecture.py
black --check tests/test_architecture.py
# Run mypy on the typed packages
mypy backend/core/models backend/core/logic/report_analysis
python scripts/scan_public_dict_apis.py
pytest -q
