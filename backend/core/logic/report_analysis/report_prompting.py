"""Stage A adjudication constants for compatibility.

The parsing-time LLM path has been removed. This module now only exports
version constants used by downstream reporting.
"""

from __future__ import annotations

import os

# ANALYSIS_PROMPT_VERSION history:
# 2: Add explicit JSON directive (Task 8)
# 1: Initial version
ANALYSIS_PROMPT_VERSION = 2
ANALYSIS_SCHEMA_VERSION = 1
PIPELINE_VERSION = int(os.getenv("PIPELINE_VERSION", "2"))

