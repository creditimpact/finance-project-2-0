"""Quick import sanity checks for core logic packages."""

import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = [
    "backend.core.logic.report_analysis.analyze_report",
    "backend.core.logic.strategy.strategy_merger",
    "backend.core.logic.letters.goodwill_preparation",
    "backend.core.logic.rendering.pdf_renderer",
    "backend.core.logic.compliance.rule_checker",
    "backend.core.logic.utils.json_utils",
    "backend.core.logic.guardrails.summary_validator",
]


def main() -> int:
    try:
        for name in MODULES:
            if name.startswith("backend.core.logic.guardrails."):
                file = (
                    ROOT
                    / "backend"
                    / "core"
                    / "logic"
                    / "guardrails"
                    / (name.rsplit(".", 1)[-1] + ".py")
                )
                SourceFileLoader(name, str(file)).load_module()
            else:
                __import__(name)
    except Exception as exc:  # pragma: no cover - diagnostic output
        print(f"import failed: {name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
