import ast
import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent

TARGETS = [
    BASE_DIR / "orchestrators.py",
]
TARGETS += list((BASE_DIR / "logic").glob("*.py"))

SUGGESTIONS = {
    "client_info": "ClientInfo",
    "client": "ClientInfo",
    "proofs_files": "ProofDocuments",
    "account": "Account",
    "acc": "Account",
    "accounts": "Account",
    "bureau_data": "BureauPayload",
    "bureau_map": "BureauPayload",
    "payload": "BureauPayload",
    "strategy": "StrategyItem",
    "context": "LetterContext",
    "artifact": "LetterArtifact",
}

results: list[dict[str, Any]] = []

for path in TARGETS:
    mod = path.relative_to(BASE_DIR).as_posix()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            func = node.name
            # parameters
            for arg in node.args.args + node.args.kwonlyargs:
                ann = arg.annotation
                if ann is None:
                    continue
                if isinstance(ann, ast.Subscript):
                    base = ann.value
                else:
                    base = ann
                if isinstance(base, ast.Name) and base.id in {"dict", "Dict"}:
                    suggested = SUGGESTIONS.get(arg.arg, "?")
                    results.append(
                        {
                            "module": mod,
                            "function": func,
                            "signature": f"param:{arg.arg}",
                            "suggested_model": suggested,
                        }
                    )
            # return annotation
            if node.returns:
                ret = node.returns
                base = ret.value if isinstance(ret, ast.Subscript) else ret
                if isinstance(base, ast.Name) and base.id in {"dict", "Dict"}:
                    results.append(
                        {
                            "module": mod,
                            "function": func,
                            "signature": "return",
                            "suggested_model": "?",
                        }
                    )

outfile = BASE_DIR / "dict_api_inventory.json"
outfile.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(json.dumps(results, indent=2))
