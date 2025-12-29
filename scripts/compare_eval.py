from __future__ import annotations

import json
import sys
from pathlib import Path


def load_metrics(path: Path) -> dict[str, float]:
    return json.loads(path.read_text())


def compare(
    pre: dict[str, float], post: dict[str, float]
) -> dict[str, tuple[float, float, float]]:
    keys = sorted(set(pre) | set(post))
    diff: dict[str, tuple[float, float, float]] = {}
    for k in keys:
        pre_v = float(pre.get(k, 0))
        post_v = float(post.get(k, 0))
        diff[k] = (pre_v, post_v, post_v - pre_v)
    return diff


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: compare_eval.py pre.json post.json")
        raise SystemExit(1)
    pre = load_metrics(Path(sys.argv[1]))
    post = load_metrics(Path(sys.argv[2]))
    for metric, values in compare(pre, post).items():
        pre_v, post_v, delta = values
        print(f"{metric}: pre={pre_v} post={post_v} delta={delta}")


if __name__ == "__main__":
    main()
