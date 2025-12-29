from __future__ import annotations

from typing import TypedDict, Optional, List


class Token(TypedDict):
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    line: int
    col: Optional[int]


class Page(TypedDict):
    number: int
    width: int
    height: int
    tokens: List[Token]

