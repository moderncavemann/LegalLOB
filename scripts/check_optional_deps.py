#!/usr/bin/env python3
"""Print optional benchmark/simulator dependency availability."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def _available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> int:
    print(f"lob_bench: {'available' if _available('lob_bench') else 'missing'}")
    print(f"jaxlob: {'available' if _available('jaxlob') else 'missing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
