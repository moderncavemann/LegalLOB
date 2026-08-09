#!/usr/bin/env python3
"""Fail if staged/tracked files exceed a GitHub-safe size threshold."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _git_files(staged: bool) -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only"] if staged else ["git", "ls-files"]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-mb", type=float, default=90.0)
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()

    max_bytes = int(args.max_mb * 1024 * 1024)
    offenders = []
    for file_name in _git_files(staged=args.staged):
        path = Path(file_name)
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > max_bytes:
            offenders.append((file_name, size))

    if offenders:
        print(f"Files exceed {args.max_mb:.1f}MB:")
        for file_name, size in offenders:
            print(f"  {size / 1024 / 1024:.1f}MB  {file_name}")
        return 1
    print(f"No {'staged' if args.staged else 'tracked'} files exceed {args.max_mb:.1f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
