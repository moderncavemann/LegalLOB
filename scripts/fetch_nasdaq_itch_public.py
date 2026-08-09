#!/usr/bin/env python3
"""Download one public Nasdaq ITCH sample file."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

try:
    from scripts.list_nasdaq_itch_public import list_public_itch
except ModuleNotFoundError:  # Direct execution adds scripts/ rather than the repo root.
    from list_nasdaq_itch_public import list_public_itch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_sha256(path: Path) -> str:
    digest = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n")
    return digest


def _download(
    url: str,
    output: Path,
    expected_size: int | None = None,
    max_segments: int = 1024,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".part")
    total = expected_size

    for segment in range(1, max_segments + 1):
        existing = tmp.stat().st_size if tmp.exists() else 0
        if expected_size is not None:
            if existing > expected_size:
                raise RuntimeError(
                    f"Partial file is larger than the listed source: {tmp} "
                    f"({existing} > {expected_size} bytes)"
                )
            if existing == expected_size:
                print(f"Partial file is already complete: {tmp}")
                tmp.replace(output)
                return

        headers = {"User-Agent": "GenLOB data preparation"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
            if segment == 1:
                print(f"Resuming at {existing / (1024 ** 3):.2f} GiB from {tmp}")

        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            status = getattr(response, "status", response.getcode())
            content_length = int(response.headers.get("content-length", 0) or 0)
            content_range = response.headers.get("content-range", "")

            if existing and status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
                if match is None or int(match.group(1)) != existing:
                    raise RuntimeError(
                        f"Invalid Content-Range while resuming {tmp}: {content_range!r}"
                    )
                mode = "ab"
                done = existing
                response_total = None if match.group(3) == "*" else int(match.group(3))
            else:
                if existing:
                    print("Server ignored the Range request; restarting the partial download.")
                mode = "wb"
                done = 0
                existing = 0
                response_total = content_length or None

            total = expected_size or response_total
            with tmp.open(mode) as f:
                while True:
                    chunk = response.read(1024 * 1024 * 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100.0 * done / total
                        print(
                            f"\rDownloaded {done / (1024 ** 3):.2f}/{total / (1024 ** 3):.2f} "
                            f"GiB ({pct:.1f}%)",
                            end="",
                        )
                        sys.stdout.flush()

        actual_size = tmp.stat().st_size
        if total is not None and actual_size == total:
            print()
            tmp.replace(output)
            return
        if actual_size <= existing:
            raise RuntimeError(
                f"Download made no progress and was retained at {tmp}: {actual_size} bytes"
            )
        if status == 200:
            raise RuntimeError(
                f"Incomplete download retained at {tmp}: "
                f"expected {total} bytes, got {actual_size}"
            )
        print(f"\nRange segment {segment} ended at {actual_size} bytes; reconnecting.")

    actual_size = tmp.stat().st_size if tmp.exists() else 0
    raise RuntimeError(
        f"Incomplete download retained at {tmp} after {max_segments} range segments: "
        f"expected {total} bytes, got {actual_size}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Example: 01302019.NASDAQ_ITCH50.gz")
    parser.add_argument("--output-dir", default="data/raw/nasdaq_itch/public")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = {row["name"]: row for row in list_public_itch()}
    if args.name not in rows:
        available = "\n".join(sorted(rows))
        raise SystemExit(f"Public ITCH file not found: {args.name}\nAvailable:\n{available}")

    output = Path(args.output_dir) / args.name
    expected_size = int(rows[args.name]["size_bytes"])
    if output.exists() and not args.force:
        actual_size = output.stat().st_size
        if actual_size != expected_size:
            raise SystemExit(
                f"Existing file has the wrong size: {output} "
                f"({actual_size} != {expected_size} bytes)"
            )
        print(f"Already exists: {output}")
        print(f"sha256: {_record_sha256(output)}")
        return 0

    print(f"Downloading {rows[args.name]['url']}")
    print(f"Output: {output}")
    _download(str(rows[args.name]["url"]), output, expected_size=expected_size)
    print(f"sha256: {_record_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
