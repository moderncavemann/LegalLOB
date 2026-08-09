#!/usr/bin/env python3
"""List public Nasdaq ITCH sample files exposed on emi.nasdaq.com."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request


DEFAULT_URL = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"


def list_public_itch(url: str = DEFAULT_URL) -> list[dict[str, object]]:
    with urllib.request.urlopen(url, timeout=30) as response:
        html = response.read().decode("latin1")
    rows = []
    pattern = re.compile(
        r"(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+"
        r"(?P<time>\d{1,2}:\d{2}\s+[AP]M)\s+"
        r"(?P<size>\d+)\s+"
        r'<A HREF="(?P<href>[^"]+)">(?P<name>[^<]+\.gz)</A>'
    )
    for match in pattern.finditer(html):
        name = match.group("name")
        if not ("ITCH" in name or name.startswith("S")):
            continue
        rows.append(
            {
                "name": name,
                "url": "https://emi.nasdaq.com" + match.group("href"),
                "size_bytes": int(match.group("size")),
                "listed_date": match.group("date"),
                "listed_time": match.group("time"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = list_public_itch(args.url)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            gib = row["size_bytes"] / (1024 ** 3)
            print(f"{row['name']}\t{gib:.2f} GiB\t{row['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
