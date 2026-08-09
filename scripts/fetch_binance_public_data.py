#!/usr/bin/env python3
"""Download Binance public spot klines and aggTrades archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://data.binance.vision/data"


def _daterange(start: date, end: date) -> list[date]:
    days: list[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _download(url: str, path: Path, timeout: int = 60) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path.stat().st_size
    tmp = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(path)
    return path.stat().st_size


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _url_and_path(root: Path, market: str, freq: str, data_type: str, symbol: str, day: date, interval: str) -> tuple[str, Path]:
    day_s = day.isoformat()
    if data_type == "klines":
        rel = Path(market) / freq / "klines" / symbol / interval / f"{symbol}-{interval}-{day_s}.zip"
        url = f"{BASE_URL}/{market}/{freq}/klines/{symbol}/{interval}/{symbol}-{interval}-{day_s}.zip"
    elif data_type == "aggTrades":
        rel = Path(market) / freq / "aggTrades" / symbol / f"{symbol}-aggTrades-{day_s}.zip"
        url = f"{BASE_URL}/{market}/{freq}/aggTrades/{symbol}/{symbol}-aggTrades-{day_s}.zip"
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
    return url, root / rel


def _download_checksum(url: str, zip_path: Path, timeout: int = 30) -> str | None:
    checksum_url = url + ".CHECKSUM"
    checksum_path = zip_path.with_name(zip_path.name + ".CHECKSUM")
    try:
        response = requests.get(checksum_url, timeout=timeout)
        if response.status_code != 200:
            return None
        checksum_path.write_text(response.text)
        return response.text.strip()
    except requests.RequestException:
        return None


def _sample_zip(path: Path, rows: int = 2) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if not names:
            return {"members": [], "sample_rows": []}
        with zf.open(names[0]) as f:
            text = f.read(4096).decode("utf-8", errors="replace").splitlines()
        sample_rows = []
        for line in text[:rows]:
            sample_rows.append(next(csv.reader([line])))
        return {"members": names[:5], "sample_rows": sample_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--data-types", nargs="+", default=["klines", "aggTrades"])
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--frequency", default="daily")
    parser.add_argument("--end-date", default=(date.today() - timedelta(days=1)).isoformat())
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output-root", default="data/raw/binance_public")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--max-total-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args()

    if args.frequency != "daily":
        raise SystemExit("Only --frequency daily is currently supported")
    end = date.fromisoformat(args.end_date)
    start = end - timedelta(days=args.days - 1)
    root = Path(args.output_root)

    planned: list[dict[str, Any]] = []
    total_head_bytes = 0
    for symbol in args.symbols:
        for data_type in args.data_types:
            for day in _daterange(start, end):
                url, path = _url_and_path(root, args.market, args.frequency, data_type, symbol, day, args.interval)
                head = requests.head(url, timeout=30, allow_redirects=True)
                if head.status_code != 200:
                    planned.append(
                        {
                            "symbol": symbol,
                            "data_type": data_type,
                            "date": day.isoformat(),
                            "url": url,
                            "path": str(path),
                            "available": False,
                            "status_code": head.status_code,
                        }
                    )
                    continue
                size = int(head.headers.get("content-length", "0") or 0)
                total_head_bytes += size
                planned.append(
                    {
                        "symbol": symbol,
                        "data_type": data_type,
                        "date": day.isoformat(),
                        "url": url,
                        "path": str(path),
                        "available": True,
                        "status_code": head.status_code,
                        "content_length": size,
                    }
                )

    if total_head_bytes > args.max_total_bytes:
        raise SystemExit(
            f"Refusing download: planned content length {total_head_bytes} exceeds "
            f"--max-total-bytes {args.max_total_bytes}"
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Binance public data",
        "base_url": BASE_URL,
        "market": args.market,
        "frequency": args.frequency,
        "symbols": args.symbols,
        "data_types": args.data_types,
        "interval": args.interval,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "planned_content_length": total_head_bytes,
        "files": [],
    }

    for item in planned:
        if not item["available"]:
            print(f"SKIP missing {item['url']} status={item['status_code']}")
            report["files"].append(item)
            continue
        path = Path(item["path"])
        print(f"Downloading {item['symbol']} {item['data_type']} {item['date']} {item['content_length']} bytes")
        size = _download(item["url"], path)
        checksum_text = _download_checksum(item["url"], path)
        sample = _sample_zip(path, rows=2)
        record = dict(item)
        record.update(
            {
                "downloaded": True,
                "size_bytes": size,
                "sha256": _sha256(path),
                "checksum_text": checksum_text,
                "zip_members": sample["members"],
                "sample_rows": sample["sample_rows"],
            }
        )
        report["files"].append(record)
        if args.sleep:
            time.sleep(args.sleep)

    report_dir = root / args.market / args.frequency
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"download_report_{start.isoformat()}_{end.isoformat()}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
