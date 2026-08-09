#!/usr/bin/env python3
"""Download a small Databento historical sample with cost guardrails."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import databento as db


def _load_env_key(env_file: Path, env_name: str) -> str:
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == env_name:
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    raise SystemExit(f"{env_name} is missing from {env_file}")


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _schema_role(schema: str) -> str:
    if schema == "mbp-10":
        return "l2_snapshot_training_validation"
    if schema == "mbo":
        return "l3_event_queue_replay_smoke"
    return "historical_data_download"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="XNAS.ITCH")
    parser.add_argument("--symbol", default="GOOG")
    parser.add_argument("--schemas", nargs="+", default=["mbp-10", "mbo"])
    parser.add_argument("--start", default="2019-01-30T00:00:00Z")
    parser.add_argument("--end", default="2019-01-30T21:00:00Z")
    parser.add_argument("--stype-in", default="raw_symbol")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--env-name", default="DATABENTO_API_KEY")
    parser.add_argument("--output-root", default="data/raw/databento")
    parser.add_argument("--max-total-cost-usd", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    client = db.Historical(_load_env_key(Path(args.env_file), args.env_name))
    output_dir = (
        Path(args.output_root)
        / _safe_component(args.dataset)
        / _safe_component(args.symbol)
        / _safe_component(args.start[:10])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    planned: list[dict[str, Any]] = []
    total_cost = 0.0
    for schema in args.schemas:
        cost = client.metadata.get_cost(
            dataset=args.dataset,
            start=args.start,
            end=args.end,
            symbols=args.symbol,
            schema=schema,
            stype_in=args.stype_in,
        )
        count = client.metadata.get_record_count(
            dataset=args.dataset,
            start=args.start,
            end=args.end,
            symbols=args.symbol,
            schema=schema,
            stype_in=args.stype_in,
        )
        filename = (
            f"{_safe_component(args.symbol)}_{_safe_component(args.dataset)}_"
            f"{args.start[:10]}_{_safe_component(schema)}_"
            f"{_safe_component(args.start[11:16])}-{_safe_component(args.end[11:16])}UTC.dbn.zst"
        )
        planned.append(
            {
                "dataset": args.dataset,
                "schema": schema,
                "symbols": args.symbol,
                "stype_in": args.stype_in,
                "start": args.start,
                "end": args.end,
                "role": _schema_role(schema),
                "estimated_cost_usd": cost,
                "record_count": count,
                "path": str(output_dir / filename),
            }
        )
        total_cost += cost

    if total_cost > args.max_total_cost_usd:
        raise SystemExit(
            f"Refusing download: estimated cost ${total_cost:.6f} exceeds "
            f"--max-total-cost-usd ${args.max_total_cost_usd:.6f}"
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Databento historical API",
        "estimated_total_cost_usd": total_cost,
        "requests": [],
    }
    for req in planned:
        path = Path(req["path"])
        print(
            f"Downloading {req['schema']} cost_usd={req['estimated_cost_usd']:.12f} "
            f"record_count={req['record_count']} -> {path}"
        )
        if args.overwrite or not path.exists() or path.stat().st_size == 0:
            client.timeseries.get_range(
                dataset=req["dataset"],
                start=req["start"],
                end=req["end"],
                symbols=req["symbols"],
                schema=req["schema"],
                stype_in=req["stype_in"],
                path=path,
            )
        stat = path.stat()
        req = dict(req)
        req.update({"size_bytes": stat.st_size, "downloaded": True})
        report["requests"].append(req)

    report_path = output_dir / "download_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
