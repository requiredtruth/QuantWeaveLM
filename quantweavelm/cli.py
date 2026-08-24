"""Command-line interface for reproducible offline experiments."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.resources import files

from .core import QuantWeaveError, atomic_json, canonical_bytes, load_json, load_jsonl
from .pipeline import run_pipeline
from .report import prompt, summary, verify


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="quantweavelm", description="Calibrate ordered market-return probability forecasts")
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="fit on calibration rows and score declared partitions")
    run.add_argument("config")
    run.add_argument("dataset")
    run.add_argument("output")
    check = commands.add_parser("verify", help="recompute and compare a report")
    check.add_argument("config")
    check.add_argument("dataset")
    check.add_argument("report")
    show = commands.add_parser("summary", help="print deterministic headline metrics")
    show.add_argument("report")
    export = commands.add_parser("prompt", help="export bounded facts for optional local commentary")
    export.add_argument("report")
    export.add_argument("output", nargs="?", default="-")
    commands.add_parser("demo", help="run the bundled synthetic experiment offline")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            report = run_pipeline(load_json(args.config), load_jsonl(args.dataset))
            atomic_json(args.output, report)
            print(f"wrote deterministic report: {args.output}")
        elif args.command == "verify":
            verify(load_json(args.config), load_jsonl(args.dataset), load_json(args.report))
            print("report verified")
        elif args.command == "summary":
            sys.stdout.write(summary(load_json(args.report)))
        elif args.command == "prompt":
            material = prompt(load_json(args.report))
            if args.output == "-":
                sys.stdout.buffer.write(canonical_bytes(material))
            else:
                atomic_json(args.output, material)
        elif args.command == "demo":
            package = files("quantweavelm.data")
            config = json.loads(package.joinpath("demo_config.json").read_text())
            rows = [json.loads(line) for line in package.joinpath("demo.jsonl").read_text().splitlines()]
            sys.stdout.write(summary(run_pipeline(config, rows)))
        return 0
    except (QuantWeaveError, OSError, KeyError, TypeError) as exc:
        print(f"quantweavelm: {exc}", file=sys.stderr)
        return 2
