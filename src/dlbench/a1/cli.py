"""Thin command interface. No automatic data download or training at import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dlbench.common.config import ConfigError, load_config, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A1 image-classification experiments and configuration checks.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-config", "prepare", "train", "evaluate"):
        child = commands.add_parser(command)
        child.add_argument("--config", type=Path, required=True, help="Model Python CONFIG file; run from repository root.")
        child.add_argument("--seed", type=int, help="Override run seed using the agreed set; never changes the split.")
        if command == "validate-config":
            child.add_argument("--strict", action="store_true", help="Reject unresolved experiment decisions.")
        elif command == "train":
            child.add_argument("--smoke", action="store_true", help="Explicit non-benchmark debug run.")
        elif command == "evaluate":
            child.add_argument("--checkpoint", type=Path, required=True)
            child.add_argument("--split", choices=("validation", "test"), default="validation")
            child.add_argument("--allow-test", action="store_true", help="Acknowledge frozen final-test evaluation.")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            from dlbench.a1.analysis import analyze_run
            analyze_run(args.run_dir)
            return 0

        config = load_config(args.config)
        if args.seed is not None:
            config["run"]["seed"] = args.seed
        if args.command == "validate-config":
            unresolved = validate_config(config, strict=args.strict)
            print(json.dumps({
                "model": config["model"]["name"],
                "protocol": config["protocol"]["id"],
                "run_seed": config["run"]["seed"],
                "run_seeds": config["budget"]["run_seeds"],
                "split_seed": config["data"]["split_seed"],
                "config_decisions_ready": not unresolved,
                "unresolved": unresolved,
                "note": "Config check only; no verification of data files, hardware or ML implementation.",
            }, indent=2))
            return 0

        strict = args.command == "train" and not args.smoke
        strict = strict or (args.command == "evaluate" and args.split == "test")
        validate_config(config, strict=strict)
        if args.command == "prepare":
            from dlbench.a1.data.dataset import prepare_data
            print(json.dumps(prepare_data(config), indent=2))
        elif args.command == "train":
            from dlbench.a1.trainer import fit
            result = fit(config, smoke=args.smoke)
            print(f"Completed run: {result.run_dir}")
        elif args.command == "evaluate":
            if args.split == "test" and not args.allow_test:
                raise ConfigError("Official test requires --allow-test after freezing model selection.")
            from dlbench.a1.trainer import evaluate_checkpoint
            result = evaluate_checkpoint(config, args.checkpoint, split=args.split)
            print(result.metrics)
        return 0
    except (ConfigError, OSError, ValueError, NotImplementedError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
