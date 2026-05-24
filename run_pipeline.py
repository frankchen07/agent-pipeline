#!/usr/bin/env python3
"""
Advisor corpus pipeline entry point.

Usage:
    python run_pipeline.py --config config/ramit-sethi.yaml
    python run_pipeline.py --config config/ramit-sethi.yaml --from-stage 4
    python run_pipeline.py --config config/ramit-sethi.yaml --only-stage 5
    python run_pipeline.py --config config/ramit-sethi.yaml --force
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Advisor corpus pipeline")
    parser.add_argument("--config", default="config/ramit-sethi.yaml", help="Config file path")
    parser.add_argument("--from-stage", type=int, default=1, metavar="N", help="Start from stage N (1-7)")
    parser.add_argument("--only-stage", type=int, metavar="N", help="Run only stage N")
    parser.add_argument("--force", action="store_true", help="Re-process even unchanged sources")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Validate sources directory exists
    sources_dir = Path(cfg["paths"]["sources"])
    if not sources_dir.exists():
        print(f"ERROR: Sources directory not found: {sources_dir}", file=sys.stderr)
        print("Create it and add your source files.", file=sys.stderr)
        sys.exit(1)

    from src.pipeline import run
    run(
        cfg=cfg,
        from_stage=args.from_stage,
        only_stage=args.only_stage,
        force=args.force,
    )

    output_dir = Path(cfg["paths"]["output"])
    print(f"\nDone. Outputs in: {output_dir.resolve()}")
    print(f"Upload folder: {(output_dir / 'project_upload').resolve()}")


if __name__ == "__main__":
    main()
