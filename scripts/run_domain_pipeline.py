#!/usr/bin/env python3
"""
Run complete post-processing pipeline for a single domain.

This script orchestrates:
1. Information Gain metrics extraction
2. Plot generation

Usage:
    python3 scripts/run_domain_pipeline.py blocksworld

    python3 scripts/run_domain_pipeline.py blocksworld \
        --consolidated-dir results/consolidated_results101225 \
        --benchmarks-dir benchmarks/olam-compatible

Prerequisites:
    - InfoGain experiment results with action logs
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        return False

    print(f"SUCCESS: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete post-processing pipeline for a single domain"
    )
    parser.add_argument(
        "domain",
        type=str,
        help="Domain name (e.g., blocksworld, depots, ferry)"
    )
    parser.add_argument(
        "--consolidated-dir",
        type=str,
        default="results/consolidated_results",
        help="Parent directory for all processed results (default: results/consolidated_results)"
    )
    parser.add_argument(
        "--infogain-results",
        type=str,
        help="Directory containing InfoGain experiment results (overrides consolidated-dir)"
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=str,
        default="benchmarks/olam-compatible",
        help="Directory containing ground truth PDDL domains"
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip InfoGain metrics extraction (use existing)"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation"
    )

    args = parser.parse_args()

    domain = args.domain
    project_root = Path(__file__).parent.parent

    # Resolve paths based on consolidated-dir
    consolidated_dir = project_root / args.consolidated_dir
    infogain_output_dir = consolidated_dir / "information_gain" / domain
    benchmarks = project_root / args.benchmarks_dir / domain

    print(f"\n{'#'*60}")
    print(f"# Domain Pipeline: {domain}")
    print(f"{'#'*60}")
    print(f"Consolidated dir: {args.consolidated_dir}")
    print(f"InfoGain output: {infogain_output_dir.relative_to(project_root)}")

    # Validate benchmarks
    if not benchmarks.exists():
        print(f"ERROR: Benchmark not found: {benchmarks}")
        sys.exit(1)

    success = True

    # Step 1: Extract InfoGain metrics
    if not args.skip_metrics:
        # Determine InfoGain source directory
        if args.infogain_results:
            infogain_consolidated = args.infogain_results
        else:
            infogain_consolidated = str(consolidated_dir)

        cmd = [
            sys.executable,
            "scripts/analyze_information_gain_metrics.py",
            "--consolidated-dir", infogain_consolidated,
            "--benchmarks-dir", args.benchmarks_dir,
            "--output-dir", str(infogain_output_dir.parent),  # Parent because script adds domain
            "--domain", domain
        ]
        if not run_command(cmd, f"Extract InfoGain metrics for {domain}"):
            success = False

    # Step 2: Generate plots (if results exist)
    if not args.skip_plots:
        if infogain_output_dir.exists():
            cmd = [
                sys.executable,
                "scripts/visualize_paper_results.py",
            ]
            if not run_command(cmd, f"Generate plots for {domain}"):
                success = False
        else:
            print(f"WARNING: InfoGain results not found: {infogain_output_dir}")
            print("Plots will be skipped")

    # Summary
    print(f"\n{'#'*60}")
    print(f"# Pipeline Complete: {domain}")
    print(f"{'#'*60}")

    if infogain_output_dir.exists():
        print(f"\nInfoGain metrics:")
        for f in sorted(infogain_output_dir.rglob("*.json")):
            print(f"  - {f.relative_to(project_root)}")

    if success:
        print("\nAll steps completed successfully")
        return 0
    else:
        print("\nSome steps failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
