#!/usr/bin/env python3
"""
Run complete post-processing pipeline for a single domain.

This script orchestrates:
1. OLAM results processing (raw exports -> metrics)
2. Information Gain metrics extraction
3. Comparison plot generation (OLAM vs InfoGain)

Usage:
    python scripts/run_domain_pipeline.py blocksworld

    python scripts/run_domain_pipeline.py blocksworld \\
        --raw-olam-results /path/to/olam_results/blocksworld \\
        --consolidated-dir results/consolidated_results101225 \\
        --benchmarks-dir benchmarks/olam-compatible

Prerequisites:
    - Raw OLAM exports OR processed OLAM results
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
        "--raw-olam-results",
        type=str,
        help="Path to raw OLAM exports (e.g., /path/to/olam_results/blocksworld)"
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
        "--skip-olam-processing",
        action="store_true",
        help="Skip OLAM processing step"
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip InfoGain metrics extraction (use existing)"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip comparison plot generation"
    )

    args = parser.parse_args()

    domain = args.domain
    project_root = Path(__file__).parent.parent

    # Resolve paths based on consolidated-dir
    consolidated_dir = project_root / args.consolidated_dir
    olam_output_dir = consolidated_dir / "olam" / domain
    infogain_output_dir = consolidated_dir / "information_gain" / domain

    # InfoGain source can be overridden
    if args.infogain_results:
        infogain_source_dir = project_root / args.infogain_results / "information_gain" / domain
    else:
        infogain_source_dir = consolidated_dir / "information_gain" / domain

    benchmarks = project_root / args.benchmarks_dir / domain

    print(f"\n{'#'*60}")
    print(f"# Domain Pipeline: {domain}")
    print(f"{'#'*60}")
    print(f"Consolidated dir: {args.consolidated_dir}")
    print(f"OLAM output: {olam_output_dir.relative_to(project_root)}")
    print(f"InfoGain output: {infogain_output_dir.relative_to(project_root)}")
    if args.raw_olam_results:
        print(f"Raw OLAM exports: {args.raw_olam_results}")

    # Validate benchmarks
    if not benchmarks.exists():
        print(f"ERROR: Benchmark not found: {benchmarks}")
        sys.exit(1)

    success = True

    # Step 1: Process OLAM exports (if raw results provided)
    if not args.skip_olam_processing and args.raw_olam_results:
        raw_olam_dir = Path(args.raw_olam_results)
        if raw_olam_dir.exists():
            cmd = [
                sys.executable,
                "scripts/process_olam_results.py",
                "--olam-results", str(raw_olam_dir),
                "--ground-truth", str(benchmarks / "domain.pddl"),
                "--benchmarks-dir", args.benchmarks_dir,
                "--output-dir", str(olam_output_dir)
            ]
            if not run_command(cmd, f"Process OLAM exports for {domain}"):
                success = False
        else:
            print(f"WARNING: Raw OLAM results not found: {raw_olam_dir}")

    # Step 2: Extract InfoGain metrics
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

    # Step 3: Generate comparison plots (only if both results exist)
    if not args.skip_plots:
        olam_results_parent = consolidated_dir / "olam"
        infogain_results_parent = consolidated_dir / "information_gain"

        if olam_output_dir.exists() and infogain_output_dir.exists():
            cmd = [
                sys.executable,
                "scripts/compare_algorithms_plots.py",
                "--olam-results", str(olam_results_parent.relative_to(project_root)),
                "--infogain-results", str(infogain_results_parent.relative_to(project_root)),
                "--output-dir", "results/comparison_plots",
                "--domain", domain
            ]
            # Add raw OLAM path for trace data (cumulative success plots)
            if args.raw_olam_results:
                raw_olam_parent = Path(args.raw_olam_results).parent
                cmd.extend(["--raw-olam-path", str(raw_olam_parent)])
            if not run_command(cmd, f"Generate comparison plots for {domain}"):
                success = False
        else:
            if not olam_output_dir.exists():
                print(f"WARNING: OLAM results not found: {olam_output_dir}")
            if not infogain_output_dir.exists():
                print(f"WARNING: InfoGain results not found: {infogain_output_dir}")
            print("Comparison plots will be skipped")

    # Summary
    print(f"\n{'#'*60}")
    print(f"# Pipeline Complete: {domain}")
    print(f"{'#'*60}")

    if infogain_output_dir.exists():
        print(f"\nInfoGain metrics:")
        for f in sorted(infogain_output_dir.rglob("*.json")):
            print(f"  - {f.relative_to(project_root)}")

    if olam_output_dir.exists():
        print(f"\nOLAM metrics:")
        for f in sorted(olam_output_dir.glob("domain_*.json")):
            print(f"  - {f.relative_to(project_root)}")

    plots_dir = project_root / "results/comparison_plots" / domain
    if plots_dir.exists():
        domain_plots = list(plots_dir.glob("*.png"))
        if domain_plots:
            print(f"\nGenerated plots:")
            for f in sorted(domain_plots):
                print(f"  - {f.relative_to(project_root)}")

    if success:
        print("\n✓ All steps completed successfully")
        return 0
    else:
        print("\n✗ Some steps failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
