#!/usr/bin/env python
"""
Run Information Gain experiments on PDDL benchmarks.

This script supports different experiment scales:
- quick: 5 simple domains, 1 problem each, 100 iterations
- standard: 10 domains, 3 problems each, 400 iterations
- full: All 23 domains, 3 problems each, 400 iterations
- custom: Specify domains, problems, and iterations

Usage:
    # Quick test (5 minutes)
    python scripts/run_full_experiments.py --mode quick

    # Standard paper experiments (2-4 hours)
    python scripts/run_full_experiments.py --mode standard

    # Full benchmark suite (8-12 hours)
    python scripts/run_full_experiments.py --mode full

    # Custom selection
    python scripts/run_full_experiments.py --domains blocksworld hanoi --problems p00 p01 --iterations 200

    # Resume from specific point
    python scripts/run_full_experiments.py --mode standard --resume-from "rover/p01"
"""

import argparse
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.runner import ExperimentRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Experiment modes
# NOTE: gripper domain excluded due to malformed PDDL type hierarchy
# (declares 'object' as custom type instead of using built-in root type)
# This causes infinite recursion in the Unified Planning PDDL parser
EXPERIMENT_MODES = {
    "quick": {
        "domains": ["blocksworld", "hanoi", "ferry", "miconic", "depots"],
        "problems": ["p00"],
        "iterations": 100,
        "description": "Quick test with simple domains"
    },
    "standard": {
        "domains": [
            "blocksworld", "depots", "driverlog", "ferry",
            "hanoi", "miconic", "n-puzzle", "satellite"
        ],
        "problems": ["p00", "p01", "p02", "p03", "p04",
                     "p05", "p06", "p07", "p08", "p09"],
        "iterations": 500,
        "description": "Standard benchmark set for papers"
    },
    "full": {
        "domains": [
            "barman", "blocksworld", "depots", "driverlog", "elevators",
            "ferry", "floortile", "gold-miner", "grid",
            "hanoi", "matching-bw", "miconic", "n-puzzle", "nomystery",
            "parking", "rover", "satellite", "sokoban", "spanner",
            "tpp", "transport", "zenotravel"
        ],
        "problems": ["p00", "p01", "p02", "p03", "p04",
                    "p05", "p06", "p07", "p08", "p09"],
        "iterations": 500,
        "description": "Complete infogain benchmark suite (excluding gripper)"
    }
}

def create_experiment_config(algorithm: str, domain: str, problem: str,
                           iterations: int, output_dir: str,
                           use_object_subset: bool = True) -> Dict[str, Any]:
    """Create experiment configuration dynamically."""
    config: Dict[str, Any] = {'experiment': {
        'name': f"{algorithm}_{domain}_{problem}",
        'algorithm': algorithm,
        'seed': 42 + hash(f"{domain}_{problem}") % 100
    }, 'domain_problem': {
        'domain': f"benchmarks/olam-compatible/{domain}/domain.pddl",
        'problem': f"benchmarks/olam-compatible/{domain}/{problem}.pddl"
    }, 'metrics': {
        'interval': 5,
        'window_size': 50
    }, 'stopping_criteria': {
        'max_iterations': iterations,
        'max_runtime_seconds': 3600,  # 1 hour max per experiment
        'convergence_check_interval': 50
    }, 'output': {
        'directory': output_dir,
        'formats': ['csv', 'json'],
        'save_learned_model': True
    }, 'algorithm_params': {
        'information_gain': {
            'selection_strategy': 'greedy',
            'max_iterations': iterations,
            'model_stability_window': min(50, iterations // 10),
            'info_gain_epsilon': 0.001,
            'success_rate_threshold': 0.98,
            'success_rate_window': min(50, iterations // 10),
            'use_object_subset': use_object_subset
        }
    }}

    # Algorithm-specific parameters

    return config

def get_all_problems(domain: str) -> List[str]:
    """Get all problem files for a domain."""
    domain_dir = Path(f"benchmarks/olam-compatible/{domain}")
    problem_files = sorted(domain_dir.glob("p*.pddl"))
    return [p.stem for p in problem_files if p.stem != "domain"]


def check_domain_availability(domain: str) -> Tuple[bool, str]:
    """Check if domain files are available and valid."""
    domain_dir = Path(f"benchmarks/olam-compatible/{domain}")
    domain_file = domain_dir / "domain.pddl"

    if not domain_dir.exists():
        return False, f"Domain directory not found: {domain_dir}"

    if not domain_file.exists():
        return False, f"Domain file not found: {domain_file}"

    # Check for at least one problem file
    problem_files = list(domain_dir.glob("p*.pddl"))
    if not problem_files:
        return False, f"No problem files found in {domain_dir}"

    return True, f"Domain {domain} OK ({len(problem_files)} problems)"

def run_single_experiment(domain: str, problem: str,
                         iterations: int, base_output_dir: str,
                         force: bool = False,
                         use_object_subset: bool = True) -> bool:
    """Run a single experiment."""
    output_dir = f"{base_output_dir}/{domain}/{problem}"

    # Check if already completed
    summary_file = Path(output_dir) / "experiment_summary.json"
    if summary_file.exists() and not force:
        logger.info(f"  Skipping (already completed): {summary_file}")
        return True

    # Check domain availability
    domain_ok, msg = check_domain_availability(domain)
    if not domain_ok:
        logger.warning(f"  {msg}")
        return False

    # Create configuration
    config = create_experiment_config("information_gain", domain, problem, iterations, output_dir, use_object_subset)

    # Save config
    config_path = Path(output_dir)
    config_path.mkdir(parents=True, exist_ok=True)
    config_file = config_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    try:
        logger.info(f"  Starting: information_gain/{domain}/{problem} ({iterations} iterations)")
        start_time = time.time()

        # Run experiment
        runner = ExperimentRunner(str(config_file))
        results = runner.run_experiment()

        elapsed = time.time() - start_time
        logger.info(f"  Completed in {elapsed:.1f}s")
        return True

    except Exception as e:
        logger.error(f"  Failed: {e}")
        logger.debug(traceback.format_exc())

        # Save error info
        error_file = config_path / "error.txt"
        with open(error_file, 'w') as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())

        return False

def estimate_runtime(num_experiments: int, iterations: int) -> str:
    """Estimate total runtime."""
    # Rough estimates based on experience
    seconds_per_iteration = 0.1  # Conservative estimate
    seconds_per_experiment = iterations * seconds_per_iteration
    total_seconds = num_experiments * seconds_per_experiment

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    return f"{hours:.0f}h {minutes:.0f}m"

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test (5 minutes):
    python scripts/run_full_experiments.py --mode quick

  Standard benchmarks (2-4 hours):
    python scripts/run_full_experiments.py --mode standard

  Custom selection:
    python scripts/run_full_experiments.py --domains blocksworld hanoi --iterations 200

  Resume from failure:
    python scripts/run_full_experiments.py --mode standard --resume-from "rover/p01"
        """
    )

    parser.add_argument("--mode", choices=list(EXPERIMENT_MODES.keys()),
                       help="Predefined experiment mode")
    parser.add_argument("--domains", nargs='+',
                       help="Custom list of domains")
    parser.add_argument("--problems", nargs='+',
                       default=["p00", "p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09"],
                       help="Problems to run (default: p00-p09)")
    parser.add_argument("--all-problems", action="store_true",
                       help="Auto-detect and run all problems in each domain")
    parser.add_argument("--iterations", type=int, default=400,
                       help="Number of iterations per experiment")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory (default: results/paper/comparison_TIMESTAMP)")
    parser.add_argument("--resume-from", type=str,
                       help="Resume from domain/problem (e.g., 'rover/p01')")
    parser.add_argument("--force", action="store_true",
                       help="Force re-run even if results exist")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be run without executing")
    parser.add_argument("--use-object-subset", action="store_true", default=False,
                       dest="use_object_subset",
                       help="Enable object subset selection (default: True)")
    parser.add_argument("--no-object-subset", action="store_false",
                       dest="use_object_subset",
                       help="Disable object subset selection")

    args = parser.parse_args()

    # Determine experiment configuration
    if args.mode:
        mode_config = EXPERIMENT_MODES[args.mode]
        domains = mode_config["domains"]
        problems = mode_config["problems"]
        iterations = mode_config["iterations"]
        logger.info(f"Running '{args.mode}' mode: {mode_config['description']}")
    elif args.domains:
        domains = args.domains
        problems = args.problems
        iterations = args.iterations
        logger.info(f"Running custom configuration")
    else:
        parser.error("Either --mode or --domains must be specified")

    # Set output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir) / "information_gain"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = f"results/comparison_{timestamp}/information_gain"

    logger.info(f"Output directory: {base_output_dir}")

    # Handle --all-problems flag
    use_all_problems = args.all_problems

    # Create experiment list
    experiments = []
    skip_until = args.resume_from
    skipping = skip_until is not None

    for domain in domains:
        domain_problems = get_all_problems(domain) if use_all_problems else problems
        for problem in domain_problems:
            exp_id = f"{domain}/{problem}"

            if skipping:
                if exp_id == skip_until:
                    skipping = False
                    logger.info(f"Resuming from {exp_id}")
                else:
                    continue

            experiments.append((domain, problem))

    # Show experiment plan
    total_experiments = len(experiments)
    logger.info(f"\nExperiment Plan:")
    logger.info(f"  Algorithm: information_gain")
    logger.info(f"  Domains: {len(domains)} domains")
    if use_all_problems:
        logger.info(f"  Problems: all available per domain")
    else:
        logger.info(f"  Problems: {len(problems)} per domain")
    logger.info(f"  Iterations: {iterations} per experiment")
    logger.info(f"  Use object subset: {args.use_object_subset}")
    logger.info(f"  Total experiments: {total_experiments}")
    logger.info(f"  Estimated runtime: {estimate_runtime(total_experiments, iterations)}")

    if args.dry_run:
        logger.info("\nDRY RUN - Experiments that would be run:")
        for alg, dom, prob in experiments[:10]:  # Show first 10
            logger.info(f"  - {alg}/{dom}/{prob}")
        if len(experiments) > 10:
            logger.info(f"  ... and {len(experiments) - 10} more")
        return 0

    # Confirm before starting
    if total_experiments > 10:
        import os
        # Auto-confirm if running in non-interactive mode
        if os.environ.get('AUTO_CONFIRM', '').lower() == 'yes' or not os.isatty(0):
            logger.info(f"Auto-confirming {total_experiments} experiments")
        else:
            response = input(f"\nAbout to run {total_experiments} experiments. Continue? (y/n): ")
            if response.lower() != 'y':
                logger.info("Aborted by user")
                return 1

    # Run experiments
    logger.info("\nStarting experiments...")
    start_time = time.time()
    successful = 0
    failed = []

    for i, (domain, problem) in enumerate(experiments, 1):
        logger.info(f"\n[{i}/{total_experiments}] information_gain/{domain}/{problem}")

        success = run_single_experiment(
            domain, problem, iterations,
            base_output_dir, args.force, args.use_object_subset
        )

        if success:
            successful += 1
        else:
            failed.append(f"information_gain/{domain}/{problem}")

    # Summary
    elapsed = time.time() - start_time
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60

    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    logger.info(f"Total runtime: {hours:.0f}h {minutes:.0f}m")
    logger.info(f"Successful: {successful}/{total_experiments}")
    logger.info(f"Failed: {len(failed)}")

    if failed:
        logger.info("\nFailed experiments:")
        for exp in failed:
            logger.info(f"  - {exp}")

    # Post-processing instructions
    if successful > 0:
        logger.info("\n" + "="*60)
    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())