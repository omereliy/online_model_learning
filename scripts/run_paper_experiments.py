#!/usr/bin/env python
"""
Run all experiments for paper comparison: OLAM vs Information Gain
Runs experiments across all domains and problem instances.
Usage: python scripts/run_paper_experiments.py [--iterations 1000]
"""

import sys
import os
import json
import yaml
import argparse
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging

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

# Experiment configuration
ALGORITHMS = ["olam", "information_gain"]
DOMAINS = ["blocksworld", "depots", "ferry", "rover"]
PROBLEMS = ["p00", "p01", "p02"]  # Three problem instances per domain
DEFAULT_ITERATIONS = 1000

def create_experiment_config(algorithm: str, domain: str, problem: str,
                           iterations: int, output_dir: str) -> Dict[str, Any]:
    """
    Create experiment configuration dynamically.

    Args:
        algorithm: Algorithm name (olam or information_gain)
        domain: Domain name
        problem: Problem instance (p00, p01, p02)
        iterations: Number of iterations
        output_dir: Output directory for results

    Returns:
        Configuration dictionary
    """
    config = {
        'experiment': {
            'name': f"{algorithm}_{domain}_{problem}_{iterations}iter",
            'algorithm': algorithm,
            'seed': 42 + ord(problem[-1])  # Different seed for each problem
        },
        'domain_problem': {
            'domain': f"benchmarks/olam-compatible/{domain}/domain.pddl",
            'problem': f"benchmarks/olam-compatible/{domain}/{problem}.pddl"
        },
        'metrics': {
            'interval': 10,  # Take snapshots every 10 actions for smooth precision/recall charts
            'window_size': 50
        },
        'stopping_criteria': {
            'max_iterations': iterations,
            'max_runtime_seconds': 7200,  # 2 hours max
            'convergence_check_interval': 50
        },
        'output': {
            'directory': output_dir,
            'formats': ['csv', 'json'],
            'save_learned_model': True,
            'track_learning_evidence': True
        }
    }

    # Algorithm-specific parameters
    if algorithm == "information_gain":
        config['algorithm_params'] = {
            'information_gain': {
                'selection_strategy': 'greedy',
                'max_iterations': iterations,
                'model_stability_window': 50,
                'info_gain_epsilon': 0.0,  # Zero epsilon as requested
                'success_rate_threshold': 0.98,
                'success_rate_window': 50
            }
        }
    elif algorithm == "olam":
        config['algorithm_params'] = {
            'olam': {
                'selection_strategy': 'greedy'
            }
        }

    return config

def run_single_experiment(config: Dict[str, Any], temp_config_path: Path) -> Dict[str, Any]:
    """
    Run a single experiment using the provided configuration.

    Args:
        config: Experiment configuration dictionary
        temp_config_path: Path to save temporary config file

    Returns:
        Experiment results dictionary
    """
    # Save config to temp file
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"  Algorithm: {config['experiment']['algorithm']}")
    logger.info(f"  Domain: {Path(config['domain_problem']['domain']).stem}")
    logger.info(f"  Problem: {Path(config['domain_problem']['problem']).stem}")

    try:
        # Run experiment
        runner = ExperimentRunner(str(temp_config_path))
        results = runner.run_experiment()

        logger.info(f"Completed: {config['experiment']['name']}")
        logger.info(f"  Success rate: {results['metrics']['success_rate']:.2%}")
        logger.info(f"  Runtime: {results['runtime_seconds']:.1f}s")
        logger.info(f"  Iterations: {results['total_iterations']}")

        return results

    except Exception as e:
        logger.error(f"Failed experiment {config['experiment']['name']}: {e}")
        return None
    finally:
        # Clean up temp config
        if temp_config_path.exists():
            temp_config_path.unlink()

def run_all_experiments(iterations: int = DEFAULT_ITERATIONS,
                       domains: List[str] = None,
                       algorithms: List[str] = None,
                       skip_existing: bool = True) -> None:
    """
    Run all experiments for the paper.

    Args:
        iterations: Number of iterations per experiment
        domains: List of domains to run (default: all)
        algorithms: List of algorithms to run (default: all)
        skip_existing: Skip experiments that already have results
    """
    # Setup
    domains = domains or DOMAINS
    algorithms = algorithms or ALGORITHMS

    # Create output directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = project_root / 'results' / 'paper' / f'comparison_{timestamp}'
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Temp config file
    temp_config_path = base_output_dir / '.temp_config.yaml'

    # Summary results
    all_results = []
    experiment_summary = {
        'timestamp': timestamp,
        'iterations': iterations,
        'experiments': []
    }

    total_experiments = len(algorithms) * len(domains) * len(PROBLEMS)
    completed = 0

    logger.info("=" * 80)
    logger.info(f"STARTING PAPER EXPERIMENTS")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Algorithms: {', '.join(algorithms)}")
    logger.info(f"Domains: {', '.join(domains)}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Output directory: {base_output_dir}")
    logger.info("=" * 80)

    # Run experiments
    for algorithm in algorithms:
        for domain in domains:
            for problem in PROBLEMS:
                completed += 1

                # Create output directory for this specific experiment
                exp_output_dir = base_output_dir / algorithm / domain / problem
                exp_output_dir.mkdir(parents=True, exist_ok=True)

                # Check if already exists
                if skip_existing:
                    existing_results = list(exp_output_dir.glob('*_summary.json'))
                    if existing_results:
                        logger.info(f"[{completed}/{total_experiments}] Skipping existing: "
                                  f"{algorithm}_{domain}_{problem}")
                        continue

                logger.info(f"\n[{completed}/{total_experiments}] "
                          f"Running: {algorithm}_{domain}_{problem}")

                # Create configuration
                config = create_experiment_config(
                    algorithm=algorithm,
                    domain=domain,
                    problem=problem,
                    iterations=iterations,
                    output_dir=str(exp_output_dir)
                )

                # Run experiment
                start_time = time.time()
                results = run_single_experiment(config, temp_config_path)
                runtime = time.time() - start_time

                if results:
                    # Add to summary
                    exp_summary = {
                        'algorithm': algorithm,
                        'domain': domain,
                        'problem': problem,
                        'iterations': results['total_iterations'],
                        'success_rate': results['metrics']['success_rate'],
                        'runtime': runtime,
                        'stopping_reason': results['stopping_reason']
                    }
                    experiment_summary['experiments'].append(exp_summary)
                    all_results.append(results)
                else:
                    logger.error(f"Failed to get results for {algorithm}_{domain}_{problem}")

    # Save overall summary
    summary_path = base_output_dir / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(experiment_summary, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info(f"Results saved to: {base_output_dir}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 80)

    # Print summary statistics
    print_summary_statistics(experiment_summary)

def print_summary_statistics(summary: Dict[str, Any]) -> None:
    """
    Print summary statistics of all experiments.

    Args:
        summary: Experiment summary dictionary
    """
    if not summary['experiments']:
        logger.warning("No experiments to summarize")
        return

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY STATISTICS")
    print("=" * 80)

    # Group by algorithm
    for algorithm in ALGORITHMS:
        algo_exps = [e for e in summary['experiments'] if e['algorithm'] == algorithm]
        if not algo_exps:
            continue

        print(f"\n{algorithm.upper()}:")

        # Group by domain
        for domain in DOMAINS:
            domain_exps = [e for e in algo_exps if e['domain'] == domain]
            if not domain_exps:
                continue

            avg_success = sum(e['success_rate'] for e in domain_exps) / len(domain_exps)
            avg_runtime = sum(e['runtime'] for e in domain_exps) / len(domain_exps)

            print(f"  {domain:12s}: success_rate={avg_success:.2%}, "
                  f"avg_runtime={avg_runtime:.1f}s")

    # Overall comparison
    print("\nOVERALL COMPARISON:")
    for algorithm in ALGORITHMS:
        algo_exps = [e for e in summary['experiments'] if e['algorithm'] == algorithm]
        if algo_exps:
            overall_success = sum(e['success_rate'] for e in algo_exps) / len(algo_exps)
            overall_runtime = sum(e['runtime'] for e in algo_exps) / len(algo_exps)
            print(f"  {algorithm:20s}: success_rate={overall_success:.2%}, "
                  f"avg_runtime={overall_runtime:.1f}s")

    print("=" * 80)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run paper experiments comparing OLAM vs Information Gain')
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS,
                       help=f'Number of iterations per experiment (default: {DEFAULT_ITERATIONS})')
    parser.add_argument('--domains', nargs='+', choices=DOMAINS,
                       help='Specific domains to run (default: all)')
    parser.add_argument('--algorithms', nargs='+', choices=ALGORITHMS,
                       help='Specific algorithms to run (default: all)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Do not skip existing experiments')

    args = parser.parse_args()

    run_all_experiments(
        iterations=args.iterations,
        domains=args.domains,
        algorithms=args.algorithms,
        skip_existing=not args.no_skip
    )

if __name__ == "__main__":
    main()