#!/usr/bin/env python
"""
Run all 300-iteration experiments across all domains and algorithms.

This script runs the full experiment suite:
- 4 domains: blocksworld, depots, gripper, logistics
- 2 algorithms: OLAM, Information Gain
- 3 problems per domain: p00, p01, p02
- Total: 24 experiments (4 domains × 2 algorithms × 3 problems)

Usage:
    python scripts/run_all_experiments.py [options]

Options:
    --algorithms ALGO [ALGO ...]  Run specific algorithms (olam, information_gain, or both)
    --domains DOMAIN [DOMAIN ...] Run specific domains (blocksworld, depots, gripper, logistics, or all)
    --problems PROB [PROB ...]    Run specific problems (0, 1, 2, or all)
    --sequential                  Run experiments sequentially (default)
    --dry-run                     Print experiment list without running
"""

import sys
import os
import logging
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.runner import ExperimentRunner


# Configuration
DOMAINS = ['blocksworld', 'depots', 'gripper', 'logistics']
ALGORITHMS = ['olam', 'information_gain']
PROBLEMS = ['p00', 'p01', 'p02']


def setup_logging(output_dir: Path) -> None:
    """Set up logging for the experiment suite."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'experiment_suite.log'

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_and_modify_config(config_path: str, problem_file: str) -> dict:
    """
    Load a config file and modify the problem file path.

    Args:
        config_path: Path to the base config file
        problem_file: Problem filename (e.g., 'p00.pddl')

    Returns:
        Modified configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert relative paths to absolute paths based on project root
    domain_path = Path(config['domain_problem']['domain'])
    if not domain_path.is_absolute():
        domain_path = project_root / domain_path
    config['domain_problem']['domain'] = str(domain_path)

    # Extract domain from original problem path
    original_problem = config['domain_problem']['problem']
    problem_path = Path(original_problem)
    if not problem_path.is_absolute():
        problem_path = project_root / problem_path
    domain_dir = problem_path.parent

    # Update problem file with absolute path
    config['domain_problem']['problem'] = str(domain_dir / problem_file)

    # Update experiment name to include problem
    problem_num = problem_file.replace('.pddl', '').replace('p', '')
    original_name = config['experiment']['name']
    config['experiment']['name'] = f"{original_name}_{problem_file.replace('.pddl', '')}"

    return config


def run_single_experiment(config_path: str, problem_file: str, output_base_dir: Path, override_iterations: int = None) -> dict:
    """
    Run a single experiment with a specific problem file.

    Args:
        config_path: Path to the config file
        problem_file: Problem filename (e.g., 'p00.pddl')
        output_base_dir: Base directory for all experiment outputs

    Returns:
        Dictionary with experiment results and metadata
    """
    logger = logging.getLogger(__name__)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load and modify config
    config = load_and_modify_config(config_path, problem_file)
    experiment_name = config['experiment']['name']

    # Override iterations if specified
    if override_iterations:
        config['stopping_criteria']['max_iterations'] = override_iterations
        if 'algorithm_params' in config:
            for algo_params in config['algorithm_params'].values():
                if 'max_iterations' in algo_params:
                    algo_params['max_iterations'] = override_iterations

    # Create experiment-specific output directory
    experiment_dir = output_base_dir / experiment_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Update config output directory
    config['output']['directory'] = str(experiment_dir)

    # Save modified config
    config_save_path = experiment_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("=" * 80)
    logger.info(f"Starting: {experiment_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Problem: {problem_file}")
    logger.info(f"Domain: {config['domain_problem']['domain']}")
    logger.info(f"Algorithm: {config['experiment']['algorithm']}")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        # Create temporary config file
        temp_config_path = experiment_dir / 'runtime_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        # Run experiment
        runner = ExperimentRunner(str(temp_config_path))
        results = runner.run_experiment()

        # Add metadata
        results['experiment_metadata'] = {
            'config_path': config_path,
            'problem_file': problem_file,
            'experiment_dir': str(experiment_dir),
            'timestamp': timestamp,
            'success': True,
            'error': None,
            'runtime_seconds': time.time() - start_time
        }

        # Log summary
        logger.info("=" * 80)
        logger.info(f"Completed: {experiment_name}")
        logger.info(f"Iterations: {results['total_iterations']}")
        logger.info(f"Stopping reason: {results['stopping_reason']}")
        logger.info(f"Runtime: {results['runtime_seconds']:.2f}s")
        logger.info(f"Success rate: {results['metrics'].get('success_rate', 'N/A')}")
        logger.info("=" * 80)

        # Save results summary
        summary_path = experiment_dir / 'results_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    except Exception as e:
        logger.error(f"Experiment failed: {experiment_name}")
        logger.error(f"Error: {e}")

        error_info = {
            'experiment_name': experiment_name,
            'config_path': config_path,
            'problem_file': problem_file,
            'timestamp': timestamp,
            'success': False,
            'error': str(e),
            'runtime_seconds': time.time() - start_time
        }

        error_path = experiment_dir / 'error.json'
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2)

        return error_info


def run_experiment_suite(algorithms=None, domains=None, problems=None, dry_run=False, override_iterations=None):
    """
    Run the full experiment suite with optional filters.

    Args:
        algorithms: List of algorithms to run (default: all)
        domains: List of domains to run (default: all)
        problems: List of problem indices to run (default: all)
        dry_run: If True, print plan without executing
        override_iterations: If set, override max_iterations in configs
    """
    # Setup basic logging for dry-run or full logging for real run
    if dry_run:
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[logging.StreamHandler()]
        )

    logger = logging.getLogger(__name__)

    # Use defaults if not specified
    algorithms = algorithms or ALGORITHMS
    domains = domains or DOMAINS
    problems = problems or ['0', '1', '2']

    # Convert problem indices to filenames
    problem_files = [f"p0{p}.pddl" for p in problems]

    # Build experiment list
    experiments = []
    for domain in domains:
        for algorithm in algorithms:
            config_name = f"{algorithm}_{domain}_300"
            config_path = project_root / 'configs' / f"{config_name}.yaml"

            if not config_path.exists():
                logger.warning(f"Config not found: {config_path}")
                continue

            for problem_file in problem_files:
                experiments.append({
                    'config_path': str(config_path),
                    'problem_file': problem_file,
                    'domain': domain,
                    'algorithm': algorithm
                })

    # Print experiment plan
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT SUITE PLAN")
    logger.info("=" * 80)
    logger.info(f"Domains: {', '.join(domains)}")
    logger.info(f"Algorithms: {', '.join(algorithms)}")
    logger.info(f"Problems: {', '.join(problem_files)}")
    logger.info(f"Total experiments: {len(experiments)}")
    if override_iterations:
        logger.info(f"Iteration override: {override_iterations}")
    logger.info("=" * 80)

    if dry_run:
        logger.info("\nExperiment list:")
        for i, exp in enumerate(experiments, 1):
            logger.info(f"  {i}. {exp['algorithm']:20s} {exp['domain']:15s} {exp['problem_file']}")
        return

    # Setup output directory and full logging for real run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base_dir = project_root / 'results' / 'full_suite' / timestamp
    setup_logging(output_base_dir)

    logger.info(f"\nOutput directory: {output_base_dir}")
    logger.info("\nStarting experiments...\n")

    # Run experiments sequentially
    results = []
    for i, exp in enumerate(experiments, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Experiment {i}/{len(experiments)}")
        logger.info(f"{'='*80}\n")

        result = run_single_experiment(
            exp['config_path'],
            exp['problem_file'],
            output_base_dir,
            override_iterations
        )
        results.append(result)

        # Short pause between experiments
        if i < len(experiments):
            logger.info("\nPausing 5 seconds before next experiment...\n")
            time.sleep(5)

    # Save combined results
    combined_path = output_base_dir / 'combined_results.json'
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUITE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total experiments: {len(experiments)}")

    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful

    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"\nResults saved to: {output_base_dir}")
    logger.info(f"Combined results: {combined_path}")
    logger.info("=" * 80)

    # Print summary table
    logger.info("\nResults Summary:")
    logger.info(f"{'Algorithm':<20} {'Domain':<15} {'Problem':<10} {'Iters':<8} {'Runtime':<10} {'Status'}")
    logger.info("-" * 80)
    for result in results:
        if result.get('success'):
            algo = result['experiment_name'].split('_')[0]
            domain = result['experiment_name'].split('_')[1]
            problem = result['experiment_metadata']['problem_file']
            iters = result.get('total_iterations', 'N/A')
            runtime = f"{result.get('runtime_seconds', 0):.1f}s"
            status = "SUCCESS"
        else:
            algo = "?"
            domain = "?"
            problem = result.get('experiment_metadata', {}).get('problem_file', '?')
            iters = "N/A"
            runtime = "N/A"
            status = "FAILED"

        logger.info(f"{algo:<20} {domain:<15} {problem:<10} {iters!s:<8} {runtime:<10} {status}")


def main():
    parser = argparse.ArgumentParser(
        description='Run all 300-iteration experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--algorithms', nargs='+', choices=ALGORITHMS + ['all'],
                       default=['all'],
                       help='Algorithms to run (default: all)')

    parser.add_argument('--domains', nargs='+', choices=DOMAINS + ['all'],
                       default=['all'],
                       help='Domains to run (default: all)')

    parser.add_argument('--problems', nargs='+', choices=['0', '1', '2', 'all'],
                       default=['all'],
                       help='Problem indices to run (default: all)')

    parser.add_argument('--dry-run', action='store_true',
                       help='Print experiment plan without running')

    parser.add_argument('--iterations', type=int,
                       help='Override max iterations for quick testing')

    args = parser.parse_args()

    # Handle 'all' keyword
    algorithms = ALGORITHMS if 'all' in args.algorithms else args.algorithms
    domains = DOMAINS if 'all' in args.domains else args.domains
    problems = ['0', '1', '2'] if 'all' in args.problems else args.problems

    # Run experiment suite
    run_experiment_suite(
        algorithms=algorithms,
        domains=domains,
        problems=problems,
        dry_run=args.dry_run,
        override_iterations=args.iterations
    )


if __name__ == "__main__":
    main()
