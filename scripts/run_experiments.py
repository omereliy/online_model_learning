#!/usr/bin/env python
"""
Run experiments with enhanced logging and metrics collection.
"""

import sys
import os
import logging
import json
import yaml
import traceback
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.runner import ExperimentRunner


def setup_logging(config: dict, experiment_name: str, timestamp: str) -> Path:
    """
    Set up comprehensive logging for the experiment.

    Args:
        config: Experiment configuration
        experiment_name: Name of the experiment
        timestamp: Timestamp for this run

    Returns:
        Path to the experiment directory
    """
    # Create experiment directory
    experiment_dir = Path(config['output']['directory']) / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Get logging config
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    console_level = getattr(logging, log_config.get('console_level', 'INFO'))
    file_level = getattr(logging, log_config.get('file_level', 'DEBUG'))

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[]
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # File handler - comprehensive log
    log_file = experiment_dir / 'experiment.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(log_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    file_handler.setFormatter(file_formatter)

    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Also create a separate debug log for detailed debugging
    debug_file = experiment_dir / 'debug.log'
    debug_handler = logging.FileHandler(debug_file)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(file_formatter)

    # Add debug handler to specific loggers
    for logger_name in ['src.experiments', 'src.algorithms', 'src.environments']:
        logger = logging.getLogger(logger_name)
        logger.addHandler(debug_handler)

    return experiment_dir


def run_single_experiment(config_path: str, override_iterations: int = None) -> dict:
    """
    Run a single experiment with comprehensive logging.

    Args:
        config_path: Path to experiment configuration
        override_iterations: Optional override for max iterations

    Returns:
        Dictionary with experiment results and metadata
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    experiment_name = config['experiment']['name']

    # Override iterations if specified
    if override_iterations:
        config['stopping_criteria']['max_iterations'] = override_iterations
        logging.info(f"Overriding max iterations to {override_iterations}")

    # Setup logging
    experiment_dir = setup_logging(config, experiment_name, timestamp)

    # Update output directory to experiment-specific directory
    config['output']['directory'] = str(experiment_dir)

    # Save the actual configuration used
    config_save_path = experiment_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Output directory: {experiment_dir}")
    logger.info(f"Domain: {config['domain_problem']['domain']}")
    logger.info(f"Problem: {config['domain_problem']['problem']}")
    logger.info(f"Algorithm: {config['experiment']['algorithm']}")
    logger.info(f"Max iterations: {config['stopping_criteria']['max_iterations']}")
    logger.info("=" * 80)

    # Create a temporary config file with updated paths
    temp_config_path = experiment_dir / 'runtime_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    try:
        # Run experiment
        runner = ExperimentRunner(str(temp_config_path))

        # Log initial state
        initial_state = runner.environment.get_state()
        logger.info(f"Initial state size: {len(initial_state)} fluents")
        logger.debug(f"Initial state: {initial_state}")

        # Run the experiment
        results = runner.run_experiment()

        # Add metadata to results
        results['experiment_metadata'] = {
            'config_path': config_path,
            'experiment_dir': str(experiment_dir),
            'timestamp': timestamp,
            'success': True,
            'error': None
        }

        # Log summary
        logger.info("=" * 80)
        logger.info("Experiment completed successfully")
        logger.info(f"Total iterations: {results['total_iterations']}")
        logger.info(f"Stopping reason: {results['stopping_reason']}")
        logger.info(f"Runtime: {results['runtime_seconds']:.2f} seconds")
        logger.info(f"Final mistake rate: {results['metrics']['mistake_rate']:.3f}")
        logger.info(f"Overall mistake rate: {results['metrics'].get('overall_mistake_rate', 'N/A')}")
        logger.info("=" * 80)

        # Save final results summary
        summary_path = experiment_dir / 'results_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        logger.error(traceback.format_exc())

        # Save error information
        error_info = {
            'experiment_name': experiment_name,
            'config_path': config_path,
            'timestamp': timestamp,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

        error_path = experiment_dir / 'error.json'
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2)

        return error_info


def run_multiple_experiments(config_paths: list, override_iterations: int = None):
    """
    Run multiple experiments sequentially.

    Args:
        config_paths: List of configuration file paths
        override_iterations: Optional override for max iterations
    """
    results = []

    for i, config_path in enumerate(config_paths, 1):
        print(f"\n{'='*80}")
        print(f"Running experiment {i}/{len(config_paths)}: {config_path}")
        print(f"{'='*80}\n")

        result = run_single_experiment(config_path, override_iterations)
        results.append(result)

        # Short break between experiments
        if i < len(config_paths):
            print("\nPausing for 5 seconds before next experiment...")
            import time
            time.sleep(5)

    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_path = Path('results') / f'combined_results_{timestamp}.json'
    combined_path.parent.mkdir(exist_ok=True)

    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"All experiments completed. Combined results saved to {combined_path}")
    print(f"{'='*80}")

    # Print summary
    print("\nSummary:")
    for result in results:
        if result.get('success', False):
            name = result.get('experiment_name', 'Unknown')
            iterations = result.get('total_iterations', 'N/A')
            runtime = result.get('runtime_seconds', 0)
            print(f"  - {name}: {iterations} iterations in {runtime:.1f}s")
        else:
            name = result.get('experiment_name', 'Unknown')
            error = result.get('error', 'Unknown error')
            print(f"  - {name}: FAILED - {error}")


def main():
    parser = argparse.ArgumentParser(description='Run online model learning experiments')
    parser.add_argument('configs', nargs='*',
                       help='Configuration files to run (default: all in configs/)')
    parser.add_argument('--iterations', type=int,
                       help='Override max iterations for quick testing')
    parser.add_argument('--list', action='store_true',
                       help='List available configurations')

    args = parser.parse_args()

    # List configurations if requested
    if args.list:
        configs_dir = project_root / 'configs'
        config_files = sorted(configs_dir.glob('experiment_*.yaml'))
        print("Available experiment configurations:")
        for config in config_files:
            print(f"  - {config.name}")
        return

    # Determine which configs to run
    if args.configs:
        config_paths = args.configs
    else:
        # Default: run all experiment configs
        configs_dir = project_root / 'configs'
        config_paths = sorted(configs_dir.glob('experiment_*.yaml'))
        config_paths = [str(p) for p in config_paths]

    if not config_paths:
        print("No configuration files found.")
        return

    # Run experiments
    if len(config_paths) == 1:
        run_single_experiment(config_paths[0], args.iterations)
    else:
        run_multiple_experiments(config_paths, args.iterations)


if __name__ == "__main__":
    main()