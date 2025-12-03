#!/usr/bin/env python3
"""
Reconstruct model snapshots for early iterations (1-10) from existing experiment data.

This script reads completed experiments from consolidated_experiments, replays the
action sequences through the learning algorithm, and saves model snapshots at each
iteration.

All input directories are read-only. Results are written to a new output directory.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import yaml
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.information_gain import InformationGainLearner
from src.environments.active_environment import ActiveEnvironment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_experiment_data(exp_dir: Path) -> Dict[str, Any]:
    """Load experiment configuration and metrics data."""
    config_file = exp_dir / "config.yaml"

    # Find metrics JSON file
    metrics_files = list((exp_dir / "experiments").glob("*_metrics.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics JSON found in {exp_dir}/experiments")

    metrics_file = metrics_files[0]

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return {
        'config': config,
        'actions': metrics.get('actions', []),
        'domain': config['domain_problem']['domain'],
        'problem': config['domain_problem']['problem'],
        'seed': config['experiment']['seed']
    }


def parse_objects_string(objects_str: str) -> List[str]:
    """Parse objects string from JSON (e.g., "['b1', 'b2']" -> ['b1', 'b2'])."""
    import ast
    return ast.literal_eval(objects_str)


def replay_and_save_models(exp_data: Dict[str, Any], output_dir: Path,
                           max_iterations: int = 10) -> bool:
    """
    Replay action sequence and save model snapshots.

    Args:
        exp_data: Experiment data including config and action sequence
        output_dir: Directory to save model checkpoints
        max_iterations: Number of iterations to reconstruct (default: 10)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize environment
        env = ActiveEnvironment(
            domain_file=exp_data['domain'],
            problem_file=exp_data['problem']
        )

        # Initialize learner (with same parameters as original experiment)
        config = exp_data['config']
        algo_params = config.get('algorithm_params', {}).get('information_gain', {})

        learner = InformationGainLearner(
            domain_file=exp_data['domain'],
            problem_file=exp_data['problem'],
            selection_strategy=algo_params.get('selection_strategy', 'greedy'),
            max_iterations=max_iterations,
            model_stability_window=algo_params.get('model_stability_window', 50),
            info_gain_epsilon=algo_params.get('info_gain_epsilon', 0.001),
            success_rate_threshold=algo_params.get('success_rate_threshold', 0.98),
            success_rate_window=algo_params.get('success_rate_window', 50)
        )

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get action sequence (limit to max_iterations)
        actions = exp_data['actions'][:max_iterations]

        if len(actions) < max_iterations:
            logger.warning(f"Only {len(actions)} actions available (requested {max_iterations})")

        # Replay each action and save model snapshot
        for i, action_data in enumerate(actions):
            iteration = i + 1  # 1-indexed for output files

            action_name = action_data['action']
            objects = parse_objects_string(action_data['objects'])
            expected_success = action_data['success']

            # Get state before action
            state_before = env.get_state()

            # Execute action in environment
            success, _ = env.execute(action_name, objects)

            # Get state after action
            state_after = env.get_state() if success else None

            # Verify observation matches recorded data
            if success != expected_success:
                logger.error(f"Observation mismatch at iteration {iteration}: "
                           f"expected {expected_success}, got {success}")
                return False

            # Record observation in learner (this automatically updates the model)
            learner.observe(state_before, action_name, objects, success, state_after)

            # Save model snapshot using export_model_snapshot (new format with metadata)
            learner.export_model_snapshot(iteration, output_dir.parent)

        logger.info(f"  Saved {len(actions)} model snapshots")
        return True

    except Exception as e:
        logger.error(f"  Failed to replay: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def process_all_experiments(input_dir: Path, output_base: Path,
                           max_iterations: int = 10) -> None:
    """
    Process all experiments in consolidated directory.

    Args:
        input_dir: Input directory (e.g., results/paper/consolidated_experiments)
        output_base: Base output directory for checkpoints
        max_iterations: Number of iterations to reconstruct
    """
    algo_dir = input_dir / "information_gain"

    if not algo_dir.exists():
        logger.error(f"Directory not found: {algo_dir}")
        return

    # Collect all experiment directories
    experiments = []
    for domain_dir in sorted(algo_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain_name = domain_dir.name

        for problem_dir in sorted(domain_dir.iterdir()):
            if not problem_dir.is_dir():
                continue

            problem_name = problem_dir.name

            # Check if experiments subdirectory exists
            if not (problem_dir / "experiments").exists():
                logger.warning(f"Skipping {domain_name}/{problem_name}: no experiments directory")
                continue

            experiments.append((domain_name, problem_name, problem_dir))

    total = len(experiments)
    logger.info(f"Found {total} experiments to process")
    logger.info(f"Reconstructing first {max_iterations} iterations for each experiment")
    logger.info("")

    # Process each experiment
    success_count = 0
    failed = []

    for idx, (domain_name, problem_name, exp_dir) in enumerate(experiments, 1):
        logger.info(f"[{idx}/{total}] {domain_name}/{problem_name}")

        try:
            # Load experiment data
            exp_data = load_experiment_data(exp_dir)

            # Create output directory
            output_dir = output_base / "information_gain" / domain_name / problem_name / "checkpoints"

            # Replay and save models
            if replay_and_save_models(exp_data, output_dir, max_iterations):
                success_count += 1
            else:
                failed.append(f"{domain_name}/{problem_name}")

        except Exception as e:
            logger.error(f"  Error: {e}")
            failed.append(f"{domain_name}/{problem_name}")

    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info(f"Completed: {success_count}/{total} successful")

    if failed:
        logger.info(f"Failed ({len(failed)}):")
        for exp in failed:
            logger.info(f"  - {exp}")

    logger.info("")
    logger.info(f"Results saved to: {output_base}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct early iteration model snapshots from completed experiments"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/paper/consolidated_experiments",
        help="Input directory containing completed experiments (read-only)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_checkpoints",
        help="Output directory for model snapshots"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to reconstruct (default: 10)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Process only specific domain (optional)"
    )
    parser.add_argument(
        "--problem",
        type=str,
        help="Process only specific problem (optional, requires --domain)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    logger.info("Model Reconstruction Script")
    logger.info("="*60)
    logger.info(f"Input (read-only): {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info("")

    if args.domain and args.problem:
        # Process single experiment
        exp_dir = input_dir / "information_gain" / args.domain / args.problem
        if not exp_dir.exists():
            logger.error(f"Experiment not found: {exp_dir}")
            return 1

        logger.info(f"Processing single experiment: {args.domain}/{args.problem}")
        exp_data = load_experiment_data(exp_dir)
        output_path = output_dir / "information_gain" / args.domain / args.problem / "checkpoints"

        if replay_and_save_models(exp_data, output_path, args.iterations):
            logger.info(f"Success! Checkpoints saved to: {output_path}")
            return 0
        else:
            logger.error("Failed to reconstruct models")
            return 1
    else:
        # Process all experiments
        process_all_experiments(input_dir, output_dir, args.iterations)
        return 0


if __name__ == "__main__":
    sys.exit(main())
