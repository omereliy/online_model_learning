#!/usr/bin/env python
"""
Recalculate precision/recall metrics from exported model snapshots.

Usage:
    python scripts/recalculate_model_metrics.py results/paper/comparison_*/
    python scripts/recalculate_model_metrics.py results/paper/comparison_20251112_120000/ --output checkpoint_metrics.csv
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Set
import logging
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.model_reconstructor import ModelReconstructor
from src.core.model_validator import ModelValidator

logger = logging.getLogger(__name__)


def find_model_snapshots(experiment_dir: Path) -> List[Path]:
    """Find all model snapshot files in experiment directory."""
    models_dir = experiment_dir / "models"
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("model_iter_*.json"))


def extract_action_schema_name(grounded_action: str) -> str:
    """Extract action schema name from grounded action.

    Examples:
        'pick-up(b1)' -> 'pick-up'
        'stack(b1,b2)' -> 'stack'
    """
    return grounded_action.split('(')[0] if '(' in grounded_action else grounded_action


def load_ground_truth(domain_file: Path, problem_file: Path) -> Dict[str, Any]:
    """Load ground truth model from PDDL domain and problem."""
    if not domain_file.exists():
        raise FileNotFoundError(f"Domain file not found: {domain_file}")
    if not problem_file.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_file}")

    validator = ModelValidator(str(domain_file), str(problem_file))
    return validator.ground_truth_models


def calculate_checkpoint_metrics(snapshot_path: Path,
                                 ground_truth: Dict[str, Any],
                                 algorithm: str,
                                 domain: str,
                                 problem: str) -> List[Dict[str, Any]]:
    """
    Calculate precision/recall for both safe and complete models.

    Args:
        snapshot_path: Path to model snapshot JSON file
        ground_truth: Ground truth action models
        algorithm: Algorithm name
        domain: Domain name
        problem: Problem name

    Returns:
        List of 2 metric dictionaries (safe and complete)
    """
    # Load and reconstruct models
    reconstructed_models = ModelReconstructor.load_and_reconstruct(snapshot_path)

    results = []
    for model in reconstructed_models:
        # Calculate precision/recall for each action
        metrics = {
            "algorithm": algorithm,
            "domain": domain,
            "problem": problem,
            "iteration": model.iteration,
            "model_type": model.model_type
        }

        # Aggregate metrics across all actions
        prec_metrics_list = []
        add_metrics_list = []
        del_metrics_list = []

        for action_name, learned_action in model.actions.items():
            # Extract action schema name from grounded action
            schema_name = extract_action_schema_name(action_name)

            if schema_name not in ground_truth:
                logger.warning(f"Action schema {schema_name} (from {action_name}) not in ground truth")
                continue

            gt_action = ground_truth[schema_name]

            # Precondition metrics
            prec_metrics = calculate_precision_recall(
                learned=learned_action.preconditions,
                ground_truth=set(gt_action.get('preconditions', []))
            )

            # Add effect metrics
            add_metrics = calculate_precision_recall(
                learned=learned_action.add_effects,
                ground_truth=set(gt_action.get('add_effects', []))
            )

            # Delete effect metrics
            del_metrics = calculate_precision_recall(
                learned=learned_action.del_effects,
                ground_truth=set(gt_action.get('del_effects', []))
            )

            prec_metrics_list.append(prec_metrics)
            add_metrics_list.append(add_metrics)
            del_metrics_list.append(del_metrics)

        # Average across actions
        if prec_metrics_list:
            # Calculate averages with safe division
            num_actions = len(prec_metrics_list)

            metrics.update({
                "precondition_precision": sum(m["precision"] for m in prec_metrics_list) / num_actions,
                "precondition_recall": sum(m["recall"] for m in prec_metrics_list) / num_actions,
                "precondition_f1": sum(m["f1"] for m in prec_metrics_list) / num_actions,
                "add_effect_precision": sum(m["precision"] for m in add_metrics_list) / num_actions,
                "add_effect_recall": sum(m["recall"] for m in add_metrics_list) / num_actions,
                "add_effect_f1": sum(m["f1"] for m in add_metrics_list) / num_actions,
                "del_effect_precision": sum(m["precision"] for m in del_metrics_list) / num_actions,
                "del_effect_recall": sum(m["recall"] for m in del_metrics_list) / num_actions,
                "del_effect_f1": sum(m["f1"] for m in del_metrics_list) / num_actions,
            })

            # Calculate overall F1 as weighted average
            total_f1_sum = (sum(m["f1"] for m in prec_metrics_list) +
                           sum(m["f1"] for m in add_metrics_list) +
                           sum(m["f1"] for m in del_metrics_list))
            total_components = num_actions * 3  # prec, add, del for each action
            metrics["overall_f1"] = total_f1_sum / total_components if total_components > 0 else 0.0
        else:
            logger.warning(f"No valid actions to calculate metrics for {algorithm}/{domain}/{problem}")
            # Set all metrics to 0 if no valid actions
            metrics.update({
                "precondition_precision": 0.0,
                "precondition_recall": 0.0,
                "precondition_f1": 0.0,
                "add_effect_precision": 0.0,
                "add_effect_recall": 0.0,
                "add_effect_f1": 0.0,
                "del_effect_precision": 0.0,
                "del_effect_recall": 0.0,
                "del_effect_f1": 0.0,
                "overall_f1": 0.0
            })

        results.append(metrics)

    return results


def calculate_precision_recall(learned: Set[str],
                               ground_truth: Set[str]) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 for a single component.

    Args:
        learned: Set of learned literals
        ground_truth: Set of ground truth literals

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    tp = len(learned & ground_truth)  # True positives
    fp = len(learned - ground_truth)  # False positives
    fn = len(ground_truth - learned)  # False negatives

    # Handle edge cases
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(ground_truth) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if len(ground_truth) == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def process_experiment_directory(experiment_dir: Path, output_csv: Path) -> None:
    """
    Process all experiments in directory and generate checkpoint_metrics.csv.

    Args:
        experiment_dir: Root experiment directory
        output_csv: Output CSV file path
    """
    all_metrics = []

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Discover experiments (algorithm/domain/problem structure)
    for algo_dir in experiment_dir.iterdir():
        if not algo_dir.is_dir():
            continue

        algorithm = algo_dir.name
        logger.info(f"Processing algorithm: {algorithm}")

        for domain_dir in algo_dir.iterdir():
            if not domain_dir.is_dir():
                continue

            domain = domain_dir.name

            for problem_dir in domain_dir.iterdir():
                if not problem_dir.is_dir():
                    continue

                problem = problem_dir.name
                logger.info(f"  Processing: {algorithm}/{domain}/{problem}")

                # Load ground truth
                domain_file = Path(f"benchmarks/olam-compatible/{domain}/domain.pddl")
                problem_file = Path(f"benchmarks/olam-compatible/{domain}/{problem}.pddl")
                if not domain_file.exists():
                    logger.warning(f"Domain file not found: {domain_file}, skipping")
                    continue
                if not problem_file.exists():
                    logger.warning(f"Problem file not found: {problem_file}, skipping")
                    continue

                try:
                    ground_truth = load_ground_truth(domain_file, problem_file)
                except Exception as e:
                    logger.error(f"Failed to load ground truth for {domain}: {e}")
                    continue

                # Process each checkpoint
                snapshots = find_model_snapshots(problem_dir)
                logger.debug(f"    Found {len(snapshots)} snapshots")

                for snapshot_path in snapshots:
                    try:
                        metrics = calculate_checkpoint_metrics(
                            snapshot_path, ground_truth, algorithm, domain, problem
                        )
                        all_metrics.extend(metrics)
                    except Exception as e:
                        logger.error(f"Failed to process {snapshot_path}: {e}")

    if not all_metrics:
        logger.warning("No metrics collected. Check if experiment directory has model snapshots.")
        return

    # Export to CSV
    df = pd.DataFrame(all_metrics)

    # Sort by algorithm, domain, problem, iteration, model_type for better organization
    df = df.sort_values(['algorithm', 'domain', 'problem', 'iteration', 'model_type'])

    df.to_csv(output_csv, index=False, float_format='%.4f')
    logger.info(f"Exported metrics to {output_csv}")
    logger.info(f"Total rows: {len(df)}")

    # Print summary statistics
    logger.info("\nSummary by algorithm and model type:")
    summary = df.groupby(['algorithm', 'model_type'])['overall_f1'].agg(['mean', 'std', 'max'])
    print(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate precision/recall metrics from model snapshots"
    )
    parser.add_argument("experiment_dir", type=Path,
                       help="Experiment directory (e.g., results/paper/comparison_*/)")
    parser.add_argument("--output", type=Path, default=Path("checkpoint_metrics.csv"),
                       help="Output CSV file (default: checkpoint_metrics.csv)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        process_experiment_directory(args.experiment_dir, args.output)
    except Exception as e:
        logger.error(f"Failed to process experiments: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()