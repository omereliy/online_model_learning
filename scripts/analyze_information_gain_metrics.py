#!/usr/bin/env python3
"""
Analyze Information Gain learning results and compute precision/recall metrics.

This script processes model checkpoints from the information gain algorithm,
compares them against ground truth PDDL domains, and computes precision/recall
for preconditions, add effects, and delete effects at each iteration.

Follows the same pattern as analyze_olam_results.py for consistency.

Input:
    results/paper/consolidated_experiments/information_gain/{domain}/{problem}/models/
    benchmarks/olam-compatible/{domain}/domain.pddl

Output:
    results/information_gain_metrics/{domain}/{problem}/metrics_per_iteration.json
    results/information_gain_metrics/{domain}/domain_metrics.json
    results/information_gain_metrics/checkpoint_metrics.csv
"""

import json
import csv
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.model_reconstructor import ModelReconstructor
from src.core.model_metrics import ModelMetrics
from src.core.model_validator import ModelValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_file: Path) -> Dict[str, Any]:
    """Load checkpoint JSON file."""
    with open(checkpoint_file) as f:
        return json.load(f)


def load_execution_stats(problem_dir: Path, up_to_iteration: int) -> Dict[str, Dict[str, int]]:
    """
    Load execution statistics from experiment metrics file.

    Args:
        problem_dir: Problem directory containing experiments/
        up_to_iteration: Count executions up to this iteration (inclusive)

    Returns:
        Dict mapping action names to {executions, successes, failures}
    """
    # Find metrics file
    experiments_dir = problem_dir / "experiments"
    if not experiments_dir.exists():
        return {}

    metrics_files = list(experiments_dir.glob("*_metrics.json"))
    if not metrics_files:
        return {}

    metrics_file = metrics_files[0]

    try:
        with open(metrics_file) as f:
            data = json.load(f)

        # Count per action up to the specified iteration
        action_stats = {}

        for action_data in data.get('actions', []):
            step = action_data.get('step', 0)
            if step >= up_to_iteration:
                break  # Stop counting after this iteration

            action_name = action_data.get('action', '')
            success = action_data.get('success', False)

            if action_name not in action_stats:
                action_stats[action_name] = {
                    'executions': 0,
                    'successes': 0,
                    'failures': 0
                }

            action_stats[action_name]['executions'] += 1
            if success:
                action_stats[action_name]['successes'] += 1
            else:
                action_stats[action_name]['failures'] += 1

        return action_stats

    except Exception as e:
        logger.warning(f"Could not load execution stats from {metrics_file}: {e}")
        return {}


def analyze_checkpoint(
    checkpoint_file: Path,
    ground_truth_domain: Path,
    ground_truth_problem: Path,
    problem_dir: Path,
    min_observations: int = 0
) -> Dict[str, Any]:
    """
    Analyze a single checkpoint file against ground truth.

    Args:
        checkpoint_file: Path to model checkpoint JSON
        ground_truth_domain: Path to ground truth domain PDDL
        ground_truth_problem: Path to ground truth problem PDDL
        problem_dir: Problem directory (for loading execution stats)
        min_observations: Minimum observations to include action in metrics

    Returns:
        Dict with metrics for safe and complete models
    """
    # Load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_file)
    iteration = checkpoint_data.get('iteration', 0)

    # Reconstruct safe and complete models
    safe_model = ModelReconstructor.reconstruct_information_gain_safe(checkpoint_data)
    complete_model = ModelReconstructor.reconstruct_information_gain_complete(checkpoint_data)

    # Compute metrics for both models (pass file paths, not ModelValidator)
    metrics_calculator = ModelMetrics(ground_truth_domain, ground_truth_problem)

    # Load execution statistics up to this iteration (for filtering)
    execution_stats = load_execution_stats(problem_dir, iteration)

    # Extract observation counts from execution stats
    observation_counts = {
        action_name: stats.get('executions', 0)
        for action_name, stats in execution_stats.items()
    }

    # Compute metrics with optional filtering
    safe_metrics = metrics_calculator.compute_metrics(
        safe_model,
        observation_counts=observation_counts,
        min_observations=min_observations
    )
    complete_metrics = metrics_calculator.compute_metrics(
        complete_model,
        observation_counts=observation_counts,
        min_observations=min_observations
    )

    return {
        'iteration': iteration,
        'safe_model': safe_metrics,
        'complete_model': complete_metrics,
        'execution_stats': execution_stats
    }


def analyze_problem(
    problem_dir: Path,
    ground_truth_domain: Path,
    ground_truth_problem: Path,
    output_dir: Path,
    min_observations: int = 0
) -> List[Dict[str, Any]]:
    """
    Analyze all checkpoints for a single problem.

    Args:
        problem_dir: Problem directory in consolidated_experiments
        ground_truth_domain: Path to ground truth domain
        ground_truth_problem: Path to ground truth problem
        output_dir: Output directory for results
        min_observations: Minimum observations to include action in metrics

    Returns:
        List of metrics per iteration
    """
    # Find checkpoint files
    models_dir = problem_dir / "models"
    if not models_dir.exists():
        logger.warning(f"No models directory found in {problem_dir}")
        return []

    checkpoint_files = sorted(models_dir.glob("model_iter_*.json"))
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found in {models_dir}")
        return []

    # Analyze each checkpoint
    iterations_metrics = []
    for checkpoint_file in checkpoint_files:
        try:
            metrics = analyze_checkpoint(
                checkpoint_file,
                ground_truth_domain,
                ground_truth_problem,
                problem_dir,
                min_observations=min_observations
            )
            iterations_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Error analyzing {checkpoint_file}: {e}")
            continue

    # Save per-iteration metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "metrics_per_iteration.json"

    # Format for JSON output
    metrics_by_iteration = {}
    for metrics in iterations_metrics:
        iteration = metrics['iteration']
        metrics_by_iteration[iteration] = {
            'safe_model': metrics['safe_model'],
            'complete_model': metrics['complete_model'],
            'execution_stats': metrics.get('execution_stats', {})
        }

    with open(output_file, 'w') as f:
        json.dump(metrics_by_iteration, f, indent=2)

    logger.info(f"  Saved metrics for {len(iterations_metrics)} iterations")

    return iterations_metrics


def aggregate_domain_metrics(
    domain_dir: Path,
    output_base: Path
) -> Dict[str, Any]:
    """
    Aggregate metrics across all problems in a domain.

    Args:
        domain_dir: Domain directory in consolidated_experiments
        output_base: Base output directory

    Returns:
        Dict with domain-level metrics
    """
    domain_name = domain_dir.name
    problem_dirs = [d for d in domain_dir.iterdir() if d.is_dir()]

    # Collect all problem metrics
    all_iterations = defaultdict(lambda: {'safe': defaultdict(list), 'complete': defaultdict(list)})

    for problem_dir in problem_dirs:
        problem_name = problem_dir.name
        metrics_file = output_base / domain_name / problem_name / "metrics_per_iteration.json"

        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            problem_metrics = json.load(f)

        # Aggregate by iteration
        for iteration_str, iter_metrics in problem_metrics.items():
            iteration = int(iteration_str)

            # Aggregate safe model metrics
            safe = iter_metrics['safe_model']
            all_iterations[iteration]['safe']['overall'].append({
                'precision': safe.get('precision', 0),
                'recall': safe.get('recall', 0),
                'f1': safe.get('f1', 0)
            })
            all_iterations[iteration]['safe']['preconditions'].append({
                'precision': safe.get('precondition_precision', 0),
                'recall': safe.get('precondition_recall', 0)
            })
            all_iterations[iteration]['safe']['effects'].append({
                'precision': safe.get('effect_precision', 0),
                'recall': safe.get('effect_recall', 0)
            })

            # Aggregate complete model metrics
            complete = iter_metrics['complete_model']
            all_iterations[iteration]['complete']['overall'].append({
                'precision': complete.get('precision', 0),
                'recall': complete.get('recall', 0),
                'f1': complete.get('f1', 0)
            })
            all_iterations[iteration]['complete']['preconditions'].append({
                'precision': complete.get('precondition_precision', 0),
                'recall': complete.get('precondition_recall', 0)
            })
            all_iterations[iteration]['complete']['effects'].append({
                'precision': complete.get('effect_precision', 0),
                'recall': complete.get('effect_recall', 0)
            })

    # Compute domain-level averages
    domain_metrics = {}
    for iteration in sorted(all_iterations.keys()):
        iter_data = all_iterations[iteration]

        domain_metrics[iteration] = {}

        for model_type in ['safe', 'complete']:
            domain_metrics[iteration][model_type] = {}

            for component in ['overall', 'preconditions', 'effects']:
                component_data = iter_data[model_type][component]

                if not component_data:
                    continue

                # Average precision, recall, F1
                avg_metrics = {
                    'precision': sum(m.get('precision', 0) for m in component_data) / len(component_data),
                    'recall': sum(m.get('recall', 0) for m in component_data) / len(component_data),
                    'problem_count': len(component_data)
                }

                # Add F1 if available
                if component == 'overall':
                    avg_metrics['f1'] = sum(m.get('f1', 0) for m in component_data) / len(component_data)

                domain_metrics[iteration][model_type][component] = avg_metrics

    # Save domain metrics
    output_dir = output_base / domain_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "domain_metrics.json"

    with open(output_file, 'w') as f:
        json.dump(domain_metrics, f, indent=2)

    logger.info(f"  Saved domain metrics across {len(problem_dirs)} problems")

    return domain_metrics


def export_csv(output_base: Path, csv_path: Path) -> None:
    """
    Export all metrics to a flat CSV file.

    Args:
        output_base: Base output directory with JSON metrics
        csv_path: Path to output CSV file
    """
    rows = []

    # Collect all problem-level metrics
    for domain_dir in sorted(output_base.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain_name = domain_dir.name

        for problem_dir in sorted(domain_dir.iterdir()):
            if not problem_dir.is_dir():
                continue

            problem_name = problem_dir.name
            metrics_file = problem_dir / "metrics_per_iteration.json"

            if not metrics_file.exists():
                continue

            with open(metrics_file) as f:
                problem_metrics = json.load(f)

            for iteration_str, iter_metrics in problem_metrics.items():
                iteration = int(iteration_str)

                for model_type in ['safe', 'complete']:
                    metrics = iter_metrics[f'{model_type}_model']

                    # Preconditions
                    prec_p = metrics.get('precondition_precision', 0)
                    prec_r = metrics.get('precondition_recall', 0)
                    prec_f1 = (2 * prec_p * prec_r / (prec_p + prec_r)) if (prec_p + prec_r) > 0 else 0
                    row = {
                        'algorithm': 'information_gain',
                        'domain': domain_name,
                        'problem': problem_name,
                        'iteration': iteration,
                        'model_type': model_type,
                        'component': 'preconditions',
                        'precision': prec_p,
                        'recall': prec_r,
                        'f1': prec_f1
                    }
                    rows.append(row)

                    # Effects (combined add + delete)
                    eff_p = metrics.get('effect_precision', 0)
                    eff_r = metrics.get('effect_recall', 0)
                    eff_f1 = (2 * eff_p * eff_r / (eff_p + eff_r)) if (eff_p + eff_r) > 0 else 0
                    row = {
                        'algorithm': 'information_gain',
                        'domain': domain_name,
                        'problem': problem_name,
                        'iteration': iteration,
                        'model_type': model_type,
                        'component': 'effects',
                        'precision': eff_p,
                        'recall': eff_r,
                        'f1': eff_f1
                    }
                    rows.append(row)

                    # Overall metrics
                    row = {
                        'algorithm': 'information_gain',
                        'domain': domain_name,
                        'problem': problem_name,
                        'iteration': iteration,
                        'model_type': model_type,
                        'component': 'overall',
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'f1': metrics.get('f1', 0)
                    }
                    rows.append(row)

    # Write CSV
    if rows:
        fieldnames = ['algorithm', 'domain', 'problem', 'iteration', 'model_type',
                     'component', 'precision', 'recall', 'f1']

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported {len(rows)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Information Gain learning metrics"
    )
    parser.add_argument(
        "--consolidated-dir",
        type=str,
        default="results/paper/consolidated_experiments",
        help="Directory containing consolidated experiments"
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=str,
        default="benchmarks/olam-compatible",
        help="Directory containing ground truth PDDL domains"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/information_gain_metrics",
        help="Output directory for metrics"
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Analyze only specific domain (optional)"
    )
    parser.add_argument(
        "--problem",
        type=str,
        help="Analyze only specific problem (requires --domain)"
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=0,
        help="Minimum observations for action to be included in metrics (default: 0, include all)"
    )

    args = parser.parse_args()

    consolidated_dir = Path(args.consolidated_dir) / "information_gain"
    benchmarks_dir = Path(args.benchmarks_dir)
    output_base = Path(args.output_dir)

    if not consolidated_dir.exists():
        logger.error(f"Consolidated directory not found: {consolidated_dir}")
        return 1

    logger.info("Information Gain Metrics Analysis")
    logger.info("=" * 60)
    logger.info(f"Input: {consolidated_dir}")
    logger.info(f"Ground truth: {benchmarks_dir}")
    logger.info(f"Output: {output_base}")
    logger.info("")

    if args.domain and args.problem:
        # Process single problem
        domain_name = args.domain
        problem_name = args.problem

        problem_dir = consolidated_dir / domain_name / problem_name
        if not problem_dir.exists():
            logger.error(f"Problem directory not found: {problem_dir}")
            return 1

        ground_truth_domain = benchmarks_dir / domain_name / "domain.pddl"
        ground_truth_problem = benchmarks_dir / domain_name / f"{problem_name}.pddl"

        if not ground_truth_domain.exists():
            logger.error(f"Ground truth domain not found: {ground_truth_domain}")
            return 1

        if not ground_truth_problem.exists():
            logger.error(f"Ground truth problem not found: {ground_truth_problem}")
            return 1

        logger.info(f"Analyzing {domain_name}/{problem_name}")
        output_dir = output_base / domain_name / problem_name
        analyze_problem(
            problem_dir, ground_truth_domain, ground_truth_problem, output_dir,
            min_observations=args.min_observations
        )

        logger.info("Done!")
        return 0

    # Process all domains
    domains = []
    for domain_dir in sorted(consolidated_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        if args.domain and domain_dir.name != args.domain:
            continue

        domain_name = domain_dir.name
        ground_truth_domain = benchmarks_dir / domain_name / "domain.pddl"

        if not ground_truth_domain.exists():
            logger.warning(f"Skipping {domain_name}: no ground truth domain found")
            continue

        domains.append((domain_name, domain_dir, ground_truth_domain))

    logger.info(f"Found {len(domains)} domains to analyze")
    logger.info("")

    # Process each domain
    for idx, (domain_name, domain_dir, ground_truth_domain) in enumerate(domains, 1):
        logger.info(f"[{idx}/{len(domains)}] {domain_name}")

        # Find all problems
        problem_dirs = [d for d in sorted(domain_dir.iterdir()) if d.is_dir()]
        logger.info(f"  Found {len(problem_dirs)} problems")

        # Analyze each problem
        for problem_dir in problem_dirs:
            problem_name = problem_dir.name
            ground_truth_problem = benchmarks_dir / domain_name / f"{problem_name}.pddl"

            if not ground_truth_problem.exists():
                logger.warning(f"  Skipping {problem_name}: no ground truth problem")
                continue

            logger.info(f"    Analyzing {problem_name}...")

            output_dir = output_base / domain_name / problem_name
            analyze_problem(
                problem_dir, ground_truth_domain, ground_truth_problem, output_dir,
                min_observations=args.min_observations
            )

        # Aggregate domain metrics
        logger.info(f"  Aggregating domain metrics...")
        aggregate_domain_metrics(domain_dir, output_base)

        logger.info("")

    # Export CSV
    logger.info("Exporting CSV...")
    csv_path = output_base / "checkpoint_metrics.csv"
    export_csv(output_base, csv_path)

    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_base}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
