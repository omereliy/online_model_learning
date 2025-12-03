#!/usr/bin/env python3
"""
Analyze Information Gain learning results and compute precision/recall metrics.

This script processes model checkpoints from the information gain algorithm,
compares them against ground truth PDDL domains, and computes:
- Preconditions precision/recall per iteration
- Effects (add/delete) precision/recall per iteration
- Domain-level aggregated metrics

Results are saved in a new directory structure:
  results/information_gain_metrics/{domain}/{problem}/metrics_per_iteration.json
  results/information_gain_metrics/{domain}/domain_metrics.json
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pddl_parser import PDDLDomainParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_ground_truth_domain(domain_file: Path) -> Dict[str, Dict[str, Set[str]]]:
    """
    Parse ground truth PDDL domain to extract action schemas.

    Returns:
        Dict mapping action names to their preconditions and effects:
        {
            "action_name": {
                "preconditions": set of precondition literals,
                "add_effects": set of add effect literals,
                "delete_effects": set of delete effect literals
            }
        }
    """
    parser = PDDLDomainParser(domain_file)
    actions = {}

    for action_name, action_schema in parser.actions.items():
        # Extract preconditions
        preconditions = set()
        if action_schema.precondition:
            # Handle different precondition formats
            prec_literals = _extract_literals(action_schema.precondition)
            preconditions = set(prec_literals)

        # Extract effects
        add_effects = set()
        delete_effects = set()
        if action_schema.effect:
            adds, deletes = _extract_effects(action_schema.effect)
            add_effects = set(adds)
            delete_effects = set(deletes)

        actions[action_name] = {
            "preconditions": preconditions,
            "add_effects": add_effects,
            "delete_effects": delete_effects
        }

    return actions


def _extract_literals(precondition) -> List[str]:
    """Extract literal strings from precondition structure."""
    literals = []

    if hasattr(precondition, '__iter__') and not isinstance(precondition, str):
        for item in precondition:
            literals.extend(_extract_literals(item))
    elif hasattr(precondition, 'predicates'):
        # AND/OR structure
        literals.extend(_extract_literals(precondition.predicates))
    elif hasattr(precondition, 'predicate'):
        # Single predicate
        lit = str(precondition)
        literals.append(lit)
    else:
        # Already a string
        lit = str(precondition)
        if lit:
            literals.append(lit)

    return literals


def _extract_effects(effect) -> Tuple[List[str], List[str]]:
    """Extract add and delete effects from effect structure."""
    add_effects = []
    delete_effects = []

    if hasattr(effect, 'add_effects'):
        add_effects = [str(e) for e in effect.add_effects]
    if hasattr(effect, 'delete_effects'):
        delete_effects = [str(e) for e in effect.delete_effects]

    return add_effects, delete_effects


def normalize_literal(literal: str) -> str:
    """
    Normalize a literal for comparison.

    Handles:
    - Removing whitespace variations
    - Converting ¬ to (not ...)
    - Standardizing parameter names
    """
    # Remove extra whitespace
    literal = ' '.join(literal.split())

    # Convert unicode negation to (not ...)
    if literal.startswith('¬'):
        pred = literal[1:]
        literal = f"(not {pred})"

    return literal


def parse_learned_model(checkpoint_file: Path) -> Dict[str, Dict[str, Set[str]]]:
    """
    Parse learned model checkpoint JSON.

    Returns same structure as parse_ground_truth_domain:
    {
        "action_name": {
            "preconditions": set of certain precondition literals,
            "add_effects": set of certain add effects,
            "delete_effects": set of certain delete effects
        }
    }
    """
    with open(checkpoint_file) as f:
        data = json.load(f)

    actions = {}

    for action_name, action_data in data.get('actions', {}).items():
        # Use only "possible" preconditions (those marked as certain/required)
        # In information gain, "possible" means literals that could be preconditions
        # We want the ones that are confirmed (not in "constraints")
        preconditions = set()
        if 'preconditions' in action_data:
            prec_data = action_data['preconditions']
            # "possible" contains literals that could be preconditions
            # "constraints" contains excluded ones
            # So actual preconditions are in "possible" minus "constraints"
            possible = set(prec_data.get('possible', []))
            constraints = set(prec_data.get('constraints', []))
            preconditions = possible - constraints

        # Add effects (certain adds only)
        add_effects = set(action_data.get('effects', {}).get('add', []))

        # Delete effects (certain deletes only)
        delete_effects = set(action_data.get('effects', {}).get('delete', []))

        actions[action_name] = {
            "preconditions": {normalize_literal(lit) for lit in preconditions},
            "add_effects": {normalize_literal(lit) for lit in add_effects},
            "delete_effects": {normalize_literal(lit) for lit in delete_effects}
        }

    return actions


def calculate_precision_recall(learned: Set[str], ground_truth: Set[str]) -> Dict[str, float]:
    """
    Calculate precision and recall.

    Precision: |learned ∩ ground_truth| / |learned|
    Recall: |learned ∩ ground_truth| / |ground_truth|
    """
    if not learned and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not learned:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = learned & ground_truth
    precision = len(intersection) / len(learned) if learned else 0.0
    recall = len(intersection) / len(ground_truth) if ground_truth else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "learned_count": len(learned),
        "ground_truth_count": len(ground_truth),
        "correct_count": len(intersection)
    }


def analyze_checkpoint(checkpoint_file: Path, ground_truth: Dict[str, Dict[str, Set[str]]]) -> Dict[str, Any]:
    """
    Analyze a single checkpoint file against ground truth.

    Returns metrics for this iteration.
    """
    learned = parse_learned_model(checkpoint_file)

    # Per-action metrics
    action_metrics = {}

    for action_name in ground_truth.keys():
        if action_name not in learned:
            # Action not yet learned
            action_metrics[action_name] = {
                "preconditions": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "add_effects": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "delete_effects": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            }
            continue

        gt = ground_truth[action_name]
        ln = learned[action_name]

        action_metrics[action_name] = {
            "preconditions": calculate_precision_recall(ln["preconditions"], gt["preconditions"]),
            "add_effects": calculate_precision_recall(ln["add_effects"], gt["add_effects"]),
            "delete_effects": calculate_precision_recall(ln["delete_effects"], gt["delete_effects"])
        }

    # Aggregate across actions
    aggregate = {
        "preconditions": {"precision": [], "recall": [], "f1": []},
        "add_effects": {"precision": [], "recall": [], "f1": []},
        "delete_effects": {"precision": [], "recall": [], "f1": []}
    }

    for action_name, metrics in action_metrics.items():
        for component in ["preconditions", "add_effects", "delete_effects"]:
            aggregate[component]["precision"].append(metrics[component]["precision"])
            aggregate[component]["recall"].append(metrics[component]["recall"])
            aggregate[component]["f1"].append(metrics[component]["f1"])

    # Average across actions
    aggregated_metrics = {}
    for component in ["preconditions", "add_effects", "delete_effects"]:
        aggregated_metrics[component] = {
            "precision": sum(aggregate[component]["precision"]) / len(aggregate[component]["precision"]),
            "recall": sum(aggregate[component]["recall"]) / len(aggregate[component]["recall"]),
            "f1": sum(aggregate[component]["f1"]) / len(aggregate[component]["f1"])
        }

    return {
        "per_action": action_metrics,
        "aggregated": aggregated_metrics
    }


def analyze_problem(problem_dir: Path, ground_truth_domain: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Analyze all checkpoints for a single problem.

    Returns:
        Dict with per-iteration metrics
    """
    # Parse ground truth
    ground_truth = parse_ground_truth_domain(ground_truth_domain)

    # Find checkpoint files
    checkpoints_dir = problem_dir / "models"
    if not checkpoints_dir.exists():
        logger.warning(f"No models directory found in {problem_dir}")
        return {}

    checkpoint_files = sorted(checkpoints_dir.glob("model_iter_*.json"))
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found in {checkpoints_dir}")
        return {}

    # Analyze each checkpoint
    iterations = {}
    for checkpoint_file in checkpoint_files:
        # Extract iteration number from filename
        # Format: model_iter_01.json or model_iter_005.json
        filename = checkpoint_file.stem
        iter_str = filename.split('_')[-1]
        iteration = int(iter_str)

        metrics = analyze_checkpoint(checkpoint_file, ground_truth)
        iterations[iteration] = metrics

    # Save per-iteration metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "metrics_per_iteration.json"

    with open(output_file, 'w') as f:
        json.dump(iterations, f, indent=2)

    logger.info(f"  Saved metrics for {len(iterations)} iterations")

    return iterations


def aggregate_domain_metrics(domain_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Aggregate metrics across all problems in a domain.

    Computes mean precision/recall across problems for each iteration.
    """
    problem_dirs = [d for d in domain_dir.iterdir() if d.is_dir()]

    # Collect all problem metrics
    all_iterations = defaultdict(lambda: defaultdict(list))

    for problem_dir in problem_dirs:
        problem_name = problem_dir.name
        metrics_file = output_dir / problem_name / "metrics_per_iteration.json"

        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            problem_metrics = json.load(f)

        # Aggregate by iteration
        for iteration, iter_metrics in problem_metrics.items():
            iteration = int(iteration)

            for component in ["preconditions", "add_effects", "delete_effects"]:
                agg = iter_metrics["aggregated"][component]
                all_iterations[iteration][component].append(agg)

    # Compute domain-level averages
    domain_metrics = {}
    for iteration in sorted(all_iterations.keys()):
        iter_data = all_iterations[iteration]

        domain_metrics[iteration] = {}
        for component in ["preconditions", "add_effects", "delete_effects"]:
            component_data = iter_data[component]

            domain_metrics[iteration][component] = {
                "precision": sum(m["precision"] for m in component_data) / len(component_data),
                "recall": sum(m["recall"] for m in component_data) / len(component_data),
                "f1": sum(m["f1"] for m in component_data) / len(component_data),
                "problem_count": len(component_data)
            }

    # Save domain metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "domain_metrics.json"

    with open(output_file, 'w') as f:
        json.dump(domain_metrics, f, indent=2)

    logger.info(f"  Saved domain metrics across {len(problem_dirs)} problems")

    return domain_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Information Gain learning results"
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

    # Find all domains
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
            logger.info(f"    Analyzing {problem_name}...")

            output_dir = output_base / domain_name / problem_name
            analyze_problem(problem_dir, ground_truth_domain, output_dir)

        # Aggregate domain metrics
        logger.info(f"  Aggregating domain metrics...")
        domain_output_dir = output_base / domain_name
        aggregate_domain_metrics(domain_dir, domain_output_dir)

        logger.info("")

    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_base}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
