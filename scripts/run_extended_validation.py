#!/usr/bin/env python
"""
Run extended 500-iteration validation for OLAM and Information Gain.
Executes experiments on depots domain and analyzes sustained learning behavior
with parameter-bound learning evidence.
"""

import sys
import logging
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def analyze_learning_evidence(experiment_name: str, algorithm_name: str):
    """
    Analyze parameter-bound learning evidence from experiment results.

    Args:
        experiment_name: Name of the experiment (from config)
        algorithm_name: Algorithm name for display
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{algorithm_name} - LEARNING EVIDENCE ANALYSIS")
    logger.info(f"{'=' * 80}")

    # Find result files
    result_dir = Path('results/experiments')
    json_files = list(result_dir.glob(f"{experiment_name}_*_metrics.json"))

    if not json_files:
        logger.warning("No detailed metrics JSON found")
        return

    # Load most recent JSON file
    json_file = sorted(json_files)[-1]
    logger.info(f"Loading learning evidence from: {json_file.name}")

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Analyze snapshots with learning evidence
    snapshots = data.get('snapshots', [])
    snapshots_with_evidence = [s for s in snapshots if 'learning_evidence' in s]

    if not snapshots_with_evidence:
        logger.warning("No learning evidence found in snapshots")
        return

    logger.info(f"\nLearning evidence captured in {len(snapshots_with_evidence)} snapshots")

    # Show initial and final state
    initial = snapshots_with_evidence[0]['learning_evidence']
    final = snapshots_with_evidence[-1]['learning_evidence']

    logger.info(f"\n{'=' * 60}")
    logger.info(f"INITIAL STATE (Iteration {snapshots_with_evidence[0]['step']})")
    logger.info(f"{'=' * 60}")

    sample_count = min(3, len(initial['actions']))
    for action_name, action_evidence in list(initial['actions'].items())[:sample_count]:
        logger.info(f"\n  Action: {action_name}")
        if 'preconditions_possible' in action_evidence:  # Information Gain
            logger.info(f"    Preconditions possible: {len(action_evidence['preconditions_possible'])} (parameter-bound)")
            if action_evidence['preconditions_possible']:
                logger.info(f"    Example: {action_evidence['preconditions_possible'][:2]}")
        elif 'preconditions_certain' in action_evidence:  # OLAM
            logger.info(f"    Preconditions certain: {len(action_evidence['preconditions_certain'])} (parameter-bound)")
            logger.info(f"    Preconditions uncertain: {len(action_evidence['preconditions_uncertain'])} (parameter-bound)")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"FINAL STATE (Iteration {snapshots_with_evidence[-1]['step']})")
    logger.info(f"{'=' * 60}")

    for action_name, action_evidence in list(final['actions'].items())[:sample_count]:
        logger.info(f"\n  Action: {action_name}")
        if 'preconditions_possible' in action_evidence:  # Information Gain
            initial_prec = len(initial['actions'][action_name]['preconditions_possible'])
            final_prec = len(action_evidence['preconditions_possible'])
            reduction = ((initial_prec - final_prec) / initial_prec * 100) if initial_prec > 0 else 0

            logger.info(f"    Preconditions possible: {final_prec} (reduced {reduction:.1f}% from {initial_prec})")
            logger.info(f"    Confirmed add effects: {len(action_evidence['effects_add_confirmed'])} (parameter-bound)")
            logger.info(f"    Confirmed del effects: {len(action_evidence['effects_del_confirmed'])} (parameter-bound)")
            logger.info(f"    Observations: {action_evidence['num_observations']}")

            if action_evidence['effects_add_confirmed']:
                logger.info(f"    Example add effects: {action_evidence['effects_add_confirmed'][:2]}")

        elif 'preconditions_certain' in action_evidence:  # OLAM
            logger.info(f"    Preconditions certain: {len(action_evidence['preconditions_certain'])} (learned)")
            logger.info(f"    Preconditions uncertain: {len(action_evidence['preconditions_uncertain'])}")
            logger.info(f"    Positive effects: {len(action_evidence['effects_positive'])} (parameter-bound)")
            logger.info(f"    Negative effects: {len(action_evidence['effects_negative'])} (parameter-bound)")

            if action_evidence['preconditions_certain']:
                logger.info(f"    Example certain preconditions: {action_evidence['preconditions_certain'][:2]}")


def run_olam_validation():
    """Run OLAM 500-iteration experiment on depots."""
    logger.info("=" * 80)
    logger.info("OLAM 500-ITERATION EXTENDED VALIDATION (DEPOTS DOMAIN)")
    logger.info("=" * 80)
    logger.info("Running experiment with parameter-bound learning evidence tracking...")

    runner = ExperimentRunner('configs/olam_depots_500iter.yaml')
    results = runner.run_experiment()

    # Basic analysis
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EXPERIMENT SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Total iterations: {results['total_iterations']}")
    logger.info(f"  Stopping reason: {results['stopping_reason']}")
    logger.info(f"  Runtime: {results['runtime_seconds']:.1f} seconds ({results['runtime_seconds']/60:.1f} minutes)")
    logger.info(f"  Total actions: {results['metrics']['total_actions']}")
    logger.info(f"  Success rate: {results['metrics']['success_rate']:.1%}")
    logger.info(f"  Final mistake rate: {results['metrics']['mistake_rate']:.1%}")

    # Analyze parameter-bound learning evidence
    analyze_learning_evidence(results['experiment_name'], "OLAM")

    return results


def run_information_gain_validation():
    """Run Information Gain 500-iteration experiment on depots."""
    logger.info("\n" + "=" * 80)
    logger.info("INFORMATION GAIN 500-ITERATION EXTENDED VALIDATION (DEPOTS DOMAIN)")
    logger.info("=" * 80)
    logger.info("Running experiment with parameter-bound learning evidence tracking...")

    runner = ExperimentRunner('configs/information_gain_depots_500iter.yaml')
    results = runner.run_experiment()

    # Basic analysis
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EXPERIMENT SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Total iterations: {results['total_iterations']}")
    logger.info(f"  Stopping reason: {results['stopping_reason']}")
    logger.info(f"  Runtime: {results['runtime_seconds']:.1f} seconds ({results['runtime_seconds']/60:.1f} minutes)")
    logger.info(f"  Total actions: {results['metrics']['total_actions']}")
    logger.info(f"  Success rate: {results['metrics']['success_rate']:.1%}")
    logger.info(f"  Final mistake rate: {results['metrics']['mistake_rate']:.1%}")

    # Analyze parameter-bound learning evidence
    analyze_learning_evidence(results['experiment_name'], "Information Gain")

    return results


def main():
    """Run both validation experiments and generate comparison summary."""

    logger.info("=" * 80)
    logger.info("PHASE 2.5: EXTENDED VALIDATION EXPERIMENTS")
    logger.info("Running 500-iteration experiments on depots domain")
    logger.info("=" * 80)

    # Run OLAM validation
    olam_results = run_olam_validation()

    # Run Information Gain validation
    ig_results = run_information_gain_validation()

    # Comparison summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY - SUSTAINED LEARNING OVER 500 ITERATIONS")
    logger.info("=" * 80)

    logger.info(f"\nOLAM:")
    logger.info(f"  Iterations completed: {olam_results['total_iterations']}")
    logger.info(f"  Success rate: {olam_results['metrics']['success_rate']:.1%}")
    logger.info(f"  Runtime: {olam_results['runtime_seconds']/60:.1f} minutes")
    logger.info(f"  Stopping reason: {olam_results['stopping_reason']}")

    logger.info(f"\nInformation Gain:")
    logger.info(f"  Iterations completed: {ig_results['total_iterations']}")
    logger.info(f"  Success rate: {ig_results['metrics']['success_rate']:.1%}")
    logger.info(f"  Runtime: {ig_results['runtime_seconds']/60:.1f} minutes")
    logger.info(f"  Stopping reason: {ig_results['stopping_reason']}")

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2.5 COMPLETE")
    logger.info("=" * 80)
    logger.info("\nDetailed parameter-bound learning traces saved to:")
    logger.info("  - results/experiments/*olam_depots_500iter*_metrics.json")
    logger.info("  - results/experiments/*information_gain_depots_500iter*_metrics.json")
    logger.info("\nEach JSON file contains:")
    logger.info("  - Complete action execution history")
    logger.info("  - Snapshots with parameter-bound learning evidence")
    logger.info("  - Preconditions: certain/uncertain/possible (parameter-bound literals)")
    logger.info("  - Effects: confirmed/possible (parameter-bound literals)")
    logger.info("\nNext steps:")
    logger.info("  1. Review learning traces for sustained learning behavior")
    logger.info("  2. Validate conservative convergence parameters worked correctly")
    logger.info("  3. Decide on Phase 3 scope (full comparison pipeline vs minimal)")


if __name__ == "__main__":
    main()
