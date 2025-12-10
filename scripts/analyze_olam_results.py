#!/usr/bin/python3
"""
Complete pipeline for analyzing OLAM results through post-processing.

This script:
1. Runs OLAM externally or uses existing results
2. Parses the execution trace
3. Reconstructs models at specified checkpoints
4. Computes metrics (precision/recall) against ground truth

Usage:
    python analyze_olam_results.py \
        --domain benchmarks/olam-compatible/blocksworld/domain.pddl \
        --problem benchmarks/olam-compatible/blocksworld/p01.pddl \
        --checkpoints 5 10 20 50 100 \
        --output-dir results/olam_analysis
"""

import argparse
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.algorithms.olam_external_runner import OLAMExternalRunner, OLAMRunResult
from src.core.olam_trace_parser import OLAMTraceParser
from src.core.olam_knowledge_reconstructor import OLAMKnowledgeReconstructor
from src.core.model_reconstructor import ModelReconstructor, ReconstructedModel
from src.core.model_metrics import ModelMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_olam_experiment(
    domain_file: Path,
    problem_file: Path,
    config: Dict,
    output_dir: Path,
    run_olam: bool = True
) -> OLAMRunResult:
    """
    Run OLAM experiment or load existing results.

    Args:
        domain_file: Path to domain PDDL
        problem_file: Path to problem PDDL
        config: OLAM configuration
        output_dir: Directory for outputs
        run_olam: If True, run OLAM; if False, expect existing results

    Returns:
        OLAMRunResult with trace and exports
    """
    if run_olam:
        logger.info("Running OLAM experiment...")
        runner = OLAMExternalRunner()
        result = runner.run_experiment(
            domain_file=domain_file,
            problem_file=problem_file,
            config=config,
            output_dir=output_dir / "olam_run"
        )

        if not result.success:
            logger.error(f"OLAM execution failed: {result.error_message}")
            sys.exit(1)

        return result
    else:
        # Load existing results
        logger.info("Loading existing OLAM results...")
        trace_file = output_dir / "olam_run" / "trace.log"
        # OLAM stores exports in complete_run/Tests/<domain>/ structure
        exports_dir = output_dir / "olam_run" / "complete_run"
        if not exports_dir.exists():
            # Fallback to old location for backward compatibility
            exports_dir = output_dir / "olam_run" / "exports"

        if not trace_file.exists():
            logger.error(f"Trace file not found: {trace_file}")
            sys.exit(1)

        return OLAMRunResult(
            success=True,
            trace_file=trace_file,
            exports_dir=exports_dir
        )


def parse_trace(trace_file: Path) -> List:
    """
    Parse OLAM execution trace.

    Args:
        trace_file: Path to OLAM trace log

    Returns:
        List of OLAMTraceStep objects
    """
    logger.info(f"Parsing trace from {trace_file}")
    parser = OLAMTraceParser()
    trace = parser.parse_log_file(trace_file)
    logger.info(f"Parsed {len(trace)} trace steps")
    return trace


def reconstruct_models_at_checkpoints(
    trace: List,
    checkpoints: List[int],
    domain_file: Path,
    problem_file: Path,
    output_dir: Path
) -> Dict[int, Dict]:
    """
    Reconstruct models at specified checkpoints.

    Args:
        trace: Parsed execution trace
        checkpoints: List of iterations to checkpoint
        domain_file: Domain file for operator extraction
        problem_file: Problem file for identifying output subdirectory
        output_dir: Directory for model snapshots

    Returns:
        Dictionary mapping checkpoint to reconstructed models
    """
    logger.info(f"Reconstructing models at checkpoints: {checkpoints}")

    reconstructor = OLAMKnowledgeReconstructor(domain_file)
    models = {}

    # Extract problem identifier (e.g., "p01" from "p01.pddl")
    problem_name = problem_file.stem  # Removes .pddl extension

    # Create per-problem directory to prevent overwrites
    models_dir = output_dir / "models" / problem_name
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving models to {models_dir}")

    for checkpoint in checkpoints:
        logger.info(f"Reconstructing at checkpoint {checkpoint}")

        # Replay trace to checkpoint
        knowledge = reconstructor.replay_to_checkpoint(trace, checkpoint)

        # Export snapshot
        snapshot = reconstructor.export_snapshot(knowledge, checkpoint)

        # Save snapshot in per-problem directory
        snapshot_path = models_dir / f"model_iter_{checkpoint:03d}.json"
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f, indent=2)

        # Reconstruct safe and complete models
        # Pass domain_file for fair safe model preconditions (uses La for unexecuted actions)
        safe = ModelReconstructor.reconstruct_olam_safe(snapshot, domain_file=domain_file)
        complete = ModelReconstructor.reconstruct_olam_complete(snapshot)

        models[checkpoint] = {
            'snapshot': snapshot,
            'safe': safe,
            'complete': complete,
            'knowledge': knowledge
        }

    return models


def compute_metrics(
    models: Dict[int, Dict],
    ground_truth_file: Path,
    problem_file: Path,
    min_observations: int = 0
) -> pd.DataFrame:
    """
    Compute precision and recall metrics for each checkpoint.

    Args:
        models: Dictionary of reconstructed models
        ground_truth_file: Path to ground truth domain
        problem_file: Path to problem file
        min_observations: Minimum observations to include action in metrics (default 0)

    Returns:
        DataFrame with metrics
    """
    logger.info("Computing metrics...")

    # Load ground truth
    metrics_calculator = ModelMetrics(ground_truth_file, problem_file)
    results = []

    for checkpoint, model_data in sorted(models.items()):
        logger.info(f"Computing metrics for checkpoint {checkpoint}")

        safe_model = model_data['safe']
        complete_model = model_data['complete']
        knowledge = model_data['knowledge']

        # Extract observation counts from knowledge for filtering
        observation_counts = {
            action_name: knowledge.observation_count.get(action_name, 0)
            for action_name in safe_model.actions.keys()
        }

        # Compute metrics for safe model (with optional filtering)
        safe_metrics = metrics_calculator.compute_metrics(
            safe_model,
            observation_counts=observation_counts,
            min_observations=min_observations
        )

        # Compute metrics for complete model (with optional filtering)
        complete_metrics = metrics_calculator.compute_metrics(
            complete_model,
            observation_counts=observation_counts,
            min_observations=min_observations
        )

        # Aggregate statistics
        total_observations = sum(knowledge.observation_count.values())
        total_successes = sum(knowledge.successful_observations.values())
        total_failures = sum(knowledge.failed_observations.values())

        results.append({
            'iteration': checkpoint,
            'observations': total_observations,
            'successes': total_successes,
            'failures': total_failures,
            'safe_precision': safe_metrics.get('precision', 0.0),
            'safe_recall': safe_metrics.get('recall', 0.0),
            'safe_f1': safe_metrics.get('f1', 0.0),
            'complete_precision': complete_metrics.get('precision', 0.0),
            'complete_recall': complete_metrics.get('recall', 0.0),
            'complete_f1': complete_metrics.get('f1', 0.0),
            'num_operators': len(knowledge.certain_precs),
            'num_certain_precs': sum(len(p) for p in knowledge.certain_precs.values()),
            'num_uncertain_precs': sum(len(p) for p in knowledge.uncertain_precs.values()),
            'num_add_effects': sum(len(e) for e in knowledge.add_effects.values()),
            'num_del_effects': sum(len(e) for e in knowledge.del_effects.values())
        })

    return pd.DataFrame(results)


def find_problem_directories(exports_dir: Path, domain_name: str) -> List[Tuple[str, Path]]:
    """
    Find all problem directories in OLAM exports.

    Args:
        exports_dir: OLAM exports directory (complete_run/)
        domain_name: Name of the domain (e.g., "gripper")

    Returns:
        List of (problem_id, problem_dir) tuples sorted by problem ID
    """
    tests_dir = exports_dir / "Tests" / domain_name
    if not tests_dir.exists():
        logger.warning(f"Tests directory not found: {tests_dir}")
        return []

    problem_dirs = []
    for item in tests_dir.iterdir():
        if item.is_dir():
            # Extract problem ID from directory name (e.g., "p01" from "2_p01_gripper_gen")
            parts = item.name.split('_')
            if len(parts) >= 2:
                problem_id = parts[1]  # e.g., "p01"
                problem_dirs.append((problem_id, item))
            else:
                logger.warning(f"Could not extract problem ID from: {item.name}")

    # Sort by problem ID for consistent ordering
    problem_dirs.sort(key=lambda x: x[0])
    logger.info(f"Found {len(problem_dirs)} problem directories for domain {domain_name}")
    return problem_dirs


def find_checkpoint_domains(problem_dir: Path, checkpoints: List[int]) -> Dict[int, Dict[str, Path]]:
    """
    Find safe and complete domain files at each checkpoint.

    Args:
        problem_dir: Directory with OLAM exports
        checkpoints: List of checkpoint iterations

    Returns:
        Dict mapping checkpoint -> {'safe': path, 'complete': path}
    """
    checkpoint_domains = {}

    for checkpoint in checkpoints:
        safe_path = problem_dir / f"domain_learned_safe_iter_{checkpoint}.pddl"
        complete_path = problem_dir / f"domain_learned_complete_iter_{checkpoint}.pddl"

        # At least one must exist
        if safe_path.exists() or complete_path.exists():
            checkpoint_domains[checkpoint] = {}
            if safe_path.exists():
                checkpoint_domains[checkpoint]['safe'] = safe_path
            if complete_path.exists():
                checkpoint_domains[checkpoint]['complete'] = complete_path

    return checkpoint_domains


def process_problem_exports(
    problem_dir: Path,
    ground_truth_domain: Path,
    problem_file: Path,
    checkpoints: List[int]
) -> Dict:
    """
    Process OLAM exports for a single problem at multiple checkpoints.
    Computes metrics for both safe and complete domains.

    Args:
        problem_dir: Directory with OLAM exports for this problem
        ground_truth_domain: Path to ground truth domain
        problem_file: Path to problem file
        checkpoints: List of checkpoint iterations to process

    Returns:
        Dictionary with checkpoint metrics for safe and complete domains
        {
            'checkpoints': {
                5: {
                    'safe': {prec/eff metrics},
                    'complete': {prec/eff metrics}
                },
                ...
            }
        }
    """
    # Find safe and complete domain files at each checkpoint
    checkpoint_domains = find_checkpoint_domains(problem_dir, checkpoints)

    if not checkpoint_domains:
        logger.warning(f"No checkpoint domains found in {problem_dir}")
        return {'checkpoints': {}}

    # Process each checkpoint
    checkpoint_metrics = {}
    metrics_calculator = ModelMetrics(ground_truth_domain, problem_file)

    for checkpoint, domains in checkpoint_domains.items():
        checkpoint_metrics[checkpoint] = {}

        # Process safe domain
        if 'safe' in domains:
            try:
                safe_metrics = metrics_calculator.compute_metrics_from_pddl(domains['safe'])
                # Extract separated precondition and effect metrics
                checkpoint_metrics[checkpoint]['safe'] = {
                    'precondition_precision': safe_metrics.get('precondition_precision', 0.0),
                    'precondition_recall': safe_metrics.get('precondition_recall', 0.0),
                    'effect_precision': safe_metrics.get('effect_precision', 0.0),
                    'effect_recall': safe_metrics.get('effect_recall', 0.0),
                    'overall_precision': safe_metrics.get('precision', 0.0),
                    'overall_recall': safe_metrics.get('recall', 0.0)
                }
            except Exception as e:
                logger.error(f"Error processing safe domain at checkpoint {checkpoint}: {e}")
                checkpoint_metrics[checkpoint]['safe'] = None

        # Process complete domain
        if 'complete' in domains:
            try:
                complete_metrics = metrics_calculator.compute_metrics_from_pddl(domains['complete'])
                # Extract separated precondition and effect metrics
                checkpoint_metrics[checkpoint]['complete'] = {
                    'precondition_precision': complete_metrics.get('precondition_precision', 0.0),
                    'precondition_recall': complete_metrics.get('precondition_recall', 0.0),
                    'effect_precision': complete_metrics.get('effect_precision', 0.0),
                    'effect_recall': complete_metrics.get('effect_recall', 0.0),
                    'overall_precision': complete_metrics.get('precision', 0.0),
                    'overall_recall': complete_metrics.get('recall', 0.0)
                }
            except Exception as e:
                logger.error(f"Error processing complete domain at checkpoint {checkpoint}: {e}")
                checkpoint_metrics[checkpoint]['complete'] = None

    return {'checkpoints': checkpoint_metrics}


def validate_against_olam_exports(
    reconstructed_models: Dict[int, Dict],
    exports_dir: Path
) -> Dict[str, float]:
    """
    Validate reconstructed models against OLAM's native exports.

    Args:
        reconstructed_models: Models reconstructed from trace
        exports_dir: Directory with OLAM's JSON exports

    Returns:
        Dictionary with validation metrics
    """
    logger.info("Validating against OLAM exports...")

    # Load OLAM's final exports
    parser = OLAMTraceParser()
    olam_exports = parser.parse_json_exports(exports_dir)

    # Reconstruct knowledge from exports
    reconstructor = OLAMKnowledgeReconstructor()
    olam_knowledge = reconstructor.reconstruct_from_exports(olam_exports)

    # Find the final checkpoint
    final_checkpoint = max(reconstructed_models.keys())
    reconstructed_knowledge = reconstructed_models[final_checkpoint]['knowledge']

    # Compare
    metrics = reconstructor.compare_knowledge(reconstructed_knowledge, olam_knowledge)

    logger.info(f"Validation similarity: {metrics.get('overall_similarity', 0):.2%}")
    return metrics


def save_per_problem_metrics(
    problem_id: str,
    metrics: Dict,
    output_dir: Path
):
    """
    Save metrics for a single problem (with checkpoint structure).

    Args:
        problem_id: Problem identifier (e.g., "p01")
        metrics: Metrics dictionary with checkpoint structure
        output_dir: Output directory
    """
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{problem_id}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics for {problem_id} to {metrics_path}")


def aggregate_domain_metrics(per_problem_metrics: Dict[str, Dict]) -> Dict:
    """
    Aggregate per-problem metrics to domain level.
    Handles checkpoints, safe/complete domains, and separated prec/effect metrics.

    Args:
        per_problem_metrics: Dictionary mapping problem_id to checkpoint metrics
            {
                'p01': {'checkpoints': {5: {'safe': {...}, 'complete': {...}}, ...}},
                ...
            }

    Returns:
        Domain-level aggregated metrics:
        {
            'checkpoints': {
                5: {
                    'safe': {
                        'preconditions': {avg/min/max/std precision/recall},
                        'effects': {avg/min/max/std precision/recall}
                    },
                    'complete': {...}
                },
                ...
            },
            'per_problem_metrics': {...}
        }
    """
    if not per_problem_metrics:
        return {'checkpoints': {}, 'per_problem_metrics': {}}

    # Helper functions
    def safe_mean(values):
        return sum(values) / len(values) if values else 0.0

    def safe_std(values):
        if not values or len(values) < 2:
            return 0.0
        mean = safe_mean(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def safe_min(values):
        return min(values) if values else 0.0

    def safe_max(values):
        return max(values) if values else 0.0

    # Collect all checkpoints across problems
    all_checkpoints = set()
    for problem_metrics in per_problem_metrics.values():
        if 'checkpoints' in problem_metrics:
            all_checkpoints.update(problem_metrics['checkpoints'].keys())

    if not all_checkpoints:
        return {'checkpoints': {}, 'per_problem_metrics': per_problem_metrics}

    # Aggregate for each checkpoint
    aggregated_checkpoints = {}

    for checkpoint in sorted(all_checkpoints):
        aggregated_checkpoints[checkpoint] = {}

        # Aggregate safe domain metrics
        safe_prec_precision = []
        safe_prec_recall = []
        safe_eff_precision = []
        safe_eff_recall = []

        # Aggregate complete domain metrics
        complete_prec_precision = []
        complete_prec_recall = []
        complete_eff_precision = []
        complete_eff_recall = []

        # Collect metrics from all problems at this checkpoint
        for problem_id, problem_metrics in per_problem_metrics.items():
            if 'checkpoints' not in problem_metrics:
                continue
            if checkpoint not in problem_metrics['checkpoints']:
                continue

            cp_metrics = problem_metrics['checkpoints'][checkpoint]

            # Safe domain
            if 'safe' in cp_metrics and cp_metrics['safe']:
                safe = cp_metrics['safe']
                safe_prec_precision.append(safe.get('precondition_precision', 0.0))
                safe_prec_recall.append(safe.get('precondition_recall', 0.0))
                safe_eff_precision.append(safe.get('effect_precision', 0.0))
                safe_eff_recall.append(safe.get('effect_recall', 0.0))

            # Complete domain
            if 'complete' in cp_metrics and cp_metrics['complete']:
                complete = cp_metrics['complete']
                complete_prec_precision.append(complete.get('precondition_precision', 0.0))
                complete_prec_recall.append(complete.get('precondition_recall', 0.0))
                complete_eff_precision.append(complete.get('effect_precision', 0.0))
                complete_eff_recall.append(complete.get('effect_recall', 0.0))

        # Build safe domain summary
        if safe_prec_precision:  # At least one safe domain at this checkpoint
            aggregated_checkpoints[checkpoint]['safe'] = {
                'problems_analyzed': len(safe_prec_precision),
                'preconditions': {
                    'avg_precision': safe_mean(safe_prec_precision),
                    'avg_recall': safe_mean(safe_prec_recall),
                    'min_precision': safe_min(safe_prec_precision),
                    'max_precision': safe_max(safe_prec_precision),
                    'min_recall': safe_min(safe_prec_recall),
                    'max_recall': safe_max(safe_prec_recall),
                    'std_precision': safe_std(safe_prec_precision),
                    'std_recall': safe_std(safe_prec_recall)
                },
                'effects': {
                    'avg_precision': safe_mean(safe_eff_precision),
                    'avg_recall': safe_mean(safe_eff_recall),
                    'min_precision': safe_min(safe_eff_precision),
                    'max_precision': safe_max(safe_eff_precision),
                    'min_recall': safe_min(safe_eff_recall),
                    'max_recall': safe_max(safe_eff_recall),
                    'std_precision': safe_std(safe_eff_precision),
                    'std_recall': safe_std(safe_eff_recall)
                }
            }

        # Build complete domain summary
        if complete_prec_precision:  # At least one complete domain at this checkpoint
            aggregated_checkpoints[checkpoint]['complete'] = {
                'problems_analyzed': len(complete_prec_precision),
                'preconditions': {
                    'avg_precision': safe_mean(complete_prec_precision),
                    'avg_recall': safe_mean(complete_prec_recall),
                    'min_precision': safe_min(complete_prec_precision),
                    'max_precision': safe_max(complete_prec_precision),
                    'min_recall': safe_min(complete_prec_recall),
                    'max_recall': safe_max(complete_prec_recall),
                    'std_precision': safe_std(complete_prec_precision),
                    'std_recall': safe_std(complete_prec_recall)
                },
                'effects': {
                    'avg_precision': safe_mean(complete_eff_precision),
                    'avg_recall': safe_mean(complete_eff_recall),
                    'min_precision': safe_min(complete_eff_precision),
                    'max_precision': safe_max(complete_eff_precision),
                    'min_recall': safe_min(complete_eff_recall),
                    'max_recall': safe_max(complete_eff_recall),
                    'std_precision': safe_std(complete_eff_precision),
                    'std_recall': safe_std(complete_eff_recall)
                }
            }

    return {
        'checkpoints': aggregated_checkpoints,
        'per_problem_metrics': per_problem_metrics
    }


def save_results(
    metrics_df: pd.DataFrame,
    validation: Dict,
    output_dir: Path,
    domain_metrics: Dict = None
):
    """
    Save analysis results.

    Args:
        metrics_df: DataFrame with checkpoint metrics
        validation: Validation results
        output_dir: Output directory
        domain_metrics: Optional domain-level metrics from per-problem analysis
    """
    # Save metrics CSV
    metrics_path = output_dir / "checkpoint_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save validation results
    validation_path = output_dir / "validation.json"
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2)
    logger.info(f"Saved validation to {validation_path}")

    # Create summary report with domain metrics if available
    summary = {
        'final_iteration': int(metrics_df['iteration'].max()),
        'total_observations': int(metrics_df['observations'].max()),
        'final_safe_precision': float(metrics_df.iloc[-1]['safe_precision']),
        'final_safe_recall': float(metrics_df.iloc[-1]['safe_recall']),
        'final_complete_precision': float(metrics_df.iloc[-1]['complete_precision']),
        'final_complete_recall': float(metrics_df.iloc[-1]['complete_recall']),
        'validation_similarity': float(validation.get('overall_similarity', 0))
    }

    # Add domain metrics if available (checkpoint-based per-problem analysis)
    if domain_metrics:
        summary['domain_checkpoints'] = domain_metrics.get('checkpoints', {})
        summary['per_problem_metrics'] = domain_metrics.get('per_problem_metrics', {})

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    # Print summary
    print("\n" + "="*60)
    print("OLAM Analysis Summary")
    print("="*60)
    print(f"Final iteration: {summary['final_iteration']}")
    print(f"Total observations: {summary['total_observations']}")
    print(f"Safe model - Precision: {summary['final_safe_precision']:.2%}, "
          f"Recall: {summary['final_safe_recall']:.2%}")
    print(f"Complete model - Precision: {summary['final_complete_precision']:.2%}, "
          f"Recall: {summary['final_complete_recall']:.2%}")
    print(f"Validation similarity: {summary['validation_similarity']:.2%}")

    # Print checkpoint-based domain metrics if available
    if domain_metrics and domain_metrics.get('checkpoints'):
        checkpoints = domain_metrics['checkpoints']
        print("\n" + "="*60)
        print("Domain-Level Checkpoint Metrics")
        print("="*60)

        for checkpoint in sorted(checkpoints.keys()):
            cp_data = checkpoints[checkpoint]
            print(f"\nCheckpoint {checkpoint}:")

            # Safe domain
            if 'safe' in cp_data:
                safe = cp_data['safe']
                print(f"  SAFE ({safe['problems_analyzed']} problems):")
                prec = safe['preconditions']
                print(f"    Prec: P={prec['avg_precision']:.2%}±{prec['std_precision']:.3f}, "
                      f"R={prec['avg_recall']:.2%}±{prec['std_recall']:.3f}")
                eff = safe['effects']
                print(f"    Eff:  P={eff['avg_precision']:.2%}±{eff['std_precision']:.3f}, "
                      f"R={eff['avg_recall']:.2%}±{eff['std_recall']:.3f}")

            # Complete domain
            if 'complete' in cp_data:
                complete = cp_data['complete']
                print(f"  COMPLETE ({complete['problems_analyzed']} problems):")
                prec = complete['preconditions']
                print(f"    Prec: P={prec['avg_precision']:.2%}±{prec['std_precision']:.3f}, "
                      f"R={prec['avg_recall']:.2%}±{prec['std_recall']:.3f}")
                eff = complete['effects']
                print(f"    Eff:  P={eff['avg_precision']:.2%}±{eff['std_precision']:.3f}, "
                      f"R={eff['avg_recall']:.2%}±{eff['std_recall']:.3f}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze OLAM results through post-processing"
    )
    parser.add_argument(
        "--domain",
        type=Path,
        required=True,
        help="Path to domain PDDL file"
    )
    parser.add_argument(
        "--problem",
        type=Path,
        required=True,
        help="Path to problem PDDL file"
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs='+',
        default=[5, 10, 20, 50, 100, 200, 300],
        help="Checkpoint iterations (default: 5 10 20 50 100 200 300)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/olam_analysis"),
        help="Output directory (default: results/olam_analysis)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=300,
        help="Maximum OLAM iterations (default: 300)"
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Don't run OLAM, use existing results"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Ground truth domain for metrics (defaults to input domain)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate against OLAM's native exports"
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=0,
        help="Minimum observations for action to be included in metrics (default: 0, include all)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.domain.exists():
        logger.error(f"Domain file not found: {args.domain}")
        sys.exit(1)
    if not args.problem.exists():
        logger.error(f"Problem file not found: {args.problem}")
        sys.exit(1)

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Configure OLAM
    config = {
        'max_iterations': args.max_iterations,
        'planner_time_limit': 240,
        'max_precs_length': 8,
        'neg_eff_assumption': False,
        'random_seed': 42
    }

    # Step 1: Run OLAM or load existing results
    result = run_olam_experiment(
        domain_file=args.domain,
        problem_file=args.problem,
        config=config,
        output_dir=args.output_dir,
        run_olam=not args.no_run
    )

    # Step 1.5: Process all OLAM problem exports (per-problem analysis)
    # Initialize domain_metrics (will be populated if per-problem analysis runs)
    domain_metrics = None

    # Extract domain name from directory (e.g., "depots" from "benchmarks/olam-compatible/depots/domain.pddl")
    domain_name = args.domain.parent.name
    ground_truth = args.ground_truth or args.domain

    # OLAM exports are in complete_run directory (not result.exports_dir which points to old location)
    actual_exports_dir = args.output_dir / "olam_run" / "complete_run"
    if actual_exports_dir.exists():
        logger.info("="*60)
        logger.info("Starting per-problem analysis of OLAM exports")
        logger.info("="*60)

        # Find all problem directories
        problem_dirs = find_problem_directories(actual_exports_dir, domain_name)

        if problem_dirs:
            per_problem_metrics = {}

            # Process each problem
            for problem_id, problem_dir in problem_dirs:
                logger.info(f"Processing problem {problem_id}...")

                # Find corresponding problem file
                # OLAM tests multiple problems, we need to find the right one
                problem_file = Path(f"benchmarks/olam-compatible/{domain_name}/{problem_id}.pddl")
                if not problem_file.exists():
                    logger.warning(f"Problem file not found: {problem_file}, using original")
                    problem_file = args.problem

                try:
                    # Process this problem's exports (OLAM saves domains every 5 steps by default)
                    # We'll detect checkpoints from the exported files
                    metrics = process_problem_exports(
                        problem_dir=problem_dir,
                        ground_truth_domain=ground_truth,
                        problem_file=problem_file,
                        checkpoints=args.checkpoints  # Pass checkpoints to find domain files
                    )

                    # Save per-problem metrics
                    save_per_problem_metrics(problem_id, metrics, args.output_dir)
                    per_problem_metrics[problem_id] = metrics

                    # Log summary for final checkpoint
                    if metrics.get('checkpoints'):
                        final_cp = max(metrics['checkpoints'].keys())
                        final_metrics = metrics['checkpoints'][final_cp]
                        if 'safe' in final_metrics and final_metrics['safe']:
                            safe = final_metrics['safe']
                            logger.info(f"  {problem_id} (checkpoint {final_cp}): "
                                      f"Safe - Prec P/R={safe['precondition_precision']:.2%}/{safe['precondition_recall']:.2%}, "
                                      f"Eff P/R={safe['effect_precision']:.2%}/{safe['effect_recall']:.2%}")

                except Exception as e:
                    logger.error(f"Error processing problem {problem_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Aggregate metrics at domain level
            if per_problem_metrics:
                logger.info("="*60)
                logger.info("Aggregating metrics at domain level")
                logger.info("="*60)

                domain_metrics = aggregate_domain_metrics(per_problem_metrics)

                # Save domain-level metrics
                domain_summary_path = args.output_dir / "domain_summary.json"
                with open(domain_summary_path, 'w') as f:
                    json.dump(domain_metrics, f, indent=2)
                logger.info(f"Saved domain metrics to {domain_summary_path}")

                # Print domain summary (checkpoint-based structure)
                checkpoints = domain_metrics.get('checkpoints', {})
                if checkpoints:
                    print("\n" + "="*60)
                    print("Domain-Level Analysis Summary (by Checkpoint)")
                    print("="*60)

                    for checkpoint in sorted(checkpoints.keys()):
                        cp_data = checkpoints[checkpoint]
                        print(f"\nCheckpoint {checkpoint}:")

                        # Safe domain
                        if 'safe' in cp_data:
                            safe = cp_data['safe']
                            print(f"  SAFE Domain ({safe['problems_analyzed']} problems):")
                            prec = safe['preconditions']
                            print(f"    Preconditions: Precision={prec['avg_precision']:.2%} ±{prec['std_precision']:.3f}, "
                                  f"Recall={prec['avg_recall']:.2%} ±{prec['std_recall']:.3f}")
                            eff = safe['effects']
                            print(f"    Effects:       Precision={eff['avg_precision']:.2%} ±{eff['std_precision']:.3f}, "
                                  f"Recall={eff['avg_recall']:.2%} ±{eff['std_recall']:.3f}")

                        # Complete domain
                        if 'complete' in cp_data:
                            complete = cp_data['complete']
                            print(f"  COMPLETE Domain ({complete['problems_analyzed']} problems):")
                            prec = complete['preconditions']
                            print(f"    Preconditions: Precision={prec['avg_precision']:.2%} ±{prec['std_precision']:.3f}, "
                                  f"Recall={prec['avg_recall']:.2%} ±{prec['std_recall']:.3f}")
                            eff = complete['effects']
                            print(f"    Effects:       Precision={eff['avg_precision']:.2%} ±{eff['std_precision']:.3f}, "
                                  f"Recall={eff['avg_recall']:.2%} ±{eff['std_recall']:.3f}")

                    print("="*60)
        else:
            logger.warning("No problem directories found in OLAM exports")

    # Step 2: Parse trace
    trace = parse_trace(result.trace_file)

    # Filter checkpoints to those within trace length
    max_iteration = len(trace)
    valid_checkpoints = [cp for cp in args.checkpoints if cp <= max_iteration]
    if not valid_checkpoints:
        logger.error(f"No valid checkpoints. Trace has {max_iteration} steps.")
        sys.exit(1)

    # Step 3: Reconstruct models at checkpoints
    models = reconstruct_models_at_checkpoints(
        trace=trace,
        checkpoints=valid_checkpoints,
        domain_file=args.domain,
        problem_file=args.problem,
        output_dir=args.output_dir
    )

    # Step 4: Compute metrics (if ground truth available)
    ground_truth = args.ground_truth or args.domain
    if ground_truth.exists():
        metrics_df = compute_metrics(
            models, ground_truth, args.problem,
            min_observations=args.min_observations
        )

        # Step 5: Validate (optional)
        validation = {}
        if args.validate and result.exports_dir and result.exports_dir.exists():
            validation = validate_against_olam_exports(models, result.exports_dir)

        # Step 6: Save results (including domain metrics if available)
        save_results(metrics_df, validation, args.output_dir, domain_metrics)
    else:
        logger.warning("No ground truth available, skipping metrics computation")
        # Just save the reconstructed models
        logger.info(f"Saved {len(models)} model snapshots to {args.output_dir}/models/")

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()