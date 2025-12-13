#!/usr/bin/env python3
"""
Process OLAM Results from External Execution.

This script reads OLAM results generated externally (not run by us).
Expected structure:
    olam_results/<domain_name>/
    ├── trace_complete.json
    ├── 1_p00_<domain>_gen/
    │   ├── trace.json
    │   └── checkpoints/
    │       ├── iter_1/
    │       │   ├── operator_certain_predicates.json
    │       │   ├── operator_uncertain_precs.json
    │       │   ├── operator_certain_positive_effects.json
    │       │   ├── operator_certain_negative_effects.json
    │       │   └── ... (6 more JSON files)
    │       ├── iter_2/
    │       └── ... (variable number, may finish early)
    └── 2_p01_<domain>_gen/

Usage:
    python process_olam_results.py \\
        --olam-results /path/to/olam_results/depots \\
        --ground-truth benchmarks/olam-compatible/depots/domain.pddl \\
        --output-dir results/olam_processed
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.olam.trace_parser import OLAMTraceParser
from src.olam.knowledge_reconstructor import OLAMKnowledgeReconstructor
from src.core.model_metrics import ModelMetrics
from src.core.model_reconstructor import ModelReconstructor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_problem_directories(olam_domain_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all problem directories in OLAM results.

    Args:
        olam_domain_dir: OLAM results directory for a domain (e.g., olam_results/depots/)

    Returns:
        List of (problem_id, problem_dir) tuples sorted by problem number
    """
    problem_dirs = []

    for item in olam_domain_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Extract problem ID from directory name (e.g., "p00" from "1_p00_depots_gen")
            parts = item.name.split('_')
            if len(parts) >= 2:
                problem_id = parts[1]  # e.g., "p00"
                problem_num = int(parts[0])  # e.g., 1
                problem_dirs.append((problem_num, problem_id, item))
            else:
                logger.warning(f"Could not parse problem directory name: {item.name}")

    # Sort by problem number
    problem_dirs.sort(key=lambda x: x[0])
    logger.info(f"Found {len(problem_dirs)} problem directories in {olam_domain_dir}")

    return [(pid, pdir) for _, pid, pdir in problem_dirs]


def detect_checkpoints(problem_dir: Path) -> List[int]:
    """
    Detect available checkpoint iterations in a problem directory.

    Args:
        problem_dir: Problem directory with checkpoints/ subdirectory

    Returns:
        Sorted list of checkpoint iteration numbers
    """
    checkpoints_dir = problem_dir / "checkpoints"
    if not checkpoints_dir.exists():
        logger.warning(f"No checkpoints directory in {problem_dir}")
        return []

    checkpoints = []
    for item in checkpoints_dir.iterdir():
        if item.is_dir() and item.name.startswith('iter_'):
            try:
                iter_num = int(item.name.split('_')[1])
                checkpoints.append(iter_num)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse checkpoint name: {item.name}")

    checkpoints.sort()
    logger.debug(f"Detected {len(checkpoints)} checkpoints in {problem_dir.name}")
    return checkpoints


def process_problem(
    problem_dir: Path,
    problem_id: str,
    ground_truth_domain: Path,
    benchmarks_dir: Path,
    domain_name: str
) -> Dict:
    """
    Process a single problem's OLAM results.

    Args:
        problem_dir: Directory with OLAM results for this problem
        problem_id: Problem identifier (e.g., "p00")
        ground_truth_domain: Path to ground truth domain PDDL
        benchmarks_dir: Path to benchmarks directory
        domain_name: Domain name (e.g., "depots")

    Returns:
        Dictionary with checkpoint metrics
    """
    logger.info(f"Processing {problem_id}...")

    # Find corresponding problem file
    problem_file = benchmarks_dir / domain_name / f"{problem_id}.pddl"
    if not problem_file.exists():
        logger.warning(f"Problem file not found: {problem_file}")
        return {'checkpoints': {}}

    # Detect available checkpoints
    checkpoints = detect_checkpoints(problem_dir)
    if not checkpoints:
        logger.warning(f"No checkpoints found for {problem_id}")
        return {'checkpoints': {}}

    logger.info(f"  Found {len(checkpoints)} checkpoints: {checkpoints[:5]}...{checkpoints[-3:] if len(checkpoints) > 5 else ''}")

    # Initialize parser and metrics
    parser = OLAMTraceParser()
    reconstructor = OLAMKnowledgeReconstructor(ground_truth_domain)
    metrics_calculator = ModelMetrics(ground_truth_domain, problem_file)

    checkpoint_metrics = {}

    # Process each checkpoint
    for checkpoint in checkpoints:
        checkpoint_dir = problem_dir / "checkpoints" / f"iter_{checkpoint}"

        try:
            # Load JSON exports from this checkpoint
            exports = parser.parse_json_exports(checkpoint_dir)

            # Reconstruct knowledge from exports
            knowledge = reconstructor.reconstruct_from_exports(exports)

            # Export as snapshot (format expected by ModelReconstructor)
            snapshot = reconstructor.export_snapshot(knowledge, checkpoint, algorithm="olam")

            # Build models (pass domain file for safe model to generate all possible predicates)
            safe_model = ModelReconstructor.reconstruct_olam_safe(snapshot, domain_file=str(ground_truth_domain))
            complete_model = ModelReconstructor.reconstruct_olam_complete(snapshot)

            # Compute metrics
            safe_metrics = metrics_calculator.compute_metrics(safe_model)
            complete_metrics = metrics_calculator.compute_metrics(complete_model)

            # DEBUG: Log metrics for first checkpoint to verify fix
            if checkpoint == checkpoints[0]:
                logger.info(f"  [DEBUG] First checkpoint ({checkpoint}) metrics:")
                logger.info(f"    Safe keys: {list(safe_metrics.keys())}")
                logger.info(f"    Safe prec P/R: {safe_metrics.get('precondition_precision', 'MISSING'):.2%} / {safe_metrics.get('precondition_recall', 'MISSING'):.2%}")
                logger.info(f"    Safe eff P/R: {safe_metrics.get('effect_precision', 'MISSING'):.2%} / {safe_metrics.get('effect_recall', 'MISSING'):.2%}")

            # Store safe and complete metrics separately
            checkpoint_metrics[checkpoint] = {
                'safe': {
                    'precondition_precision': safe_metrics.get('precondition_precision', 0.0),
                    'precondition_recall': safe_metrics.get('precondition_recall', 0.0),
                    'effect_precision': safe_metrics.get('effect_precision', 0.0),
                    'effect_recall': safe_metrics.get('effect_recall', 0.0),
                    'overall_precision': safe_metrics.get('precision', 0.0),
                    'overall_recall': safe_metrics.get('recall', 0.0)
                },
                'complete': {
                    'precondition_precision': complete_metrics.get('precondition_precision', 0.0),
                    'precondition_recall': complete_metrics.get('precondition_recall', 0.0),
                    'effect_precision': complete_metrics.get('effect_precision', 0.0),
                    'effect_recall': complete_metrics.get('effect_recall', 0.0),
                    'overall_precision': complete_metrics.get('precision', 0.0),
                    'overall_recall': complete_metrics.get('recall', 0.0)
                },
                'safe_detailed': safe_metrics.get('detailed_per_action', {}),
                'complete_detailed': complete_metrics.get('detailed_per_action', {})
            }

        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint} for {problem_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"  Processed {len(checkpoint_metrics)}/{len(checkpoints)} checkpoints")
    return {'checkpoints': checkpoint_metrics}


def aggregate_metrics(per_problem_metrics: Dict[str, Dict]) -> Dict:
    """
    Aggregate per-problem metrics to domain level.

    Args:
        per_problem_metrics: Dict mapping problem_id to checkpoint metrics

    Returns:
        Aggregated domain-level metrics
    """
    if not per_problem_metrics:
        return {'checkpoints': {}}

    # Collect all checkpoints
    all_checkpoints = set()
    for metrics in per_problem_metrics.values():
        if 'checkpoints' in metrics:
            all_checkpoints.update(metrics['checkpoints'].keys())

    if not all_checkpoints:
        return {'checkpoints': {}}

    aggregated = {}

    for checkpoint in sorted(all_checkpoints):
        # Collect values for this checkpoint across all problems
        safe_prec_p, safe_prec_r = [], []
        safe_eff_p, safe_eff_r = [], []
        complete_prec_p, complete_prec_r = [], []
        complete_eff_p, complete_eff_r = [], []

        for problem_id, prob_metrics in per_problem_metrics.items():
            if 'checkpoints' not in prob_metrics or checkpoint not in prob_metrics['checkpoints']:
                continue

            cp = prob_metrics['checkpoints'][checkpoint]

            if 'safe' in cp:
                safe_prec_p.append(cp['safe']['precondition_precision'])
                safe_prec_r.append(cp['safe']['precondition_recall'])
                safe_eff_p.append(cp['safe']['effect_precision'])
                safe_eff_r.append(cp['safe']['effect_recall'])

            if 'complete' in cp:
                complete_prec_p.append(cp['complete']['precondition_precision'])
                complete_prec_r.append(cp['complete']['precondition_recall'])
                complete_eff_p.append(cp['complete']['effect_precision'])
                complete_eff_r.append(cp['complete']['effect_recall'])

        def avg(vals):
            return sum(vals) / len(vals) if vals else 0.0

        aggregated[checkpoint] = {}

        if safe_prec_p:
            aggregated[checkpoint]['safe'] = {
                'problems_count': len(safe_prec_p),
                'preconditions': {
                    'avg_precision': avg(safe_prec_p),
                    'avg_recall': avg(safe_prec_r)
                },
                'effects': {
                    'avg_precision': avg(safe_eff_p),
                    'avg_recall': avg(safe_eff_r)
                }
            }

        if complete_prec_p:
            aggregated[checkpoint]['complete'] = {
                'problems_count': len(complete_prec_p),
                'preconditions': {
                    'avg_precision': avg(complete_prec_p),
                    'avg_recall': avg(complete_prec_r)
                },
                'effects': {
                    'avg_precision': avg(complete_eff_p),
                    'avg_recall': avg(complete_eff_r)
                }
            }

    return {'checkpoints': aggregated, 'per_problem': per_problem_metrics}


def main():
    parser = argparse.ArgumentParser(
        description="Process OLAM results from external execution"
    )
    parser.add_argument(
        "--olam-results",
        type=Path,
        required=True,
        help="Path to OLAM results domain directory (e.g., olam_results/depots/)"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to ground truth domain PDDL"
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=Path("benchmarks/olam-compatible"),
        help="Path to benchmarks directory (default: benchmarks/olam-compatible)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/olam_processed"),
        help="Output directory (default: results/olam_processed)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.olam_results.exists():
        logger.error(f"OLAM results directory not found: {args.olam_results}")
        sys.exit(1)

    if not args.ground_truth.exists():
        logger.error(f"Ground truth domain not found: {args.ground_truth}")
        sys.exit(1)

    # Extract domain name from results directory
    domain_name = args.olam_results.name
    logger.info(f"Processing domain: {domain_name}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all problem directories
    problem_dirs = find_problem_directories(args.olam_results)

    if not problem_dirs:
        logger.error("No problem directories found")
        sys.exit(1)

    # Process each problem
    per_problem_metrics = {}

    for problem_id, problem_dir in problem_dirs:
        try:
            metrics = process_problem(
                problem_dir=problem_dir,
                problem_id=problem_id,
                ground_truth_domain=args.ground_truth,
                benchmarks_dir=args.benchmarks_dir,
                domain_name=domain_name
            )

            per_problem_metrics[problem_id] = metrics

            # Save separate files for safe and complete metrics
            # Extract safe metrics only
            safe_metrics_only = {
                'checkpoints': {
                    cp: data['safe'] for cp, data in metrics['checkpoints'].items()
                }
            }

            # Extract complete metrics only
            complete_metrics_only = {
                'checkpoints': {
                    cp: data['complete'] for cp, data in metrics['checkpoints'].items()
                }
            }

            # Extract detailed metrics
            detailed_metrics = {
                'checkpoints': {
                    cp: {
                        'safe': data.get('safe_detailed', {}),
                        'complete': data.get('complete_detailed', {})
                    }
                    for cp, data in metrics['checkpoints'].items()
                }
            }

            # Save safe metrics
            safe_output = args.output_dir / f"{problem_id}_safe_metrics.json"
            with open(safe_output, 'w') as f:
                json.dump(safe_metrics_only, f, indent=2)
            logger.info(f"  Saved safe metrics to {safe_output}")

            # Save complete metrics
            complete_output = args.output_dir / f"{problem_id}_complete_metrics.json"
            with open(complete_output, 'w') as f:
                json.dump(complete_metrics_only, f, indent=2)
            logger.info(f"  Saved complete metrics to {complete_output}")

            # Save detailed metrics (for debugging)
            detailed_output = args.output_dir / f"{problem_id}_detailed_metrics.json"
            with open(detailed_output, 'w') as f:
                json.dump(detailed_metrics, f, indent=2)
            logger.info(f"  Saved detailed metrics to {detailed_output}")

        except Exception as e:
            logger.error(f"Failed to process {problem_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate to domain level
    logger.info("\n" + "="*60)
    logger.info("Aggregating metrics at domain level")
    logger.info("="*60)

    domain_metrics = aggregate_metrics(per_problem_metrics)

    # Save domain-level results (combined)
    domain_output = args.output_dir / "domain_metrics.json"
    with open(domain_output, 'w') as f:
        json.dump(domain_metrics, f, indent=2)
    logger.info(f"Saved domain metrics to {domain_output}")

    # Save separate safe and complete domain metrics
    if 'checkpoints' in domain_metrics:
        safe_domain_metrics = {
            'checkpoints': {
                cp: {'safe': data.get('safe', {})}
                for cp, data in domain_metrics['checkpoints'].items()
            }
        }
        complete_domain_metrics = {
            'checkpoints': {
                cp: {'complete': data.get('complete', {})}
                for cp, data in domain_metrics['checkpoints'].items()
            }
        }

        safe_domain_output = args.output_dir / "domain_safe_metrics.json"
        with open(safe_domain_output, 'w') as f:
            json.dump(safe_domain_metrics, f, indent=2)
        logger.info(f"Saved safe domain metrics to {safe_domain_output}")

        complete_domain_output = args.output_dir / "domain_complete_metrics.json"
        with open(complete_domain_output, 'w') as f:
            json.dump(complete_domain_metrics, f, indent=2)
        logger.info(f"Saved complete domain metrics to {complete_domain_output}")

    # Print summary
    print("\n" + "="*60)
    print(f"Domain: {domain_name}")
    print("="*60)

    if 'checkpoints' in domain_metrics:
        for checkpoint in sorted(domain_metrics['checkpoints'].keys()):
            cp = domain_metrics['checkpoints'][checkpoint]
            print(f"\nCheckpoint {checkpoint}:")

            if 'safe' in cp:
                safe = cp['safe']
                prec = safe['preconditions']
                eff = safe['effects']
                print(f"  SAFE ({safe['problems_count']} problems):")
                print(f"    Prec: P={prec['avg_precision']:.2%}, R={prec['avg_recall']:.2%}")
                print(f"    Eff:  P={eff['avg_precision']:.2%}, R={eff['avg_recall']:.2%}")

            if 'complete' in cp:
                complete = cp['complete']
                prec = complete['preconditions']
                eff = complete['effects']
                print(f"  COMPLETE ({complete['problems_count']} problems):")
                print(f"    Prec: P={prec['avg_precision']:.2%}, R={prec['avg_recall']:.2%}")
                print(f"    Eff:  P={eff['avg_precision']:.2%}, R={eff['avg_recall']:.2%}")

    print("="*60)
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
