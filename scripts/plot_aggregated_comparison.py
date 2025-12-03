#!/usr/bin/env python3
"""
Generate aggregated comparison plots across all domains for OLAM vs Information Gain.

Creates:
1. Aggregated Preconditions Precision/Recall comparison
2. Aggregated Effects Precision/Recall comparison
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import defaultdict


def load_all_domain_metrics(base_dir: Path, algorithm: str) -> Dict:
    """
    Load metrics for all domains for a given algorithm.

    Args:
        base_dir: Base results directory
        algorithm: 'olam' or 'infogain'

    Returns:
        Dict mapping domain names to their metrics
    """
    if algorithm == 'olam':
        results_dir = base_dir / "olam-results"
    else:
        results_dir = base_dir / "information_gain_metrics"

    domain_metrics = {}

    for domain_dir in sorted(results_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain_name = domain_dir.name

        try:
            if algorithm == 'olam':
                # Load OLAM metrics
                safe_file = domain_dir / "domain_safe_metrics.json"
                complete_file = domain_dir / "domain_complete_metrics.json"

                with open(safe_file) as f:
                    safe_metrics = json.load(f)
                with open(complete_file) as f:
                    complete_metrics = json.load(f)

                domain_metrics[domain_name] = {
                    "safe": safe_metrics,
                    "complete": complete_metrics
                }
            else:
                # Load Information Gain metrics - aggregate from problems
                problem_dirs = sorted([d for d in domain_dir.iterdir() if d.is_dir()])

                all_iterations = defaultdict(lambda: {"safe": [], "complete": []})

                for problem_dir in problem_dirs:
                    metrics_file = problem_dir / "metrics_per_iteration.json"
                    if not metrics_file.exists():
                        continue

                    with open(metrics_file) as f:
                        data = json.load(f)

                    for iteration, iter_data in data.items():
                        iteration = int(iteration)

                        # Extract safe model metrics
                        safe = iter_data.get("safe_model", {})
                        all_iterations[iteration]["safe"].append({
                            "precondition_precision": safe.get("precondition_precision", 0),
                            "precondition_recall": safe.get("precondition_recall", 0),
                            "effect_precision": safe.get("effect_precision", 0),
                            "effect_recall": safe.get("effect_recall", 0),
                        })

                        # Extract complete model metrics
                        complete = iter_data.get("complete_model", {})
                        all_iterations[iteration]["complete"].append({
                            "precondition_precision": complete.get("precondition_precision", 0),
                            "precondition_recall": complete.get("precondition_recall", 0),
                            "effect_precision": complete.get("effect_precision", 0),
                            "effect_recall": complete.get("effect_recall", 0),
                        })

                # Average across problems
                averaged_metrics = {"safe": {}, "complete": {}}
                for iteration in sorted(all_iterations.keys()):
                    for model_type in ["safe", "complete"]:
                        metrics_list = all_iterations[iteration][model_type]
                        if metrics_list:
                            averaged_metrics[model_type][str(iteration)] = {
                                "precondition_precision": np.mean([m["precondition_precision"] for m in metrics_list]),
                                "precondition_recall": np.mean([m["precondition_recall"] for m in metrics_list]),
                                "effect_precision": np.mean([m["effect_precision"] for m in metrics_list]),
                                "effect_recall": np.mean([m["effect_recall"] for m in metrics_list]),
                            }

                domain_metrics[domain_name] = {"metrics": averaged_metrics}

        except Exception as e:
            print(f"Warning: Could not load {domain_name}: {e}")
            continue

    return domain_metrics


def aggregate_across_domains(domain_metrics: Dict, algorithm: str, metric_type: str) -> Dict:
    """
    Aggregate metrics across all domains.

    Args:
        domain_metrics: Dict of domain_name -> metrics
        algorithm: 'olam' or 'infogain'
        metric_type: 'preconditions' or 'effects'

    Returns:
        Dict with aggregated safe and complete metrics per iteration
    """
    # Collect all data points per iteration
    iteration_data = defaultdict(lambda: {
        "safe_precision": [], "safe_recall": [],
        "complete_precision": [], "complete_recall": []
    })

    for domain_name, metrics in domain_metrics.items():
        if algorithm == 'olam':
            safe_checkpoints = metrics["safe"]["checkpoints"]
            complete_checkpoints = metrics["complete"]["checkpoints"]

            for checkpoint_id in safe_checkpoints.keys():
                iteration = int(checkpoint_id)

                safe_data = safe_checkpoints[checkpoint_id]["safe"]
                complete_data = complete_checkpoints[checkpoint_id]["complete"]

                metric_category = metric_type  # 'preconditions' or 'effects'

                iteration_data[iteration]["safe_precision"].append(
                    safe_data[metric_category]["avg_precision"]
                )
                iteration_data[iteration]["safe_recall"].append(
                    safe_data[metric_category]["avg_recall"]
                )
                iteration_data[iteration]["complete_precision"].append(
                    complete_data[metric_category]["avg_precision"]
                )
                iteration_data[iteration]["complete_recall"].append(
                    complete_data[metric_category]["avg_recall"]
                )

        else:  # infogain
            safe_metrics = metrics["metrics"]["safe"]
            complete_metrics = metrics["metrics"]["complete"]

            # Determine key names
            if metric_type == "preconditions":
                precision_key = "precondition_precision"
                recall_key = "precondition_recall"
            else:
                precision_key = "effect_precision"
                recall_key = "effect_recall"

            for iteration_str in safe_metrics.keys():
                iteration = int(iteration_str)

                iteration_data[iteration]["safe_precision"].append(
                    safe_metrics[iteration_str][precision_key]
                )
                iteration_data[iteration]["safe_recall"].append(
                    safe_metrics[iteration_str][recall_key]
                )
                iteration_data[iteration]["complete_precision"].append(
                    complete_metrics[iteration_str][precision_key]
                )
                iteration_data[iteration]["complete_recall"].append(
                    complete_metrics[iteration_str][recall_key]
                )

    # Average across domains for each iteration
    aggregated = {
        "iterations": [],
        "safe_precision": [], "safe_recall": [],
        "complete_precision": [], "complete_recall": []
    }

    for iteration in sorted(iteration_data.keys()):
        data = iteration_data[iteration]

        # Only include iterations with data from at least one domain
        if data["safe_precision"]:
            aggregated["iterations"].append(iteration)
            aggregated["safe_precision"].append(np.mean(data["safe_precision"]))
            aggregated["safe_recall"].append(np.mean(data["safe_recall"]))
            aggregated["complete_precision"].append(np.mean(data["complete_precision"]))
            aggregated["complete_recall"].append(np.mean(data["complete_recall"]))

    return aggregated


def plot_aggregated_comparison(
    olam_agg: Dict,
    infogain_agg: Dict,
    metric_type: str,
    output_dir: Path
):
    """
    Create aggregated comparison plot.

    Args:
        olam_agg: Aggregated OLAM metrics
        infogain_agg: Aggregated InfoGain metrics
        metric_type: 'preconditions' or 'effects'
        output_dir: Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Aggregated Across All Domains - {metric_type.title()} Metrics',
                 fontsize=16, fontweight='bold')

    # Plot Precision
    ax1.plot(olam_agg["iterations"], olam_agg["safe_precision"],
             'o-', label='OLAM Safe', linewidth=2.5, markersize=7, color='#1f77b4')
    ax1.plot(olam_agg["iterations"], olam_agg["complete_precision"],
             's--', label='OLAM Complete', linewidth=2.5, markersize=7, color='#ff7f0e')
    ax1.plot(infogain_agg["iterations"], infogain_agg["safe_precision"],
             '^-', label='InfoGain Safe', linewidth=2.5, markersize=7, color='#2ca02c')
    ax1.plot(infogain_agg["iterations"], infogain_agg["complete_precision"],
             'd--', label='InfoGain Complete', linewidth=2.5, markersize=7, color='#d62728')

    ax1.set_xlabel('Steps', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax1.set_title('Precision Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(-0.05, 1.05)

    # Plot Recall
    ax2.plot(olam_agg["iterations"], olam_agg["safe_recall"],
             'o-', label='OLAM Safe', linewidth=2.5, markersize=7, color='#1f77b4')
    ax2.plot(olam_agg["iterations"], olam_agg["complete_recall"],
             's--', label='OLAM Complete', linewidth=2.5, markersize=7, color='#ff7f0e')
    ax2.plot(infogain_agg["iterations"], infogain_agg["safe_recall"],
             '^-', label='InfoGain Safe', linewidth=2.5, markersize=7, color='#2ca02c')
    ax2.plot(infogain_agg["iterations"], infogain_agg["complete_recall"],
             'd--', label='InfoGain Complete', linewidth=2.5, markersize=7, color='#d62728')

    ax2.set_xlabel('Steps', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Recall', fontsize=13, fontweight='bold')
    ax2.set_title('Recall Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    output_file = output_dir / f"aggregated_{metric_type}_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main execution function."""
    base_dir = Path("/home/omer/projects/online_model_learning/results")
    output_dir = base_dir / "comparison_plots"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Loading OLAM metrics from all domains...")
    olam_metrics = load_all_domain_metrics(base_dir, "olam")
    print(f"  Loaded {len(olam_metrics)} domains: {sorted(olam_metrics.keys())}")

    print("\nLoading Information Gain metrics from all domains...")
    infogain_metrics = load_all_domain_metrics(base_dir, "infogain")
    print(f"  Loaded {len(infogain_metrics)} domains: {sorted(infogain_metrics.keys())}")

    # Find common domains
    common_domains = set(olam_metrics.keys()) & set(infogain_metrics.keys())

    # Exclude problematic domains
    excluded_domains = {"sokoban"}
    common_domains = common_domains - excluded_domains

    print(f"\nCommon domains ({len(common_domains)}): {sorted(common_domains)}")
    if excluded_domains:
        print(f"Excluded domains: {sorted(excluded_domains)}")

    # Filter to only common domains
    olam_metrics = {k: v for k, v in olam_metrics.items() if k in common_domains}
    infogain_metrics = {k: v for k, v in infogain_metrics.items() if k in common_domains}

    print("\n" + "=" * 70)
    print("Generating Aggregated Preconditions Comparison...")
    print("=" * 70)

    olam_prec_agg = aggregate_across_domains(olam_metrics, "olam", "preconditions")
    infogain_prec_agg = aggregate_across_domains(infogain_metrics, "infogain", "preconditions")

    plot_aggregated_comparison(olam_prec_agg, infogain_prec_agg, "preconditions", output_dir)

    print("\n" + "=" * 70)
    print("Generating Aggregated Effects Comparison...")
    print("=" * 70)

    olam_eff_agg = aggregate_across_domains(olam_metrics, "olam", "effects")
    infogain_eff_agg = aggregate_across_domains(infogain_metrics, "infogain", "effects")

    plot_aggregated_comparison(olam_eff_agg, infogain_eff_agg, "effects", output_dir)

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Aggregated across {len(common_domains)} domains")
    print(f"OLAM iterations: {len(olam_prec_agg['iterations'])}")
    print(f"InfoGain iterations: {len(infogain_prec_agg['iterations'])}")
    print("\nPlots saved to:", output_dir)


if __name__ == "__main__":
    main()
