#!/usr/bin/env python3
"""
Create dual-axis plots comparing OLAM vs InfoGain for each specific metric.

Creates 4 plots per domain:
1. Precondition Precision: OLAM (left axis) vs InfoGain (right axis)
2. Precondition Recall: OLAM (left axis) vs InfoGain (right axis)
3. Effects Precision: OLAM (left axis) vs InfoGain (right axis)
4. Effects Recall: OLAM (left axis) vs InfoGain (right axis)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from collections import defaultdict


def load_domain_metrics(domain_name: str, algorithm: str) -> Dict:
    """
    Load metrics for a domain and algorithm.

    Returns:
        Dict with iterations -> {safe/complete metrics for prec/eff precision/recall}
    """
    base_dir = Path("/home/omer/projects/online_model_learning/results")

    if algorithm == "infogain":
        domain_dir = base_dir / "information_gain_metrics" / domain_name
        problem_dirs = sorted([d for d in domain_dir.iterdir() if d.is_dir()])

        all_iterations = defaultdict(lambda: {
            "safe_prec_precision": [], "safe_prec_recall": [],
            "safe_eff_precision": [], "safe_eff_recall": [],
            "complete_prec_precision": [], "complete_prec_recall": [],
            "complete_eff_precision": [], "complete_eff_recall": []
        })

        for problem_dir in problem_dirs:
            metrics_file = problem_dir / "metrics_per_iteration.json"
            if not metrics_file.exists():
                continue

            with open(metrics_file) as f:
                data = json.load(f)

            for iteration_str, iter_data in data.items():
                iteration = int(iteration_str)

                safe = iter_data.get("safe_model", {})
                all_iterations[iteration]["safe_prec_precision"].append(safe.get("precondition_precision", 0))
                all_iterations[iteration]["safe_prec_recall"].append(safe.get("precondition_recall", 0))
                all_iterations[iteration]["safe_eff_precision"].append(safe.get("effect_precision", 0))
                all_iterations[iteration]["safe_eff_recall"].append(safe.get("effect_recall", 0))

                complete = iter_data.get("complete_model", {})
                all_iterations[iteration]["complete_prec_precision"].append(complete.get("precondition_precision", 0))
                all_iterations[iteration]["complete_prec_recall"].append(complete.get("precondition_recall", 0))
                all_iterations[iteration]["complete_eff_precision"].append(complete.get("effect_precision", 0))
                all_iterations[iteration]["complete_eff_recall"].append(complete.get("effect_recall", 0))

        result = {}
        for iteration in sorted(all_iterations.keys()):
            data = all_iterations[iteration]
            result[iteration] = {
                "safe_prec_precision": np.mean(data["safe_prec_precision"]),
                "safe_prec_recall": np.mean(data["safe_prec_recall"]),
                "safe_eff_precision": np.mean(data["safe_eff_precision"]),
                "safe_eff_recall": np.mean(data["safe_eff_recall"]),
                "complete_prec_precision": np.mean(data["complete_prec_precision"]),
                "complete_prec_recall": np.mean(data["complete_prec_recall"]),
                "complete_eff_precision": np.mean(data["complete_eff_precision"]),
                "complete_eff_recall": np.mean(data["complete_eff_recall"])
            }

        return result

    else:  # olam
        domain_dir = base_dir / "olam-results" / domain_name

        safe_file = domain_dir / "domain_safe_metrics.json"
        complete_file = domain_dir / "domain_complete_metrics.json"

        with open(safe_file) as f:
            safe_metrics = json.load(f)
        with open(complete_file) as f:
            complete_metrics = json.load(f)

        result = {}
        for checkpoint_str in safe_metrics["checkpoints"].keys():
            checkpoint = int(checkpoint_str)

            safe_data = safe_metrics["checkpoints"][checkpoint_str]["safe"]
            complete_data = complete_metrics["checkpoints"][checkpoint_str]["complete"]

            result[checkpoint] = {
                "safe_prec_precision": safe_data["preconditions"]["avg_precision"],
                "safe_prec_recall": safe_data["preconditions"]["avg_recall"],
                "safe_eff_precision": safe_data["effects"]["avg_precision"],
                "safe_eff_recall": safe_data["effects"]["avg_recall"],
                "complete_prec_precision": complete_data["preconditions"]["avg_precision"],
                "complete_prec_recall": complete_data["preconditions"]["avg_recall"],
                "complete_eff_precision": complete_data["effects"]["avg_precision"],
                "complete_eff_recall": complete_data["effects"]["avg_recall"]
            }

        return result


def plot_dual_axis_metric_comparison(
    domain_name: str,
    olam_data: Dict,
    infogain_data: Dict,
    metric_name: str,
    output_dir: Path
):
    """
    Create dual-axis plot comparing OLAM vs InfoGain for a specific metric.

    Args:
        domain_name: Domain name
        olam_data: OLAM metrics by iteration
        infogain_data: InfoGain metrics by iteration
        metric_name: One of 'prec_precision', 'prec_recall', 'eff_precision', 'eff_recall'
        output_dir: Output directory
    """
    # Determine metric keys and labels
    metric_map = {
        "prec_precision": {
            "safe_key": "safe_prec_precision",
            "complete_key": "complete_prec_precision",
            "title": "Precondition Precision",
            "ylabel": "Precision"
        },
        "prec_recall": {
            "safe_key": "safe_prec_recall",
            "complete_key": "complete_prec_recall",
            "title": "Precondition Recall",
            "ylabel": "Recall"
        },
        "eff_precision": {
            "safe_key": "safe_eff_precision",
            "complete_key": "complete_eff_precision",
            "title": "Effects Precision",
            "ylabel": "Precision"
        },
        "eff_recall": {
            "safe_key": "safe_eff_recall",
            "complete_key": "complete_eff_recall",
            "title": "Effects Recall",
            "ylabel": "Recall"
        }
    }

    metric_info = metric_map[metric_name]
    safe_key = metric_info["safe_key"]
    complete_key = metric_info["complete_key"]

    # Extract OLAM data
    olam_iterations = sorted(olam_data.keys())
    olam_safe = [olam_data[i][safe_key] for i in olam_iterations]
    olam_complete = [olam_data[i][complete_key] for i in olam_iterations]

    # Extract InfoGain data
    infogain_iterations = sorted(infogain_data.keys())
    infogain_safe = [infogain_data[i][safe_key] for i in infogain_iterations]
    infogain_complete = [infogain_data[i][complete_key] for i in infogain_iterations]

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.suptitle(f'{domain_name.upper()} - {metric_info["title"]} Comparison',
                 fontsize=16, fontweight='bold')

    # Left y-axis: OLAM
    ax1.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax1.set_ylabel(f'OLAM {metric_info["ylabel"]}', fontsize=13, fontweight='bold', color='#1f77b4')

    line1 = ax1.plot(olam_iterations, olam_safe, 'o-', label='OLAM Safe',
                     linewidth=2.5, markersize=7, color='#1f77b4')
    line2 = ax1.plot(olam_iterations, olam_complete, 's--', label='OLAM Complete',
                     linewidth=2.5, markersize=7, color='#ff7f0e')

    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Right y-axis: InfoGain
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'InfoGain {metric_info["ylabel"]}', fontsize=13, fontweight='bold', color='#2ca02c')

    line3 = ax2.plot(infogain_iterations, infogain_safe, '^-', label='InfoGain Safe',
                     linewidth=2.5, markersize=7, color='#2ca02c')
    line4 = ax2.plot(infogain_iterations, infogain_complete, 'd--', label='InfoGain Complete',
                     linewidth=2.5, markersize=7, color='#d62728')

    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.set_ylim(-0.05, 1.05)

    # Combined legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=11)

    plt.tight_layout()

    output_file = output_dir / f"{domain_name}_{metric_name}_dual_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main execution function."""
    base_dir = Path("/home/omer/projects/online_model_learning/results")
    output_dir = base_dir / "comparison_plots" / "correlation_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Domains to analyze (excluding sokoban)
    domains = ["blocksworld", "depots", "driverlog", "ferry", "hanoi", "miconic", "satellite"]

    # Metrics to plot
    metrics = ["prec_precision", "prec_recall", "eff_precision", "eff_recall"]

    print("=" * 70)
    print("Generating Dual-Axis Algorithm Comparison Plots")
    print("=" * 70)

    for domain in domains:
        print(f"\nProcessing domain: {domain}")

        try:
            # Load data for both algorithms
            olam_data = load_domain_metrics(domain, "olam")
            infogain_data = load_domain_metrics(domain, "infogain")

            if not olam_data or not infogain_data:
                print(f"  Skipping {domain} - missing data")
                continue

            # Create plot for each metric
            for metric in metrics:
                plot_dual_axis_metric_comparison(
                    domain, olam_data, infogain_data, metric, output_dir
                )

        except Exception as e:
            print(f"  Error processing {domain}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generated {len(domains)} × {len(metrics)} = {len(domains) * len(metrics)} dual-axis comparison plots")
    print(f"  - Precondition Precision comparison")
    print(f"  - Precondition Recall comparison")
    print(f"  - Effects Precision comparison")
    print(f"  - Effects Recall comparison")
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
