#!/usr/bin/env python3
"""
Analyze correlation between execution success/failures and model precision/recall.

Creates:
1. Scatter plots: Mistakes vs Precision/Recall
2. Dual-axis time series: Metrics and execution outcomes together
3. Statistical correlation analysis
4. Learning efficiency plots
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats


def load_domain_data_with_execution(domain_name: str) -> Dict:
    """
    Load metrics and execution stats for a domain.

    Returns:
        Dict with iterations -> {metrics, successes, failures}
    """
    base_dir = Path("/home/omer/projects/online_model_learning/results")

    # Load from information_gain_metrics
    domain_dir = base_dir / "information_gain_metrics" / domain_name
    problem_dirs = sorted([d for d in domain_dir.iterdir() if d.is_dir()])

    # Aggregate across problems
    all_iterations = defaultdict(lambda: {
        "safe_prec_precision": [], "safe_prec_recall": [],
        "safe_eff_precision": [], "safe_eff_recall": [],
        "complete_prec_precision": [], "complete_prec_recall": [],
        "complete_eff_precision": [], "complete_eff_recall": [],
        "successes": 0, "failures": 0
    })

    for problem_dir in problem_dirs:
        metrics_file = problem_dir / "metrics_per_iteration.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            data = json.load(f)

        for iteration_str, iter_data in data.items():
            iteration = int(iteration_str)

            # Safe model metrics
            safe = iter_data.get("safe_model", {})
            all_iterations[iteration]["safe_prec_precision"].append(safe.get("precondition_precision", 0))
            all_iterations[iteration]["safe_prec_recall"].append(safe.get("precondition_recall", 0))
            all_iterations[iteration]["safe_eff_precision"].append(safe.get("effect_precision", 0))
            all_iterations[iteration]["safe_eff_recall"].append(safe.get("effect_recall", 0))

            # Complete model metrics
            complete = iter_data.get("complete_model", {})
            all_iterations[iteration]["complete_prec_precision"].append(complete.get("precondition_precision", 0))
            all_iterations[iteration]["complete_prec_recall"].append(complete.get("precondition_recall", 0))
            all_iterations[iteration]["complete_eff_precision"].append(complete.get("effect_precision", 0))
            all_iterations[iteration]["complete_eff_recall"].append(complete.get("effect_recall", 0))

            # Execution stats (already cumulative per problem)
            exec_stats = iter_data.get("execution_stats", {})
            for action_stats in exec_stats.values():
                all_iterations[iteration]["successes"] += action_stats.get("successes", 0)
                all_iterations[iteration]["failures"] += action_stats.get("failures", 0)

    # Average metrics across problems
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
            "complete_eff_recall": np.mean(data["complete_eff_recall"]),
            "successes": data["successes"],
            "failures": data["failures"]
        }

    return result


def plot_scatter_correlation(
    domain_name: str,
    data: Dict,
    metric_type: str,
    output_dir: Path
):
    """
    Create scatter plots showing mistakes vs precision/recall.

    Args:
        domain_name: Domain name
        data: Iteration data with metrics and execution stats
        metric_type: 'preconditions' or 'effects'
        output_dir: Output directory
    """
    iterations = sorted(data.keys())

    # Extract data
    failures = [data[i]["failures"] for i in iterations]
    successes = [data[i]["successes"] for i in iterations]
    mistake_rate = [f / (s + f) if (s + f) > 0 else 0
                    for s, f in zip(successes, failures)]

    # Get appropriate metric keys
    if metric_type == "preconditions":
        safe_prec_key = "safe_prec_precision"
        safe_rec_key = "safe_prec_recall"
        complete_prec_key = "complete_prec_precision"
        complete_rec_key = "complete_prec_recall"
    else:
        safe_prec_key = "safe_eff_precision"
        safe_rec_key = "safe_eff_recall"
        complete_prec_key = "complete_eff_precision"
        complete_rec_key = "complete_eff_recall"

    safe_precision = [data[i][safe_prec_key] for i in iterations]
    safe_recall = [data[i][safe_rec_key] for i in iterations]
    complete_precision = [data[i][complete_prec_key] for i in iterations]
    complete_recall = [data[i][complete_rec_key] for i in iterations]

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'{domain_name.upper()} (Information Gain) - {metric_type.title()} vs Execution Outcomes',
                 fontsize=16, fontweight='bold')

    # Color by iteration (gradient)
    colors = plt.cm.viridis(np.linspace(0, 1, len(iterations)))

    # Top-left: Failures vs Precision
    ax = axes[0, 0]
    ax.scatter(failures, safe_precision, c=colors, s=100, alpha=0.7, label='Safe', marker='o')
    ax.scatter(failures, complete_precision, c=colors, s=100, alpha=0.7, label='Complete', marker='^')
    ax.set_xlabel('Cumulative Failures', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax.set_title('Failures vs Precision', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Failures vs Recall
    ax = axes[0, 1]
    ax.scatter(failures, safe_recall, c=colors, s=100, alpha=0.7, label='Safe', marker='o')
    ax.scatter(failures, complete_recall, c=colors, s=100, alpha=0.7, label='Complete', marker='^')
    ax.set_xlabel('Cumulative Failures', fontsize=11, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax.set_title('Failures vs Recall', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Mistake Rate vs Precision
    ax = axes[1, 0]
    ax.scatter(mistake_rate, safe_precision, c=colors, s=100, alpha=0.7, label='Safe', marker='o')
    ax.scatter(mistake_rate, complete_precision, c=colors, s=100, alpha=0.7, label='Complete', marker='^')
    ax.set_xlabel('Mistake Rate (failures / total)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax.set_title('Mistake Rate vs Precision', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Mistake Rate vs Recall
    ax = axes[1, 1]
    ax.scatter(mistake_rate, safe_recall, c=colors, s=100, alpha=0.7, label='Safe', marker='o')
    ax.scatter(mistake_rate, complete_recall, c=colors, s=100, alpha=0.7, label='Complete', marker='^')
    ax.set_xlabel('Mistake Rate (failures / total)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax.set_title('Mistake Rate vs Recall', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add colorbar for iteration gradient
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=min(iterations), vmax=max(iterations)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('Iteration', fontsize=11, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / f"{domain_name}_{metric_type}_scatter_correlation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_dual_axis_timeseries(
    domain_name: str,
    data: Dict,
    metric_type: str,
    output_dir: Path
):
    """
    Create dual-axis time series: metrics on left, execution outcomes on right.

    Args:
        domain_name: Domain name
        data: Iteration data with metrics and execution stats
        metric_type: 'preconditions' or 'effects'
        output_dir: Output directory
    """
    iterations = sorted(data.keys())

    # Extract data
    failures = [data[i]["failures"] for i in iterations]
    successes = [data[i]["successes"] for i in iterations]

    # Get appropriate metric keys
    if metric_type == "preconditions":
        safe_prec_key = "safe_prec_precision"
        safe_rec_key = "safe_prec_recall"
    else:
        safe_prec_key = "safe_eff_precision"
        safe_rec_key = "safe_eff_recall"

    safe_precision = [data[i][safe_prec_key] for i in iterations]
    safe_recall = [data[i][safe_rec_key] for i in iterations]

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.suptitle(f'{domain_name.upper()} (Information Gain) - {metric_type.title()} Metrics vs Execution',
                 fontsize=14, fontweight='bold')

    # Left y-axis: Metrics
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision / Recall', fontsize=12, fontweight='bold', color='black')

    line1 = ax1.plot(iterations, safe_precision, 'o-', label='Precision (Safe)',
                     linewidth=2.5, markersize=7, color='#1f77b4')
    line2 = ax1.plot(iterations, safe_recall, 's-', label='Recall (Safe)',
                     linewidth=2.5, markersize=7, color='#ff7f0e')

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Right y-axis: Execution outcomes
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Count', fontsize=12, fontweight='bold', color='black')

    line3 = ax2.plot(iterations, successes, '^--', label='Successes',
                     linewidth=2, markersize=6, color='#2ca02c', alpha=0.7)
    line4 = ax2.plot(iterations, failures, 'd--', label='Failures',
                     linewidth=2, markersize=6, color='#d62728', alpha=0.7)

    ax2.tick_params(axis='y', labelcolor='black')

    # Combined legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=11)

    plt.tight_layout()

    output_file = output_dir / f"{domain_name}_{metric_type}_dual_axis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def compute_correlations(
    domain_name: str,
    data: Dict,
    metric_type: str
) -> Dict:
    """
    Compute statistical correlations between execution outcomes and metrics.

    Returns:
        Dict with correlation coefficients and p-values
    """
    iterations = sorted(data.keys())

    # Extract data
    failures = np.array([data[i]["failures"] for i in iterations])
    successes = np.array([data[i]["successes"] for i in iterations])
    total = successes + failures
    mistake_rate = np.array([f / t if t > 0 else 0 for f, t in zip(failures, total)])

    # Get appropriate metric keys
    if metric_type == "preconditions":
        safe_prec_key = "safe_prec_precision"
        safe_rec_key = "safe_prec_recall"
    else:
        safe_prec_key = "safe_eff_precision"
        safe_rec_key = "safe_eff_recall"

    safe_precision = np.array([data[i][safe_prec_key] for i in iterations])
    safe_recall = np.array([data[i][safe_rec_key] for i in iterations])

    results = {
        "domain": domain_name,
        "algorithm": "information_gain",
        "metric_type": metric_type,
        "n_iterations": len(iterations)
    }

    # Pearson correlations
    if len(failures) > 2:
        # Failures vs Precision
        r, p = stats.pearsonr(failures, safe_precision)
        results["failures_vs_precision_pearson_r"] = r
        results["failures_vs_precision_pearson_p"] = p

        # Failures vs Recall
        r, p = stats.pearsonr(failures, safe_recall)
        results["failures_vs_recall_pearson_r"] = r
        results["failures_vs_recall_pearson_p"] = p

        # Mistake rate vs Precision
        r, p = stats.pearsonr(mistake_rate, safe_precision)
        results["mistake_rate_vs_precision_pearson_r"] = r
        results["mistake_rate_vs_precision_pearson_p"] = p

        # Mistake rate vs Recall
        r, p = stats.pearsonr(mistake_rate, safe_recall)
        results["mistake_rate_vs_recall_pearson_r"] = r
        results["mistake_rate_vs_recall_pearson_p"] = p

        # Spearman correlations (for non-linear relationships)
        r, p = stats.spearmanr(failures, safe_precision)
        results["failures_vs_precision_spearman_r"] = r
        results["failures_vs_precision_spearman_p"] = p

        r, p = stats.spearmanr(failures, safe_recall)
        results["failures_vs_recall_spearman_r"] = r
        results["failures_vs_recall_spearman_p"] = p

    return results


def main():
    """Main execution function."""
    base_dir = Path("/home/omer/projects/online_model_learning/results")
    output_dir = base_dir / "comparison_plots" / "correlation_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Domains to analyze (excluding sokoban)
    domains = ["blocksworld", "depots", "driverlog", "ferry", "hanoi", "miconic", "satellite"]
    metric_types = ["preconditions", "effects"]

    all_correlations = []

    for domain in domains:
        print(f"\n{'='*70}")
        print(f"Processing domain: {domain}")
        print('='*70)

        try:
            # Load data
            data = load_domain_data_with_execution(domain)

            if not data:
                print(f"  No data available for {domain}")
                continue

            for metric_type in metric_types:
                print(f"  Metric: {metric_type}")

                # Create scatter plots
                plot_scatter_correlation(domain, data, metric_type, output_dir)

                # Create dual-axis time series
                plot_dual_axis_timeseries(domain, data, metric_type, output_dir)

                # Compute correlations
                corr = compute_correlations(domain, data, metric_type)
                all_correlations.append(corr)

        except Exception as e:
            print(f"  Error processing {domain}: {e}")
            import traceback
            traceback.print_exc()

    # Save correlation results
    corr_file = output_dir / "correlation_statistics.json"
    with open(corr_file, 'w') as f:
        json.dump(all_correlations, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Correlation statistics saved to: {corr_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Generated {len(domains)} × {len(metric_types)} × 2 = {len(domains) * len(metric_types) * 2} plots")
    print(f"  - Scatter correlation plots")
    print(f"  - Dual-axis time series plots")
    print(f"Computed correlations for {len(all_correlations)} combinations")
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
