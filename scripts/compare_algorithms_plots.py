#!/usr/bin/env python3
"""
Generate comparison plots between OLAM and Information Gain algorithms.

This script creates:
1. Precision/Recall comparison plots for preconditions and effects
2. Cumulative success plots
3. Analysis of metrics vs execution outcomes

Usage:
    python scripts/compare_algorithms_plots.py

    python scripts/compare_algorithms_plots.py \\
        --olam-results results/consolidated_results101225/olam \\
        --infogain-results results/consolidated_results101225/information_gain \\
        --domain blocksworld
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict


def load_olam_domain_metrics(domain_path: Path) -> Dict:
    """Load OLAM domain-level metrics."""
    # OLAM results are in subdirectories: olam-results/blocksworld/, olam-results/depots/, etc.
    metrics_file = domain_path / "domain_safe_metrics.json"
    complete_file = domain_path / "domain_complete_metrics.json"

    with open(metrics_file) as f:
        safe_metrics = json.load(f)

    with open(complete_file) as f:
        complete_metrics = json.load(f)

    return {"safe": safe_metrics, "complete": complete_metrics}


def load_infogain_domain_metrics(domain_path: Path) -> Dict:
    """Load Information Gain domain-level aggregated metrics."""
    # Aggregate from all problem directories
    problem_dirs = sorted([d for d in domain_path.iterdir() if d.is_dir()])

    all_iterations = defaultdict(lambda: {"safe": [], "complete": []})

    # Per-problem execution stats: {problem_name: {iteration: {action: {successes, failures}}}}
    per_problem_execution_stats = {}

    # Track which problems reached which iterations for proper averaging
    # {iteration: [{successes: X, failures: Y}, ...]} - list of totals per problem
    execution_stats_per_iteration = defaultdict(list)

    for problem_dir in problem_dirs:
        metrics_file = problem_dir / "metrics_per_iteration.json"
        if not metrics_file.exists():
            continue

        problem_name = problem_dir.name
        per_problem_execution_stats[problem_name] = {}

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

            # Extract execution stats - store per problem
            exec_stats = iter_data.get("execution_stats", {})
            per_problem_execution_stats[problem_name][iteration] = exec_stats

            # Sum successes/failures across actions for this problem at this iteration
            iter_successes = sum(stats.get("successes", 0) for stats in exec_stats.values())
            iter_failures = sum(stats.get("failures", 0) for stats in exec_stats.values())
            execution_stats_per_iteration[iteration].append({
                "successes": iter_successes,
                "failures": iter_failures,
                "problem": problem_name
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

    # Average execution stats across problems that reached each iteration
    averaged_execution_stats = {}
    for iteration in sorted(execution_stats_per_iteration.keys()):
        problem_stats = execution_stats_per_iteration[iteration]
        num_problems = len(problem_stats)
        if num_problems > 0:
            averaged_execution_stats[iteration] = {
                "successes": np.mean([p["successes"] for p in problem_stats]),
                "failures": np.mean([p["failures"] for p in problem_stats]),
                "num_problems": num_problems
            }

    return {
        "metrics": averaged_metrics,
        "execution_stats": averaged_execution_stats,  # Now properly averaged
        "per_problem_execution_stats": per_problem_execution_stats  # For per-problem plots
    }


def plot_precision_recall_comparison(
    domain_name: str,
    olam_metrics: Dict,
    infogain_data: Dict,
    output_dir: Path,
    metric_type: str = "preconditions"
):
    """
    Create comparison plots for precision and recall.

    Args:
        domain_name: Name of the domain
        olam_metrics: OLAM metrics dict with 'safe' and 'complete' keys
        infogain_data: Information Gain data dict
        output_dir: Directory to save plots
        metric_type: 'preconditions' or 'effects'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{domain_name.upper()} - {metric_type.title()} Metrics', fontsize=16, fontweight='bold')

    # Determine metric category based on type
    metric_category = "preconditions" if metric_type == "preconditions" else "effects"

    # Extract OLAM data
    # Structure: safe/complete -> checkpoints -> checkpoint_id -> safe/complete -> preconditions/effects
    olam_safe_checkpoints = olam_metrics["safe"]["checkpoints"]
    olam_complete_checkpoints = olam_metrics["complete"]["checkpoints"]

    olam_steps = sorted([int(k) for k in olam_safe_checkpoints.keys()])

    # Extract precision and recall from nested structure
    olam_safe_precision = [olam_safe_checkpoints[str(s)]["safe"][metric_category]["avg_precision"] for s in olam_steps]
    olam_safe_recall = [olam_safe_checkpoints[str(s)]["safe"][metric_category]["avg_recall"] for s in olam_steps]
    olam_complete_precision = [olam_complete_checkpoints[str(s)]["complete"][metric_category]["avg_precision"] for s in olam_steps]
    olam_complete_recall = [olam_complete_checkpoints[str(s)]["complete"][metric_category]["avg_recall"] for s in olam_steps]

    # Extract Information Gain data
    # Structure: safe/complete -> iteration -> {precondition_precision, precondition_recall, effect_precision, effect_recall}
    infogain_metrics = infogain_data["metrics"]
    infogain_safe = infogain_metrics["safe"]
    infogain_complete = infogain_metrics["complete"]

    # Determine InfoGain metric keys (uses underscore notation)
    if metric_type == "preconditions":
        precision_key = "precondition_precision"
        recall_key = "precondition_recall"
    else:
        precision_key = "effect_precision"
        recall_key = "effect_recall"

    infogain_steps = sorted([int(k) for k in infogain_safe.keys()])
    infogain_safe_precision = [infogain_safe[str(s)][precision_key] for s in infogain_steps]
    infogain_safe_recall = [infogain_safe[str(s)][recall_key] for s in infogain_steps]
    infogain_complete_precision = [infogain_complete[str(s)][precision_key] for s in infogain_steps]
    infogain_complete_recall = [infogain_complete[str(s)][recall_key] for s in infogain_steps]

    # Plot Precision
    ax1.plot(olam_steps, olam_safe_precision, 'o-', label='OLAM Safe', linewidth=2, markersize=6, color='#1f77b4')
    ax1.plot(olam_steps, olam_complete_precision, 's--', label='OLAM Complete', linewidth=2, markersize=6, color='#ff7f0e')
    ax1.plot(infogain_steps, infogain_safe_precision, '^-', label='InfoGain Safe', linewidth=2, markersize=6, color='#2ca02c')
    ax1.plot(infogain_steps, infogain_complete_precision, 'd--', label='InfoGain Complete', linewidth=2, markersize=6, color='#d62728')

    ax1.set_xlabel('Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title('Precision Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Plot Recall
    ax2.plot(olam_steps, olam_safe_recall, 'o-', label='OLAM Safe', linewidth=2, markersize=6, color='#1f77b4')
    ax2.plot(olam_steps, olam_complete_recall, 's--', label='OLAM Complete', linewidth=2, markersize=6, color='#ff7f0e')
    ax2.plot(infogain_steps, infogain_safe_recall, '^-', label='InfoGain Safe', linewidth=2, markersize=6, color='#2ca02c')
    ax2.plot(infogain_steps, infogain_complete_recall, 'd--', label='InfoGain Complete', linewidth=2, markersize=6, color='#d62728')

    ax2.set_xlabel('Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_title('Recall Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    output_file = output_dir / f"{domain_name}_{metric_type}_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def load_olam_traces(domain_name: str, raw_olam_path: str = None) -> Dict:
    """Load OLAM trace data for cumulative success/failure calculation.

    Args:
        domain_name: Name of the domain
        raw_olam_path: Path to raw OLAM results directory (parent of domain dirs)
    """
    if raw_olam_path:
        olam_results_path = Path(raw_olam_path) / domain_name
    else:
        # Legacy hardcoded path for backward compatibility
        olam_results_path = Path(f"/home/omer/projects/olam_results/{domain_name}")

    if not olam_results_path.exists():
        return {}

    # Aggregate traces from all problems
    iteration_stats = defaultdict(lambda: {"successes": 0, "failures": 0})

    for problem_dir in sorted(olam_results_path.iterdir()):
        if not problem_dir.is_dir():
            continue

        trace_file = problem_dir / "trace.json"
        if not trace_file.exists():
            continue

        # Read JSON Lines format
        with open(trace_file) as f:
            for line in f:
                entry = json.loads(line)
                iteration = entry.get("iter")
                success = entry.get("success", False)

                if iteration is not None:
                    if success:
                        iteration_stats[iteration]["successes"] += 1
                    else:
                        iteration_stats[iteration]["failures"] += 1

    return dict(iteration_stats)


def plot_cumulative_success(
    domain_name: str,
    infogain_data: Dict,
    output_dir: Path,
    raw_olam_path: str = None
):
    """
    Plot cumulative successes and failures over iterations for OLAM vs InfoGain.
    Also generates per-problem plots.

    Args:
        domain_name: Name of the domain
        infogain_data: Information Gain data with execution_stats (averaged) and per_problem_execution_stats
        output_dir: Directory to save plots (should be domain-specific subdirectory)
        raw_olam_path: Path to raw OLAM results for trace data
    """
    # Load InfoGain execution stats (now properly averaged across problems)
    infogain_stats = infogain_data.get("execution_stats", {})
    per_problem_stats = infogain_data.get("per_problem_execution_stats", {})

    # Load OLAM trace data
    olam_stats = load_olam_traces(domain_name, raw_olam_path)

    if not infogain_stats and not olam_stats:
        print(f"No execution stats available for {domain_name}")
        return

    # === Plot 1: Averaged cumulative success/failure ===
    fig, ax = plt.subplots(figsize=(14, 7))

    # Process InfoGain data - now using averaged stats
    if infogain_stats:
        iterations = sorted([int(k) for k in infogain_stats.keys()])
        cumulative_successes = []
        cumulative_failures = []
        num_problems_list = []

        # Compute cumulative sums of the averaged values
        cum_success = 0
        cum_failure = 0
        for iteration in iterations:
            iter_stats = infogain_stats[iteration]
            # These are per-iteration averages, accumulate them
            cum_success += iter_stats["successes"]
            cum_failure += iter_stats["failures"]
            cumulative_successes.append(cum_success)
            cumulative_failures.append(cum_failure)
            num_problems_list.append(iter_stats.get("num_problems", 1))

        # Add number of problems at final iteration to legend
        final_num_problems = num_problems_list[-1] if num_problems_list else 0
        ax.plot(iterations, cumulative_successes, 'o-',
                label=f'InfoGain Successes (avg of {final_num_problems} problems)',
                linewidth=2.5, markersize=6, color='#2ca02c')
        ax.plot(iterations, cumulative_failures, 's-',
                label=f'InfoGain Failures (avg of {final_num_problems} problems)',
                linewidth=2.5, markersize=6, color='#d62728')

    # Process OLAM data
    if olam_stats:
        iterations = sorted([int(k) for k in olam_stats.keys()])
        cumulative_successes = []
        cumulative_failures = []

        total_success = 0
        total_failure = 0

        for iteration in iterations:
            iter_data = olam_stats[iteration]
            total_success += iter_data["successes"]
            total_failure += iter_data["failures"]

            cumulative_successes.append(total_success)
            cumulative_failures.append(total_failure)

        ax.plot(iterations, cumulative_successes, '^-', label='OLAM Successes',
                linewidth=2.5, markersize=6, color='#1f77b4')
        ax.plot(iterations, cumulative_failures, 'd-', label='OLAM Failures',
                linewidth=2.5, markersize=6, color='#ff7f0e')

    ax.set_xlabel('Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Count (averaged across problems)', fontsize=12, fontweight='bold')
    ax.set_title(f'{domain_name.upper()} - Cumulative Success/Failure (Averaged)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"{domain_name}_cumulative_success.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # === Plot 2: Per-problem cumulative success/failure ===
    if per_problem_stats:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Use a colormap for different problems
        colors = plt.cm.tab20(np.linspace(0, 1, len(per_problem_stats)))

        for idx, (problem_name, problem_data) in enumerate(sorted(per_problem_stats.items())):
            if not problem_data:
                continue

            iterations = sorted([int(k) for k in problem_data.keys()])
            cumulative_successes = []
            cumulative_failures = []

            cum_success = 0
            cum_failure = 0
            for iteration in iterations:
                iter_stats = problem_data[iteration]
                # Sum across actions for this iteration
                iter_success = sum(stats.get("successes", 0) for stats in iter_stats.values())
                iter_failure = sum(stats.get("failures", 0) for stats in iter_stats.values())
                cum_success += iter_success
                cum_failure += iter_failure
                cumulative_successes.append(cum_success)
                cumulative_failures.append(cum_failure)

            ax1.plot(iterations, cumulative_successes, '-', label=problem_name,
                     linewidth=1.5, color=colors[idx], alpha=0.7)
            ax2.plot(iterations, cumulative_failures, '-', label=problem_name,
                     linewidth=1.5, color=colors[idx], alpha=0.7)

        ax1.set_xlabel('Steps', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Cumulative Successes', fontsize=11, fontweight='bold')
        ax1.set_title(f'{domain_name.upper()} - InfoGain Per-Problem Successes', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8, ncol=2)

        ax2.set_xlabel('Steps', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Failures', fontsize=11, fontweight='bold')
        ax2.set_title(f'{domain_name.upper()} - InfoGain Per-Problem Failures', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8, ncol=2)

        plt.tight_layout()

        per_problem_file = output_dir / f"{domain_name}_per_problem_success_failure.png"
        plt.savefig(per_problem_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {per_problem_file}")
        plt.close()


def suggest_mistake_correlation_approach():
    """
    Suggest approach for analyzing precision/recall as function of mistakes and successes.
    """
    approach = """
APPROACH: Precision/Recall vs Mistakes/Success Analysis
========================================================

1. DATA COLLECTION:
   - For each iteration i, collect:
     * Precision and Recall (preconditions and effects separately)
     * Number of successful action executions up to iteration i
     * Number of failed action executions up to iteration i
     * Mistake rate = failures / (successes + failures)

2. VISUALIZATION OPTIONS:

   A. Scatter Plots:
      - X-axis: Cumulative mistakes (or mistake rate)
      - Y-axis: Precision or Recall
      - Color code by: iteration number (gradient)
      - Separate plots for:
        * Safe model vs Complete model
        * Preconditions vs Effects
        * OLAM vs Information Gain

      Expected insight: Does precision/recall improve despite mistakes?
      Does higher mistake rate correlate with learning progress?

   B. Dual-Axis Time Series:
      - X-axis: Steps/iterations
      - Left Y-axis: Precision/Recall
      - Right Y-axis: Cumulative success/failure count
      - Plot both metrics on same chart with different colors

      Expected insight: Visual correlation between execution outcomes
      and model quality over time.

   C. Heatmap Analysis:
      - Rows: Precision bins (0-0.2, 0.2-0.4, ..., 0.8-1.0)
      - Columns: Success rate bins (0-20%, 20-40%, ..., 80-100%)
      - Cell values: Number of (domain, iteration) pairs in that bin

      Expected insight: Distribution showing if high precision
      correlates with high success rates.

   D. Learning Efficiency Plot:
      - X-axis: Number of mistakes encountered
      - Y-axis: Precision gain or Recall gain
      - Points: Each iteration as a point
      - Fit curve: Show learning rate per mistake

      Expected insight: How efficiently does each algorithm learn
      from failures?

3. STATISTICAL ANALYSIS:
   - Pearson correlation: mistake_rate vs precision/recall
   - Spearman rank correlation: for non-linear relationships
   - Linear regression: predict precision from cumulative failures
   - Compare slopes between OLAM and Information Gain

4. COMPARATIVE METRICS:
   - Mistakes-to-convergence: How many mistakes needed to reach
     90% precision/recall?
   - Success-weighted learning: Precision gain per successful execution
   - Failure-weighted learning: Precision gain per failed execution
   - Recovery rate: How quickly does precision improve after failures?

5. IMPLEMENTATION NOTES:
   - Need to align iteration numbers between OLAM and InfoGain
     (they may have different checkpoint frequencies)
   - Consider normalizing by domain complexity
     (some domains inherently have more failures)
   - Aggregate across multiple problems to get domain-level trends
   - Use confidence intervals when comparing algorithms

6. KEY QUESTIONS TO ANSWER:
   - Does learning require failures, or do successes also contribute?
   - Which algorithm learns more efficiently from mistakes?
   - Is there a minimum mistake threshold for convergence?
   - Do mistakes in early iterations have more impact than later ones?
"""
    return approach


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate comparison plots between OLAM and Information Gain algorithms"
    )
    parser.add_argument(
        "--olam-results",
        type=str,
        default="results/olam-results",
        help="Directory containing OLAM results (default: results/olam-results)"
    )
    parser.add_argument(
        "--infogain-results",
        type=str,
        default="results/information_gain_metrics",
        help="Directory containing InfoGain results (default: results/information_gain_metrics)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison_plots",
        help="Output directory for plots (default: results/comparison_plots)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Process single domain (optional, processes all common domains if not specified)"
    )
    parser.add_argument(
        "--raw-olam-path",
        type=str,
        help="Path to raw OLAM results for trace data (e.g., /path/to/olam_results)"
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path("/home/omer/projects/online_model_learning")
    olam_results_dir = base_dir / args.olam_results
    infogain_results_dir = base_dir / args.infogain_results
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    # Check if directories exist
    if not olam_results_dir.exists():
        print(f"ERROR: OLAM results directory not found: {olam_results_dir}")
        return 1
    if not infogain_results_dir.exists():
        print(f"ERROR: InfoGain results directory not found: {infogain_results_dir}")
        return 1

    # Get all domains (intersection of both algorithms)
    olam_domains = set([d.name for d in olam_results_dir.iterdir() if d.is_dir()])
    infogain_domains = set([d.name for d in infogain_results_dir.iterdir() if d.is_dir()])

    # Filter to single domain if specified
    if args.domain:
        if args.domain not in olam_domains:
            print(f"ERROR: Domain '{args.domain}' not found in OLAM results")
            return 1
        if args.domain not in infogain_domains:
            print(f"ERROR: Domain '{args.domain}' not found in InfoGain results")
            return 1
        common_domains = [args.domain]
    else:
        common_domains = sorted(olam_domains & infogain_domains)

    print(f"Found {len(common_domains)} common domains: {common_domains}")
    print()

    # Process each domain
    for domain in common_domains:
        print(f"Processing domain: {domain}")
        print("-" * 60)

        # Create domain-specific output directory
        domain_output_dir = output_dir / domain
        domain_output_dir.mkdir(exist_ok=True, parents=True)

        try:
            # Load metrics
            olam_metrics = load_olam_domain_metrics(olam_results_dir / domain)
            infogain_data = load_infogain_domain_metrics(infogain_results_dir / domain)

            # Create comparison plots for preconditions
            plot_precision_recall_comparison(
                domain, olam_metrics, infogain_data, domain_output_dir, "preconditions"
            )

            # Create comparison plots for effects
            plot_precision_recall_comparison(
                domain, olam_metrics, infogain_data, domain_output_dir, "effects"
            )

            # Create cumulative success plots
            plot_cumulative_success(domain, infogain_data, domain_output_dir, args.raw_olam_path)

            print()

        except Exception as e:
            print(f"Error processing {domain}: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Print suggested approach for mistake correlation
    print("\n" + "=" * 80)
    print(suggest_mistake_correlation_approach())
    print("=" * 80)

    # Save approach to file
    approach_file = output_dir / "mistake_correlation_approach.txt"
    with open(approach_file, 'w') as f:
        f.write(suggest_mistake_correlation_approach())
    print(f"\nApproach saved to: {approach_file}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
