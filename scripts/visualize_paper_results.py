#!/usr/bin/env python
"""
Generate publication-ready visualizations for OLAM vs Information Gain comparison.
Produces charts and tables for paper analysis.

Usage: python scripts/visualize_paper_results.py <results_directory>
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.pddl_io import PDDLReader

# Setup plotting style for publication
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')  # Fallback if seaborn style not available
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

# Constants
ALGORITHMS = ["olam", "information_gain"]
DOMAINS = ["blocksworld", "depots"]  # Excluding logistics (OLAM fails) and gripper (corrupted files)
PROBLEMS = ["p00", "p01", "p02"]
ALGORITHM_LABELS = {
    "olam": "OLAM",
    "information_gain": "Information Gain"
}
COLORS = {
    "olam": "#1f77b4",
    "information_gain": "#ff7f0e"
}


class PaperVisualizer:
    """Generate publication-ready visualizations for experimental results."""

    def __init__(self, results_dir: Path):
        """
        Initialize the visualizer.

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = results_dir
        self.output_dir = results_dir / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        self.data = self._load_all_data()

    def _load_all_data(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Load all experiment data from results directory.

        Returns:
            Nested dictionary: algorithm -> domain -> problem -> data
        """
        data = {}

        for algorithm in ALGORITHMS:
            data[algorithm] = {}

            for domain in DOMAINS:
                data[algorithm][domain] = {}

                for problem in PROBLEMS:
                    # Find experiment directory
                    exp_dir = self.results_dir / algorithm / domain / problem

                    if not exp_dir.exists():
                        print(f"Warning: Missing data for {algorithm}/{domain}/{problem}")
                        continue

                    # Load metrics JSON (files are in experiments/ subdirectory)
                    metrics_files = list(exp_dir.glob('experiments/*_metrics.json'))
                    if metrics_files:
                        with open(metrics_files[0], 'r') as f:
                            metrics_data = json.load(f)
                            data[algorithm][domain][problem] = metrics_data

                    # Load summary (files are in experiments/ subdirectory)
                    summary_files = list(exp_dir.glob('experiments/*_summary.json'))
                    if summary_files:
                        with open(summary_files[0], 'r') as f:
                            summary_data = json.load(f)
                            if problem in data[algorithm][domain]:
                                data[algorithm][domain][problem]['summary'] = summary_data

        return data

    def generate_success_rate_charts(self) -> None:
        """Generate success rate charts for each domain and aggregated."""
        print("\nGenerating success rate charts...")

        # 1. Per-domain charts
        for domain in DOMAINS:
            self._plot_domain_success_rate(domain)

        # 2. Aggregated chart for all domains
        self._plot_aggregated_success_rate()

        print("Success rate charts saved to:", self.output_dir)

    def _plot_domain_success_rate(self, domain: str) -> None:
        """
        Plot success rate over iterations for a specific domain.

        Args:
            domain: Domain name
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        convergence_points = {}  # Track convergence for each algorithm

        for algorithm in ALGORITHMS:
            # Aggregate data across problems
            all_iterations = []
            all_success_rates = []
            convergence_iters = []

            for problem in PROBLEMS:
                if problem not in self.data[algorithm].get(domain, {}):
                    continue

                exp_data = self.data[algorithm][domain][problem]
                if 'actions' not in exp_data or not exp_data['actions']:
                    continue

                # Calculate cumulative success rate
                actions_df = pd.DataFrame(exp_data['actions'])
                if len(actions_df) == 0:
                    continue

                actions_df['cumulative_success_rate'] = (
                    actions_df['success'].expanding().mean()
                )

                all_iterations.append(actions_df.index.values)
                all_success_rates.append(actions_df['cumulative_success_rate'].values)

                # Track convergence point (last iteration)
                convergence_iters.append(len(actions_df))

            if all_success_rates:
                # Store average convergence point
                avg_convergence = int(np.mean(convergence_iters))
                convergence_points[algorithm] = avg_convergence
                # Average across problems
                max_len = max(len(sr) for sr in all_success_rates)
                padded_rates = []
                for sr in all_success_rates:
                    # Pad with last value if needed
                    if len(sr) < max_len:
                        padded = np.pad(sr, (0, max_len - len(sr)),
                                      mode='constant', constant_values=sr[-1])
                    else:
                        padded = sr[:max_len]
                    padded_rates.append(padded)

                avg_success_rate = np.mean(padded_rates, axis=0)
                iterations = np.arange(len(avg_success_rate))

                # Plot with confidence interval
                std_success_rate = np.std(padded_rates, axis=0)
                ax.plot(iterations, avg_success_rate,
                       label=ALGORITHM_LABELS[algorithm],
                       color=COLORS[algorithm], linewidth=2)
                ax.fill_between(iterations,
                              avg_success_rate - std_success_rate,
                              avg_success_rate + std_success_rate,
                              color=COLORS[algorithm], alpha=0.2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cumulative Success Rate')
        ax.set_title(f'Success Rate Comparison - {domain.capitalize()} Domain')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # Add horizontal lines showing convergence performance level
        for algorithm in ALGORITHMS:
            if algorithm in convergence_points and convergence_points[algorithm] < 1000:
                # Find the success rate at convergence
                for problem in PROBLEMS:
                    if problem not in self.data[algorithm].get(domain, {}):
                        continue
                    exp_data = self.data[algorithm][domain][problem]
                    if 'actions' in exp_data and exp_data['actions']:
                        actions_df = pd.DataFrame(exp_data['actions'])
                        if len(actions_df) > 0:
                            final_rate = actions_df['success'].expanding().mean().iloc[-1]
                            ax.axhline(y=final_rate, color=COLORS[algorithm], linestyle='--',
                                      alpha=0.5, linewidth=1.5,
                                      label=f'{ALGORITHM_LABELS[algorithm]} converged @{convergence_points[algorithm]} (rate={final_rate:.3f})')
                            break  # Only need one problem to show the line
                # Re-do legend to include convergence lines
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc='lower right')

        # Save figure
        fig_path = self.output_dir / f'success_rate_{domain}.pdf'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        # Also save as PNG for preview
        fig_path_png = self.output_dir / f'success_rate_{domain}.png'
        fig.savefig(fig_path_png, bbox_inches='tight', dpi=150)

    def _plot_aggregated_success_rate(self) -> None:
        """Plot aggregated success rate across all domains."""
        fig, ax = plt.subplots(figsize=(12, 7))

        for algorithm in ALGORITHMS:
            all_domain_rates = []

            for domain in DOMAINS:
                domain_rates = []

                for problem in PROBLEMS:
                    if problem not in self.data[algorithm].get(domain, {}):
                        continue

                    exp_data = self.data[algorithm][domain][problem]
                    if 'actions' not in exp_data or not exp_data['actions']:
                        continue

                    actions_df = pd.DataFrame(exp_data['actions'])
                    if len(actions_df) == 0:
                        continue

                    actions_df['cumulative_success_rate'] = (
                        actions_df['success'].expanding().mean()
                    )
                    domain_rates.append(actions_df['cumulative_success_rate'].values)

                if domain_rates:
                    # Average across problems for this domain
                    max_len = max(len(sr) for sr in domain_rates)
                    padded_rates = []
                    for sr in domain_rates:
                        if len(sr) < max_len:
                            padded = np.pad(sr, (0, max_len - len(sr)),
                                          mode='constant', constant_values=sr[-1])
                        else:
                            padded = sr[:max_len]
                        padded_rates.append(padded)
                    avg_domain_rate = np.mean(padded_rates, axis=0)
                    all_domain_rates.append(avg_domain_rate)

            if all_domain_rates:
                # Average across all domains
                max_len = max(len(sr) for sr in all_domain_rates)
                padded_rates = []
                for sr in all_domain_rates:
                    if len(sr) < max_len:
                        padded = np.pad(sr, (0, max_len - len(sr)),
                                      mode='constant', constant_values=sr[-1])
                    else:
                        padded = sr[:max_len]
                    padded_rates.append(padded)

                avg_success_rate = np.mean(padded_rates, axis=0)
                iterations = np.arange(len(avg_success_rate))

                ax.plot(iterations, avg_success_rate,
                       label=ALGORITHM_LABELS[algorithm],
                       color=COLORS[algorithm], linewidth=2.5)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cumulative Success Rate')
        ax.set_title('Aggregated Success Rate Comparison Across All Domains')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # Save figure
        fig_path = self.output_dir / 'success_rate_aggregated.pdf'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        # Also save as PNG
        fig_path_png = self.output_dir / 'success_rate_aggregated.png'
        fig.savefig(fig_path_png, bbox_inches='tight', dpi=150)

    def generate_success_tables(self) -> None:
        """Generate success rate tables for paper."""
        print("\nGenerating success rate tables...")

        # Create comprehensive comparison table
        table_data = []

        for domain in DOMAINS:
            for algorithm in ALGORITHMS:
                row = {'Domain': domain.capitalize(),
                      'Algorithm': ALGORITHM_LABELS[algorithm]}

                success_rates = []
                final_rates = []

                for problem in PROBLEMS:
                    if problem not in self.data[algorithm].get(domain, {}):
                        continue

                    exp_data = self.data[algorithm][domain][problem]
                    if 'actions' in exp_data and exp_data['actions']:
                        actions_df = pd.DataFrame(exp_data['actions'])
                        if len(actions_df) > 0:
                            final_rate = actions_df['success'].mean()
                            success_rates.append(final_rate)
                            final_rates.append(final_rate)

                if success_rates:
                    row['Avg Success Rate'] = f"{np.mean(success_rates):.3f}"
                    row['Std Dev'] = f"{np.std(success_rates):.3f}"

                    # Add checkpoints
                    for checkpoint in [100, 300, 500, 1000]:
                        checkpoint_rates = []
                        for problem in PROBLEMS:
                            if problem not in self.data[algorithm].get(domain, {}):
                                continue
                            exp_data = self.data[algorithm][domain][problem]
                            if 'actions' in exp_data and exp_data['actions']:
                                actions_df = pd.DataFrame(exp_data['actions'])
                                if len(actions_df) >= checkpoint:
                                    rate = actions_df.iloc[:checkpoint]['success'].mean()
                                    checkpoint_rates.append(rate)

                        if checkpoint_rates:
                            row[f'@{checkpoint}'] = f"{np.mean(checkpoint_rates):.3f}"

                    table_data.append(row)

        # Convert to DataFrame and save
        table_df = pd.DataFrame(table_data)
        table_path = self.output_dir / 'success_rate_table.csv'
        table_df.to_csv(table_path, index=False)

        # Also create LaTeX table for paper (skip if jinja2 not available)
        try:
            latex_table = table_df.to_latex(index=False, float_format="%.3f")
            latex_path = self.output_dir / 'success_rate_table.tex'
            with open(latex_path, 'w') as f:
                f.write(latex_table)
        except ImportError:
            print("Note: Skipping LaTeX table generation (jinja2 not installed)")

        print(f"Success rate table saved to: {table_path}")
        print("\nSuccess Rate Summary Table:")
        print(table_df.to_string(index=False))

    def generate_precision_recall_charts(self) -> None:
        """Generate precision and recall charts over iterations by comparing to ground truth."""
        print("\nGenerating precision and recall charts over iterations (vs ground truth)...")

        # Create figures for precision and recall (dynamic layout based on number of domains)
        num_domains = len(DOMAINS)
        fig_prec, axes_prec = plt.subplots(1, num_domains, figsize=(9 * num_domains, 5))
        if num_domains == 1:
            axes_prec = [axes_prec]

        fig_rec, axes_rec = plt.subplots(1, num_domains, figsize=(9 * num_domains, 5))
        if num_domains == 1:
            axes_rec = [axes_rec]

        # Also create aggregated charts
        fig_agg, (ax_prec_agg, ax_rec_agg) = plt.subplots(1, 2, figsize=(14, 6))

        all_algo_prec_data = {algo: [] for algo in ALGORITHMS}
        all_algo_rec_data = {algo: [] for algo in ALGORITHMS}

        for idx, domain in enumerate(DOMAINS):
            ax_prec = axes_prec[idx]
            ax_rec = axes_rec[idx]

            # Load ground truth for this domain
            domain_file = project_root / f"benchmarks/olam-compatible/{domain}/domain.pddl"
            problem_file = project_root / f"benchmarks/olam-compatible/{domain}/p00.pddl"  # Any problem file
            try:
                reader = PDDLReader()
                ground_truth_domain, _ = reader.parse_domain_and_problem(str(domain_file), str(problem_file))
            except Exception as e:
                print(f"Warning: Could not load ground truth for {domain}: {e}")
                continue

            for algorithm in ALGORITHMS:
                domain_prec_curves = []
                domain_rec_curves = []

                for problem in PROBLEMS:
                    if problem not in self.data[algorithm].get(domain, {}):
                        continue

                    exp_data = self.data[algorithm][domain][problem]

                    # Extract precision/recall from snapshots
                    if 'snapshots' in exp_data:
                        snapshots = exp_data['snapshots']

                        iterations = []
                        prec_precisions = []  # Precision for preconditions
                        prec_recalls = []     # Recall for preconditions
                        eff_precisions = []   # Precision for effects
                        eff_recalls = []      # Recall for effects

                        for snapshot in snapshots:
                            if 'learning_evidence' not in snapshot:
                                continue

                            iteration = snapshot['step']
                            evidence = snapshot['learning_evidence']

                            # Calculate precision/recall against ground truth
                            all_prec_precision = []
                            all_prec_recall = []
                            all_eff_precision = []
                            all_eff_recall = []

                            for action_name, learned_data in evidence.get('actions', {}).items():
                                # Get ground truth for this action
                                true_action = ground_truth_domain.get_action(action_name)
                                if not true_action:
                                    continue

                                true_prec_raw = true_action.preconditions
                                true_add_raw = true_action.add_effects
                                true_del_raw = true_action.del_effects

                                # Normalize ground truth (IMPORTANT!)
                                true_prec = set(self._normalize_pddl_literal(lit) for lit in true_prec_raw)
                                true_add = set(self._normalize_pddl_literal(lit) for lit in true_add_raw)
                                true_del = set(self._normalize_pddl_literal(lit) for lit in true_del_raw)

                                # Get learned preconditions and effects
                                learned_prec = set()
                                learned_add = set()
                                learned_del = set()

                                # Extract learned preconditions (handle different formats)
                                # For OLAM
                                if 'preconditions_certain' in learned_data:
                                    learned_prec.update(self._normalize_pddl_literal(lit) for lit in learned_data['preconditions_certain'])
                                    # Also include uncertain as learned (optimistic)
                                    learned_prec.update(self._normalize_pddl_literal(lit) for lit in learned_data.get('preconditions_uncertain', []))

                                    # Effects for OLAM
                                    learned_add.update(self._normalize_pddl_literal(lit) for lit in learned_data.get('effects_positive', []))
                                    learned_del.update(self._normalize_pddl_literal(lit) for lit in learned_data.get('effects_negative', []))

                                # For Information Gain
                                elif 'preconditions_possible' in learned_data:
                                    # Use the possible set as learned preconditions
                                    learned_prec.update(self._normalize_pddl_literal(lit) for lit in learned_data['preconditions_possible'])

                                    # Effects for InfoGain
                                    learned_add.update(self._normalize_pddl_literal(lit) for lit in learned_data.get('effects_add_confirmed', []))
                                    learned_del.update(self._normalize_pddl_literal(lit) for lit in learned_data.get('effects_del_confirmed', []))

                                # Calculate precision and recall for preconditions
                                if true_prec or learned_prec:
                                    tp_prec = len(true_prec & learned_prec)
                                    fp_prec = len(learned_prec - true_prec)
                                    fn_prec = len(true_prec - learned_prec)

                                    prec_precision = tp_prec / (tp_prec + fp_prec) if (tp_prec + fp_prec) > 0 else 0.0
                                    prec_recall = tp_prec / (tp_prec + fn_prec) if (tp_prec + fn_prec) > 0 else 0.0

                                    all_prec_precision.append(prec_precision)
                                    all_prec_recall.append(prec_recall)

                                # Calculate precision and recall for effects (combine add and del)
                                true_eff = true_add | set(f"del({e})" for e in true_del)
                                learned_eff = learned_add | set(f"del({e})" for e in learned_del)

                                if true_eff or learned_eff:
                                    tp_eff = len(true_eff & learned_eff)
                                    fp_eff = len(learned_eff - true_eff)
                                    fn_eff = len(true_eff - learned_eff)

                                    eff_precision = tp_eff / (tp_eff + fp_eff) if (tp_eff + fp_eff) > 0 else 0.0
                                    eff_recall = tp_eff / (tp_eff + fn_eff) if (tp_eff + fn_eff) > 0 else 0.0

                                    all_eff_precision.append(eff_precision)
                                    all_eff_recall.append(eff_recall)

                            # Average metrics across all actions
                            if all_prec_precision:
                                avg_prec_precision = np.mean(all_prec_precision)
                                avg_prec_recall = np.mean(all_prec_recall)
                            else:
                                avg_prec_precision = 0.0
                                avg_prec_recall = 0.0

                            if all_eff_precision:
                                avg_eff_precision = np.mean(all_eff_precision)
                                avg_eff_recall = np.mean(all_eff_recall)
                            else:
                                avg_eff_precision = 0.0
                                avg_eff_recall = 0.0

                            iterations.append(iteration)
                            prec_precisions.append(avg_prec_precision)
                            prec_recalls.append(avg_prec_recall)
                            eff_precisions.append(avg_eff_precision)
                            eff_recalls.append(avg_eff_recall)

                        if iterations:
                            # Keep precondition and effect metrics SEPARATE (no averaging!)
                            domain_prec_curves.append({
                                'preconditions': np.array(prec_precisions),
                                'effects': np.array(eff_precisions)
                            })
                            domain_rec_curves.append({
                                'preconditions': np.array(prec_recalls),
                                'effects': np.array(eff_recalls)
                            })

                # Average across problems and plot for this domain (preconditions and effects separately)
                if domain_prec_curves:
                    # Extract precondition and effect curves separately
                    prec_only_curves = [curve['preconditions'] for curve in domain_prec_curves]
                    eff_only_curves = [curve['effects'] for curve in domain_prec_curves]
                    prec_rec_only_curves = [curve['preconditions'] for curve in domain_rec_curves]
                    eff_rec_only_curves = [curve['effects'] for curve in domain_rec_curves]

                    # Average precondition curves
                    max_len = max(len(curve) for curve in prec_only_curves)
                    padded_prec = []
                    for curve in prec_only_curves:
                        if len(curve) < max_len:
                            padded = np.pad(curve, (0, max_len - len(curve)),
                                          mode='constant', constant_values=curve[-1])
                        else:
                            padded = curve[:max_len]
                        padded_prec.append(padded)
                    avg_prec_prec = np.mean(padded_prec, axis=0)

                    # Average effect precision curves
                    padded_eff = []
                    for curve in eff_only_curves:
                        if len(curve) < max_len:
                            padded = np.pad(curve, (0, max_len - len(curve)),
                                          mode='constant', constant_values=curve[-1])
                        else:
                            padded = curve[:max_len]
                        padded_eff.append(padded)
                    avg_eff_prec = np.mean(padded_eff, axis=0)

                    # Average precondition recall curves
                    padded_prec_rec = []
                    for curve in prec_rec_only_curves:
                        if len(curve) < max_len:
                            padded = np.pad(curve, (0, max_len - len(curve)),
                                          mode='constant', constant_values=curve[-1])
                        else:
                            padded = curve[:max_len]
                        padded_prec_rec.append(padded)
                    avg_prec_rec = np.mean(padded_prec_rec, axis=0)

                    # Average effect recall curves
                    padded_eff_rec = []
                    for curve in eff_rec_only_curves:
                        if len(curve) < max_len:
                            padded = np.pad(curve, (0, max_len - len(curve)),
                                          mode='constant', constant_values=curve[-1])
                        else:
                            padded = curve[:max_len]
                        padded_eff_rec.append(padded)
                    avg_eff_rec = np.mean(padded_eff_rec, axis=0)

                    # Create smooth x-axis (iterations)
                    x_iterations = np.linspace(0, 1000, len(avg_prec_prec))

                    # Plot SEPARATE lines for preconditions and effects
                    ax_prec.plot(x_iterations, avg_prec_prec,
                               label=f'{ALGORITHM_LABELS[algorithm]} (Prec)',
                               color=COLORS[algorithm], linewidth=2, linestyle='-')
                    ax_prec.plot(x_iterations, avg_eff_prec,
                               label=f'{ALGORITHM_LABELS[algorithm]} (Eff)',
                               color=COLORS[algorithm], linewidth=2, linestyle='--')

                    ax_rec.plot(x_iterations, avg_prec_rec,
                              label=f'{ALGORITHM_LABELS[algorithm]} (Prec)',
                              color=COLORS[algorithm], linewidth=2, linestyle='-')
                    ax_rec.plot(x_iterations, avg_eff_rec,
                              label=f'{ALGORITHM_LABELS[algorithm]} (Eff)',
                              color=COLORS[algorithm], linewidth=2, linestyle='--')

                    # Store for aggregated plot (store both preconditions and effects)
                    all_algo_prec_data[algorithm].append({'preconditions': avg_prec_prec, 'effects': avg_eff_prec})
                    all_algo_rec_data[algorithm].append({'preconditions': avg_prec_rec, 'effects': avg_eff_rec})

            # Configure domain-specific precision plot
            ax_prec.set_xlabel('Iteration')
            ax_prec.set_ylabel('Precision')
            ax_prec.set_title(f'{domain.capitalize()} Domain - Precision')
            ax_prec.legend()
            ax_prec.grid(True, alpha=0.3)
            ax_prec.set_ylim([0, 1.05])

            # Configure domain-specific recall plot
            ax_rec.set_xlabel('Iteration')
            ax_rec.set_ylabel('Recall')
            ax_rec.set_title(f'{domain.capitalize()} Domain - Recall')
            ax_rec.legend()
            ax_rec.grid(True, alpha=0.3)
            ax_rec.set_ylim([0, 1.05])

        # Create aggregated plots (preconditions and effects separately)
        for algorithm in ALGORITHMS:
            if all_algo_prec_data[algorithm]:
                # Extract preconditions and effects from dictionaries
                prec_only_curves = [curve['preconditions'] for curve in all_algo_prec_data[algorithm]]
                eff_only_curves = [curve['effects'] for curve in all_algo_prec_data[algorithm]]
                prec_rec_only_curves = [curve['preconditions'] for curve in all_algo_rec_data[algorithm]]
                eff_rec_only_curves = [curve['effects'] for curve in all_algo_rec_data[algorithm]]

                # Find max length
                max_len = max(len(curve) for curve in prec_only_curves)

                # Pad precondition precision curves
                padded_prec_prec = []
                for curve in prec_only_curves:
                    if len(curve) < max_len:
                        padded = np.pad(curve, (0, max_len - len(curve)),
                                      mode='constant', constant_values=curve[-1])
                    else:
                        padded = curve[:max_len]
                    padded_prec_prec.append(padded)

                # Pad effect precision curves
                padded_eff_prec = []
                for curve in eff_only_curves:
                    if len(curve) < max_len:
                        padded = np.pad(curve, (0, max_len - len(curve)),
                                      mode='constant', constant_values=curve[-1])
                    else:
                        padded = curve[:max_len]
                    padded_eff_prec.append(padded)

                # Pad precondition recall curves
                padded_prec_rec = []
                for curve in prec_rec_only_curves:
                    if len(curve) < max_len:
                        padded = np.pad(curve, (0, max_len - len(curve)),
                                      mode='constant', constant_values=curve[-1])
                    else:
                        padded = curve[:max_len]
                    padded_prec_rec.append(padded)

                # Pad effect recall curves
                padded_eff_rec = []
                for curve in eff_rec_only_curves:
                    if len(curve) < max_len:
                        padded = np.pad(curve, (0, max_len - len(curve)),
                                      mode='constant', constant_values=curve[-1])
                    else:
                        padded = curve[:max_len]
                    padded_eff_rec.append(padded)

                # Average across domains
                overall_avg_prec_prec = np.mean(padded_prec_prec, axis=0)
                overall_avg_eff_prec = np.mean(padded_eff_prec, axis=0)
                overall_avg_prec_rec = np.mean(padded_prec_rec, axis=0)
                overall_avg_eff_rec = np.mean(padded_eff_rec, axis=0)
                x_iterations = np.linspace(0, 1000, len(overall_avg_prec_prec))

                # Plot aggregated precision (separate lines for preconditions and effects)
                ax_prec_agg.plot(x_iterations, overall_avg_prec_prec,
                               label=f'{ALGORITHM_LABELS[algorithm]} (Prec)',
                               color=COLORS[algorithm], linewidth=2.5, linestyle='-')
                ax_prec_agg.plot(x_iterations, overall_avg_eff_prec,
                               label=f'{ALGORITHM_LABELS[algorithm]} (Eff)',
                               color=COLORS[algorithm], linewidth=2.5, linestyle='--')

                # Plot aggregated recall (separate lines for preconditions and effects)
                ax_rec_agg.plot(x_iterations, overall_avg_prec_rec,
                              label=f'{ALGORITHM_LABELS[algorithm]} (Prec)',
                              color=COLORS[algorithm], linewidth=2.5, linestyle='-')
                ax_rec_agg.plot(x_iterations, overall_avg_eff_rec,
                              label=f'{ALGORITHM_LABELS[algorithm]} (Eff)',
                              color=COLORS[algorithm], linewidth=2.5, linestyle='--')

        # Configure aggregated plots
        ax_prec_agg.set_xlabel('Iteration')
        ax_prec_agg.set_ylabel('Precision')
        ax_prec_agg.set_title('Aggregated Precision Across All Domains')
        ax_prec_agg.legend(fontsize=11)
        ax_prec_agg.grid(True, alpha=0.3)
        ax_prec_agg.set_ylim([0, 1.05])

        ax_rec_agg.set_xlabel('Iteration')
        ax_rec_agg.set_ylabel('Recall')
        ax_rec_agg.set_title('Aggregated Recall Across All Domains')
        ax_rec_agg.legend(fontsize=11)
        ax_rec_agg.grid(True, alpha=0.3)
        ax_rec_agg.set_ylim([0, 1.05])

        # Save figures
        fig_prec.suptitle('Precision Evolution Over Iterations', fontsize=14)
        fig_prec.tight_layout()
        fig_prec.savefig(self.output_dir / 'precision_by_domain.pdf', bbox_inches='tight')
        fig_prec.savefig(self.output_dir / 'precision_by_domain.png', bbox_inches='tight', dpi=150)
        plt.close(fig_prec)

        fig_rec.suptitle('Recall Evolution Over Iterations', fontsize=14)
        fig_rec.tight_layout()
        fig_rec.savefig(self.output_dir / 'recall_by_domain.pdf', bbox_inches='tight')
        fig_rec.savefig(self.output_dir / 'recall_by_domain.png', bbox_inches='tight', dpi=150)
        plt.close(fig_rec)

        fig_agg.tight_layout()
        fig_agg.savefig(self.output_dir / 'precision_recall_aggregated.pdf', bbox_inches='tight')
        fig_agg.savefig(self.output_dir / 'precision_recall_aggregated.png', bbox_inches='tight', dpi=150)
        plt.close(fig_agg)

        print("Precision and recall charts saved to:", self.output_dir)

    @staticmethod
    def _normalize_pddl_literal(literal: str) -> str:
        """
        Normalize PDDL literal from various formats to match ground truth format.
        Also canonicalizes variable parameters to standard names (?x, ?y, ?z, ?w).

        Converts:
        - "(clear ?x)" → "clear(?x)" (OLAM PDDL format)
        - "(handempty)" → "handempty()" (OLAM zero-arity)
        - "handempty" → "handempty()" (Information Gain zero-arity)
        - "at(?u,?p)" → "at(?x,?y)" (canonicalize parameters by position)
        - "(not (clear ?x))" → "clear(?x)" (OLAM delete effects)

        Args:
            literal: PDDL literal string in various formats

        Returns:
            Normalized literal string with canonical variable names
        """
        if not literal:
            return literal

        # Handle delete effects with (not ...) wrapper (OLAM format)
        if literal.startswith('(not '):
            # Extract the inner literal: (not (clear ?x)) → (clear ?x)
            inner_start = literal.find('(', 4)  # Skip "(not "
            if inner_start != -1:
                inner_end = literal.rfind(')')
                if inner_end != -1:
                    literal = literal[inner_start:inner_end]

        # Remove outer parentheses if present (OLAM format)
        literal = literal.strip()
        if literal.startswith('(') and literal.endswith(')'):
            literal = literal[1:-1].strip()

        # Parse predicate and parameters
        if '(' in literal and literal.endswith(')'):
            # Format: predicate(param1,param2,...)
            paren_idx = literal.index('(')
            predicate = literal[:paren_idx]
            params_str = literal[paren_idx+1:-1]
            params = [p.strip() for p in params_str.split(',')] if params_str else []
        else:
            # Format: predicate param1 param2 ... (space-separated)
            parts = literal.split()
            if not parts:
                return literal
            predicate = parts[0]
            params = parts[1:] if len(parts) > 1 else []

        # Canonicalize variable parameters by order of first appearance
        # This ensures at(?x,?p) and at(?x,?u) both become at(?x,?y) when compared
        canonical_vars = ['?x', '?y', '?z', '?w', '?v', '?u', '?t', '?s']
        var_mapping = {}
        next_canonical_idx = 0

        normalized_params = []
        for param in params:
            if param.startswith('?'):
                # Variable parameter - canonicalize by order of first appearance
                if param not in var_mapping:
                    var_mapping[param] = canonical_vars[next_canonical_idx] if next_canonical_idx < len(canonical_vars) else f'?v{next_canonical_idx}'
                    next_canonical_idx += 1
                normalized_params.append(var_mapping[param])
            else:
                # Non-variable parameter (constant) - keep as-is
                normalized_params.append(param)

        # Reconstruct: predicate(param1,param2,...) or predicate()
        if normalized_params:
            return f"{predicate}({','.join(normalized_params)})"
        else:
            return f"{predicate}()"

    def calculate_precision_recall(self) -> pd.DataFrame:
        """
        Calculate syntactic precision and recall for learned models using final learned_model.json files.
        Compares learned models against ground truth using the paper's methodology:
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)

        Returns:
            DataFrame with precision/recall metrics
        """
        print("\nCalculating final precision and recall metrics (syntactic similarity vs ground truth)...")

        metrics_data = []

        for algorithm in ALGORITHMS:
            for domain in DOMAINS:
                # Load ground truth for this domain
                domain_file = project_root / f"benchmarks/olam-compatible/{domain}/domain.pddl"
                problem_file = project_root / f"benchmarks/olam-compatible/{domain}/p00.pddl"

                try:
                    reader = PDDLReader()
                    ground_truth_domain, _ = reader.parse_domain_and_problem(str(domain_file), str(problem_file))
                except Exception as e:
                    print(f"Warning: Could not load ground truth for {domain}: {e}")
                    continue

                domain_metrics = {
                    'Algorithm': ALGORITHM_LABELS[algorithm],
                    'Domain': domain.capitalize()
                }

                precondition_precisions = []
                precondition_recalls = []
                effect_precisions = []
                effect_recalls = []

                for problem in PROBLEMS:
                    if problem not in self.data[algorithm].get(domain, {}):
                        continue

                    # Load learned model from learned_model.json
                    exp_dir = self.results_dir / algorithm / domain / problem / 'experiments'
                    learned_model_path = exp_dir / 'learned_model.json'

                    if not learned_model_path.exists():
                        print(f"Warning: No learned_model.json for {algorithm}/{domain}/{problem}")
                        continue

                    with open(learned_model_path, 'r') as f:
                        learned_model = json.load(f)

                    # Calculate precision/recall for each action against ground truth
                    action_prec_precisions = []
                    action_prec_recalls = []
                    action_eff_precisions = []
                    action_eff_recalls = []

                    for action_name, action_data in learned_model.get('actions', {}).items():
                        # Get ground truth for this action
                        true_action = ground_truth_domain.get_action(action_name)
                        if not true_action:
                            continue

                        true_prec_raw = true_action.preconditions
                        true_add_raw = true_action.add_effects
                        true_del_raw = true_action.del_effects

                        # Normalize ground truth (IMPORTANT!)
                        true_prec = set(self._normalize_pddl_literal(lit) for lit in true_prec_raw)
                        true_add = set(self._normalize_pddl_literal(lit) for lit in true_add_raw)
                        true_del = set(self._normalize_pddl_literal(lit) for lit in true_del_raw)

                        # Extract learned preconditions and effects based on algorithm format
                        learned_prec = set()
                        learned_add = set()
                        learned_del = set()

                        # OLAM learned model structure
                        if 'preconditions' in action_data and 'certain' in action_data['preconditions']:
                            # Use only certain preconditions for OLAM (normalize format)
                            raw_prec = action_data['preconditions']['certain']
                            learned_prec.update(self._normalize_pddl_literal(lit) for lit in raw_prec)

                            # Effects for OLAM (normalize format)
                            raw_add = action_data['effects'].get('positive', [])
                            raw_del = action_data['effects'].get('negative', [])
                            learned_add.update(self._normalize_pddl_literal(lit) for lit in raw_add)
                            learned_del.update(self._normalize_pddl_literal(lit) for lit in raw_del)

                        # Information Gain learned model structure
                        elif 'preconditions' in action_data and 'possible' in action_data['preconditions']:
                            # For InfoGain: use only the constrained preconditions
                            # If there are constraints, the preconditions are more certain
                            possible = action_data['preconditions'].get('possible', [])
                            constraints = action_data['preconditions'].get('constraints', [])

                            # Use all possible preconditions as learned (normalize format)
                            learned_prec.update(self._normalize_pddl_literal(lit) for lit in possible)

                            # Effects for InfoGain: use confirmed effects only (normalize format)
                            raw_add = action_data['effects'].get('add', [])
                            raw_del = action_data['effects'].get('delete', [])
                            learned_add.update(self._normalize_pddl_literal(lit) for lit in raw_add)
                            learned_del.update(self._normalize_pddl_literal(lit) for lit in raw_del)

                        # Calculate precision and recall for preconditions
                        tp_prec = len(true_prec & learned_prec)
                        fp_prec = len(learned_prec - true_prec)
                        fn_prec = len(true_prec - learned_prec)

                        if (tp_prec + fp_prec) > 0:
                            prec_precision = tp_prec / (tp_prec + fp_prec)
                            action_prec_precisions.append(prec_precision)

                        if (tp_prec + fn_prec) > 0:
                            prec_recall = tp_prec / (tp_prec + fn_prec)
                            action_prec_recalls.append(prec_recall)

                        # Calculate precision and recall for effects
                        # Combine add and delete effects for comparison
                        true_eff = true_add | set(f"del({e})" for e in true_del)
                        learned_eff = learned_add | set(f"del({e})" for e in learned_del)

                        tp_eff = len(true_eff & learned_eff)
                        fp_eff = len(learned_eff - true_eff)
                        fn_eff = len(true_eff - learned_eff)

                        if (tp_eff + fp_eff) > 0:
                            eff_precision = tp_eff / (tp_eff + fp_eff)
                            action_eff_precisions.append(eff_precision)

                        if (tp_eff + fn_eff) > 0:
                            eff_recall = tp_eff / (tp_eff + fn_eff)
                            action_eff_recalls.append(eff_recall)

                    # Average across all actions for this problem
                    if action_prec_precisions:
                        precondition_precisions.append(np.mean(action_prec_precisions))
                    if action_prec_recalls:
                        precondition_recalls.append(np.mean(action_prec_recalls))
                    if action_eff_precisions:
                        effect_precisions.append(np.mean(action_eff_precisions))
                    if action_eff_recalls:
                        effect_recalls.append(np.mean(action_eff_recalls))

                # Average across problems for this domain
                if precondition_precisions:
                    domain_metrics['Precondition Precision'] = f"{np.mean(precondition_precisions):.3f}"
                if precondition_recalls:
                    domain_metrics['Precondition Recall'] = f"{np.mean(precondition_recalls):.3f}"
                if effect_precisions:
                    domain_metrics['Effect Precision'] = f"{np.mean(effect_precisions):.3f}"
                if effect_recalls:
                    domain_metrics['Effect Recall'] = f"{np.mean(effect_recalls):.3f}"

                metrics_data.append(domain_metrics)

        metrics_df = pd.DataFrame(metrics_data)

        # Save metrics
        metrics_path = self.output_dir / 'precision_recall_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)

        print("\nSyntactic Precision/Recall Metrics (vs Ground Truth):")
        print(metrics_df.to_string(index=False))

        return metrics_df

    def generate_runtime_analysis(self) -> None:
        """Generate runtime analysis charts and tables."""
        print("\nGenerating runtime analysis...")

        runtime_data = []

        for algorithm in ALGORITHMS:
            for domain in DOMAINS:
                domain_runtimes = []

                for problem in PROBLEMS:
                    if problem not in self.data[algorithm].get(domain, {}):
                        continue

                    exp_data = self.data[algorithm][domain][problem]
                    if 'actions' in exp_data and exp_data['actions']:
                        actions_df = pd.DataFrame(exp_data['actions'])
                        if len(actions_df) > 0:
                            avg_runtime = actions_df['runtime'].mean()
                            total_runtime = actions_df['runtime'].sum()
                            domain_runtimes.append({
                                'avg_per_action': avg_runtime,
                                'total': total_runtime
                            })

                if domain_runtimes:
                    runtime_data.append({
                        'Algorithm': ALGORITHM_LABELS[algorithm],
                        'Domain': domain.capitalize(),
                        'Avg Runtime/Action (s)': f"{np.mean([r['avg_per_action'] for r in domain_runtimes]):.4f}",
                        'Total Runtime (s)': f"{np.mean([r['total'] for r in domain_runtimes]):.1f}"
                    })

        # Create runtime comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Average runtime per action
        runtime_df = pd.DataFrame(runtime_data)
        runtime_pivot = runtime_df.pivot(index='Domain',
                                        columns='Algorithm',
                                        values='Avg Runtime/Action (s)')

        # Convert to numeric
        for col in runtime_pivot.columns:
            runtime_pivot[col] = pd.to_numeric(runtime_pivot[col].str.replace('f', ''))

        runtime_pivot.plot(kind='bar', ax=ax1)
        ax1.set_title('Average Runtime per Action')
        ax1.set_xlabel('Domain')
        ax1.set_ylabel('Runtime (seconds)')
        ax1.legend(title='Algorithm')
        ax1.grid(True, alpha=0.3)

        # Total runtime
        total_pivot = runtime_df.pivot(index='Domain',
                                      columns='Algorithm',
                                      values='Total Runtime (s)')
        for col in total_pivot.columns:
            total_pivot[col] = pd.to_numeric(total_pivot[col])

        total_pivot.plot(kind='bar', ax=ax2)
        ax2.set_title('Total Runtime')
        ax2.set_xlabel('Domain')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.legend(title='Algorithm')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = self.output_dir / 'runtime_analysis.pdf'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        # Save table
        runtime_table_path = self.output_dir / 'runtime_table.csv'
        runtime_df.to_csv(runtime_table_path, index=False)

        print("\nRuntime Analysis Table:")
        print(runtime_df.to_string(index=False))

    def generate_success_count_charts(self) -> None:
        """Generate simple cumulative counts of successful vs failed transitions."""
        print("\nGenerating success count charts...")

        num_domains = len(DOMAINS)
        fig, axes = plt.subplots(1, num_domains, figsize=(9 * num_domains, 5))
        if num_domains == 1:
            axes = [axes]

        for idx, domain in enumerate(DOMAINS):
            ax = axes[idx]

            for algorithm in ALGORITHMS:
                # Collect cumulative counts across problems
                all_success_counts = []
                all_failure_counts = []

                for problem in PROBLEMS:
                    if problem not in self.data[algorithm].get(domain, {}):
                        continue

                    exp_data = self.data[algorithm][domain][problem]
                    if 'actions' not in exp_data or not exp_data['actions']:
                        continue

                    actions_df = pd.DataFrame(exp_data['actions'])
                    if len(actions_df) == 0:
                        continue

                    # Calculate cumulative counts
                    success_cumsum = actions_df['success'].cumsum()
                    failure_cumsum = (~actions_df['success']).cumsum()

                    all_success_counts.append(success_cumsum.values)
                    all_failure_counts.append(failure_cumsum.values)

                if all_success_counts:
                    # Average across problems
                    max_len = max(len(sc) for sc in all_success_counts)

                    # Pad and average success counts
                    padded_success = []
                    for sc in all_success_counts:
                        if len(sc) < max_len:
                            padded = np.pad(sc, (0, max_len - len(sc)),
                                          mode='constant', constant_values=sc[-1])
                        else:
                            padded = sc[:max_len]
                        padded_success.append(padded)
                    avg_success = np.mean(padded_success, axis=0)

                    # Pad and average failure counts
                    padded_failure = []
                    for fc in all_failure_counts:
                        if len(fc) < max_len:
                            padded = np.pad(fc, (0, max_len - len(fc)),
                                          mode='constant', constant_values=fc[-1])
                        else:
                            padded = fc[:max_len]
                        padded_failure.append(padded)
                    avg_failure = np.mean(padded_failure, axis=0)

                    iterations = np.arange(len(avg_success))

                    # Plot success and failure counts
                    ax.plot(iterations, avg_success,
                           label=f'{ALGORITHM_LABELS[algorithm]} (Success)',
                           color=COLORS[algorithm], linewidth=2, linestyle='-')
                    ax.plot(iterations, avg_failure,
                           label=f'{ALGORITHM_LABELS[algorithm]} (Failure)',
                           color=COLORS[algorithm], linewidth=2, linestyle='--', alpha=0.7)

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cumulative Count')
            ax.set_title(f'{domain.capitalize()} Domain')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Cumulative Success/Failure Counts', fontsize=14)
        plt.tight_layout()

        fig_path = self.output_dir / 'success_failure_counts.pdf'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        # Also save as PNG
        fig_path_png = self.output_dir / 'success_failure_counts.png'
        fig.savefig(fig_path_png, bbox_inches='tight', dpi=150)

        print(f"Success count charts saved to: {self.output_dir}")

    def generate_learning_curve_analysis(self) -> None:
        """Generate learning curve analysis showing how quickly models are learned."""
        print("\nGenerating learning curve analysis...")

        num_domains = len(DOMAINS)
        fig, axes = plt.subplots(1, num_domains, figsize=(9 * num_domains, 5))
        if num_domains == 1:
            axes = [axes]

        for idx, domain in enumerate(DOMAINS):
            ax = axes[idx]

            for algorithm in ALGORITHMS:
                mistake_rates = []

                for problem in PROBLEMS:
                    if problem not in self.data[algorithm].get(domain, {}):
                        continue

                    exp_data = self.data[algorithm][domain][problem]
                    if 'actions' not in exp_data or not exp_data['actions']:
                        continue

                    actions_df = pd.DataFrame(exp_data['actions'])
                    if len(actions_df) == 0:
                        continue

                    # Calculate mistake rate over sliding windows
                    window_size = 50
                    mistake_rate = []
                    for i in range(window_size, len(actions_df) + 1, 10):
                        window = actions_df.iloc[i-window_size:i]
                        rate = (~window['success']).mean()
                        mistake_rate.append(rate)

                    if mistake_rate:
                        mistake_rates.append(mistake_rate)

                if mistake_rates:
                    # Average across problems
                    max_len = max(len(mr) for mr in mistake_rates)
                    padded_rates = []
                    for mr in mistake_rates:
                        if len(mr) < max_len:
                            padded = np.pad(mr, (0, max_len - len(mr)),
                                          mode='constant', constant_values=mr[-1])
                        else:
                            padded = mr[:max_len]
                        padded_rates.append(padded)

                    avg_mistake_rate = np.mean(padded_rates, axis=0)
                    iterations = np.arange(len(avg_mistake_rate)) * 10 + 50

                    ax.plot(iterations, avg_mistake_rate,
                           label=ALGORITHM_LABELS[algorithm],
                           color=COLORS[algorithm], linewidth=2)

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mistake Rate (50-step window)')
            ax.set_title(f'{domain.capitalize()} Domain')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        plt.suptitle('Learning Curves - Mistake Rate Over Time', fontsize=14)
        plt.tight_layout()

        fig_path = self.output_dir / 'learning_curves.pdf'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        # Also save as PNG
        fig_path_png = self.output_dir / 'learning_curves.png'
        fig.savefig(fig_path_png, bbox_inches='tight', dpi=150)

    def generate_all_visualizations(self) -> None:
        """Generate all visualizations and analyses."""
        print("=" * 80)
        print("GENERATING PAPER VISUALIZATIONS")
        print("=" * 80)

        # 1. Success rate charts and tables
        self.generate_success_rate_charts()
        self.generate_success_tables()

        # 2. Success/failure cumulative counts
        self.generate_success_count_charts()

        # 3. Learning curves
        self.generate_learning_curve_analysis()

        # 4. Precision/Recall evolution charts
        self.generate_precision_recall_charts()

        # 5. Precision/Recall final metrics table
        self.calculate_precision_recall()

        # 6. Runtime analysis
        self.generate_runtime_analysis()

        print("\n" + "=" * 80)
        print(f"All visualizations saved to: {self.output_dir}")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate paper visualizations for experiment results')
    parser.add_argument('results_dir', nargs='?',
                       help='Path to results directory')
    parser.add_argument('--latest', action='store_true',
                       help='Use the latest results directory')

    args = parser.parse_args()

    if args.latest or not args.results_dir:
        # Find the latest results directory
        results_base = project_root / 'results' / 'paper'
        if results_base.exists():
            result_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()],
                               key=lambda x: x.stat().st_mtime, reverse=True)
            if result_dirs:
                results_dir = result_dirs[0]
                print(f"Using latest results directory: {results_dir}")
            else:
                print("No results directories found")
                sys.exit(1)
        else:
            print(f"Results directory not found: {results_base}")
            sys.exit(1)
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            sys.exit(1)

    # Generate visualizations
    visualizer = PaperVisualizer(results_dir)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()