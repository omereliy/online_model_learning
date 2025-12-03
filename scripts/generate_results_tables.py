#!/usr/bin/env python3
"""
Generate comprehensive results tables and success rate chart from experiment results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def load_experiment_data(results_dir):
    """Load all experiment data from summary and metrics files."""
    results_dir = Path(results_dir)

    algorithms = ['olam', 'information_gain']
    domains = ['blocksworld', 'depots']
    problems = ['p00', 'p01', 'p02']

    data = {}

    for algorithm in algorithms:
        data[algorithm] = {}
        for domain in domains:
            data[algorithm][domain] = {}
            for problem in problems:
                exp_dir = results_dir / algorithm / domain / problem / 'experiments'

                # Find summary file
                summary_files = list(exp_dir.glob('*_summary.json'))
                if not summary_files:
                    continue

                summary_file = summary_files[0]

                # Find metrics CSV file
                metrics_files = list(exp_dir.glob('*_metrics.csv'))
                if not metrics_files:
                    continue

                metrics_file = metrics_files[0]

                # Load summary
                with open(summary_file) as f:
                    summary = json.load(f)

                # Load metrics CSV
                try:
                    metrics_df = pd.read_csv(metrics_file)
                except Exception as e:
                    print(f"Error loading {metrics_file}: {e}")
                    continue

                data[algorithm][domain][problem] = {
                    'summary': summary,
                    'metrics': metrics_df
                }

    return data

def create_summary_table(data):
    """Create comprehensive summary table."""
    rows = []

    for algorithm in ['olam', 'information_gain']:
        for domain in ['blocksworld', 'depots']:
            for problem in ['p00', 'p01', 'p02']:
                if problem not in data[algorithm][domain]:
                    continue

                summary = data[algorithm][domain][problem]['summary']
                metrics = summary.get('metrics', {})

                row = {
                    'Algorithm': algorithm.replace('_', ' ').title(),
                    'Domain': domain.title(),
                    'Problem': problem,
                    'Iterations': summary.get('total_iterations', 'N/A'),
                    'Success Rate': f"{metrics.get('success_rate', 0):.3f}",
                    'Successful': metrics.get('successful_actions', 'N/A'),
                    'Failed': metrics.get('failed_actions', 'N/A'),
                    'Runtime (s)': f"{summary.get('runtime_seconds', 0):.2f}",
                    'Stopping Reason': summary.get('stopping_reason', 'N/A')
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    return df

def create_cumulative_success_table(data):
    """Create table showing cumulative successful actions at key iteration milestones."""
    rows = []

    milestones = [10, 50, 100, 150, 200, 500, 1000]

    for algorithm in ['olam', 'information_gain']:
        for domain in ['blocksworld', 'depots']:
            for problem in ['p00', 'p01', 'p02']:
                if problem not in data[algorithm][domain]:
                    continue

                metrics_df = data[algorithm][domain][problem]['metrics']
                summary = data[algorithm][domain][problem]['summary']

                row = {
                    'Algorithm': algorithm.replace('_', ' ').title(),
                    'Domain': domain.title(),
                    'Problem': problem,
                }

                # Get cumulative success at each milestone
                if 'successful_actions' in metrics_df.columns:
                    # metrics_df uses step column which is in intervals of 10
                    for milestone in milestones:
                        step_idx = milestone // 10
                        if step_idx < len(metrics_df):
                            cumulative = metrics_df.iloc[step_idx]['successful_actions']
                            row[f'Success@{milestone}'] = int(cumulative)
                        else:
                            # Use final value if milestone beyond convergence
                            total_iters = summary.get('total_iterations', 0)
                            if milestone <= total_iters:
                                row[f'Success@{milestone}'] = int(metrics_df.iloc[-1]['successful_actions'])
                            else:
                                row[f'Success@{milestone}'] = '-'

                rows.append(row)

    df = pd.DataFrame(rows)
    return df

def create_cumulative_comparison_table(data):
    """Create aggregated comparison table with cumulative metrics."""
    rows = []

    for algorithm in ['olam', 'information_gain']:
        for domain in ['blocksworld', 'depots']:
            cumulative_success = []
            total_iterations = []
            converged_at = []

            for problem in ['p00', 'p01', 'p02']:
                if problem not in data[algorithm][domain]:
                    continue

                summary = data[algorithm][domain][problem]['summary']
                metrics_df = data[algorithm][domain][problem]['metrics']

                # Final cumulative success
                if 'successful_actions' in metrics_df.columns:
                    final_success = int(metrics_df.iloc[-1]['successful_actions'])
                    cumulative_success.append(final_success)

                total_iterations.append(summary.get('total_iterations', 0))
                converged_at.append(summary.get('total_iterations', 0))

            if cumulative_success:
                row = {
                    'Algorithm': algorithm.replace('_', ' ').title(),
                    'Domain': domain.title(),
                    'Avg Cumulative Success': f"{np.mean(cumulative_success):.1f}",
                    'Total Success (All Problems)': int(np.sum(cumulative_success)),
                    'Avg Iterations to Converge': f"{np.mean(converged_at):.0f}",
                    'Success per 100 Iterations': f"{(np.sum(cumulative_success) / np.sum(total_iterations) * 100):.1f}"
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    return df

def create_learning_progress_table(data):
    """Create table showing learning progress at different stages."""
    rows = []

    stages = [
        (0, 50, 'Early (0-50)'),
        (50, 100, 'Mid (50-100)'),
        (100, 200, 'Late (100-200)'),
        (200, 1000, 'Final (200-1000)')
    ]

    for algorithm in ['olam', 'information_gain']:
        for domain in ['blocksworld', 'depots']:
            for stage_start, stage_end, stage_name in stages:
                stage_successes = []

                for problem in ['p00', 'p01', 'p02']:
                    if problem not in data[algorithm][domain]:
                        continue

                    metrics_df = data[algorithm][domain][problem]['metrics']
                    summary = data[algorithm][domain][problem]['summary']
                    total_iters = summary.get('total_iterations', 0)

                    if 'successful_actions' in metrics_df.columns and stage_start < total_iters:
                        # Find indices for this stage
                        start_idx = min(stage_start // 10, len(metrics_df) - 1)
                        end_idx = min(stage_end // 10, len(metrics_df) - 1)

                        if end_idx < len(metrics_df):
                            success_at_start = metrics_df.iloc[start_idx]['successful_actions'] if start_idx > 0 else 0
                            success_at_end = metrics_df.iloc[end_idx]['successful_actions']
                            stage_gain = success_at_end - success_at_start
                            stage_successes.append(int(stage_gain))

                if stage_successes:
                    row = {
                        'Algorithm': algorithm.replace('_', ' ').title(),
                        'Domain': domain.title(),
                        'Stage': stage_name,
                        'Avg Success Gained': f"{np.mean(stage_successes):.1f}",
                        'Total Success Gained': int(np.sum(stage_successes)),
                        'Success per Iteration': f"{(np.sum(stage_successes) / ((stage_end - stage_start) * len(stage_successes))):.3f}"
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df

def create_learning_metrics_table(data, results_dir):
    """Create learning metrics table from learned models."""
    results_dir = Path(results_dir)
    rows = []

    for algorithm in ['olam', 'information_gain']:
        for domain in ['blocksworld', 'depots']:
            for problem in ['p00', 'p01', 'p02']:
                model_file = results_dir / algorithm / domain / problem / 'experiments' / 'learned_model.json'

                if not model_file.exists():
                    continue

                with open(model_file) as f:
                    model = json.load(f)

                actions = model.get('actions', {})

                # Calculate metrics based on algorithm
                if algorithm == 'olam':
                    # OLAM stores actions as lifted schemas
                    total_actions = len(actions)
                    total_precs = sum(
                        len(act.get('preconditions', {}).get('certain', [])) +
                        len(act.get('preconditions', {}).get('uncertain', []))
                        for act in actions.values()
                    )
                    total_effects = sum(
                        len(act.get('effects', {}).get('positive', [])) +
                        len(act.get('effects', {}).get('negative', []))
                        for act in actions.values()
                    )
                else:
                    # Information Gain
                    total_actions = len(actions)
                    total_precs = sum(
                        len(act.get('preconditions', {}).get('possible', []))
                        for act in actions.values()
                    )
                    total_effects = sum(
                        len(act.get('effects', {}).get('add', [])) +
                        len(act.get('effects', {}).get('delete', []))
                        for act in actions.values()
                    )

                row = {
                    'Algorithm': algorithm.replace('_', ' ').title(),
                    'Domain': domain.title(),
                    'Problem': problem,
                    'Actions Learned': total_actions,
                    'Preconditions': total_precs,
                    'Effects': total_effects,
                    'Total Literals': total_precs + total_effects
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    return df

def plot_success_rate_over_iterations(data, output_dir):
    """Plot cumulative successful actions over iterations for all experiments."""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cumulative Successful Actions Over Iterations', fontsize=16, fontweight='bold')

    colors = {
        'olam': '#FF6B6B',
        'information_gain': '#4ECDC4'
    }

    plot_configs = [
        ('blocksworld', 'Blocksworld', axes[0, 0]),
        ('depots', 'Depots', axes[0, 1]),
    ]

    # Aggregated plots
    for domain, title, ax in plot_configs:
        for algorithm in ['olam', 'information_gain']:
            all_cumulative_success = []

            for problem in ['p00', 'p01', 'p02']:
                if problem not in data[algorithm][domain]:
                    continue

                metrics_df = data[algorithm][domain][problem]['metrics']

                # Get cumulative successful actions (Y-axis = total successful actions up to iteration i)
                if 'successful_actions' in metrics_df.columns:
                    cumulative_success = metrics_df['successful_actions'].values
                    all_cumulative_success.append(cumulative_success)

            if all_cumulative_success:
                # Pad to same length
                max_len = max(len(cs) for cs in all_cumulative_success)
                padded = [np.pad(cs, (0, max_len - len(cs)), mode='edge')
                         for cs in all_cumulative_success]

                # Calculate mean and std
                mean_cs = np.mean(padded, axis=0)
                std_cs = np.std(padded, axis=0)
                iterations = np.arange(0, len(mean_cs) * 10, 10)[:len(mean_cs)]

                label = 'OLAM' if algorithm == 'olam' else 'Information Gain'
                ax.plot(iterations, mean_cs, label=label,
                       color=colors[algorithm], linewidth=2)
                ax.fill_between(iterations,
                               mean_cs - std_cs,
                               mean_cs + std_cs,
                               color=colors[algorithm], alpha=0.2)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Cumulative Successful Actions', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    # Overall aggregated
    ax_all = axes[1, 0]
    for algorithm in ['olam', 'information_gain']:
        all_cumulative_success = []

        for domain in ['blocksworld', 'depots']:
            for problem in ['p00', 'p01', 'p02']:
                if problem not in data[algorithm][domain]:
                    continue

                metrics_df = data[algorithm][domain][problem]['metrics']

                if 'successful_actions' in metrics_df.columns:
                    cumulative_success = metrics_df['successful_actions'].values
                    all_cumulative_success.append(cumulative_success)

        if all_cumulative_success:
            max_len = max(len(cs) for cs in all_cumulative_success)
            padded = [np.pad(cs, (0, max_len - len(cs)), mode='edge')
                     for cs in all_cumulative_success]

            mean_cs = np.mean(padded, axis=0)
            std_cs = np.std(padded, axis=0)
            iterations = np.arange(0, len(mean_cs) * 10, 10)[:len(mean_cs)]

            label = 'OLAM' if algorithm == 'olam' else 'Information Gain'
            ax_all.plot(iterations, mean_cs, label=label,
                       color=colors[algorithm], linewidth=2)
            ax_all.fill_between(iterations,
                               mean_cs - std_cs,
                               mean_cs + std_cs,
                               color=colors[algorithm], alpha=0.2)

    ax_all.set_xlabel('Iteration', fontsize=12)
    ax_all.set_ylabel('Cumulative Successful Actions', fontsize=12)
    ax_all.set_title('Overall (All Domains)', fontsize=14, fontweight='bold')
    ax_all.legend(fontsize=11)
    ax_all.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Save
    output_file = output_dir / 'cumulative_success_iterations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Cumulative success chart saved to: {output_file}")

    output_file_pdf = output_dir / 'cumulative_success_iterations.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Cumulative success chart saved to: {output_file_pdf}")

    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_results_tables.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_dir = Path(results_dir) / 'visualizations'
    output_dir.mkdir(exist_ok=True)

    print(f"Loading experiment data from {results_dir}...")
    data = load_experiment_data(results_dir)

    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    summary_table = create_summary_table(data)
    print(summary_table.to_string(index=False))

    # Save to CSV
    csv_file = output_dir / 'summary_table.csv'
    summary_table.to_csv(csv_file, index=False)
    print(f"\nSummary table saved to: {csv_file}")

    print("\n" + "="*80)
    print("LEARNING METRICS TABLE")
    print("="*80)
    learning_table = create_learning_metrics_table(data, results_dir)
    print(learning_table.to_string(index=False))

    # Save to CSV
    csv_file = output_dir / 'learning_metrics_table.csv'
    learning_table.to_csv(csv_file, index=False)
    print(f"\nLearning metrics table saved to: {csv_file}")

    print("\n" + "="*80)
    print("CUMULATIVE SUCCESS AT KEY MILESTONES")
    print("="*80)
    cumulative_table = create_cumulative_success_table(data)
    print(cumulative_table.to_string(index=False))

    csv_file = output_dir / 'cumulative_success_milestones.csv'
    cumulative_table.to_csv(csv_file, index=False)
    print(f"\nCumulative success milestones saved to: {csv_file}")

    print("\n" + "="*80)
    print("CUMULATIVE COMPARISON TABLE")
    print("="*80)
    comparison_table = create_cumulative_comparison_table(data)
    print(comparison_table.to_string(index=False))

    csv_file = output_dir / 'cumulative_comparison.csv'
    comparison_table.to_csv(csv_file, index=False)
    print(f"\nCumulative comparison saved to: {csv_file}")

    print("\n" + "="*80)
    print("LEARNING PROGRESS BY STAGE")
    print("="*80)
    progress_table = create_learning_progress_table(data)
    print(progress_table.to_string(index=False))

    csv_file = output_dir / 'learning_progress_stages.csv'
    progress_table.to_csv(csv_file, index=False)
    print(f"\nLearning progress by stage saved to: {csv_file}")

    print("\n" + "="*80)
    print("GENERATING CUMULATIVE SUCCESS CHART")
    print("="*80)
    plot_success_rate_over_iterations(data, output_dir)

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
