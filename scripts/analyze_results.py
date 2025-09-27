#!/usr/bin/env python
"""
Analyze experiment results with multiple window sizes and comprehensive metrics.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_experiment_results(experiment_dir: Path) -> Dict[str, Any]:
    """
    Load all results from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with all experiment data
    """
    results = {}

    # Load summary
    summary_file = experiment_dir / 'results_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            results['summary'] = json.load(f)

    # Load metrics CSV
    csv_files = list(experiment_dir.glob('*_metrics.csv'))
    if csv_files:
        results['metrics_df'] = pd.read_csv(csv_files[0])

    # Load raw metrics JSON
    json_files = list(experiment_dir.glob('*_metrics.json'))
    if json_files:
        with open(json_files[0], 'r') as f:
            results['metrics_json'] = json.load(f)

    # Load config
    config_file = experiment_dir / 'config.yaml'
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            results['config'] = yaml.safe_load(f)

    return results


def analyze_mistake_rates_windows(metrics_df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """
    Calculate mistake rates for multiple window sizes.

    Args:
        metrics_df: DataFrame with action records
        windows: List of window sizes to analyze

    Returns:
        DataFrame with mistake rates for each window
    """
    if windows is None:
        windows = [5, 10, 25, 50, 100]

    analysis = []

    for window in windows:
        if len(metrics_df) >= window:
            # Calculate mistake rate for each possible window position
            for i in range(window, len(metrics_df) + 1):
                window_data = metrics_df.iloc[i-window:i]
                mistake_rate = (~window_data['success']).mean()
                analysis.append({
                    'step': i,
                    'window_size': window,
                    'mistake_rate': mistake_rate,
                    'window_start': i - window,
                    'window_end': i
                })

    return pd.DataFrame(analysis)


def analyze_action_types(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance by action type.

    Args:
        metrics_df: DataFrame with action records

    Returns:
        DataFrame with per-action statistics
    """
    if 'action' not in metrics_df.columns:
        return pd.DataFrame()

    action_stats = []
    for action in metrics_df['action'].unique():
        action_data = metrics_df[metrics_df['action'] == action]
        stats = {
            'action': action,
            'total_attempts': len(action_data),
            'successes': action_data['success'].sum(),
            'failures': (~action_data['success']).sum(),
            'success_rate': action_data['success'].mean(),
            'failure_rate': (~action_data['success']).mean(),
            'avg_runtime': action_data['runtime'].mean(),
            'std_runtime': action_data['runtime'].std()
        }
        action_stats.append(stats)

    return pd.DataFrame(action_stats).sort_values('total_attempts', ascending=False)


def analyze_learning_progress(metrics_df: pd.DataFrame, segment_size: int = 50) -> pd.DataFrame:
    """
    Analyze learning progress over segments.

    Args:
        metrics_df: DataFrame with action records
        segment_size: Size of each segment for analysis

    Returns:
        DataFrame with segment-wise statistics
    """
    segments = []
    num_segments = len(metrics_df) // segment_size + (1 if len(metrics_df) % segment_size else 0)

    for i in range(num_segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, len(metrics_df))
        segment_data = metrics_df.iloc[start:end]

        if len(segment_data) > 0:
            segments.append({
                'segment': i + 1,
                'start_step': start,
                'end_step': end - 1,
                'actions': len(segment_data),
                'mistake_rate': (~segment_data['success']).mean(),
                'avg_runtime': segment_data['runtime'].mean(),
                'cumulative_mistakes': segment_data['cumulative_mistakes'].iloc[-1] if 'cumulative_mistakes' in segment_data else None,
                'overall_mistake_rate': segment_data['overall_mistake_rate'].iloc[-1] if 'overall_mistake_rate' in segment_data else None
            })

    return pd.DataFrame(segments)


def print_analysis_report(experiment_dir: Path):
    """
    Print comprehensive analysis report for an experiment.

    Args:
        experiment_dir: Path to experiment directory
    """
    results = load_experiment_results(experiment_dir)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT ANALYSIS REPORT")
    print(f"Directory: {experiment_dir}")
    print("=" * 80)

    # Basic information
    if 'config' in results:
        config = results['config']
        print(f"\nExperiment: {config['experiment']['name']}")
        print(f"Algorithm: {config['experiment']['algorithm']}")
        print(f"Domain: {Path(config['domain_problem']['domain']).name}")
        print(f"Problem: {Path(config['domain_problem']['problem']).name}")

    if 'summary' in results:
        summary = results['summary']
        print(f"\nTotal iterations: {summary.get('total_iterations', 'N/A')}")
        print(f"Runtime: {summary.get('runtime_seconds', 0):.2f} seconds")
        print(f"Stopping reason: {summary.get('stopping_reason', 'N/A')}")

    # Load detailed metrics
    if 'metrics_json' in results:
        metrics_json = results['metrics_json']
        if 'actions' in metrics_json:
            metrics_df = pd.DataFrame(metrics_json['actions'])

            # Overall statistics
            print("\n" + "-" * 40)
            print("OVERALL STATISTICS")
            print("-" * 40)
            total_actions = len(metrics_df)
            total_successes = metrics_df['success'].sum() if 'success' in metrics_df else 0
            total_failures = total_actions - total_successes

            print(f"Total actions: {total_actions}")
            print(f"Successful actions: {total_successes}")
            print(f"Failed actions: {total_failures}")
            print(f"Overall success rate: {total_successes/total_actions:.3f}")
            print(f"Overall failure rate: {total_failures/total_actions:.3f}")

            if 'cumulative_mistakes' in metrics_df.columns:
                final_cumulative = metrics_df['cumulative_mistakes'].iloc[-1]
                print(f"Cumulative mistakes (from start): {final_cumulative}")

            if 'overall_mistake_rate' in metrics_df.columns:
                final_overall_rate = metrics_df['overall_mistake_rate'].iloc[-1]
                print(f"Final overall mistake rate: {final_overall_rate:.3f}")

            # Action type analysis
            print("\n" + "-" * 40)
            print("ACTION TYPE ANALYSIS")
            print("-" * 40)
            action_stats = analyze_action_types(metrics_df)
            if not action_stats.empty:
                print("\nTop 5 most attempted actions:")
                for _, row in action_stats.head(5).iterrows():
                    print(f"  {row['action']}: {row['total_attempts']} attempts, "
                          f"{row['success_rate']:.2%} success rate")

            # Window analysis
            print("\n" + "-" * 40)
            print("MISTAKE RATES BY WINDOW SIZE")
            print("-" * 40)
            windows = [5, 10, 25, 50, 100]
            for window in windows:
                if len(metrics_df) >= window:
                    # Calculate for last window
                    last_window = metrics_df.tail(window)
                    mistake_rate = (~last_window['success']).mean()
                    print(f"Last {window} actions: {mistake_rate:.3f} mistake rate")

            # Learning progress analysis
            print("\n" + "-" * 40)
            print("LEARNING PROGRESS (50-action segments)")
            print("-" * 40)
            progress = analyze_learning_progress(metrics_df, segment_size=50)
            if not progress.empty:
                for _, row in progress.iterrows():
                    print(f"Segment {row['segment']}: Steps {row['start_step']}-{row['end_step']}")
                    print(f"  Mistake rate: {row['mistake_rate']:.3f}")
                    if row['overall_mistake_rate'] is not None:
                        print(f"  Overall rate at end: {row['overall_mistake_rate']:.3f}")

                # Check for improvement
                if len(progress) >= 2:
                    first_rate = progress.iloc[0]['mistake_rate']
                    last_rate = progress.iloc[-1]['mistake_rate']
                    improvement = first_rate - last_rate
                    print(f"\nImprovement from first to last segment: {improvement:.3f}")
                    if improvement > 0:
                        print(f"  ✓ Learning shows improvement ({improvement:.1%} reduction in mistakes)")
                    elif improvement < 0:
                        print(f"  ✗ Performance degraded ({-improvement:.1%} increase in mistakes)")
                    else:
                        print(f"  - No change in performance")

    print("\n" + "=" * 80)


def compare_experiments(experiment_dirs: List[Path]):
    """
    Compare multiple experiments.

    Args:
        experiment_dirs: List of experiment directories to compare
    """
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    comparison = []

    for exp_dir in experiment_dirs:
        results = load_experiment_results(exp_dir)

        row = {'experiment': exp_dir.name}

        if 'summary' in results:
            summary = results['summary']
            row['iterations'] = summary.get('total_iterations', 0)
            row['runtime'] = summary.get('runtime_seconds', 0)

        if 'metrics_json' in results and 'actions' in results['metrics_json']:
            metrics_df = pd.DataFrame(results['metrics_json']['actions'])
            row['total_actions'] = len(metrics_df)
            row['overall_success_rate'] = metrics_df['success'].mean() if 'success' in metrics_df else 0

            if 'overall_mistake_rate' in metrics_df.columns:
                row['final_mistake_rate'] = metrics_df['overall_mistake_rate'].iloc[-1]

            if 'cumulative_mistakes' in metrics_df.columns:
                row['total_mistakes'] = metrics_df['cumulative_mistakes'].iloc[-1]

        comparison.append(row)

    comparison_df = pd.DataFrame(comparison)

    if not comparison_df.empty:
        print("\n" + comparison_df.to_string(index=False))

        # Find best performer
        if 'final_mistake_rate' in comparison_df.columns:
            best_idx = comparison_df['final_mistake_rate'].idxmin()
            best = comparison_df.iloc[best_idx]
            print(f"\nBest performer: {best['experiment']}")
            print(f"  Final mistake rate: {best['final_mistake_rate']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('experiment', nargs='?',
                       help='Experiment directory to analyze')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple experiments')
    parser.add_argument('--latest', action='store_true',
                       help='Analyze the latest experiment')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all experiments')

    args = parser.parse_args()

    results_dir = project_root / 'results'

    if args.latest:
        # Find latest experiment
        exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()],
                         key=lambda x: x.stat().st_mtime, reverse=True)
        if exp_dirs:
            print_analysis_report(exp_dirs[0])
        else:
            print("No experiments found.")

    elif args.all:
        # Analyze all experiments
        exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
        for exp_dir in exp_dirs:
            print_analysis_report(exp_dir)

        if len(exp_dirs) > 1:
            compare_experiments(exp_dirs)

    elif args.compare:
        # Compare specific experiments
        exp_dirs = [results_dir / exp for exp in args.compare]
        compare_experiments(exp_dirs)

    elif args.experiment:
        # Analyze specific experiment
        exp_dir = results_dir / args.experiment
        if exp_dir.exists():
            print_analysis_report(exp_dir)
        else:
            print(f"Experiment directory not found: {exp_dir}")

    else:
        # Default: show available experiments
        exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
        if exp_dirs:
            print("Available experiments:")
            for exp_dir in exp_dirs:
                print(f"  - {exp_dir.name}")
            print("\nUsage: python analyze_results.py <experiment_name>")
            print("   or: python analyze_results.py --latest")
            print("   or: python analyze_results.py --all")
        else:
            print("No experiments found in results/")


if __name__ == "__main__":
    main()