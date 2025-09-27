#!/usr/bin/env python
"""
Test experiment with mock environment only (no OLAM).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.metrics import MetricsCollector
from src.environments.mock_environment import MockEnvironment
import json


def run_mock_experiment(iterations: int = 50):
    """Run a simple experiment with mock environment."""
    print("Running mock experiment...")

    # Initialize components
    metrics = MetricsCollector(interval=1, window_size=25)  # Record every action
    env = MockEnvironment(success_rate=0.7, seed=42)

    # Run experiment
    for step in range(iterations):
        # Get state and applicable actions
        state = env.get_state()
        actions = env.get_applicable_actions()

        if actions:
            action, objects = actions[0]  # Simple: take first available action
        else:
            action, objects = 'wait', []

        # Execute and record
        success, runtime = env.execute(action, objects)
        metrics.record_action(step, action, objects, success, runtime)

        # Progress report every 10 steps
        if step % 10 == 0:
            mistake_rate = metrics.compute_overall_mistake_rate()
            print(f"Step {step}: Overall mistake rate = {mistake_rate:.3f}")

    # Final analysis
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)

    # Overall stats
    print(f"\nTotal actions: {metrics.total_actions}")
    print(f"Cumulative mistakes: {metrics.cumulative_mistakes}")
    print(f"Overall mistake rate: {metrics.compute_overall_mistake_rate():.3f}")

    # Window analysis
    print("\nMistake rates by window:")
    window_rates = metrics.compute_mistake_rates_multiple_windows([5, 10, 25])
    for window, rate in window_rates.items():
        if metrics.total_actions >= window:
            print(f"  Last {window} actions: {rate:.3f}")

    # Action type stats
    print("\nAction type statistics:")
    for action, stats in metrics.action_type_stats.items():
        success_rate = 1 - (stats['failures'] / stats['total'])
        print(f"  {action}: {stats['total']} attempts, {success_rate:.2%} success")

    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / 'mock_experiment_results.json'
    metrics.export(str(output_file), format='json')
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run_mock_experiment(50)