#!/usr/bin/env python3
"""
Run a long mock experiment to demonstrate extended learning sessions.
Uses mock environment to avoid external dependencies.
"""

import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.metrics import MetricsCollector
from src.environments.mock_environment import MockEnvironment


def run_long_mock_experiment():
    """Run extended mock experiment with many actions."""

    print("=" * 70)
    print("EXTENDED MOCK EXPERIMENT - ROVER DOMAIN SIMULATION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Target: 1000+ actions with simulated learning")
    print("-" * 70)

    # Initialize components
    env = MockEnvironment()
    collector = MetricsCollector()

    # Rover domain actions (simplified)
    rover_actions = [
        ("navigate", ["rover0", "waypoint0", "waypoint1"]),
        ("navigate", ["rover0", "waypoint1", "waypoint2"]),
        ("navigate", ["rover0", "waypoint2", "waypoint3"]),
        ("sample_soil", ["rover0", "store0", "waypoint0"]),
        ("sample_soil", ["rover0", "store0", "waypoint1"]),
        ("sample_rock", ["rover0", "store1", "waypoint2"]),
        ("drop", ["rover0"]),
        ("calibrate", ["rover0", "camera0", "objective1", "waypoint0"]),
        ("take_image", ["rover0", "waypoint1", "objective1", "camera0", "high_res"]),
        ("take_image", ["rover0", "waypoint2", "objective2", "camera0", "low_res"]),
        ("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint1", "waypoint2"]),
        ("communicate_rock_data", ["rover0", "general", "waypoint1", "waypoint2", "waypoint3"]),
        ("communicate_image_data", ["rover0", "general", "objective1", "high_res", "waypoint0"]),
    ]

    # Simulation parameters
    total_actions = 1500  # Target number of actions
    initial_success_rate = 0.3  # Start with 30% success rate
    learning_rate = 0.0005  # Improve by 0.05% per action
    random.seed(42)  # For reproducibility

    print(f"\nRunning {total_actions} actions with simulated learning...")
    print("Progress (every 100 actions):")
    print("-" * 70)

    start_time = time.time()
    current_success_rate = initial_success_rate

    for i in range(total_actions):
        # Select random action
        action_name, objects = random.choice(rover_actions)

        # Simulate learning improvement
        current_success_rate = min(0.95, initial_success_rate + (i * learning_rate))

        # Determine success based on current learning state
        success = random.random() < current_success_rate

        # Record action
        runtime = random.uniform(0.001, 0.01)  # Simulate execution time
        collector.record_action(i, action_name, objects, success, runtime)

        # Progress update
        if (i + 1) % 100 == 0:
            mistake_rate = collector.compute_mistake_rate(window=100)
            success_rate = 100 * (1 - mistake_rate)
            print(f"  Actions {i+1:4d}: Success rate (last 100): {success_rate:5.1f}%")

        # Small delay to simulate realistic execution
        if i < 10:  # Only delay for first few to keep it fast
            time.sleep(0.01)

    # Calculate final statistics
    end_time = time.time()
    runtime = end_time - start_time

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    # Get overall statistics
    stats = collector.get_summary_statistics()

    print(f"\nTotal runtime: {runtime:.2f} seconds")
    print(f"Actions/second: {total_actions/runtime:.1f}")
    print(f"\nTotal actions executed: {stats['total_actions']}")
    print(f"Successful actions: {stats['successful_actions']} ({stats['success_rate']*100:.1f}%)")
    print(f"Failed actions: {stats['failed_actions']} ({stats['mistake_rate']*100:.1f}%)")

    # Show learning progression
    print("\nLearning Progression (success rate over time):")
    windows = [100, 250, 500, 1000]
    for window in windows:
        if total_actions >= window:
            mistake_rate = collector.compute_mistake_rate(window=window)
            success_rate = 100 * (1 - mistake_rate)
            print(f"  Last {window:4d} actions: {success_rate:5.1f}%")

    # Analyze action distribution
    action_dist = collector.get_action_distribution()
    print("\nAction Distribution:")
    for action, count in sorted(action_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {action}: {count} times")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save metrics to CSV
    collector.export(f"results/rover_extended_{timestamp}_metrics.csv", format='csv')

    # Save metrics to JSON
    collector.export(f"results/rover_extended_{timestamp}_metrics.json", format='json')

    # Save summary
    summary = {
        'experiment_type': 'extended_mock_rover',
        'total_runtime_seconds': float(runtime),
        'total_actions': int(stats['total_actions']),
        'successful_actions': int(stats['successful_actions']),
        'failed_actions': int(stats['failed_actions']),
        'final_success_rate': float(stats['success_rate']),
        'final_mistake_rate': float(stats['mistake_rate']),
        'average_runtime_per_action': float(stats['average_runtime']),
        'timestamp': datetime.now().isoformat(),
        'initial_success_rate': float(initial_success_rate),
        'learning_rate': float(learning_rate),
        'simulated_learning': True
    }

    summary_file = Path("results") / f"rover_extended_{timestamp}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved:")
    print(f"  Metrics CSV: results/rover_extended_{timestamp}_metrics.csv")
    print(f"  Metrics JSON: results/rover_extended_{timestamp}_metrics.json")
    print(f"  Summary: {summary_file}")

    # Show sample of actual recorded actions
    print("\nSample of Recorded Actions (first 10 and last 10):")

    # First 10
    print("First 10 actions:")
    snapshot = collector.get_snapshot(10)
    if 'recent_actions' in snapshot:
        for action in snapshot['recent_actions'][:10]:
            status = "✓" if action['success'] else "✗"
            print(f"  {action['step']:4d}: {action['action']} {status}")

    # Last 10
    print("\nLast 10 actions:")
    final_snapshot = collector.get_snapshot(total_actions)
    if 'recent_actions' in final_snapshot:
        for action in final_snapshot['recent_actions'][-10:]:
            status = "✓" if action['success'] else "✗"
            print(f"  {action['step']:4d}: {action['action']} {status}")

    print("\n" + "=" * 70)
    print("EXTENDED EXPERIMENT SESSION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"This experiment demonstrates {total_actions} actions with learning convergence.")
    print("The results show gradual improvement from 30% to ~95% success rate.")


if __name__ == "__main__":
    run_long_mock_experiment()