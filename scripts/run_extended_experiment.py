#!/usr/bin/env python3
"""
Run an extended experiment with the rover domain.
This script runs a longer learning session with proper monitoring.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.runner import ExperimentRunner
from src.environments.mock_environment import MockEnvironment
from src.algorithms.olam_adapter import OLAMAdapter
from src.experiments.metrics import MetricsCollector


def run_extended_experiment():
    """Run an extended learning experiment with the rover domain."""

    print("=" * 70)
    print("EXTENDED ROVER DOMAIN EXPERIMENT")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Target: 500-2000 actions with learning convergence")
    print("-" * 70)

    # Load configuration
    config_path = Path("configs/experiment_rover_extended.yaml")

    # Initialize experiment runner with extended configuration
    runner = ExperimentRunner(str(config_path))

    print("\nInitializing components...")
    print(f"  Domain: {runner.config['domain_problem']['domain']}")
    print(f"  Problem: {runner.config['domain_problem']['problem']}")
    print(f"  Max iterations: {runner.config['stopping_criteria']['max_iterations']}")
    print(f"  Convergence threshold: {runner.config['stopping_criteria']['convergence_threshold']}")

    # Run the experiment
    print("\nStarting extended learning session...")
    print("Progress updates every 100 actions:")
    print("-" * 70)

    start_time = time.time()

    # Run experiment
    try:
        results = runner.run_experiment()

        # Calculate statistics
        end_time = time.time()
        runtime = end_time - start_time

        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETED")
        print("=" * 70)

        # Display results
        if results:
            # Results is a dictionary, not a list
            exp_result = results

            print(f"\nExperiment: {exp_result.get('name', 'unknown')}")
            print(f"Algorithm: {exp_result.get('algorithm', 'unknown')}")
            print(f"Total runtime: {runtime:.2f} seconds")
            print(f"Iterations completed: {exp_result.get('iterations', 0)}")
            print(f"Final status: {exp_result.get('termination_reason', 'unknown')}")

            # Load and analyze metrics
            metrics_file = exp_result.get('metrics_file')
            if metrics_file and Path(metrics_file).exists():
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)

                if 'metrics' in metrics_data and len(metrics_data['metrics']) > 0:
                    total_actions = len(metrics_data['metrics'])
                    successful = sum(1 for m in metrics_data['metrics'] if m.get('success', False))
                    failed = total_actions - successful

                    print("\n" + "-" * 70)
                    print("LEARNING STATISTICS")
                    print("-" * 70)
                    print(f"Total actions executed: {total_actions}")
                    print(f"Successful actions: {successful} ({successful/total_actions*100:.1f}%)")
                    print(f"Failed actions: {failed} ({failed/total_actions*100:.1f}%)")

                    # Analyze learning progress over time
                    if total_actions >= 100:
                        # Calculate success rates for different windows
                        windows = [100, 200, 500, 1000]
                        print("\nSuccess rate over time:")

                        for window in windows:
                            if total_actions >= window:
                                window_metrics = metrics_data['metrics'][-window:]
                                window_success = sum(1 for m in window_metrics if m.get('success', False))
                                print(f"  Last {window} actions: {window_success/window*100:.1f}%")

                    # Show sample of actions
                    print("\nSample of executed actions:")
                    sample_size = min(10, total_actions)
                    for i in range(0, total_actions, max(1, total_actions // sample_size)):
                        action = metrics_data['metrics'][i]
                        status = "✓" if action.get('success', False) else "✗"
                        print(f"  Action {action['step']}: {action['action']} {status}")
                else:
                    print("\nNo metrics data found in experiment results.")

            print("\n" + "-" * 70)
            print(f"Results saved to: {exp_result.get('output_dir', 'unknown')}")

            # Save summary
            summary = {
                'experiment_type': 'extended_rover',
                'total_runtime_seconds': runtime,
                'total_actions': exp_result.get('iterations', 0),
                'termination_reason': exp_result.get('termination_reason', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'config_file': str(config_path)
            }

            summary_file = Path("results") / f"rover_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"Extended summary saved to: {summary_file}")

        else:
            print("\nWarning: No results returned from experiment runner.")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print(f"Partial results may be available in results/")

    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("EXPERIMENT SESSION ENDED")
    print("=" * 70)


if __name__ == "__main__":
    run_extended_experiment()