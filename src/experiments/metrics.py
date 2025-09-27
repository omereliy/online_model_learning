"""
Metrics collection and tracking for experiment framework.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and tracks metrics during experiments.

    Tracks action executions, success rates, runtime performance,
    and provides export functionality for analysis.
    """

    def __init__(self, interval: int = 10, window_size: int = 50):
        """
        Initialize the metrics collector.

        Args:
            interval: How often to collect snapshot metrics (in steps)
            window_size: Size of sliding window for mistake rate calculation
        """
        self.interval = interval
        self.window_size = window_size

        # Main metrics storage
        self.metrics_df = pd.DataFrame()
        self.snapshots = []

        # Cumulative metrics
        self.cumulative_mistakes = 0
        self.total_actions = 0

        # Per-action type tracking
        self.action_type_stats = {}

        # Thread safety - use RLock to prevent deadlocks when methods call each other
        self._lock = threading.RLock()

        logger.info(f"Initialized MetricsCollector with interval={interval}, window_size={window_size}")

    def record_action(self, step: int, action: str, objects: List[str],
                     success: bool, runtime: float) -> None:
        """
        Record a single action execution.

        Args:
            step: Current step number
            action: Action name
            objects: Objects involved in the action
            success: Whether the action succeeded
            runtime: Time taken to execute the action
        """
        with self._lock:
            # Update cumulative metrics
            self.total_actions += 1
            if not success:
                self.cumulative_mistakes += 1

            # Update per-action type stats
            if action not in self.action_type_stats:
                self.action_type_stats[action] = {'total': 0, 'failures': 0}
            self.action_type_stats[action]['total'] += 1
            if not success:
                self.action_type_stats[action]['failures'] += 1

            # Record to DataFrame
            new_row = pd.DataFrame([{
                'step': step,
                'action': action,
                'objects': objects,
                'success': success,
                'runtime': runtime,
                'timestamp': datetime.now(),
                'cumulative_mistakes': self.cumulative_mistakes,
                'overall_mistake_rate': self.cumulative_mistakes / self.total_actions
            }])

            self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)

        logger.debug(f"Recorded action at step {step}: {action}({objects}) - Success: {success}")

    def compute_mistake_rate(self, window: Optional[int] = None) -> float:
        """
        Calculate the mistake rate over the sliding window.

        Args:
            window: Window size to use (default: self.window_size)

        Returns:
            Fraction of failed actions in the last window_size actions
        """
        if window is None:
            window = self.window_size

        with self._lock:
            if len(self.metrics_df) == 0 or window == 0:
                return 0.0

            # Get last window_size actions
            window_df = self.metrics_df.tail(window)

            if len(window_df) == 0:
                return 0.0

            failures = (~window_df['success']).sum()
            return failures / len(window_df)

    def compute_overall_mistake_rate(self) -> float:
        """
        Calculate the overall mistake rate from the beginning.

        Returns:
            Total mistakes / total actions
        """
        with self._lock:
            if self.total_actions == 0:
                return 0.0
            return self.cumulative_mistakes / self.total_actions

    def compute_mistake_rates_multiple_windows(self, windows: List[int] = None) -> Dict[int, float]:
        """
        Calculate mistake rates for multiple window sizes.

        Args:
            windows: List of window sizes (default: [5, 10, 25, 50, 100])

        Returns:
            Dictionary mapping window size to mistake rate
        """
        if windows is None:
            windows = [5, 10, 25, 50, 100]

        rates = {}
        for window in windows:
            rates[window] = self.compute_mistake_rate(window)
        return rates

    def compute_average_runtime(self) -> float:
        """
        Calculate average runtime over all actions.

        Returns:
            Average runtime in seconds
        """
        with self._lock:
            if len(self.metrics_df) == 0:
                return 0.0

            return self.metrics_df['runtime'].mean()

    def should_collect(self, step: int) -> bool:
        """
        Check if metrics should be collected at this step.

        Args:
            step: Current step number

        Returns:
            True if metrics should be collected at this step
        """
        return step % self.interval == 0

    def get_snapshot(self, step: int) -> Dict[str, Any]:
        """
        Get current metrics snapshot.

        Args:
            step: Current step number

        Returns:
            Dictionary containing current metrics
        """
        with self._lock:
            total_actions = len(self.metrics_df)

            if total_actions > 0:
                successful_actions = self.metrics_df['success'].sum()
                failed_actions = total_actions - successful_actions
            else:
                successful_actions = 0
                failed_actions = 0

            # Calculate mistake rates for multiple windows
            window_rates = self.compute_mistake_rates_multiple_windows()

            snapshot = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'total_actions': total_actions,
                'successful_actions': successful_actions,
                'failed_actions': failed_actions,
                'cumulative_mistakes': self.cumulative_mistakes,
                'overall_mistake_rate': self.compute_overall_mistake_rate(),
                'mistake_rate': self.compute_mistake_rate(),  # Default window
                'mistake_rates_windows': window_rates,
                'average_runtime': self.compute_average_runtime(),
                'action_type_stats': dict(self.action_type_stats)
            }

            return snapshot

    def collect_snapshot(self, step: int) -> None:
        """
        Collect and store a snapshot at the current step.

        Args:
            step: Current step number
        """
        snapshot = self.get_snapshot(step)
        with self._lock:
            self.snapshots.append(snapshot)

        logger.debug(f"Collected snapshot at step {step}: mistake_rate={snapshot['mistake_rate']:.3f}")

    def get_action_distribution(self) -> Dict[str, int]:
        """
        Get distribution of action types.

        Returns:
            Dictionary mapping action names to counts
        """
        with self._lock:
            if len(self.metrics_df) == 0:
                return {}

            return self.metrics_df['action'].value_counts().to_dict()

    def export(self, filepath: str, format: str = 'csv') -> None:
        """
        Export metrics to file.

        Args:
            filepath: Path to save the metrics
            format: Export format ('csv' or 'json')

        Raises:
            ValueError: If format is not supported
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if format == 'csv':
                # Export snapshots as CSV
                if self.snapshots:
                    snapshots_df = pd.DataFrame(self.snapshots)
                    snapshots_df.to_csv(filepath, index=False)
                    logger.info(f"Exported metrics to CSV: {filepath}")
                else:
                    # If no snapshots, export raw metrics
                    self.metrics_df.to_csv(filepath, index=False)
                    logger.info(f"Exported raw metrics to CSV: {filepath}")

            elif format == 'json':
                # Export both snapshots and actions
                export_data = {
                    'snapshots': self.snapshots,
                    'actions': self.metrics_df.to_dict(orient='records'),
                    'summary': {
                        'total_actions': len(self.metrics_df),
                        'final_mistake_rate': self.compute_mistake_rate(),
                        'average_runtime': self.compute_average_runtime(),
                        'action_distribution': self.get_action_distribution()
                    }
                }

                # Convert objects lists to strings for JSON serialization
                for action in export_data['actions']:
                    if isinstance(action.get('objects'), list):
                        action['objects'] = str(action['objects'])
                    if 'timestamp' in action:
                        action['timestamp'] = str(action['timestamp'])

                # Custom JSON encoder to handle numpy types
                import numpy as np

                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return super().default(obj)

                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, cls=NumpyEncoder)
                    logger.info(f"Exported metrics to JSON: {filepath}")

            else:
                raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")

    def reset(self) -> None:
        """Reset the metrics collector to initial state."""
        with self._lock:
            self.metrics_df = pd.DataFrame()
            self.snapshots = []

        logger.info("Reset metrics collector")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of collected metrics.

        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            if len(self.metrics_df) == 0:
                return {
                    'total_actions': 0,
                    'successful_actions': 0,
                    'failed_actions': 0,
                    'success_rate': 0.0,
                    'mistake_rate': 0.0,
                    'average_runtime': 0.0,
                    'runtime_std': 0.0,
                    'action_distribution': {}
                }

            total = len(self.metrics_df)
            successful = self.metrics_df['success'].sum()

            return {
                'total_actions': total,
                'successful_actions': successful,
                'failed_actions': total - successful,
                'success_rate': successful / total,
                'mistake_rate': self.compute_mistake_rate(),
                'average_runtime': self.metrics_df['runtime'].mean(),
                'runtime_std': self.metrics_df['runtime'].std(),
                'action_distribution': self.get_action_distribution()
            }

    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot metrics evolution over time.

        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed. Cannot generate plots.")
            return

        if not self.snapshots:
            logger.warning("No snapshots to plot")
            return

        with self._lock:
            snapshots_df = pd.DataFrame(self.snapshots)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot mistake rate
        axes[0].plot(snapshots_df['step'], snapshots_df['mistake_rate'], 'b-', label='Mistake Rate')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Mistake Rate')
        axes[0].set_title('Mistake Rate Over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot average runtime
        axes[1].plot(snapshots_df['step'], snapshots_df['average_runtime'], 'r-', label='Avg Runtime')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Runtime (seconds)')
        axes[1].set_title('Average Runtime Over Time')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved metrics plot to {save_path}")
        else:
            plt.show()