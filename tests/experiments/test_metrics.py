"""
Test suite for MetricsCollector - comprehensive testing for experiment metrics.
"""

import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMetricsCollector:
    """Test suite for MetricsCollector class."""

    def test_initialization(self):
        """Test MetricsCollector initialization with default and custom parameters."""
        from src.experiments.metrics import MetricsCollector

        # Test default initialization
        collector = MetricsCollector()
        assert collector.interval == 10
        assert collector.window_size == 50
        assert isinstance(collector.metrics_df, pd.DataFrame)
        assert len(collector.metrics_df) == 0

        # Test custom initialization
        collector = MetricsCollector(interval=5, window_size=100)
        assert collector.interval == 5
        assert collector.window_size == 100

    def test_record_action(self):
        """Test recording individual action executions."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector()

        # Record successful action
        collector.record_action(
            step=1,
            action="pick-up",
            objects=["a"],
            success=True,
            runtime=0.05
        )

        # Check that action was recorded
        assert len(collector.metrics_df) == 1
        assert collector.metrics_df.iloc[0]['step'] == 1
        assert collector.metrics_df.iloc[0]['action'] == "pick-up"
        assert collector.metrics_df.iloc[0]['objects'] == ["a"]
        assert collector.metrics_df.iloc[0]['success']
        assert collector.metrics_df.iloc[0]['runtime'] == 0.05

        # Record failed action
        collector.record_action(
            step=2,
            action="stack",
            objects=["a", "b"],
            success=False,
            runtime=0.03
        )

        assert len(collector.metrics_df) == 2
        assert collector.metrics_df.iloc[1]['success'] == False

    def test_compute_mistake_rate(self):
        """Test mistake rate calculation over sliding window."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector(window_size=5)

        # No actions recorded yet
        assert collector.compute_mistake_rate() == 0.0

        # Record some actions (3 successes, 2 failures)
        collector.record_action(1, "pick-up", ["a"], True, 0.01)
        collector.record_action(2, "stack", ["a", "b"], False, 0.02)
        collector.record_action(3, "put-down", ["a"], True, 0.01)
        collector.record_action(4, "unstack", ["c", "d"], False, 0.03)
        collector.record_action(5, "pick-up", ["b"], True, 0.01)

        # Mistake rate should be 2/5 = 0.4
        assert collector.compute_mistake_rate() == 0.4

        # Add more actions to test sliding window
        collector.record_action(6, "stack", ["b", "c"], True, 0.02)
        collector.record_action(7, "put-down", ["b"], True, 0.01)

        # Last 5 actions: steps 3-7 (4 successes, 1 failure)
        # Mistake rate should be 1/5 = 0.2
        assert collector.compute_mistake_rate() == 0.2

    def test_compute_average_runtime(self):
        """Test average runtime calculation."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector()

        # No actions recorded
        assert collector.compute_average_runtime() == 0.0

        # Record actions with different runtimes
        collector.record_action(1, "pick-up", ["a"], True, 0.10)
        collector.record_action(2, "stack", ["a", "b"], False, 0.20)
        collector.record_action(3, "put-down", ["a"], True, 0.30)

        # Average should be (0.10 + 0.20 + 0.30) / 3 = 0.20
        assert abs(collector.compute_average_runtime() - 0.20) < 0.001

    def test_should_collect(self):
        """Test checking if metrics should be collected at a given step."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector(interval=10)

        # Should collect at step 0 (initial)
        assert collector.should_collect(0)

        # Should not collect at non-interval steps
        assert collector.should_collect(5) == False
        assert collector.should_collect(7) == False

        # Should collect at interval steps
        assert collector.should_collect(10)
        assert collector.should_collect(20)
        assert collector.should_collect(100)

    def test_get_snapshot(self):
        """Test getting current metrics snapshot."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector(window_size=3)

        # Record some actions
        collector.record_action(1, "pick-up", ["a"], True, 0.10)
        collector.record_action(2, "stack", ["a", "b"], False, 0.20)
        collector.record_action(3, "put-down", ["a"], True, 0.15)

        # Get snapshot
        snapshot = collector.get_snapshot(step=3)

        # Check snapshot contents
        assert snapshot['step'] == 3
        assert snapshot['total_actions'] == 3
        assert snapshot['mistake_rate'] == pytest.approx(1 / 3, rel=0.01)
        assert snapshot['average_runtime'] == pytest.approx(0.15, rel=0.01)
        assert 'timestamp' in snapshot

        # Additional metrics
        assert snapshot['successful_actions'] == 2
        assert snapshot['failed_actions'] == 1

    def test_get_action_distribution(self):
        """Test getting distribution of action types."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector()

        # Record various actions
        collector.record_action(1, "pick-up", ["a"], True, 0.1)
        collector.record_action(2, "pick-up", ["b"], True, 0.1)
        collector.record_action(3, "stack", ["a", "b"], False, 0.2)
        collector.record_action(4, "pick-up", ["c"], True, 0.1)

        distribution = collector.get_action_distribution()

        assert distribution['pick-up'] == 3
        assert distribution['stack'] == 1
        assert sum(distribution.values()) == 4

    def test_export_csv(self, temp_dir):
        """Test exporting metrics to CSV format."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector()

        # Record some actions
        collector.record_action(1, "pick-up", ["a"], True, 0.10)
        collector.record_action(2, "stack", ["a", "b"], False, 0.20)

        # Collect snapshots
        collector.collect_snapshot(1)
        collector.collect_snapshot(2)

        # Export to CSV
        csv_path = temp_dir / "metrics.csv"
        collector.export(str(csv_path), format='csv')

        # Verify file exists and can be loaded
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 2  # Two snapshots
        assert 'step' in df.columns
        assert 'mistake_rate' in df.columns

    def test_export_json(self, temp_dir):
        """Test exporting metrics to JSON format."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector()

        # Record some actions
        collector.record_action(1, "pick-up", ["a"], True, 0.10)
        collector.record_action(2, "stack", ["a", "b"], False, 0.20)

        # Collect snapshots
        collector.collect_snapshot(1)
        collector.collect_snapshot(2)

        # Export to JSON
        json_path = temp_dir / "metrics.json"
        collector.export(str(json_path), format='json')

        # Verify file exists and can be loaded
        assert json_path.exists()
        with open(json_path, 'r') as f:
            data = json.load(f)

        assert 'snapshots' in data
        assert len(data['snapshots']) == 2
        assert 'actions' in data
        assert len(data['actions']) == 2

    def test_export_invalid_format(self, temp_dir):
        """Test handling of invalid export format."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector()

        # Should raise ValueError for unsupported format
        with pytest.raises(ValueError, match="Unsupported format"):
            collector.export(str(temp_dir / "metrics.txt"), format='txt')

    def test_reset(self):
        """Test resetting the metrics collector."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector()

        # Record some data
        collector.record_action(1, "pick-up", ["a"], True, 0.10)
        collector.collect_snapshot(1)

        assert len(collector.metrics_df) > 0
        assert len(collector.snapshots) > 0

        # Reset
        collector.reset()

        assert len(collector.metrics_df) == 0
        assert len(collector.snapshots) == 0

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector(window_size=1)

        # Empty collector
        assert collector.compute_mistake_rate() == 0.0
        assert collector.compute_average_runtime() == 0.0

        # Single action
        collector.record_action(1, "pick-up", ["a"], False, 1.0)
        assert collector.compute_mistake_rate() == 1.0
        assert collector.compute_average_runtime() == 1.0

        # Window size edge case
        collector = MetricsCollector(window_size=0)
        collector.record_action(1, "pick-up", ["a"], True, 0.1)
        assert collector.compute_mistake_rate() == 0.0  # No window

    def test_metrics_over_time(self):
        """Test tracking metrics evolution over time."""
        from src.experiments.metrics import MetricsCollector

        collector = MetricsCollector(interval=2, window_size=3)

        # Simulate learning improvement over time
        # Early: high failure rate
        for i in range(1, 6):
            success = i > 2  # First 2 fail, then succeed
            collector.record_action(i, f"action-{i}", ["obj"], success, 0.1)

            if collector.should_collect(i):
                collector.collect_snapshot(i)

        # Check that mistake rate decreases over time
        snapshots = collector.snapshots
        assert len(snapshots) > 1

        # Later snapshots should have lower mistake rates
        if len(snapshots) >= 2:
            assert snapshots[-1]['mistake_rate'] < snapshots[0]['mistake_rate']

    def test_concurrent_metrics(self):
        """Test thread-safety of metrics collection."""
        from src.experiments.metrics import MetricsCollector
        import threading

        collector = MetricsCollector()

        def record_actions(start_step):
            for i in range(5):
                collector.record_action(
                    start_step + i,
                    "action",
                    ["obj"],
                    True,
                    0.01
                )

        # Create threads that record actions concurrently
        threads = [
            threading.Thread(target=record_actions, args=(i * 10,))
            for i in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded all actions without errors
        assert len(collector.metrics_df) == 15  # 3 threads * 5 actions
