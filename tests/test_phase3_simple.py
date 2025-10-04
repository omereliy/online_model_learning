"""
Simple test for Phase 3 components without OLAM dependency.
"""

from src.environments.mock_environment import MockEnvironment
from src.experiments.metrics import MetricsCollector
import pytest
import tempfile
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPhase3Simple:
    """Simple tests for Phase 3 components."""

    def test_metrics_collector_basic(self):
        """Test basic MetricsCollector functionality."""
        collector = MetricsCollector(interval=5, window_size=10)

        # Record some actions
        for i in range(10):
            success = i % 3 != 0  # Every 3rd action fails
            collector.record_action(
                step=i,
                action="test-action",
                objects=["obj"],
                success=success,
                runtime=0.05
            )

        # Check metrics
        mistake_rate = collector.compute_mistake_rate()
        assert 0.0 <= mistake_rate <= 1.0

        avg_runtime = collector.compute_average_runtime()
        assert avg_runtime == pytest.approx(0.05, rel=0.01)

        # Check snapshots
        collector.collect_snapshot(10)
        assert len(collector.snapshots) == 1

    def test_mock_environment_basic(self):
        """Test basic MockEnvironment functionality."""
        env = MockEnvironment(success_rate=0.8, seed=42)

        # Get initial state
        state = env.get_state()
        assert len(state) > 0
        assert 'handempty' in state or 'holding_a' in state

        # Execute action
        success, runtime = env.execute('pick-up', ['a'])
        assert isinstance(success, bool)
        assert 0.0 <= runtime <= 1.0

        # Check applicable actions
        actions = env.get_applicable_actions()
        assert len(actions) >= 0

        # Test reset
        env.reset()
        new_state = env.get_state()
        assert len(new_state) > 0

    def test_metrics_export(self, temp_dir):
        """Test metrics export functionality."""
        collector = MetricsCollector()

        # Add some data
        for i in range(5):
            collector.record_action(i, f"action-{i}", ["obj"], True, 0.1)

        collector.collect_snapshot(0)
        collector.collect_snapshot(5)

        # Export to CSV
        csv_path = temp_dir / "test_metrics.csv"
        collector.export(str(csv_path), format='csv')
        assert csv_path.exists()

        # Export to JSON
        json_path = temp_dir / "test_metrics.json"
        collector.export(str(json_path), format='json')
        assert json_path.exists()

    def test_environment_state_updates(self):
        """Test that environment state updates correctly."""
        env = MockEnvironment(success_rate=1.0)  # Always succeed

        initial_state = env.get_state()

        # Pick up a block
        if 'clear_a' in initial_state and 'handempty' in initial_state:
            success, _ = env.execute('pick-up', ['a'])
            if success:
                new_state = env.get_state()
                assert 'holding_a' in new_state
                assert 'clear_a' not in new_state
                assert 'handempty' not in new_state

    def test_metrics_sliding_window(self):
        """Test sliding window for mistake rate."""
        collector = MetricsCollector(window_size=3)

        # Add actions: F, T, T, F, T (F=False, T=True)
        actions = [False, True, True, False, True]
        for i, success in enumerate(actions):
            collector.record_action(i, "action", ["obj"], success, 0.1)

        # Last 3 actions: T, F, T -> mistake rate = 1/3
        mistake_rate = collector.compute_mistake_rate()
        assert mistake_rate == pytest.approx(1 / 3, rel=0.01)

    def test_complete_workflow(self, temp_dir):
        """Test complete workflow without experiment runner."""
        # Initialize components
        metrics = MetricsCollector(interval=2, window_size=5)
        env = MockEnvironment(success_rate=0.7, seed=123)

        # Simulate experiment loop
        for step in range(10):
            # Get state
            state = env.get_state()

            # Get applicable actions (simple selection)
            actions = env.get_applicable_actions()
            if actions:
                action, objects = actions[0]
            else:
                action, objects = 'wait', []

            # Execute action
            success, runtime = env.execute(action, objects)

            # Record metrics
            metrics.record_action(step, action, objects, success, runtime)

            # Collect snapshot if needed
            if metrics.should_collect(step):
                metrics.collect_snapshot(step)

        # Export results
        output_path = temp_dir / "workflow_metrics.json"
        metrics.export(str(output_path), format='json')

        # Verify results
        assert output_path.exists()
        assert len(metrics.metrics_df) == 10
        assert len(metrics.snapshots) > 0

        # Get summary
        summary = metrics.get_summary_statistics()
        assert summary['total_actions'] == 10
        assert 0.0 <= summary['mistake_rate'] <= 1.0
