"""
Test suite for ExperimentRunner - comprehensive testing for experiment orchestration.
"""

import pytest
import yaml
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExperimentRunner:
    """Test suite for ExperimentRunner class."""

    @pytest.fixture
    def sample_config(self):
        """Sample experiment configuration."""
        return {
            'experiment': {
                'name': 'test_experiment',
                'algorithm': 'olam',
                'seed': 42
            },
            'domain_problem': {
                'domain': 'benchmarks/olam-compatible/blocksworld/domain.pddl',
                'problem': 'benchmarks/olam-compatible/blocksworld/p01.pddl'
            },
            'algorithm_params': {
                'olam': {
                    'max_iterations': 100,
                    'eval_frequency': 10
                }
            },
            'metrics': {
                'interval': 10,
                'window_size': 50
            },
            'stopping_criteria': {
                'max_iterations': 100,
                'max_runtime_seconds': 300,
                'convergence_check_interval': 20
            },
            'output': {
                'directory': 'results/',
                'formats': ['csv', 'json'],
                'save_learned_model': True
            }
        }

    @pytest.fixture
    def config_file(self, temp_dir, sample_config):
        """Create a temporary config file."""
        config_path = temp_dir / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        return str(config_path)

    def test_initialization(self, config_file):
        """Test ExperimentRunner initialization."""
        from src.experiments.runner import ExperimentRunner

        runner = ExperimentRunner(config_file)

        assert runner.config is not None
        assert runner.config['experiment']['name'] == 'test_experiment'
        assert runner.metrics is not None
        assert runner.learner is not None

    def test_load_config(self, config_file, sample_config):
        """Test configuration loading and validation."""
        from src.experiments.runner import ExperimentRunner

        runner = ExperimentRunner(config_file)
        config = runner.config

        assert config['experiment']['algorithm'] == 'olam'
        assert config['metrics']['interval'] == 10
        assert config['stopping_criteria']['max_iterations'] == 100

    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        from src.experiments.runner import ExperimentRunner

        with pytest.raises(FileNotFoundError):
            runner = ExperimentRunner("nonexistent.yaml")

    def test_config_validation(self, temp_dir):
        """Test configuration validation for missing required fields."""
        from src.experiments.runner import ExperimentRunner

        # Config missing required fields
        invalid_config = {
            'experiment': {
                'name': 'test'
                # Missing 'algorithm'
            }
        }

        config_path = temp_dir / "invalid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError, match="Missing required.*algorithm"):
            runner = ExperimentRunner(str(config_path))

    @patch('src.experiments.runner.OLAMAdapter')
    def test_init_learner_olam(self, mock_olam, config_file):
        """Test OLAM learner initialization."""
        from src.experiments.runner import ExperimentRunner

        mock_instance = MagicMock()
        mock_olam.return_value = mock_instance

        runner = ExperimentRunner(config_file)

        # Check that OLAM was initialized with correct parameters
        mock_olam.assert_called_once()
        call_args = mock_olam.call_args
        # OLAMAdapter uses keyword arguments
        assert 'domain.pddl' in call_args.kwargs['domain_file']
        assert 'p01.pddl' in call_args.kwargs['problem_file']

    def test_unsupported_algorithm(self, temp_dir):
        """Test handling of unsupported algorithm."""
        from src.experiments.runner import ExperimentRunner

        config = {
            'experiment': {
                'name': 'test',
                'algorithm': 'unknown_algo'
            },
            'domain_problem': {
                'domain': 'test.pddl',
                'problem': 'test.pddl'
            }
        }

        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="Unsupported algorithm"):
            runner = ExperimentRunner(str(config_path))

    @patch('src.experiments.runner.MockEnvironment')
    def test_run_experiment_basic(self, mock_env_class, config_file):
        """Test basic experiment execution."""
        from src.experiments.runner import ExperimentRunner

        # Setup mocks
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_env.get_state.return_value = {'clear_a', 'on_b_c'}
        mock_env.execute.return_value = (True, 0.05)  # success, runtime

        with patch.object(ExperimentRunner, '_init_learner') as mock_init:
            mock_learner = MagicMock()
            mock_learner.select_action.return_value = ('pick-up', ['a'])
            mock_learner.has_converged.return_value = False
            mock_init.return_value = mock_learner

            runner = ExperimentRunner(config_file)
            runner.config['stopping_criteria']['max_iterations'] = 5

            results = runner.run_experiment()

            # Verify experiment ran
            assert results is not None
            assert 'experiment_name' in results
            assert 'total_iterations' in results
            assert results['total_iterations'] == 5
            assert mock_learner.select_action.call_count == 5

    def test_stopping_criteria_max_iterations(self, config_file):
        """Test stopping on max iterations."""
        from src.experiments.runner import ExperimentRunner

        with patch.object(ExperimentRunner, '_init_learner') as mock_init:
            mock_learner = MagicMock()
            mock_learner.select_action.return_value = ('action', ['obj'])
            mock_learner.has_converged.return_value = False
            mock_init.return_value = mock_learner

            runner = ExperimentRunner(config_file)
            runner.config['stopping_criteria']['max_iterations'] = 10

            with patch.object(runner, '_execute_action', return_value=(True, 0.01)):
                results = runner.run_experiment()

            assert results['total_iterations'] == 10
            assert results['stopping_reason'] == 'max_iterations'

    def test_stopping_criteria_convergence(self, config_file):
        """Test stopping on convergence."""
        from src.experiments.runner import ExperimentRunner

        with patch.object(ExperimentRunner, '_init_learner') as mock_init:
            mock_learner = MagicMock()
            mock_learner.select_action.return_value = ('action', ['obj'])

            # Converge after 5 iterations
            convergence_sequence = [False] * 5 + [True] * 100
            mock_learner.has_converged.side_effect = convergence_sequence
            mock_init.return_value = mock_learner

            runner = ExperimentRunner(config_file)
            runner.config['stopping_criteria']['convergence_check_interval'] = 1

            with patch.object(runner, '_execute_action', return_value=(True, 0.01)):
                results = runner.run_experiment()

            assert results['total_iterations'] <= 10
            assert results['stopping_reason'] == 'convergence'

    def test_stopping_criteria_timeout(self, config_file):
        """Test stopping on timeout."""
        from src.experiments.runner import ExperimentRunner

        with patch.object(ExperimentRunner, '_init_learner') as mock_init:
            mock_learner = MagicMock()
            mock_learner.select_action.return_value = ('action', ['obj'])
            mock_learner.has_converged.return_value = False
            mock_init.return_value = mock_learner

            runner = ExperimentRunner(config_file)
            runner.config['stopping_criteria']['max_runtime_seconds'] = 0.1

            # Simulate slow execution
            def slow_execute(*args):
                time.sleep(0.05)
                return (True, 0.05)

            with patch.object(runner, '_execute_action', side_effect=slow_execute):
                results = runner.run_experiment()

            assert results['stopping_reason'] == 'timeout'
            assert results['runtime_seconds'] >= 0.1

    def test_metrics_collection_intervals(self, config_file):
        """Test that metrics are collected at correct intervals."""
        from src.experiments.runner import ExperimentRunner

        with patch.object(ExperimentRunner, '_init_learner') as mock_init:
            mock_learner = MagicMock()
            mock_learner.select_action.return_value = ('action', ['obj'])
            mock_learner.has_converged.return_value = False
            mock_init.return_value = mock_learner

            runner = ExperimentRunner(config_file)
            runner.config['metrics']['interval'] = 3
            runner.metrics.interval = 3  # Also update the metrics collector directly
            runner.config['stopping_criteria']['max_iterations'] = 10

            with patch.object(runner, '_execute_action', return_value=(True, 0.01)):
                with patch.object(runner.metrics, 'collect_snapshot') as mock_collect:
                    results = runner.run_experiment()

                    # Should collect at steps 0, 3, 6, 9
                    collected_steps = [call[0][0] for call in mock_collect.call_args_list]
                    assert 0 in collected_steps
                    assert 3 in collected_steps
                    assert 6 in collected_steps
                    assert 9 in collected_steps

    def test_results_export(self, config_file, temp_dir):
        """Test exporting results to different formats."""
        from src.experiments.runner import ExperimentRunner

        with patch.object(ExperimentRunner, '_init_learner') as mock_init:
            mock_learner = MagicMock()
            mock_learner.select_action.return_value = ('action', ['obj'])
            mock_learner.has_converged.return_value = False
            mock_learner.get_learned_model.return_value = {'actions': {}}
            mock_init.return_value = mock_learner

            runner = ExperimentRunner(config_file)
            runner.config['output']['directory'] = str(temp_dir)
            runner.config['stopping_criteria']['max_iterations'] = 5

            with patch.object(runner, '_execute_action', return_value=(True, 0.01)):
                results = runner.run_experiment()

            # Check that results were exported
            output_dir = Path(runner.config['output']['directory'])
            assert output_dir.exists()

            # Check for exported files
            csv_files = list(output_dir.glob('*.csv'))
            json_files = list(output_dir.glob('*.json'))

            assert len(csv_files) > 0
            assert len(json_files) > 0

    def test_learned_model_saving(self, config_file, temp_dir):
        """Test saving the learned model."""
        from src.experiments.runner import ExperimentRunner

        learned_model = {
            'actions': {
                'pick-up': {
                    'preconditions': ['clear', 'handempty'],
                    'effects': ['holding', '-clear']
                }
            },
            'predicates': ['clear', 'holding', 'handempty']
        }

        with patch.object(ExperimentRunner, '_init_learner') as mock_init:
            mock_learner = MagicMock()
            mock_learner.select_action.return_value = ('action', ['obj'])
            mock_learner.has_converged.return_value = True
            mock_learner.get_learned_model.return_value = learned_model
            mock_init.return_value = mock_learner

            runner = ExperimentRunner(config_file)
            runner.config['output']['directory'] = str(temp_dir)
            runner.config['output']['save_learned_model'] = True
            runner.config['stopping_criteria']['max_iterations'] = 1

            with patch.object(runner, '_execute_action', return_value=(True, 0.01)):
                results = runner.run_experiment()

            # Check that model was saved
            model_file = Path(runner.config['output']['directory']) / 'learned_model.json'
            assert model_file.exists()

            with open(model_file, 'r') as f:
                saved_model = json.load(f)

            assert saved_model['actions']['pick-up'] is not None
            assert 'clear' in saved_model['predicates']

    def test_error_recovery(self, config_file):
        """Test error handling and recovery during experiment."""
        from src.experiments.runner import ExperimentRunner

        with patch.object(ExperimentRunner, '_init_learner') as mock_init:
            mock_learner = MagicMock()

            # Simulate error in action selection
            mock_learner.select_action.side_effect = [
                ('action', ['obj']),  # First call succeeds
                RuntimeError("Test error"),  # Second call fails
                ('action', ['obj']),  # Recovery
            ]
            mock_learner.has_converged.return_value = False
            mock_init.return_value = mock_learner

            runner = ExperimentRunner(config_file)
            runner.config['stopping_criteria']['max_iterations'] = 3

            with patch.object(runner, '_execute_action', return_value=(True, 0.01)):
                with patch.object(runner, '_handle_error') as mock_error:
                    results = runner.run_experiment()

                    # Error should be handled
                    mock_error.assert_called_once()

    def test_experiment_reproducibility(self, config_file):
        """Test that experiments are reproducible with same seed."""
        from src.experiments.runner import ExperimentRunner
        import random
        import numpy as np

        def run_with_seed(seed):
            with patch.object(ExperimentRunner, '_init_learner') as mock_init:
                mock_learner = MagicMock()

                # Use random for action selection
                def random_action(*args):
                    actions = ['pick-up', 'put-down', 'stack']
                    return (random.choice(actions), ['obj'])

                mock_learner.select_action.side_effect = random_action
                mock_learner.has_converged.return_value = False
                mock_init.return_value = mock_learner

                runner = ExperimentRunner(config_file)
                runner.config['experiment']['seed'] = seed
                runner.config['stopping_criteria']['max_iterations'] = 10

                # Set seeds
                runner._set_random_seed(seed)

                with patch.object(runner, '_execute_action', return_value=(True, 0.01)):
                    results = runner.run_experiment()

                return results

        # Run twice with same seed
        results1 = run_with_seed(42)
        results2 = run_with_seed(42)

        # Should produce same results
        assert results1['total_iterations'] == results2['total_iterations']

    def test_integration_with_mock_environment(self, config_file):
        """Test full integration with mock environment."""
        from src.experiments.runner import ExperimentRunner
        from src.environments.mock_environment import MockEnvironment

        runner = ExperimentRunner(config_file)
        runner.config['stopping_criteria']['max_iterations'] = 20

        # Run with actual mock environment (no mocking)
        results = runner.run_experiment()

        assert results is not None
        assert results['total_iterations'] <= 20
        assert 'metrics' in results
        assert 'runtime_seconds' in results

    def test_parallel_experiments(self, config_file):
        """Test running multiple experiments in parallel."""
        from src.experiments.runner import ExperimentRunner
        import threading

        results = []

        def run_experiment(config, index):
            with patch.object(ExperimentRunner, '_init_learner') as mock_init:
                mock_learner = MagicMock()
                mock_learner.select_action.return_value = ('action', ['obj'])
                mock_learner.has_converged.return_value = False
                mock_init.return_value = mock_learner

                runner = ExperimentRunner(config)
                runner.config['experiment']['name'] = f"experiment_{index}"
                runner.config['stopping_criteria']['max_iterations'] = 5

                with patch.object(runner, '_execute_action', return_value=(True, 0.01)):
                    result = runner.run_experiment()
                    results.append(result)

        # Run 3 experiments in parallel
        threads = []
        for i in range(3):
            t = threading.Thread(target=run_experiment, args=(config_file, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All experiments should complete
        assert len(results) == 3
        for result in results:
            assert result['total_iterations'] == 5

    @patch('src.experiments.runner.MetricsCollector')
    def test_custom_metrics_collector(self, mock_metrics_class, config_file):
        """Test using custom metrics collector configuration."""
        from src.experiments.runner import ExperimentRunner

        mock_metrics = MagicMock()
        mock_metrics_class.return_value = mock_metrics

        with patch.object(ExperimentRunner, '_init_learner') as mock_init:
            mock_learner = MagicMock()
            mock_init.return_value = mock_learner

            runner = ExperimentRunner(config_file)

            # Check metrics initialized with config params
            mock_metrics_class.assert_called_with(
                interval=10,
                window_size=50
            )

    def test_experiment_metadata(self, config_file):
        """Test that experiment metadata is properly recorded."""
        from src.experiments.runner import ExperimentRunner
        import platform

        with patch.object(ExperimentRunner, '_init_learner') as mock_init:
            mock_learner = MagicMock()
            mock_learner.select_action.return_value = ('action', ['obj'])
            mock_learner.has_converged.return_value = False
            mock_init.return_value = mock_learner

            runner = ExperimentRunner(config_file)
            runner.config['stopping_criteria']['max_iterations'] = 1

            with patch.object(runner, '_execute_action', return_value=(True, 0.01)):
                results = runner.run_experiment()

            # Check metadata
            assert 'experiment_name' in results
            assert 'algorithm' in results
            assert 'domain_file' in results
            assert 'problem_file' in results
            assert 'start_time' in results
            assert 'end_time' in results
            assert 'platform' in results
            assert results['platform'] == platform.platform()
