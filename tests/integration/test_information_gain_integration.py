"""
Integration tests for Information Gain Algorithm with ExperimentRunner.
Tests end-to-end experiment execution with all selection strategies.
"""

import pytest
import yaml
from pathlib import Path
import tempfile
import shutil

from src.experiments.runner import ExperimentRunner
from src.algorithms.information_gain import InformationGainLearner


@pytest.fixture
def temp_results_dir():
    """Create temporary directory for test results."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def base_config():
    """Base configuration for Information Gain experiments."""
    return {
        'experiment': {
            'name': 'test_info_gain',
            'algorithm': 'information_gain',
            'seed': 42,
            'environment_type': 'mock'  # Use mock for faster testing
        },
        'domain_problem': {
            'domain': 'benchmarks/olam-compatible/blocksworld/domain.pddl',
            'problem': 'benchmarks/olam-compatible/blocksworld/p01.pddl'
        },
        'algorithm_params': {
            'information_gain': {
                'selection_strategy': 'greedy',
                'max_iterations': 50
            }
        },
        'metrics': {
            'interval': 10,
            'window_size': 20
        },
        'stopping_criteria': {
            'max_iterations': 50,
            'max_runtime_seconds': 60,
            'convergence_check_interval': 10
        },
        'output': {
            'directory': 'results/',
            'formats': ['json'],
            'save_learned_model': True
        }
    }


def create_config_file(config_dict, temp_dir):
    """Create a temporary config file from dictionary."""
    config_path = Path(temp_dir) / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    return str(config_path)


class TestInformationGainIntegration:
    """Test Information Gain learner integration with ExperimentRunner."""

    def test_runner_initializes_info_gain_learner(self, base_config, temp_results_dir):
        """Test that ExperimentRunner properly initializes Information Gain learner."""
        base_config['output']['directory'] = temp_results_dir
        config_path = create_config_file(base_config, temp_results_dir)

        runner = ExperimentRunner(config_path)

        assert runner.learner is not None
        assert isinstance(runner.learner, InformationGainLearner)
        assert runner.learner.selection_strategy == 'greedy'

    def test_greedy_strategy_experiment(self, base_config, temp_results_dir):
        """Test end-to-end experiment with greedy selection strategy."""
        base_config['algorithm_params']['information_gain']['selection_strategy'] = 'greedy'
        base_config['output']['directory'] = temp_results_dir
        config_path = create_config_file(base_config, temp_results_dir)

        runner = ExperimentRunner(config_path)
        results = runner.run_experiment()

        # Verify experiment completed
        assert results is not None
        assert 'total_iterations' in results
        assert results['total_iterations'] > 0
        assert results['algorithm'] == 'information_gain'

        # Verify metrics were collected
        assert 'metrics' in results
        assert results['metrics'] is not None

    def test_epsilon_greedy_strategy_experiment(self, base_config, temp_results_dir):
        """Test end-to-end experiment with epsilon-greedy selection strategy."""
        base_config['algorithm_params']['information_gain'] = {
            'selection_strategy': 'epsilon_greedy',
            'epsilon': 0.1,
            'max_iterations': 50
        }
        base_config['output']['directory'] = temp_results_dir
        config_path = create_config_file(base_config, temp_results_dir)

        runner = ExperimentRunner(config_path)

        # Verify epsilon parameter was passed
        assert runner.learner.epsilon == 0.1
        assert runner.learner.selection_strategy == 'epsilon_greedy'

        results = runner.run_experiment()

        # Verify experiment completed
        assert results is not None
        assert results['total_iterations'] > 0

    def test_boltzmann_strategy_experiment(self, base_config, temp_results_dir):
        """Test end-to-end experiment with Boltzmann selection strategy."""
        base_config['algorithm_params']['information_gain'] = {
            'selection_strategy': 'boltzmann',
            'temperature': 1.0,
            'max_iterations': 50
        }
        base_config['output']['directory'] = temp_results_dir
        config_path = create_config_file(base_config, temp_results_dir)

        runner = ExperimentRunner(config_path)

        # Verify temperature parameter was passed
        assert runner.learner.temperature == 1.0
        assert runner.learner.selection_strategy == 'boltzmann'

        results = runner.run_experiment()

        # Verify experiment completed
        assert results is not None
        assert results['total_iterations'] > 0

    def test_learned_model_export(self, base_config, temp_results_dir):
        """Test that learned model is properly exported."""
        base_config['output']['directory'] = temp_results_dir
        base_config['output']['save_learned_model'] = True
        config_path = create_config_file(base_config, temp_results_dir)

        runner = ExperimentRunner(config_path)
        results = runner.run_experiment()

        # Check that learned model file exists
        model_path = Path(temp_results_dir) / 'learned_model.json'
        assert model_path.exists()

    def test_metrics_collection(self, base_config, temp_results_dir):
        """Test that metrics are collected during experiment."""
        base_config['metrics']['interval'] = 1  # Collect every action
        base_config['output']['directory'] = temp_results_dir
        config_path = create_config_file(base_config, temp_results_dir)

        runner = ExperimentRunner(config_path)
        results = runner.run_experiment()

        # Verify metrics were collected
        assert 'metrics' in results
        metrics = results['metrics']

        # Check that we have action data
        assert 'total_actions' in metrics or 'success_rate' in metrics

    def test_convergence_detection(self, base_config, temp_results_dir):
        """Test that convergence is properly detected."""
        base_config['stopping_criteria']['max_iterations'] = 1000  # High enough to potentially converge
        base_config['stopping_criteria']['convergence_check_interval'] = 10
        base_config['output']['directory'] = temp_results_dir
        config_path = create_config_file(base_config, temp_results_dir)

        runner = ExperimentRunner(config_path)
        results = runner.run_experiment()

        # Verify stopping reason is recorded
        assert 'stopping_reason' in results
        assert results['stopping_reason'] in ['convergence', 'max_iterations', 'timeout']

    def test_action_selection_produces_valid_actions(self, base_config, temp_results_dir):
        """Test that selected actions are valid."""
        base_config['stopping_criteria']['max_iterations'] = 10
        base_config['output']['directory'] = temp_results_dir
        config_path = create_config_file(base_config, temp_results_dir)

        runner = ExperimentRunner(config_path)

        # Run a few iterations manually
        state = runner.environment.get_state()
        action, objects = runner.learner.select_action(state)

        # Verify action is not empty
        assert action is not None
        assert action != ""

        # Verify objects list exists (may be empty for some actions)
        assert objects is not None
        assert isinstance(objects, list)


class TestMultipleDomains:
    """Test Information Gain learner on multiple PDDL domains."""

    @pytest.fixture
    def gripper_config(self, base_config, temp_results_dir):
        """Configuration for gripper domain."""
        config = base_config.copy()
        config['domain_problem'] = {
            'domain': 'benchmarks/olam-compatible/gripper/domain.pddl',
            'problem': 'benchmarks/olam-compatible/gripper/p01.pddl'
        }
        config['output']['directory'] = temp_results_dir
        return config

    def test_gripper_domain(self, gripper_config, temp_results_dir):
        """Test Information Gain learner on gripper domain."""
        config_path = create_config_file(gripper_config, temp_results_dir)

        runner = ExperimentRunner(config_path)
        results = runner.run_experiment()

        assert results is not None
        assert results['total_iterations'] > 0


class TestPerformance:
    """Performance-related integration tests."""

    def test_experiment_completes_within_timeout(self, base_config, temp_results_dir):
        """Test that experiment completes within configured timeout."""
        base_config['stopping_criteria']['max_runtime_seconds'] = 30
        base_config['stopping_criteria']['max_iterations'] = 100
        base_config['output']['directory'] = temp_results_dir
        config_path = create_config_file(base_config, temp_results_dir)

        runner = ExperimentRunner(config_path)
        results = runner.run_experiment()

        # Verify it completed
        assert results is not None
        # Verify runtime is reasonable
        assert results['runtime_seconds'] < 60  # Should be well under timeout
