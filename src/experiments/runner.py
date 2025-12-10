"""
Experiment runner for online action model learning framework.
"""

import yaml
import json
import logging
import time
import platform
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from ..algorithms.base_learner import BaseActionModelLearner
from ..algorithms.information_gain import InformationGainLearner
from ..environments.active_environment import ActiveEnvironment
from ..environments.mock_environment import MockEnvironment
from .metrics import MetricsCollector

# OLAM has been refactored to external post-processing approach
# Import only if needed for backward compatibility
try:
    from ..algorithms.olam_adapter import OLAMAdapter
    OLAM_AVAILABLE = True
except ImportError:
    OLAM_AVAILABLE = False

logger = logging.getLogger(__name__)

# Model snapshot checkpoints for post-processing analysis
CHECKPOINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35,
               40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250,
               300, 350, 400, 450, 500]


class ExperimentRunner:
    """
    Orchestrates experiments for comparing action model learning algorithms.

    Handles configuration loading, algorithm initialization, metrics collection,
    and result export.
    """

    def __init__(self, config_path: str, verbose_debug: bool = False):
        """
        Initialize the experiment runner.

        Args:
            config_path: Path to YAML configuration file
            verbose_debug: Enable detailed logging for validation/debugging
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._validate_config()
        self.verbose_debug = verbose_debug

        # Setup verbose logging if requested
        if self.verbose_debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("VERBOSE DEBUG MODE ENABLED - Detailed logging active")

        # Set random seed if specified
        if 'seed' in self.config['experiment']:
            self._set_random_seed(self.config['experiment']['seed'])

        # Initialize components
        self.learner = self._init_learner()
        self.metrics = MetricsCollector(
            interval=self.config['metrics']['interval'],
            window_size=self.config['metrics']['window_size'],
            track_learning_evidence=self.config['output'].get('track_learning_evidence', False)
        )
        self.environment = self._init_environment()

        # Tracking
        self.start_time = None
        self.end_time = None

        logger.info(f"Initialized ExperimentRunner for {self.config['experiment']['name']}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    def _validate_config(self) -> None:
        """
        Validate that configuration has all required fields.

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = {
            'experiment': ['name', 'algorithm'],
            'domain_problem': ['domain', 'problem'],
            'metrics': [],  # Has defaults
            'stopping_criteria': [],  # Has defaults
            'output': []  # Has defaults
        }

        for section, fields in required_fields.items():
            if section not in self.config and fields:
                raise ValueError(f"Missing required configuration section: {section}")

            for field in fields:
                if field not in self.config.get(section, {}):
                    raise ValueError(f"Missing required field: {section}.{field}")

        # Set defaults
        self._set_config_defaults()

    def _set_config_defaults(self) -> None:
        """Set default values for optional configuration fields."""
        defaults = {
            'metrics': {
                'interval': 10,
                'window_size': 50
            },
            'stopping_criteria': {
                'max_iterations': 1000,
                'max_runtime_seconds': 3600,
                'convergence_check_interval': 100
            },
            'output': {
                'directory': 'results/',
                'formats': ['csv', 'json'],
                'save_learned_model': True
            }
        }

        for section, section_defaults in defaults.items():
            if section not in self.config:
                self.config[section] = {}

            for key, value in section_defaults.items():
                if key not in self.config[section]:
                    self.config[section][key] = value

    def _init_learner(self) -> BaseActionModelLearner:
        """
        Initialize the learning algorithm.

        Returns:
            Initialized learner instance

        Raises:
            ValueError: If algorithm is not supported
        """
        algorithm = self.config['experiment']['algorithm']
        domain_file = self.config['domain_problem']['domain']
        problem_file = self.config['domain_problem']['problem']

        # Get algorithm-specific parameters
        algo_params = self.config.get('algorithm_params', {}).get(algorithm, {})

        if algorithm == 'olam':
            if not OLAM_AVAILABLE:
                raise RuntimeError(
                    "OLAM has been refactored to post-processing approach. "
                    "Use scripts/analyze_olam_results.py for OLAM experiments."
                )
            learner = OLAMAdapter(
                domain_file=domain_file,
                problem_file=problem_file,
                **algo_params
            )
            logger.info("Initialized OLAM adapter")

        elif algorithm == 'information_gain':
            # Set max_iterations if not in algo_params
            if 'max_iterations' not in algo_params:
                algo_params['max_iterations'] = self.config['stopping_criteria']['max_iterations']

            learner = InformationGainLearner(
                domain_file=domain_file,
                problem_file=problem_file,
                **algo_params
            )
            logger.info("Initialized Information Gain learner")

        elif algorithm == 'model_learner':
            # TODO: Phase 6 - ModelLearner adapter
            raise NotImplementedError("ModelLearner adapter not yet implemented")

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return learner

    def _init_environment(self):
        """
        Initialize the environment based on configuration.

        Returns:
            Environment instance (ActiveEnvironment or MockEnvironment)
        """
        domain_file = self.config['domain_problem']['domain']
        problem_file = self.config['domain_problem']['problem']

        # Check for environment type in config (default to 'active' for real execution)
        env_type = self.config.get('experiment', {}).get('environment_type', 'active')

        if env_type == 'mock':
            # Use mock environment for testing
            environment = MockEnvironment(domain_file, problem_file)
            logger.info(f"Initialized MockEnvironment for testing with domain: {domain_file}")
        else:
            # Use ActiveEnvironment for actual experiments
            environment = ActiveEnvironment(domain_file, problem_file)
            logger.info(f"Initialized ActiveEnvironment with domain: {domain_file}")
            logger.info("Using real PDDL execution")

        return environment

    def _set_random_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Set random seed to {seed}")

    def run_experiment(self) -> Dict[str, Any]:
        """
        Run the complete experiment.

        Returns:
            Dictionary containing experiment results
        """
        logger.info(f"Starting experiment: {self.config['experiment']['name']}")
        self.start_time = datetime.now()
        start_perf_time = time.perf_counter()

        # Initialize results tracking
        results = {
            'experiment_name': self.config['experiment']['name'],
            'algorithm': self.config['experiment']['algorithm'],
            'domain_file': self.config['domain_problem']['domain'],
            'problem_file': self.config['domain_problem']['problem'],
            'start_time': self.start_time.isoformat(),
            'platform': platform.platform()
        }

        # Main experiment loop
        iteration = 0
        stopping_reason = None

        while iteration < self.config['stopping_criteria']['max_iterations']:
            # Collect metrics for this iteration
            if self.metrics.should_collect(iteration):
                self.metrics.collect_snapshot(iteration, learner=self.learner)

            # Get current state
            state = self.environment.get_state()

            # Verbose debug: log hypothesis space info
            if self.verbose_debug and iteration % 5 == 0:
                self._log_hypothesis_space_info(iteration)

            try:
                # Select action
                action, objects = self.learner.select_action(state)

                if self.verbose_debug:
                    logger.debug(f"Iteration {iteration}: Selected {action}({','.join(objects)})")

                # Check for convergence signal
                if action == "no_action":
                    logger.info(f"Algorithm signaled convergence (no information gain) at iteration {iteration}")
                    stopping_reason = 'convergence'
                    break

                # Execute action
                success, runtime = self._execute_action(action, objects)

                if self.verbose_debug:
                    logger.debug(f"  Execution: {'SUCCESS' if success else 'FAILURE'} (runtime: {runtime:.3f}s)")

                # Record metrics
                self.metrics.record_action(
                    step=iteration,
                    action=action,
                    objects=objects,
                    success=success,
                    runtime=runtime
                )

                # Observe result for learning
                if success:
                    next_state = self.environment.get_state()
                else:
                    next_state = None

                self.learner.observe(
                    state=state,
                    action=action,
                    objects=objects,
                    success=success,
                    next_state=next_state
                )

                # Export model snapshot at checkpoints
                if iteration + 1 in CHECKPOINTS:  # +1 because iteration is 0-based
                    try:
                        self.learner.export_model_snapshot(
                            iteration=iteration + 1,
                            output_dir=Path(self.config['output']['directory'])
                        )
                        if self.verbose_debug:
                            logger.debug(f"Exported model snapshot at iteration {iteration + 1}")
                    except Exception as e:
                        logger.warning(f"Failed to export model at iteration {iteration + 1}: {e}")
                        # Continue experiment even if export fails

            except Exception as e:
                logger.error(f"Error at iteration {iteration}: {e}")
                self._handle_error(e, iteration)

            iteration += 1

            # Check other stopping criteria after incrementing
            if self._check_convergence(iteration):
                stopping_reason = 'convergence'
                break

            elapsed = time.perf_counter() - start_perf_time
            if elapsed >= self.config['stopping_criteria']['max_runtime_seconds']:
                stopping_reason = 'timeout'
                break

        # Set stopping reason if we hit max iterations
        if stopping_reason is None and iteration >= self.config['stopping_criteria']['max_iterations']:
            stopping_reason = 'max_iterations'

        # Final metrics collection
        self.metrics.collect_snapshot(iteration, learner=self.learner)

        # Complete results
        self.end_time = datetime.now()
        results.update({
            'end_time': self.end_time.isoformat(),
            'total_iterations': iteration,
            'stopping_reason': stopping_reason or 'completed',
            'runtime_seconds': time.perf_counter() - start_perf_time,
            'metrics': self.metrics.get_summary_statistics()
        })

        # Export results
        self._export_results(results)

        logger.info(f"Experiment completed: {iteration} iterations, reason: {stopping_reason}")
        return results


    def _should_stop(self, iteration: int, start_time: float) -> bool:
        """
        Check if experiment should stop.

        Args:
            iteration: Current iteration
            start_time: Experiment start time

        Returns:
            True if should stop
        """
        # Check max iterations
        if iteration >= self.config['stopping_criteria']['max_iterations']:
            return True

        # Check timeout
        elapsed = time.perf_counter() - start_time
        if elapsed >= self.config['stopping_criteria']['max_runtime_seconds']:
            return True

        # Check convergence
        if self._check_convergence(iteration):
            return True

        return False

    def _execute_action(self, action: str, objects: List[str]) -> Tuple[bool, float]:
        """
        Execute an action in the environment.

        Args:
            action: Action name to execute
            objects: Objects involved in the action

        Returns:
            Tuple of (success, runtime)
        """
        return self.environment.execute(action, objects)

    def _check_convergence(self, iteration: int) -> bool:
        """
        Check if the learner has converged.

        Args:
            iteration: Current iteration

        Returns:
            True if converged
        """
        check_interval = self.config['stopping_criteria'].get('convergence_check_interval', 100)

        if iteration % check_interval == 0 and iteration > 0:
            return self.learner.has_converged()

        return False

    def _log_hypothesis_space_info(self, iteration: int) -> None:
        """
        Log detailed hypothesis space information for debugging.

        Args:
            iteration: Current iteration number
        """
        if not hasattr(self.learner, 'get_learned_model'):
            return

        model = self.learner.get_learned_model()
        actions = model.get('actions', {})

        # Count statistics
        total_actions = len(actions)
        with_certain = sum(1 for a in actions.values() if a['preconditions'].get('certain', []))
        with_uncertain = sum(1 for a in actions.values() if a['preconditions'].get('uncertain', []))
        unexplored = sum(1 for a in actions.values() if not a['preconditions'].get('certain', []) and not a['preconditions'].get('uncertain', []))

        # For OLAM, check how many actions are filtered
        filtered = 0
        if hasattr(self.learner, 'learner') and hasattr(self.learner.learner, 'compute_not_executable_actionsJAVA'):
            filtered = len(self.learner.learner.compute_not_executable_actionsJAVA())

        logger.info(f"\n--- HYPOTHESIS SPACE (Iteration {iteration}) ---")
        logger.info(f"Total actions: {total_actions}")
        logger.info(f"Actions filtered as non-executable: {filtered}")
        logger.info(f"Actions with certain preconditions: {with_certain}")
        logger.info(f"Actions with uncertain preconditions: {with_uncertain}")
        logger.info(f"Unexplored actions: {unexplored}")
        logger.info(f"Hypothesis space reduction: {filtered}/{total_actions} = {(filtered/total_actions*100):.1f}% filtered")

        # Sample a learned action for detail
        for action_name, data in list(actions.items())[:1]:
            if data['preconditions'].get('certain'):
                logger.debug(f"Example learned action: {action_name}")
                logger.debug(f"  Certain: {data['preconditions']['certain'][:3]}")
                logger.debug(f"  Uncertain: {data['preconditions']['uncertain'][:3] if data['preconditions'].get('uncertain') else []}")

    def _handle_error(self, error: Exception, iteration: int) -> None:
        """
        Handle errors during experiment execution.

        Args:
            error: Exception that occurred
            iteration: Current iteration when error occurred
        """
        import traceback
        logger.error(f"Handled error at iteration {iteration}: {error}")
        logger.error(f"Traceback:\n{''.join(traceback.format_tb(error.__traceback__))}")
        # For now, just log. In production, might want to save state or retry

    def _export_results(self, results: Dict[str, Any]) -> None:
        """
        Export experiment results to files.

        Args:
            results: Experiment results dictionary
        """
        # Determine if this is a test or real experiment
        experiment_name = self.config['experiment']['name']
        is_test = 'test' in experiment_name.lower()

        # Route to appropriate subdirectory
        base_output_dir = Path(self.config['output']['directory'])
        if is_test:
            output_dir = base_output_dir / 'tests'
        else:
            output_dir = base_output_dir / 'experiments'

        output_dir.mkdir(parents=True, exist_ok=True)

        # Clean up old test files if this is a test run
        if is_test:
            self._cleanup_old_test_results(output_dir)

        # Generate filename with timestamp
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        base_name = f"{experiment_name}_{timestamp}"

        # Export metrics
        for format in self.config['output']['formats']:
            if format == 'csv':
                csv_path = output_dir / f"{base_name}_metrics.csv"
                self.metrics.export(str(csv_path), format='csv')

            elif format == 'json':
                json_path = output_dir / f"{base_name}_metrics.json"
                self.metrics.export(str(json_path), format='json')

        # Save experiment summary
        summary_path = output_dir / f"{base_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save learned model if requested
        if self.config['output'].get('save_learned_model', False):
            try:
                learned_model = self.learner.get_learned_model()
                model_path = output_dir / 'learned_model.json'
                with open(model_path, 'w') as f:
                    json.dump(learned_model, f, indent=2, default=str)
                logger.info(f"Saved learned model to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save learned model: {e}")

        logger.info(f"Exported results to {output_dir}")

    def _cleanup_old_test_results(self, test_dir: Path) -> None:
        """
        Remove old test result files to keep the test directory clean.

        Args:
            test_dir: Path to the test results directory
        """
        if not test_dir.exists():
            return

        # Remove all old test result files
        for pattern in ['test_*.csv', 'test_*.json']:
            for old_file in test_dir.glob(pattern):
                try:
                    old_file.unlink()
                    logger.debug(f"Cleaned up old test file: {old_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove old test file {old_file}: {e}")

    def _compile_results(self) -> Dict[str, Any]:
        """
        Compile final experiment results.

        Returns:
            Dictionary with complete results
        """
        return {
            'experiment_config': self.config,
            'metrics_summary': self.metrics.get_summary_statistics(),
            'snapshots': self.metrics.snapshots,
            'learner_statistics': self.learner.get_statistics()
        }