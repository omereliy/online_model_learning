"""
Validation tests for convergence detection with conservative settings.

Tests that convergence detection works reliably during long experiments:
- No premature convergence (false positives)
- Stable convergence criteria
- Conservative parameters prevent early termination
"""

import pytest
from unittest.mock import Mock
from pathlib import Path

from src.algorithms.information_gain import InformationGainLearner


@pytest.fixture
def test_domain_file():
    """Get path to test domain file."""
    return "benchmarks/olam-compatible/blocksworld/domain.pddl"


@pytest.fixture
def test_problem_file():
    """Get path to test problem file."""
    return "benchmarks/olam-compatible/blocksworld/p01.pddl"


class TestConservativeConvergence:
    """Test convergence with conservative parameters."""

    def test_conservative_defaults_prevent_early_convergence(self, test_domain_file, test_problem_file):
        """Test that default conservative parameters prevent premature convergence."""
        # Create learner with default conservative parameters
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000
        )

        # Verify conservative defaults are set
        assert learner.MODEL_STABILITY_WINDOW == 50
        assert learner.INFO_GAIN_EPSILON == 0.001
        assert learner.SUCCESS_RATE_THRESHOLD == 0.98
        assert learner.SUCCESS_RATE_WINDOW == 50

        # Simulate partial learning (would trigger convergence with aggressive settings)
        learner.iteration_count = 100

        # Add 10 stable iterations (not enough for conservative window of 50)
        snapshot = {action: pre_set.copy() for action, pre_set in learner.pre.items()}
        for _ in range(10):
            learner._model_snapshot_history.append(snapshot)

        # Low gain but above conservative threshold
        learner._last_max_gain = 0.005
        learner.observation_count = 1

        # 95% success rate (below conservative 98% threshold)
        learner._success_history = [True] * 19 + [False]  # 95% success

        # Should NOT converge - conservative settings prevent premature convergence
        assert learner.has_converged() is False

    def test_aggressive_settings_allow_early_convergence(self, test_domain_file, test_problem_file):
        """Test that aggressive parameters allow earlier convergence."""
        # Create learner with aggressive parameters
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000,
            model_stability_window=10,
            info_gain_epsilon=0.01,
            success_rate_threshold=0.95,
            success_rate_window=20
        )

        # Same partial learning scenario as above
        learner.iteration_count = 100

        # 10 stable iterations (enough for aggressive window)
        snapshot = {action: pre_set.copy() for action, pre_set in learner.pre.items()}
        for _ in range(10):
            learner._model_snapshot_history.append(snapshot)

        # Low gain below aggressive threshold
        learner._last_max_gain = 0.005
        learner.observation_count = 1

        # 95% success rate (meets aggressive threshold)
        learner._success_history = [True] * 19 + [False]  # 95% success

        # Should converge - aggressive settings allow early termination
        assert learner.has_converged() is True

    def test_long_run_stability_window_requirement(self, test_domain_file, test_problem_file):
        """Test that large stability window requires sustained model stability."""
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000,
            model_stability_window=50
        )

        learner.iteration_count = 100

        # Add 48 stable iterations (below window size)
        # Note: has_converged() calls _check_model_stability() which adds one more snapshot
        # So 48 + 1 = 49 (just below 50 window size)
        snapshot = {action: pre_set.copy() for action, pre_set in learner.pre.items()}
        for _ in range(48):
            learner._model_snapshot_history.append(snapshot)

        # All other criteria met
        learner._last_max_gain = 0.0001
        learner.observation_count = 1
        learner._success_history = [True] * 50

        # Should NOT converge - need full 50 iteration window (48 + 1 from check = 49)
        assert learner.has_converged() is False

        # Now with 49 snapshots in history, next check adds 1 more = 50 total
        learner.iteration_count = 101

        # Now should converge - all 3 criteria met with full window (49 + 1 = 50)
        assert learner.has_converged() is True

    def test_strict_epsilon_threshold(self, test_domain_file, test_problem_file):
        """Test that conservative epsilon requires very low information gain."""
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000,
            info_gain_epsilon=0.001
        )

        learner.iteration_count = 100

        # Model stable
        snapshot = {action: pre_set.copy() for action, pre_set in learner.pre.items()}
        for _ in range(50):
            learner._model_snapshot_history.append(snapshot)

        # High success rate
        learner._success_history = [True] * 50
        learner.observation_count = 1

        # Gain just above conservative threshold
        learner._last_max_gain = 0.0011

        # Should NOT converge - info gain too high
        assert learner.has_converged() is False

        # Lower gain below threshold
        learner._last_max_gain = 0.0009

        # Now should converge
        assert learner.has_converged() is True

    def test_high_success_rate_threshold(self, test_domain_file, test_problem_file):
        """Test that conservative success rate threshold requires high accuracy."""
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000,
            success_rate_threshold=0.98
        )

        learner.iteration_count = 100

        # Model stable
        snapshot = {action: pre_set.copy() for action, pre_set in learner.pre.items()}
        for _ in range(50):
            learner._model_snapshot_history.append(snapshot)

        # Low info gain
        learner._last_max_gain = 0.0001
        learner.observation_count = 1

        # 97% success rate (below 98% threshold)
        learner._success_history = [True] * 48 + [False, False]  # 96% with 50 window

        # Should NOT converge - success rate too low
        assert learner.has_converged() is False

        # Increase to 98%
        learner._success_history = [True] * 49 + [False]  # 98% with 50 window

        # Now should converge
        assert learner.has_converged() is True


class TestConvergenceWithLearning:
    """Test convergence behavior during actual learning."""

    def test_early_iterations_do_not_converge(self, test_domain_file, test_problem_file):
        """Test that convergence doesn't trigger in early learning phases."""
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000
        )

        # Early in learning (< stability window)
        for i in range(1, 40):
            learner.iteration_count = i
            # Even with other criteria potentially met
            learner._last_max_gain = 0.0001
            learner.observation_count = i
            learner._success_history = [True] * min(i, 50)

            # Should not converge - not enough history
            assert learner.has_converged() is False

    def test_max_iterations_forces_convergence(self, test_domain_file, test_problem_file):
        """Test that max iterations always forces convergence."""
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000
        )

        # Set to max iterations
        learner.iteration_count = 1000

        # Even with no other criteria met
        learner._model_snapshot_history = []
        learner._last_max_gain = 1.0  # High gain
        learner._success_history = [False] * 50  # Low success

        # Should still converge - max iterations reached
        assert learner.has_converged() is True

    def test_model_instability_prevents_convergence(self, test_domain_file, test_problem_file):
        """Test that model changes prevent convergence even with other criteria."""
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000
        )

        learner.iteration_count = 100

        # Unstable model - preconditions keep changing
        for i in range(50):
            # Create slightly different snapshots (simulating ongoing learning)
            snapshot = {action: pre_set.copy() for action, pre_set in learner.pre.items()}
            # Modify one action's preconditions each iteration
            if i % 10 == 0 and snapshot:
                first_action = list(snapshot.keys())[0]
                snapshot[first_action] = snapshot[first_action].copy()
                # Remove one literal to simulate learning
                if snapshot[first_action]:
                    snapshot[first_action].pop()
            learner._model_snapshot_history.append(snapshot)

        # Other criteria met
        learner._last_max_gain = 0.0001
        learner.observation_count = 1
        learner._success_history = [True] * 50

        # Should NOT converge - model not stable
        assert learner.has_converged() is False


class TestConvergenceCriteriaIndependence:
    """Test that all three criteria must be met independently."""

    def test_only_model_stability_insufficient(self, test_domain_file, test_problem_file):
        """Test that model stability alone doesn't trigger convergence."""
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000
        )

        learner.iteration_count = 100

        # Model stable
        snapshot = {action: pre_set.copy() for action, pre_set in learner.pre.items()}
        for _ in range(50):
            learner._model_snapshot_history.append(snapshot)

        # High info gain
        learner._last_max_gain = 0.5
        learner.observation_count = 1

        # Low success rate
        learner._success_history = [False] * 50

        # Should NOT converge - only 1 of 3 criteria met
        assert learner.has_converged() is False

    def test_only_low_gain_insufficient(self, test_domain_file, test_problem_file):
        """Test that low information gain alone doesn't trigger convergence."""
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000
        )

        learner.iteration_count = 100

        # Unstable model
        learner._model_snapshot_history = []

        # Low info gain
        learner._last_max_gain = 0.0001
        learner.observation_count = 1

        # Low success rate
        learner._success_history = [False] * 50

        # Should NOT converge - only 1 of 3 criteria met
        assert learner.has_converged() is False

    def test_only_high_success_insufficient(self, test_domain_file, test_problem_file):
        """Test that high success rate alone doesn't trigger convergence."""
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000
        )

        learner.iteration_count = 100

        # Unstable model
        learner._model_snapshot_history = []

        # High info gain
        learner._last_max_gain = 0.5
        learner.observation_count = 1

        # High success rate
        learner._success_history = [True] * 50

        # Should NOT converge - only 1 of 3 criteria met
        assert learner.has_converged() is False

    def test_two_criteria_insufficient(self, test_domain_file, test_problem_file):
        """Test that any 2 of 3 criteria is insufficient (requires all 3)."""
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000
        )

        learner.iteration_count = 100

        # Scenario 1: Model stable + low gain, but low success
        snapshot = {action: pre_set.copy() for action, pre_set in learner.pre.items()}
        for _ in range(50):
            learner._model_snapshot_history.append(snapshot)
        learner._last_max_gain = 0.0001
        learner.observation_count = 1
        learner._success_history = [False] * 50

        assert learner.has_converged() is False

        # Scenario 2: Model stable + high success, but high gain
        learner._success_history = [True] * 50
        learner._last_max_gain = 0.5

        assert learner.has_converged() is False

        # Scenario 3: Low gain + high success, but unstable model
        learner._model_snapshot_history = []
        learner._last_max_gain = 0.0001

        assert learner.has_converged() is False

    def test_all_three_criteria_required(self, test_domain_file, test_problem_file):
        """Test that all 3 criteria must be met for convergence."""
        learner = InformationGainLearner(
            domain_file=test_domain_file,
            problem_file=test_problem_file,
            max_iterations=1000
        )

        learner.iteration_count = 100

        # All 3 criteria met
        snapshot = {action: pre_set.copy() for action, pre_set in learner.pre.items()}
        for _ in range(50):
            learner._model_snapshot_history.append(snapshot)
        learner._last_max_gain = 0.0001
        learner.observation_count = 1
        learner._success_history = [True] * 50

        # Should converge - all 3 criteria met
        assert learner.has_converged() is True
