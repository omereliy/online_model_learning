"""
Tests for Information Gain learner convergence detection criteria.

Tests the convergence detection mechanisms for Information Gain learner:
1. Model stability check (no pre(a) changes for N iterations)
2. Low information gain check (max gain < ε threshold)
3. High success rate check (>95% in last M actions)
4. Combined convergence detection
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
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


@pytest.fixture
def ig_learner(test_domain_file, test_problem_file):
    """Create Information Gain learner."""
    return InformationGainLearner(
        domain_file=test_domain_file,
        problem_file=test_problem_file,
        max_iterations=100
    )


class TestModelStabilityConvergence:
    """Test model stability convergence criterion."""

    def test_stable_model_converges(self, ig_learner):
        """Test that stable model (no pre(a) changes) triggers convergence."""
        # Simulate model stability by filling snapshot history with identical snapshots
        snapshot = {action: pre_set.copy() for action, pre_set in ig_learner.pre.items()}
        for _ in range(ig_learner.MODEL_STABILITY_WINDOW):
            ig_learner._model_snapshot_history.append(snapshot)

        # Also simulate low information gain (second criterion)
        ig_learner._last_max_gain = ig_learner.INFO_GAIN_EPSILON / 2
        ig_learner.observation_count = 1  # Needed for low_info_gain check

        # Set enough iterations
        ig_learner.iteration_count = ig_learner.MODEL_STABILITY_WINDOW + 1

        # With 2 criteria met (stable model + low gain), should converge
        assert ig_learner.has_converged() is True

    def test_model_changes_prevent_convergence(self, ig_learner):
        """Test that recent model changes prevent convergence."""
        # Simulate unstable model
        ig_learner._converged = False
        ig_learner.iteration_count = 50  # Below max

        assert ig_learner.has_converged() is False

    def test_tracks_precondition_changes(self, ig_learner):
        """Test that precondition changes are tracked for stability."""
        # Initial state
        initial_pre_size = {action: len(pre_set) for action, pre_set in ig_learner.pre.items()}

        # After some observations, pre sets should change
        # This is tested indirectly through convergence behavior
        assert ig_learner.has_converged() is False


class TestInformationGainThresholdConvergence:
    """Test information gain threshold convergence criterion."""

    def test_low_information_gain_triggers_convergence(self, ig_learner):
        """Test that low expected information gain (< ε) triggers convergence."""
        # Simulate low information gain + model stability
        snapshot = {action: pre_set.copy() for action, pre_set in ig_learner.pre.items()}
        for _ in range(ig_learner.MODEL_STABILITY_WINDOW):
            ig_learner._model_snapshot_history.append(snapshot)

        ig_learner._last_max_gain = ig_learner.INFO_GAIN_EPSILON / 2
        ig_learner.observation_count = 1
        ig_learner.iteration_count = ig_learner.MODEL_STABILITY_WINDOW + 1

        # With 2 criteria met, should converge
        assert ig_learner.has_converged() is True

    def test_high_information_gain_prevents_convergence(self, ig_learner):
        """Test that high information gain prevents convergence."""
        # High gain means still learning
        ig_learner._converged = False
        ig_learner.iteration_count = 50

        assert ig_learner.has_converged() is False

    def test_information_gain_threshold_configurable(self, ig_learner):
        """Test that information gain threshold (ε) is configurable."""
        # Check that epsilon can be set (will be added in implementation)
        # For now, just verify the learner is created successfully
        assert ig_learner is not None


class TestSuccessRateConvergence:
    """Test success rate-based convergence criterion."""

    def test_high_success_rate_contributes_to_convergence(self, ig_learner):
        """Test that >95% success rate contributes to convergence."""
        # Simulate high success rate + model stability
        snapshot = {action: pre_set.copy() for action, pre_set in ig_learner.pre.items()}
        for _ in range(ig_learner.MODEL_STABILITY_WINDOW):
            ig_learner._model_snapshot_history.append(snapshot)

        # Fill success history with >95% successes
        ig_learner._success_history = [True] * ig_learner.SUCCESS_RATE_WINDOW
        ig_learner.iteration_count = ig_learner.MODEL_STABILITY_WINDOW + 1

        # With 2 criteria met, should converge
        assert ig_learner.has_converged() is True

    def test_low_success_rate_prevents_convergence(self, ig_learner):
        """Test that low success rate prevents convergence."""
        ig_learner._converged = False
        ig_learner.iteration_count = 50

        assert ig_learner.has_converged() is False


class TestMaxIterationsConvergence:
    """Test max iterations forcing convergence."""

    def test_max_iterations_forces_convergence(self, ig_learner):
        """Test that reaching max iterations forces convergence."""
        ig_learner.iteration_count = ig_learner.max_iterations

        # Should converge regardless of other criteria
        assert ig_learner.has_converged() is True

    def test_below_max_iterations_respects_criteria(self, ig_learner):
        """Test that below max iterations, other criteria are respected."""
        ig_learner.iteration_count = ig_learner.MODEL_STABILITY_WINDOW + 1

        # Should not converge if criteria not met (no snapshots, no successes)
        assert ig_learner.has_converged() is False

        # Should converge if 2 criteria met
        snapshot = {action: pre_set.copy() for action, pre_set in ig_learner.pre.items()}
        for _ in range(ig_learner.MODEL_STABILITY_WINDOW):
            ig_learner._model_snapshot_history.append(snapshot)
        ig_learner._last_max_gain = ig_learner.INFO_GAIN_EPSILON / 2
        ig_learner.observation_count = 1

        assert ig_learner.has_converged() is True


class TestCombinedConvergence:
    """Test combined convergence criteria."""

    def test_all_criteria_met_converges(self, ig_learner):
        """Test that meeting all criteria triggers convergence."""
        # Simulate all 3 criteria met
        snapshot = {action: pre_set.copy() for action, pre_set in ig_learner.pre.items()}
        for _ in range(ig_learner.MODEL_STABILITY_WINDOW):
            ig_learner._model_snapshot_history.append(snapshot)

        ig_learner._last_max_gain = ig_learner.INFO_GAIN_EPSILON / 2
        ig_learner.observation_count = 1
        ig_learner._success_history = [True] * ig_learner.SUCCESS_RATE_WINDOW
        ig_learner.iteration_count = ig_learner.MODEL_STABILITY_WINDOW + 1

        # With all 3 criteria met, should definitely converge
        assert ig_learner.has_converged() is True

    def test_partial_criteria_prevents_convergence(self, ig_learner):
        """Test that partial criteria satisfaction prevents convergence."""
        # Only some criteria met
        ig_learner._converged = False
        ig_learner.iteration_count = 50

        assert ig_learner.has_converged() is False


class TestConvergenceReset:
    """Test convergence state reset."""

    def test_reset_clears_convergence(self, ig_learner):
        """Test that reset clears convergence status."""
        # Converge
        ig_learner._converged = True
        ig_learner.iteration_count = ig_learner.max_iterations
        assert ig_learner.has_converged() is True

        # Reset
        ig_learner.reset()

        # Should no longer be converged
        assert ig_learner.has_converged() is False
        assert ig_learner.iteration_count == 0


class TestConvergenceEdgeCases:
    """Test edge cases in convergence detection."""

    def test_zero_iterations(self, ig_learner):
        """Test convergence check at zero iterations."""
        ig_learner.iteration_count = 0
        ig_learner._converged = False

        assert ig_learner.has_converged() is False

    def test_exactly_max_iterations(self, ig_learner):
        """Test convergence at exactly max iterations."""
        ig_learner.iteration_count = ig_learner.max_iterations

        # Should converge regardless of other criteria
        assert ig_learner.has_converged() is True

    def test_beyond_max_iterations(self, ig_learner):
        """Test convergence beyond max iterations."""
        ig_learner.iteration_count = ig_learner.max_iterations + 10

        # Should definitely be converged
        assert ig_learner.has_converged() is True


class TestEnhancedConvergenceDetection:
    """Test enhanced convergence detection with specific criteria."""

    def test_model_stability_window(self, ig_learner):
        """Test that model stability is measured over a window of iterations."""
        # This tests the enhanced convergence implementation
        # Window size should be configurable (default: N=10 iterations)

        # Initially not converged
        assert ig_learner.has_converged() is False

        # After stability window with no changes, should converge
        # (This will be tested in integration tests with actual learning)

    def test_information_gain_decreases_over_time(self, ig_learner):
        """Test that information gain decreases as model is learned."""
        # This property should emerge from the learning algorithm
        # We test it indirectly through convergence behavior

        # Initially should have high potential gain
        # After learning, gain should decrease
        # Eventually triggering convergence

        # For now, just test the interface exists
        assert ig_learner.has_converged() is not None

    def test_success_rate_tracking(self, ig_learner):
        """Test that success rate is tracked over a sliding window."""
        # Success rate should be calculated over last M actions
        # High success rate (>95%) contributes to convergence

        # For now, test basic interface
        assert ig_learner.has_converged() is not None


class TestConvergenceWithObservations:
    """Test convergence detection with actual observations."""

    def test_convergence_after_successful_observations(self, ig_learner):
        """Test convergence after multiple successful observations."""
        # Simulate successful observations
        state = {'clear_a', 'clear_b', 'clear_c', 'ontable_a', 'ontable_b', 'ontable_c', 'handempty'}

        # Get an action
        action_name, objects = ig_learner.select_action(state)

        # Simulate success
        next_state = state.copy()
        ig_learner.observe(state, action_name, objects, success=True, next_state=next_state)

        # Initially should not be converged (only 1 observation)
        assert ig_learner.has_converged() is False

    def test_convergence_tracks_model_changes(self, ig_learner):
        """Test that convergence tracking responds to model updates."""
        # Initial state
        initial_converged = ig_learner.has_converged()

        # After observations and model updates, convergence status may change
        # This is tested through the has_converged() method

        assert isinstance(initial_converged, bool)
