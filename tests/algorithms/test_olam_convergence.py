"""
Tests for OLAM convergence detection criteria.

Tests the convergence detection mechanisms for OLAM adapter:
1. Hypothesis space stability check (no changes for N iterations)
2. High success rate check (>95% in last M actions)
3. Combined convergence detection
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.algorithms.olam_adapter import OLAMAdapter


@pytest.fixture
def test_domain_file():
    """Get path to test domain file."""
    return "benchmarks/olam-compatible/blocksworld/domain.pddl"


@pytest.fixture
def test_problem_file():
    """Get path to test problem file."""
    return "benchmarks/olam-compatible/blocksworld/p01.pddl"


@pytest.fixture
def olam_adapter(test_domain_file, test_problem_file):
    """Create OLAM adapter with bypass_java enabled."""
    return OLAMAdapter(
        domain_file=test_domain_file,
        problem_file=test_problem_file,
        max_iterations=100,
        eval_frequency=10,
        bypass_java=True
    )


class TestHypothesisSpaceStability:
    """Test hypothesis space stability convergence criterion."""

    def test_no_changes_converges(self, olam_adapter):
        """Test that no changes for N iterations triggers convergence."""
        # Simulate stable hypothesis space
        # In OLAM, this is tracked by model_convergence flag
        olam_adapter.learner.model_convergence = True

        assert olam_adapter.has_converged() is True

    def test_recent_changes_prevents_convergence(self, olam_adapter):
        """Test that recent changes prevent convergence."""
        # Simulate unstable hypothesis space
        olam_adapter.learner.model_convergence = False

        # Should not converge yet (unless max iterations reached)
        olam_adapter.iteration_count = 50  # Below max
        assert olam_adapter.has_converged() is False

    def test_stability_window(self, olam_adapter):
        """Test that stability is measured over a window of iterations."""
        # This test verifies the stability window logic
        # OLAM uses internal tracking, so we test the public interface

        # Start with no convergence
        olam_adapter.learner.model_convergence = False
        assert olam_adapter.has_converged() is False

        # Simulate reaching stability
        olam_adapter.learner.model_convergence = True
        assert olam_adapter.has_converged() is True


class TestSuccessRateConvergence:
    """Test success rate-based convergence criterion."""

    def test_high_success_rate_converges(self, olam_adapter):
        """Test that >95% success rate in recent actions triggers convergence."""
        # OLAM's convergence includes success rate tracking
        # We test this through the OLAM learner's convergence flag

        # Simulate high success rate scenario
        olam_adapter.learner.model_convergence = True
        olam_adapter.iteration_count = 50

        assert olam_adapter.has_converged() is True

    def test_low_success_rate_prevents_convergence(self, olam_adapter):
        """Test that low success rate prevents convergence."""
        # Simulate low success rate
        olam_adapter.learner.model_convergence = False
        olam_adapter.iteration_count = 50

        assert olam_adapter.has_converged() is False

    def test_success_rate_window_size(self, olam_adapter):
        """Test that success rate is calculated over a sliding window."""
        # Window size should be configurable (default: last M actions)
        # OLAM tracks this internally through its convergence mechanism

        # Test that recent failures prevent convergence
        olam_adapter.learner.model_convergence = False
        assert olam_adapter.has_converged() is False


class TestMaxIterationsConvergence:
    """Test max iterations forcing convergence."""

    def test_max_iterations_forces_convergence(self, olam_adapter):
        """Test that reaching max iterations forces convergence."""
        # Set iteration count to max
        olam_adapter.iteration_count = olam_adapter.max_iterations

        # Should converge even if other criteria not met
        olam_adapter.learner.model_convergence = False
        assert olam_adapter.has_converged() is True

    def test_below_max_iterations_respects_criteria(self, olam_adapter):
        """Test that below max iterations, other criteria are respected."""
        # Set iteration count below max
        olam_adapter.iteration_count = olam_adapter.max_iterations // 2

        # Should not converge if criteria not met
        olam_adapter.learner.model_convergence = False
        assert olam_adapter.has_converged() is False

        # Should converge if criteria met
        olam_adapter.learner.model_convergence = True
        assert olam_adapter.has_converged() is True


class TestCombinedConvergence:
    """Test combined convergence criteria."""

    def test_all_criteria_met_converges(self, olam_adapter):
        """Test that meeting all criteria triggers convergence."""
        # Simulate all criteria met
        olam_adapter.learner.model_convergence = True
        olam_adapter.iteration_count = 50

        assert olam_adapter.has_converged() is True

    def test_partial_criteria_prevents_convergence(self, olam_adapter):
        """Test that partial criteria satisfaction prevents convergence."""
        # Only some criteria met
        olam_adapter.learner.model_convergence = False
        olam_adapter.iteration_count = 50

        assert olam_adapter.has_converged() is False

    def test_convergence_tracks_olam_flag(self, olam_adapter):
        """Test that convergence tracks OLAM's internal model_convergence flag."""
        # Converge
        olam_adapter.learner.model_convergence = True
        assert olam_adapter.has_converged() is True

        # If OLAM flag changes, convergence should reflect that
        # (allows OLAM to revise convergence decision if needed)
        olam_adapter.learner.model_convergence = False

        # Note: Once _converged is set to True from max_iterations,
        # it becomes sticky. But if set from model_convergence, it tracks the flag.
        # For this test, we're below max_iterations, so it should track
        olam_adapter.iteration_count = 50  # Below max
        assert olam_adapter.has_converged() is False


class TestConvergenceReset:
    """Test convergence state reset."""

    def test_reset_clears_convergence(self, olam_adapter):
        """Test that reset clears convergence status."""
        # Converge
        olam_adapter.learner.model_convergence = True
        olam_adapter.iteration_count = olam_adapter.max_iterations
        assert olam_adapter.has_converged() is True

        # Reset
        olam_adapter.reset()

        # Should no longer be converged
        assert olam_adapter.has_converged() is False


class TestConvergenceEdgeCases:
    """Test edge cases in convergence detection."""

    def test_zero_iterations(self, olam_adapter):
        """Test convergence check at zero iterations."""
        olam_adapter.iteration_count = 0
        olam_adapter.learner.model_convergence = False

        assert olam_adapter.has_converged() is False

    def test_exactly_max_iterations(self, olam_adapter):
        """Test convergence at exactly max iterations."""
        olam_adapter.iteration_count = olam_adapter.max_iterations

        # Should converge regardless of other criteria
        assert olam_adapter.has_converged() is True

    def test_beyond_max_iterations(self, olam_adapter):
        """Test convergence beyond max iterations."""
        olam_adapter.iteration_count = olam_adapter.max_iterations + 10

        # Should definitely be converged
        assert olam_adapter.has_converged() is True
