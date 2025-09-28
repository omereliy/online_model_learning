"""
Test OLAM adapter's proper learning behavior.
Ensures OLAM learns from experience without accessing ground truth.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Set

from src.algorithms.olam_adapter import OLAMAdapter
from src.environments.pddl_environment import PDDLEnvironment


class TestOLAMLearningBehavior:
    """Test that OLAM exhibits proper online learning behavior."""

    @pytest.fixture
    def setup_olam(self):
        """Setup OLAM adapter with rover domain."""
        domain = '/home/omer/projects/domains/rover/domain.pddl'
        problem = '/home/omer/projects/domains/rover/pfile1.pddl'

        # Create OLAM without PDDL environment (no cheating)
        olam = OLAMAdapter(
            domain_file=domain,
            problem_file=problem,
            bypass_java=True
        )

        # Create environment separately for testing
        env = PDDLEnvironment(domain, problem)

        return olam, env

    def test_olam_starts_with_uncertainty(self, setup_olam):
        """Test that OLAM initially has high uncertainty (filters few actions)."""
        olam, env = setup_olam

        # Get initial state
        state = env.get_state()

        # Check how many actions OLAM thinks are non-executable initially
        # Should be very few since it hasn't learned yet
        if hasattr(olam, 'learner'):
            non_executable = olam.learner.compute_not_executable_actionsJAVA()
            total_actions = len(olam.action_list)

            # Initially should filter < 20% of actions (high uncertainty)
            assert len(non_executable) < total_actions * 0.2, \
                f"OLAM filtered {len(non_executable)}/{total_actions} actions initially - too certain!"

    def test_olam_learns_from_failures(self, setup_olam):
        """Test that OLAM learns preconditions from failed actions."""
        olam, env = setup_olam
        state = env.get_state()

        # Try an action that should fail
        # Pick an action that requires preconditions not met
        action = "communicate_rock_data"
        objects = ["rover0", "general", "waypoint0", "waypoint1", "waypoint2"]

        # Execute and observe failure
        success, _ = env.execute(action, objects)
        assert not success, "Expected action to fail for testing"

        # Let OLAM observe the failure
        olam.observe(state, action, objects, False, state)

        # Check that OLAM updated its model
        model = olam.get_learned_model()
        actions = model.get('actions', {})

        # Should have learned something about this action
        action_key = f"{action}({','.join(objects)})"
        if action_key in actions:
            preconds = actions[action_key].get('preconditions', {})
            # After failure, should have some uncertain preconditions
            assert preconds.get('uncertain', []) or preconds.get('certain', []), \
                "OLAM didn't learn from failure"

    def test_olam_improves_over_time(self, setup_olam):
        """Test that OLAM's success rate improves with learning."""
        olam, env = setup_olam

        successes_early = 0
        successes_late = 0
        early_iterations = 20
        late_iterations = 20

        # Early phase - should have lower success rate
        for i in range(early_iterations):
            state = env.get_state()
            action, objects = olam.select_action(state)
            success, _ = env.execute(action, objects)
            if success:
                successes_early += 1
            olam.observe(state, action, objects, success, env.get_state() if success else state)

        # Continue learning for a while
        for i in range(100):
            state = env.get_state()
            action, objects = olam.select_action(state)
            success, _ = env.execute(action, objects)
            olam.observe(state, action, objects, success, env.get_state() if success else state)

        # Late phase - should have higher success rate
        for i in range(late_iterations):
            state = env.get_state()
            action, objects = olam.select_action(state)
            success, _ = env.execute(action, objects)
            if success:
                successes_late += 1
            olam.observe(state, action, objects, success, env.get_state() if success else state)

        early_rate = successes_early / early_iterations
        late_rate = successes_late / late_iterations

        # Success rate should improve significantly
        assert late_rate > early_rate * 1.5, \
            f"Success rate didn't improve enough: {early_rate:.1%} -> {late_rate:.1%}"

        # Late success rate should be reasonable (not perfect due to exploration)
        assert late_rate > 0.5, f"Late success rate too low: {late_rate:.1%}"
        assert late_rate < 0.95, f"Late success rate suspiciously high: {late_rate:.1%} (possible cheating?)"

    def test_no_ground_truth_access(self, setup_olam):
        """Test that OLAM doesn't access ground truth for action filtering."""
        olam, env = setup_olam

        # OLAM should not have pddl_environment attribute
        assert not hasattr(olam, 'pddl_environment') or olam.pddl_environment is None, \
            "OLAM has access to PDDL environment (ground truth)"

        # Check that bypass doesn't use environment
        if hasattr(olam.learner, 'compute_not_executable_actionsJAVA'):
            # Call the bypass method
            state = env.get_state()
            olam._update_simulator_state(olam._up_state_to_olam(state))
            non_executable = olam.learner.compute_not_executable_actionsJAVA()

            # If it returns many non-executable actions initially, it's cheating
            total_actions = len(olam.action_list)
            assert len(non_executable) < total_actions * 0.5, \
                f"Bypass returned {len(non_executable)}/{total_actions} non-executable - using ground truth?"

    def test_filtering_increases_with_learning(self, setup_olam):
        """Test that action filtering increases as OLAM learns."""
        olam, env = setup_olam

        # Get initial filtering
        state = env.get_state()
        olam._update_simulator_state(olam._up_state_to_olam(state))
        initial_filtered = len(olam.learner.compute_not_executable_actionsJAVA())

        # Learn for a while
        for i in range(50):
            state = env.get_state()
            action, objects = olam.select_action(state)
            success, _ = env.execute(action, objects)
            olam.observe(state, action, objects, success, env.get_state() if success else state)

        # Get filtering after learning
        state = env.get_state()
        olam._update_simulator_state(olam._up_state_to_olam(state))
        later_filtered = len(olam.learner.compute_not_executable_actionsJAVA())

        # Should filter more actions after learning
        assert later_filtered > initial_filtered, \
            f"Filtering didn't increase: {initial_filtered} -> {later_filtered}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])