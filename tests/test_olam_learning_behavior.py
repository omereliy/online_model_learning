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
from src.core.domain_analyzer import DomainAnalyzer


class TestOLAMLearningBehavior:
    """Test that OLAM exhibits proper online learning behavior."""

    @pytest.fixture
    def setup_olam(self):
        """Setup OLAM adapter with OLAM-compatible gripper domain."""
        # Use gripper domain which is known to work well with OLAM
        domain = '/home/omer/projects/online_model_learning/domains/olam_compatible/gripper.pddl'
        problem = '/home/omer/projects/online_model_learning/domains/olam_compatible/gripper/1_p00_gripper_gen.pddl'

        # Verify domain compatibility
        analyzer = DomainAnalyzer(domain, problem)
        analyzer.analyze()
        assert analyzer.is_compatible_with('olam'), "Domain not compatible with OLAM"

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

            # Initially should filter 0 actions (complete uncertainty)
            assert len(non_executable) == 0, \
                f"OLAM filtered {len(non_executable)}/{total_actions} actions initially - should start with complete uncertainty!"

    def test_olam_learns_from_failures(self, setup_olam):
        """Test that OLAM learns preconditions from failed actions."""
        olam, env = setup_olam
        state = env.get_state()

        # Track learning from multiple failures
        failed_actions = []
        learned_something = False

        # Try multiple actions until we get failures and learning
        for attempt in range(20):
            action, objects = olam.select_action(state)
            success, _ = env.execute(action, objects)

            if not success:
                failed_actions.append(f"{action}({','.join(objects)})")
                # Let OLAM observe the failure
                olam.observe(state, action, objects, False, state)

                # Check if OLAM learned from this failure
                model = olam.get_learned_model()
                for action_key in failed_actions:
                    if action_key in model['actions']:
                        preconds = model['actions'][action_key].get('preconditions', {})
                        if preconds.get('uncertain', []) or preconds.get('certain', []):
                            learned_something = True
                            break

                if learned_something:
                    break
            else:
                # Success - let OLAM learn from it too
                olam.observe(state, action, objects, True, env.get_state())

        assert failed_actions, "No failures occurred in 20 attempts - cannot test failure learning"
        assert learned_something, f"OLAM didn't learn from {len(failed_actions)} failures"

    def test_olam_learns_and_filters_actions(self, setup_olam):
        """Test that OLAM learns from experience and filters actions accordingly.

        NOTE: We do NOT test success rates as they depend on domain characteristics.
        We only verify that OLAM:
        1. Starts with no filtering (complete uncertainty)
        2. Learns from observations
        3. Increases filtering over time based on learned model
        """
        olam, env = setup_olam

        # Track filtering progression (not success rate)
        initial_filtered = None
        mid_filtered = None
        final_filtered = None

        # Initial check - should have minimal or no filtering
        state = env.get_state()
        olam._update_simulator_state(olam._up_state_to_olam(state))
        initial_filtered = len(olam.learner.compute_not_executable_actionsJAVA())
        total_actions = len(olam.action_list)

        print(f"Initial: {initial_filtered}/{total_actions} actions filtered")

        # Learning phase - let OLAM observe and learn
        for i in range(50):
            state = env.get_state()
            action, objects = olam.select_action(state)
            success, _ = env.execute(action, objects)
            olam.observe(state, action, objects, success, env.get_state() if success else state)

        # Mid-point check
        state = env.get_state()
        olam._update_simulator_state(olam._up_state_to_olam(state))
        mid_filtered = len(olam.learner.compute_not_executable_actionsJAVA())
        print(f"After 50 iterations: {mid_filtered}/{total_actions} actions filtered")

        # Continue learning
        for i in range(50):
            state = env.get_state()
            action, objects = olam.select_action(state)
            success, _ = env.execute(action, objects)
            olam.observe(state, action, objects, success, env.get_state() if success else state)

        # Final check
        state = env.get_state()
        olam._update_simulator_state(olam._up_state_to_olam(state))
        final_filtered = len(olam.learner.compute_not_executable_actionsJAVA())
        print(f"After 100 iterations: {final_filtered}/{total_actions} actions filtered")

        # Verify learning progression (filtering should increase)
        assert final_filtered >= mid_filtered, \
            f"Filtering decreased: {mid_filtered} -> {final_filtered}"
        assert final_filtered >= initial_filtered, \
            f"No learning occurred: filtering stayed at {initial_filtered}"

        # Check that OLAM actually learned something
        model = olam.get_learned_model()
        actions_with_knowledge = 0
        for action_data in model['actions'].values():
            if action_data['preconditions'].get(
                    'certain') or action_data['preconditions'].get('uncertain'):
                actions_with_knowledge += 1

        assert actions_with_knowledge > 0, "OLAM didn't learn any preconditions"
        print(f"Learned preconditions for {actions_with_knowledge} actions")

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

        # Track filtering over time
        filtering_progression = []

        for iteration in range(0, 60, 10):
            # Run 10 iterations
            for i in range(10):
                state = env.get_state()
                action, objects = olam.select_action(state)
                success, _ = env.execute(action, objects)
                olam.observe(state, action, objects, success, env.get_state() if success else state)

            # Check filtering at this point
            state = env.get_state()
            olam._update_simulator_state(olam._up_state_to_olam(state))
            filtered = len(olam.learner.compute_not_executable_actionsJAVA())
            filtering_progression.append(filtered)

            print(f"Iteration {iteration+10}: {filtered}/{len(olam.action_list)} actions filtered")

        # Should show increasing trend
        assert filtering_progression[-1] > filtering_progression[0], \
            f"Filtering didn't increase over time: {filtering_progression}"

        # Should have filtered at least some actions after 60 iterations
        assert filtering_progression[-1] > 0, \
            f"No actions filtered even after 60 iterations of learning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
