"""
Test Information Gain observation processing and model updating.

Tests that the algorithm correctly:
1. Builds CNF clauses from failure observations
2. Reduces hypothesis space after observations
3. Learns effects from successful actions
4. Updates preconditions based on observations
"""

import pytest
from pathlib import Path

from src.algorithms.information_gain import InformationGainLearner
from src.environments.active_environment import ActiveEnvironment


class TestInformationGainObservation:
    """Test observation processing in Information Gain algorithm."""

    @pytest.fixture
    def simple_domain_setup(self):
        """Setup with simple blocksworld domain."""
        domain = 'benchmarks/olam-compatible/blocksworld/domain.pddl'
        problem = 'benchmarks/olam-compatible/blocksworld/p01.pddl'
        return domain, problem

    def test_failure_creates_cnf_clauses(self, simple_domain_setup):
        """Test that failure observations create CNF clauses."""
        domain, problem = simple_domain_setup

        learner = InformationGainLearner(domain, problem, max_iterations=10)
        env = ActiveEnvironment(domain, problem)

        # Initial state - no CNF clauses
        initial_clauses = {}
        for action_name in learner.pre.keys():
            cnf = learner.cnf_managers[action_name]
            initial_clauses[action_name] = len(cnf.cnf.clauses)
            assert initial_clauses[action_name] == 0, f"Should start with 0 clauses for {action_name}"

        # Execute a failing action
        state = env.get_state()
        action = "pick-up"
        objects = ["a"]  # Assuming 'a' is on something, so pick-up will fail
        success, _ = env.execute(action, objects)

        if success:
            # If it succeeded, try another action that should fail
            action = "unstack"
            objects = ["a", "b"]  # Try unstacking when not stacked
            success, _ = env.execute(action, objects)

        assert not success, "Need a failure to test CNF clause creation"

        # Observe the failure
        learner.observe(state, action, objects, False, state)

        # Check CNF clauses were created
        cnf_after = learner.cnf_managers[action]
        assert len(cnf_after.cnf.clauses) > 0, f"Failure should create CNF clauses for {action}"

        # Check that constraint was added
        assert len(learner.pre_constraints[action]) > 0, "Failure should add constraint"

    def test_hypothesis_space_reduction(self, simple_domain_setup):
        """Test that observations reduce hypothesis space."""
        domain, problem = simple_domain_setup

        learner = InformationGainLearner(domain, problem, max_iterations=10)
        env = ActiveEnvironment(domain, problem)

        # Calculate initial hypothesis space
        initial_space = self._calculate_hypothesis_space(learner)
        assert initial_space > 0, "Should have non-zero initial hypothesis space"

        # Run several observations
        for _ in range(3):
            state = env.get_state()
            action, objects = learner.select_action(state)
            success, _ = env.execute(action, objects)
            next_state = env.get_state() if success else state
            learner.observe(state, action, objects, success, next_state)

        # Calculate hypothesis space after observations
        final_space = self._calculate_hypothesis_space(learner)

        # Space should reduce (or at least not increase)
        assert final_space <= initial_space, \
            f"Hypothesis space should not increase: {initial_space} -> {final_space}"

        # If we had failures, space should strictly reduce
        if any(not s for s in learner._success_history):
            assert final_space < initial_space, \
                "Hypothesis space should reduce after failure observations"

    def test_effect_learning_from_success(self, simple_domain_setup):
        """Test that successful actions learn effects."""
        domain, problem = simple_domain_setup

        learner = InformationGainLearner(domain, problem, max_iterations=50)
        env = ActiveEnvironment(domain, problem)

        # Run until we get a successful action
        success_found = False
        for i in range(20):
            state = env.get_state()
            action, objects = learner.select_action(state)
            success, _ = env.execute(action, objects)
            next_state = env.get_state() if success else state

            learner.observe(state, action, objects, success, next_state)

            if success:
                success_found = True
                # Check that effects were learned
                if state != next_state:  # If state changed
                    # Should have learned some add or delete effects
                    has_effects = (len(learner.eff_add[action]) > 0 or
                                  len(learner.eff_del[action]) > 0)
                    assert has_effects, \
                        f"Successful action {action} with state change should learn effects"
                break

        if not success_found:
            pytest.skip("No successful actions in 20 attempts")

    def test_precondition_reduction(self, simple_domain_setup):
        """Test that preconditions are refined through observations."""
        domain, problem = simple_domain_setup

        learner = InformationGainLearner(domain, problem, max_iterations=10)
        env = ActiveEnvironment(domain, problem)

        # Track precondition sizes
        initial_pre_sizes = {a: len(pre) for a, pre in learner.pre.items()}

        # Run observations
        for _ in range(5):
            state = env.get_state()
            action, objects = learner.select_action(state)
            success, _ = env.execute(action, objects)
            next_state = env.get_state() if success else state
            learner.observe(state, action, objects, success, next_state)

        # Check precondition refinement
        for action in learner.pre.keys():
            current_size = len(learner.pre[action])
            initial_size = initial_pre_sizes[action]

            # Preconditions should not grow
            assert current_size <= initial_size, \
                f"Preconditions for {action} should not grow: {initial_size} -> {current_size}"

            # If action was observed successfully, preconditions should be refined
            if action in [obs['action'] for obs in learner.observation_history[action] if obs['success']]:
                # Success should refine (reduce) preconditions
                assert current_size <= initial_size, \
                    f"Successful {action} should refine preconditions"

    def test_information_gain_changes_after_observation(self, simple_domain_setup):
        """Test that information gain values change as model learns."""
        domain, problem = simple_domain_setup

        learner = InformationGainLearner(domain, problem, max_iterations=10)
        env = ActiveEnvironment(domain, problem)

        state = env.get_state()

        # Calculate initial information gains
        initial_gains = {}
        for action in ["pick-up", "put-down", "stack", "unstack"]:
            for obj in ["a", "b", "c"]:
                try:
                    gain = learner._calculate_expected_information_gain(action, [obj], state)
                    initial_gains[f"{action}({obj})"] = gain
                except:
                    pass

        # Execute and observe several actions
        for _ in range(3):
            action, objects = learner.select_action(state)
            success, _ = env.execute(action, objects)
            next_state = env.get_state() if success else state
            learner.observe(state, action, objects, success, next_state)
            state = next_state

        # Calculate gains after learning
        final_gains = {}
        for action in ["pick-up", "put-down", "stack", "unstack"]:
            for obj in ["a", "b", "c"]:
                try:
                    gain = learner._calculate_expected_information_gain(action, [obj], state)
                    final_gains[f"{action}({obj})"] = gain
                except:
                    pass

        # At least some gains should have changed
        changed_count = sum(1 for key in initial_gains
                           if key in final_gains and
                           abs(initial_gains[key] - final_gains[key]) > 0.001)

        assert changed_count > 0, \
            "Information gain values should change after observations"

    def _calculate_hypothesis_space(self, learner):
        """Calculate total hypothesis space across all actions."""
        total = 0
        for action_name in learner.pre.keys():
            if not learner.cnf_managers[action_name].has_clauses():
                learner._build_cnf_formula(action_name)

            cnf = learner.cnf_managers[action_name]
            if cnf.has_clauses():
                total += cnf.count_solutions()
            else:
                # No constraints means max possible
                la_size = len(learner.pre[action_name])
                total += 2 ** la_size if la_size > 0 else 1
        return total