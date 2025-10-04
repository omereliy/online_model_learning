"""
Test suite for PDDL environment implementation.
Tests real PDDL action execution using Unified Planning Framework.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Set, Tuple

from src.environments.pddl_environment import PDDLEnvironment


class TestPDDLEnvironment:
    """Test suite for PDDL environment functionality."""

    @pytest.fixture
    def simple_blocksworld_domain(self) -> str:
        """Create a simple blocksworld domain for testing."""
        return """
(define (domain blocksworld)
  (:requirements :strips :typing)
  (:types block)
  (:predicates
    (on ?x - block ?y - block)
    (ontable ?x - block)
    (clear ?x - block)
    (handempty)
    (holding ?x - block))

  (:action pick-up
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x)))

  (:action put-down
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x)))

  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y)) (clear ?x) (handempty) (on ?x ?y)))

  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y)))))
"""

    @pytest.fixture
    def simple_blocksworld_problem(self) -> str:
        """Create a simple blocksworld problem for testing."""
        return """
(define (problem blocks-3)
  (:domain blocksworld)
  (:objects a b c - block)
  (:init
    (on b a)
    (on c b)
    (clear c)
    (ontable a)
    (handempty))
  (:goal (and (on a b) (on b c))))
"""

    @pytest.fixture
    def temp_pddl_files(self, simple_blocksworld_domain, simple_blocksworld_problem):
        """Create temporary PDDL files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_file = Path(tmpdir) / "domain.pddl"
            problem_file = Path(tmpdir) / "problem.pddl"

            domain_file.write_text(simple_blocksworld_domain)
            problem_file.write_text(simple_blocksworld_problem)

            yield str(domain_file), str(problem_file)

    def test_environment_initialization(self, temp_pddl_files):
        """Test that environment can be initialized with PDDL files."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        assert env is not None
        assert env.domain_file == domain_file
        assert env.problem_file == problem_file
        assert env.current_state is not None

    def test_get_initial_state(self, temp_pddl_files):
        """Test getting initial state in correct format."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        state = env.get_state()

        # Should return set of fluent strings
        assert isinstance(state, set)
        assert len(state) > 0

        # Check expected initial state fluents
        expected_fluents = {'on_b_a', 'on_c_b', 'clear_c', 'ontable_a', 'handempty'}
        assert state == expected_fluents

    def test_execute_valid_action(self, temp_pddl_files):
        """Test executing a valid action with satisfied preconditions."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        initial_state = env.get_state()

        # Execute unstack(c, b) - should be valid in initial state
        success, runtime = env.execute('unstack', ['c', 'b'])

        assert success is True
        assert runtime > 0

        # Check state changed correctly
        new_state = env.get_state()
        assert new_state != initial_state
        assert 'holding_c' in new_state
        assert 'clear_b' in new_state
        assert 'on_c_b' not in new_state
        assert 'handempty' not in new_state

    def test_execute_invalid_action(self, temp_pddl_files):
        """Test executing action with unsatisfied preconditions."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        initial_state = env.get_state()

        # Try to pick-up a block that's not clear
        success, runtime = env.execute('pick-up', ['a'])

        assert success is False
        assert runtime > 0

        # State should not change on failed action
        new_state = env.get_state()
        assert new_state == initial_state

    def test_action_sequence(self, temp_pddl_files):
        """Test executing a sequence of actions."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        # Execute sequence: unstack(c,b), put-down(c), unstack(b,a)
        success1, _ = env.execute('unstack', ['c', 'b'])
        assert success1 is True
        assert 'holding_c' in env.get_state()

        success2, _ = env.execute('put-down', ['c'])
        assert success2 is True
        assert 'ontable_c' in env.get_state()
        assert 'clear_c' in env.get_state()

        success3, _ = env.execute('unstack', ['b', 'a'])
        assert success3 is True
        assert 'holding_b' in env.get_state()
        assert 'clear_a' in env.get_state()

    def test_reset_to_initial_state(self, temp_pddl_files):
        """Test resetting environment to initial state."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        initial_state = env.get_state()

        # Execute some actions
        env.execute('unstack', ['c', 'b'])
        env.execute('put-down', ['c'])

        modified_state = env.get_state()
        assert modified_state != initial_state

        # Reset
        env.reset()

        reset_state = env.get_state()
        assert reset_state == initial_state

    def test_get_applicable_actions(self, temp_pddl_files):
        """Test getting list of applicable actions in current state."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        applicable = env.get_applicable_actions()

        # Should return list of (action, parameters) tuples
        assert isinstance(applicable, list)
        assert len(applicable) > 0

        # In initial state, should be able to unstack(c,b)
        assert ('unstack', ['c', 'b']) in applicable

        # Should NOT be able to pick-up(a) since it's not clear
        assert ('pick-up', ['a']) not in applicable

    def test_state_format_consistency(self, temp_pddl_files):
        """Test that state format is consistent with OLAM adapter expectations."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        state = env.get_state()

        # All fluents should be strings in format: predicate_param1_param2
        for fluent in state:
            assert isinstance(fluent, str)
            # Check format (predicate and params separated by underscore)
            parts = fluent.split('_')
            assert len(parts) >= 1  # At least predicate name

    def test_goal_checking(self, temp_pddl_files):
        """Test checking if goal is reached."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        # Initially goal should not be satisfied
        assert env.is_goal_reached() is False

        # Execute actions to reach goal
        # Goal: (on a b) (on b c)
        # Need: unstack(c,b), put-down(c), unstack(b,a), stack(b,c), stack(a,b)
        env.execute('unstack', ['c', 'b'])
        env.execute('put-down', ['c'])
        env.execute('unstack', ['b', 'a'])
        env.execute('stack', ['b', 'c'])
        env.execute('pick-up', ['a'])
        env.execute('stack', ['a', 'b'])

        # Now goal should be satisfied
        assert env.is_goal_reached() is True

    def test_get_all_grounded_actions(self, temp_pddl_files):
        """Test getting all grounded actions for the domain."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        all_actions = env.get_all_grounded_actions()

        # Should have all combinations of actions and objects
        assert isinstance(all_actions, list)
        assert len(all_actions) > 0

        # Check some expected actions exist
        assert ('pick-up', ['a']) in all_actions
        assert ('pick-up', ['b']) in all_actions
        assert ('pick-up', ['c']) in all_actions
        assert ('stack', ['a', 'b']) in all_actions
        assert ('stack', ['a', 'c']) in all_actions

        # Should not have invalid combinations (can't stack object on itself)
        assert ('stack', ['a', 'a']) not in all_actions

    def test_state_changes_tracking(self, temp_pddl_files):
        """Test that environment properly tracks state changes."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        # Track state history
        state_history = [env.get_state()]

        # Execute several actions
        actions = [
            ('unstack', ['c', 'b']),
            ('put-down', ['c']),
            ('unstack', ['b', 'a']),
            ('put-down', ['b'])
        ]

        for action, params in actions:
            success, _ = env.execute(action, params)
            if success:
                state_history.append(env.get_state())

        # All states should be different
        for i, state1 in enumerate(state_history):
            for j, state2 in enumerate(state_history):
                if i != j:
                    assert state1 != state2, f"States {i} and {j} should be different"

    def test_handle_nonexistent_action(self, temp_pddl_files):
        """Test handling of non-existent action names."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        # Try to execute non-existent action
        success, runtime = env.execute('fly', ['a', 'b'])

        assert success is False
        assert runtime >= 0

    def test_handle_wrong_parameters(self, temp_pddl_files):
        """Test handling of wrong parameter count or types."""
        domain_file, problem_file = temp_pddl_files
        env = PDDLEnvironment(domain_file, problem_file)

        # Wrong number of parameters for pick-up (expects 1, giving 2)
        success, runtime = env.execute('pick-up', ['a', 'b'])

        assert success is False
        assert runtime >= 0

        # Non-existent object
        success, runtime = env.execute('pick-up', ['z'])

        assert success is False
        assert runtime >= 0


class TestPDDLEnvironmentWithRover:
    """Test PDDL environment with more complex Rover domain."""

    @pytest.fixture
    def rover_files(self):
        """Use actual rover domain files."""
        domain_file = "/home/omer/projects/domains/rover/domain.pddl"
        problem_file = "/home/omer/projects/domains/rover/pfile1.pddl"

        # Check if files exist, skip tests if not
        if not Path(domain_file).exists() or not Path(problem_file).exists():
            pytest.skip("Rover domain files not found")

        return domain_file, problem_file

    def test_rover_domain_initialization(self, rover_files):
        """Test initializing environment with complex Rover domain."""
        domain_file, problem_file = rover_files
        env = PDDLEnvironment(domain_file, problem_file)

        assert env is not None
        state = env.get_state()

        # Rover domain should have many fluents
        assert len(state) > 10

        # Check some expected rover predicates
        state_strings = list(state)
        assert any('at_' in s for s in state_strings)  # at predicates
        assert any('empty' in s for s in state_strings)  # empty store

    def test_rover_action_execution(self, rover_files):
        """Test executing rover-specific actions."""
        domain_file, problem_file = rover_files
        env = PDDLEnvironment(domain_file, problem_file)

        # Get applicable actions
        applicable = env.get_applicable_actions()

        # Should have some applicable actions
        assert len(applicable) > 0

        # Try to execute first applicable action
        if applicable:
            action, params = applicable[0]
            success, runtime = env.execute(action, params)

            # Should succeed since it was applicable
            assert success is True
            assert runtime > 0
