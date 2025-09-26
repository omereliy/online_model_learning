"""
Comprehensive test suite for OLAM adapter implementation.
Tests all aspects of OLAM integration following TDD approach.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
from typing import Set, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.base_learner import BaseActionModelLearner


class TestOLAMAdapter:
    """Test suite for OLAM adapter implementation."""

    # ========== Test Category A: Basic Functionality Tests ==========

    def test_olam_import(self):
        """Test that OLAM can be imported successfully."""
        # This will be implemented when OLAM adapter is created
        # For now, we test that we can add OLAM path
        olam_path = '/home/omer/projects/OLAM'
        assert os.path.exists(olam_path), "OLAM repository not found"

        # Test that we can add to sys.path
        if olam_path not in sys.path:
            sys.path.append(olam_path)
        assert olam_path in sys.path

    def test_base_interface_compliance(self):
        """Test that OLAM adapter implements BaseActionModelLearner interface."""
        # This will verify that OLAMAdapter properly inherits from BaseActionModelLearner
        # and implements all required methods
        from src.algorithms.base_learner import BaseActionModelLearner

        # When OLAMAdapter is implemented, we'll test:
        # assert issubclass(OLAMAdapter, BaseActionModelLearner)
        # And verify all abstract methods are implemented

        # For now, test that base class exists and has required methods
        required_methods = ['select_action', 'observe', 'get_learned_model']
        for method in required_methods:
            assert hasattr(BaseActionModelLearner, method)

    @pytest.fixture
    def sample_domain_problem(self, temp_dir, blocksworld_domain, blocksworld_problem):
        """Create temporary PDDL files for testing."""
        domain_file = temp_dir / "domain.pddl"
        problem_file = temp_dir / "problem.pddl"

        domain_file.write_text(blocksworld_domain)
        problem_file.write_text(blocksworld_problem)

        return str(domain_file), str(problem_file)

    def test_initialization(self, sample_domain_problem):
        """Test OLAM adapter initialization with domain and problem files."""
        domain_file, problem_file = sample_domain_problem

        # When OLAMAdapter is implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # assert adapter.domain_file == domain_file
        # assert adapter.problem_file == problem_file
        # assert adapter.learner is not None  # OLAM Learner initialized
        # assert adapter.action_list is not None  # Actions extracted

        # For now, test file existence
        assert os.path.exists(domain_file)
        assert os.path.exists(problem_file)

    # ========== Test Category B: State Format Conversion Tests ==========

    def test_up_state_to_olam_empty_state(self):
        """Test conversion of empty UP state to OLAM format."""
        # UP state: empty set of fluents
        up_state = set()

        # Expected OLAM format: empty list
        expected_olam = []

        # When implemented:
        # adapter = OLAMAdapter(...)
        # result = adapter._up_state_to_olam(up_state)
        # assert result == expected_olam

    def test_up_state_to_olam_simple_state(self):
        """Test conversion of simple UP state to OLAM format."""
        # UP state: set of true fluents
        up_state = {'clear_a', 'on_a_b', 'ontable_b', 'handempty'}

        # Expected OLAM format: PDDL strings
        expected_olam = [
            '(clear a)',
            '(handempty)',
            '(on a b)',
            '(ontable b)'
        ]

        # When implemented:
        # adapter = OLAMAdapter(...)
        # result = adapter._up_state_to_olam(up_state)
        # assert sorted(result) == sorted(expected_olam)

    def test_olam_state_to_up(self):
        """Test conversion of OLAM state back to UP format."""
        # OLAM state: list of PDDL predicate strings
        olam_state = ['(clear a)', '(on a b)', '(ontable b)', '(handempty)']

        # Expected UP format: set of fluent strings
        expected_up = {'clear_a', 'on_a_b', 'ontable_b', 'handempty'}

        # When implemented:
        # adapter = OLAMAdapter(...)
        # result = adapter._olam_state_to_up(olam_state)
        # assert result == expected_up

    # ========== Test Category C: Action Format Conversion Tests ==========

    def test_up_action_to_olam_no_params(self):
        """Test conversion of parameterless UP action to OLAM format."""
        action_name = "handempty"
        objects = []

        expected_olam = "handempty()"

        # When implemented:
        # adapter = OLAMAdapter(...)
        # result = adapter._up_action_to_olam(action_name, objects)
        # assert result == expected_olam

    def test_up_action_to_olam_single_param(self):
        """Test conversion of single parameter UP action to OLAM format."""
        action_name = "pick-up"
        objects = ["a"]

        expected_olam = "pick-up(a)"

        # When implemented:
        # adapter = OLAMAdapter(...)
        # result = adapter._up_action_to_olam(action_name, objects)
        # assert result == expected_olam

    def test_up_action_to_olam_multiple_params(self):
        """Test conversion of multi-parameter UP action to OLAM format."""
        action_name = "stack"
        objects = ["a", "b"]

        expected_olam = "stack(a,b)"

        # When implemented:
        # adapter = OLAMAdapter(...)
        # result = adapter._up_action_to_olam(action_name, objects)
        # assert result == expected_olam

    def test_olam_action_to_up(self):
        """Test conversion of OLAM action string back to UP format."""
        test_cases = [
            ("pick-up(a)", ("pick-up", ["a"])),
            ("stack(a,b)", ("stack", ["a", "b"])),
            ("handempty()", ("handempty", []))
        ]

        for olam_action, expected in test_cases:
            # When implemented:
            # adapter = OLAMAdapter(...)
            # result = adapter._olam_action_to_up(olam_action)
            # assert result == expected
            pass

    # ========== Test Category D: Action Selection Tests ==========

    def test_action_selection_returns_valid_action(self):
        """Test that action selection returns a valid action."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # state = {'clear_a', 'ontable_a', 'handempty'}
        # action, objects = adapter.select_action(state)
        # assert isinstance(action, str)
        # assert isinstance(objects, list)
        # assert action in ['pick-up', 'put-down', 'stack', 'unstack']
        pass

    def test_action_selection_exploration(self):
        """Test that action selection explores different actions."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # state = {'clear_a', 'clear_b', 'ontable_a', 'ontable_b', 'handempty'}
        #
        # actions_selected = []
        # for _ in range(10):
        #     action, objects = adapter.select_action(state)
        #     actions_selected.append((action, tuple(objects)))
        #
        # # Should explore multiple actions
        # unique_actions = set(actions_selected)
        # assert len(unique_actions) > 1
        pass

    def test_action_selection_empty_state(self):
        """Test action selection with empty state."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # state = set()  # Empty state
        # action, objects = adapter.select_action(state)
        # # Should still return some action (exploration)
        # assert action is not None
        pass

    def test_action_selection_handles_no_applicable_actions(self):
        """Test action selection when no actions are applicable."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # # State where nothing can be done (contrived)
        # state = {'holding_a', 'holding_b'}  # Can't hold two things
        # action, objects = adapter.select_action(state)
        # # Should still return something (for exploration)
        # assert action is not None
        pass

    # ========== Test Category E: Learning from Observations Tests ==========

    def test_learn_from_successful_action(self):
        """Test learning from a successful action execution."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # state = {'clear_a', 'ontable_a', 'handempty'}
        # action = 'pick-up'
        # objects = ['a']
        # next_state = {'holding_a'}
        #
        # adapter.observe(state, action, objects, success=True, next_state=next_state)
        #
        # model = adapter.get_learned_model()
        # # Should have learned something about pick-up action
        # assert 'pick-up' in model['actions']
        pass

    def test_learn_failed_action_preconditions(self):
        """Test learning from a failed action execution."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # state = {'on_a_b', 'clear_a'}  # Missing handempty for pick-up
        # action = 'pick-up'
        # objects = ['a']
        #
        # adapter.observe(state, action, objects, success=False, next_state=None)
        #
        # model = adapter.get_learned_model()
        # # Should have learned that pick-up needs handempty
        # assert 'pick-up' in model['actions']
        pass

    def test_learn_action_effects(self):
        """Test learning of action effects from observations."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # state = {'clear_a', 'ontable_a', 'handempty'}
        # action = 'pick-up'
        # objects = ['a']
        # next_state = {'holding_a'}
        #
        # adapter.observe(state, action, objects, success=True, next_state=next_state)
        #
        # model = adapter.get_learned_model()
        # pick_up_effects = model['actions']['pick-up']['effects']
        # # Should learn add and delete effects
        # assert 'holding_a' in pick_up_effects['add']
        # assert 'clear_a' in pick_up_effects['delete']
        pass

    def test_model_refinement_over_multiple_observations(self):
        """Test that model improves with more observations."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        #
        # # Initial observation
        # adapter.observe(state1, action1, objects1, True, next_state1)
        # model1 = adapter.get_learned_model()
        #
        # # More observations
        # for obs in observations:
        #     adapter.observe(obs['state'], obs['action'], obs['objects'],
        #                   obs['success'], obs['next_state'])
        #
        # model2 = adapter.get_learned_model()
        # # Model should be more complete
        # assert len(model2['actions']) >= len(model1['actions'])
        pass

    # ========== Test Category F: Edge Cases and Error Handling ==========

    def test_empty_state_handling(self):
        """Test adapter handles empty states correctly."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # empty_state = set()
        #
        # # Should not crash
        # action, objects = adapter.select_action(empty_state)
        # adapter.observe(empty_state, action, objects, False, None)
        pass

    def test_invalid_action_handling(self):
        """Test adapter handles invalid actions gracefully."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # state = {'clear_a', 'ontable_a', 'handempty'}
        #
        # # Invalid action
        # adapter.observe(state, 'invalid-action', ['a'], False, None)
        # # Should not crash
        pass

    def test_contradictory_observations(self):
        """Test adapter handles contradictory observations."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # state = {'clear_a', 'ontable_a', 'handempty'}
        # action = 'pick-up'
        # objects = ['a']
        #
        # # First observation: success with one outcome
        # adapter.observe(state, action, objects, True, {'holding_a'})
        #
        # # Contradictory observation: same state/action, different outcome
        # adapter.observe(state, action, objects, True, {'holding_a', 'clear_b'})
        #
        # # Should handle gracefully, refine model
        pass

    def test_iteration_limit(self):
        """Test that adapter respects iteration limits."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file, max_iterations=10)
        #
        # for i in range(20):
        #     if adapter.has_converged():
        #         break
        #     state = get_current_state()
        #     action, objects = adapter.select_action(state)
        #     adapter.observe(state, action, objects, True, next_state)
        #
        # assert adapter.iteration_count <= 10 or adapter.has_converged()
        pass

    def test_convergence_detection(self):
        """Test that adapter detects convergence correctly."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        #
        # # Train until convergence
        # while not adapter.has_converged() and adapter.iteration_count < 1000:
        #     state = get_current_state()
        #     action, objects = adapter.select_action(state)
        #     success, next_state = execute_action(action, objects)
        #     adapter.observe(state, action, objects, success, next_state)
        #
        # if adapter.has_converged():
        #     model = adapter.get_learned_model()
        #     # Converged model should be stable
        #     assert model['statistics']['converged'] == True
        pass

    # ========== Test Category G: Plan Execution Tests ==========

    def test_plan_generation_with_learned_model(self):
        """Test that learned model can be used for planning."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        #
        # # Learn a model
        # train_adapter(adapter)
        #
        # model = adapter.get_learned_model()
        # # Use model for planning
        # plan = generate_plan_from_model(model, initial_state, goal_state)
        # assert len(plan) > 0
        pass

    def test_plan_execution_monitoring(self):
        """Test monitoring plan execution with learned model."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        #
        # # Execute plan and learn
        # for action in plan:
        #     state = get_current_state()
        #     success, next_state = execute(action)
        #     adapter.observe(state, action['name'], action['objects'],
        #                   success, next_state)
        #
        # # Model should improve with plan execution
        pass

    def test_replanning_on_failure(self):
        """Test replanning when action fails."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # state = initial_state
        # goal = goal_state
        #
        # while not goal_achieved(state, goal):
        #     action, objects = adapter.select_action(state)
        #     success, next_state = execute(action, objects)
        #     adapter.observe(state, action, objects, success, next_state)
        #
        #     if success:
        #         state = next_state
        #     else:
        #         # Should trigger replanning/exploration
        #         pass
        pass

    def test_goal_achievement_validation(self):
        """Test that adapter helps achieve goals."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # # Train adapter
        # # Try to achieve goal
        # # Validate goal is achieved
        pass

    # ========== Test Category H: Integration Tests ==========

    @pytest.mark.parametrize("domain", ["blocksworld", "gripper", "rover"])
    def test_integration_with_standard_domains(self, domain):
        """Test adapter with standard planning domains."""
        # When implemented:
        # domain_file = f"benchmarks/{domain}/domain.pddl"
        # problem_file = f"benchmarks/{domain}/p01.pddl"
        #
        # if os.path.exists(domain_file):
        #     adapter = OLAMAdapter(domain_file, problem_file)
        #     # Run learning episode
        #     run_learning_episode(adapter)
        #
        #     model = adapter.get_learned_model()
        #     assert len(model['actions']) > 0
        pass

    def test_action_sequence_execution(self):
        """Test executing sequence of actions."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # action_sequence = [
        #     ('pick-up', ['a']),
        #     ('stack', ['a', 'b']),
        #     ('pick-up', ['c']),
        #     ('stack', ['c', 'a'])
        # ]
        #
        # for action, objects in action_sequence:
        #     state = get_current_state()
        #     success, next_state = execute(action, objects)
        #     adapter.observe(state, action, objects, success, next_state)
        pass

    def test_model_accuracy_over_time(self):
        """Test that model accuracy improves over time."""
        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # accuracy_history = []
        #
        # for episode in range(10):
        #     run_episode(adapter)
        #     accuracy = evaluate_model_accuracy(adapter.get_learned_model())
        #     accuracy_history.append(accuracy)
        #
        # # Accuracy should generally improve
        # assert accuracy_history[-1] >= accuracy_history[0]
        pass

    def test_compatibility_with_up_simulator(self):
        """Test that adapter works with UP's SequentialSimulator."""
        # When implemented:
        # from unified_planning.engines import SequentialSimulator
        #
        # adapter = OLAMAdapter(domain_file, problem_file)
        # simulator = SequentialSimulator(problem)
        #
        # state = simulator.get_initial_state()
        # action, objects = adapter.select_action(convert_up_state(state))
        # # Should work together
        pass

    # ========== Test Category I: Performance Tests ==========

    def test_action_selection_performance(self):
        """Test that action selection is fast enough."""
        import time

        # When implemented:
        # adapter = OLAMAdapter(domain_file, problem_file)
        # state = {'clear_a', 'clear_b', 'clear_c', 'ontable_a',
        #          'ontable_b', 'ontable_c', 'handempty'}
        #
        # start = time.time()
        # for _ in range(100):
        #     action, objects = adapter.select_action(state)
        # elapsed = time.time() - start
        #
        # # Should be less than 1s per action (100 actions < 100s)
        # assert elapsed < 100
        pass

    def test_memory_usage(self):
        """Test that adapter doesn't use excessive memory."""
        # When implemented:
        # import psutil
        # import os
        #
        # process = psutil.Process(os.getpid())
        # initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        #
        # adapter = OLAMAdapter(domain_file, problem_file)
        # # Run many observations
        # for _ in range(1000):
        #     state = generate_random_state()
        #     action, objects = adapter.select_action(state)
        #     adapter.observe(state, action, objects, True, state)
        #
        # final_memory = process.memory_info().rss / 1024 / 1024  # MB
        # memory_increase = final_memory - initial_memory
        #
        # # Should not use more than 100MB
        # assert memory_increase < 100
        pass

    def test_olam_functionality_preserved(self):
        """Test that OLAM's core functionality is preserved through adapter."""
        # When implemented:
        # # Direct OLAM usage
        # olam_learner = create_olam_learner_directly()
        # olam_result = run_olam_learning(olam_learner)
        #
        # # Through adapter
        # adapter = OLAMAdapter(domain_file, problem_file)
        # adapter_result = run_adapter_learning(adapter)
        #
        # # Should produce similar results
        # assert compare_models(olam_result, adapter_result)
        pass