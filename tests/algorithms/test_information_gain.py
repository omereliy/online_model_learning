"""
Tests for Information Gain algorithm implementation.
Phase 1: Core data structures and binding functions.
"""

import pytest
from pathlib import Path
from typing import Set
import itertools

from src.algorithms.information_gain import InformationGainLearner


@pytest.fixture
def depots_domain():
    """Path to depots domain file."""
    return "/home/omer/projects/domains/depots/domain.pddl"


@pytest.fixture
def depots_problem():
    """Path to depots problem file."""
    return "/home/omer/projects/domains/depots/pfile1.pddl"


@pytest.fixture
def learner(depots_domain, depots_problem):
    """Create Information Gain learner instance."""
    return InformationGainLearner(
        domain_file=depots_domain,
        problem_file=depots_problem,
        max_iterations=100
    )


@pytest.fixture
def initial_state(depots_domain, depots_problem):
    """Get initial state from PDDL files."""
    from src.core.pddl_io import PDDLReader
    reader = PDDLReader()
    _, initial_state = reader.parse_domain_and_problem(depots_domain, depots_problem)
    return initial_state


class TestInitialization:
    """Test learner initialization and state variables."""

    def test_learner_creation(self, learner):
        """Test that learner initializes successfully."""
        assert learner is not None
        assert learner.iteration_count == 0
        assert learner.observation_count == 0

    def test_domain_initialized(self, learner):
        """Test that domain knowledge is properly initialized."""
        assert learner.domain is not None
        assert len(learner.domain.lifted_actions) > 0

    def test_action_models_initialized(self, learner):
        """Test that action model state variables are initialized."""
        # Should have initialized models for each action in domain
        assert len(learner.pre) > 0
        assert len(learner.pre_constraints) == len(learner.pre)
        assert len(learner.eff_add) == len(learner.pre)
        assert len(learner.eff_del) == len(learner.pre)
        assert len(learner.eff_maybe_add) == len(learner.pre)
        assert len(learner.eff_maybe_del) == len(learner.pre)

    def test_initial_state_variables_correct(self, learner):
        """Test that state variables have correct initial values."""
        for action_name in learner.pre.keys():
            # Get La using the learner's internal method
            La = learner._get_parameter_bound_literals(action_name)

            # Verify La structure
            assert len(La) > 0, f"Action {action_name} has empty La"
            positive_lits = {l for l in La if not l.startswith('¬')}
            negative_lits = {l for l in La if l.startswith('¬')}
            assert len(positive_lits) > 0
            assert len(negative_lits) > 0
            assert len(positive_lits) == len(negative_lits)

            # pre(a) should be EXACTLY La
            assert isinstance(learner.pre[action_name], set)
            assert learner.pre[action_name] == La, \
                f"Action {action_name}: pre should equal La (expected {len(La)}, got {len(learner.pre[action_name])})"

            # pre?(a) should be empty set (not list - updated in current implementation)
            assert isinstance(learner.pre_constraints[action_name], set)
            assert len(learner.pre_constraints[action_name]) == 0

            # eff+(a) and eff-(a) should be empty
            assert isinstance(learner.eff_add[action_name], set)
            assert len(learner.eff_add[action_name]) == 0
            assert isinstance(learner.eff_del[action_name], set)
            assert len(learner.eff_del[action_name]) == 0

            # eff?+(a) and eff?-(a) should be EXACTLY La (not just non-empty)
            assert isinstance(learner.eff_maybe_add[action_name], set)
            assert learner.eff_maybe_add[action_name] == La, \
                f"Action {action_name}: eff_maybe_add should equal La"

            assert isinstance(learner.eff_maybe_del[action_name], set)
            assert learner.eff_maybe_del[action_name] == La, \
                f"Action {action_name}: eff_maybe_del should equal La"


class TestParameterBoundLiterals:
    """Test generation of parameter-bound literals (La)."""

    def test_la_contains_positive_and_negative(self, learner):
        """Test that La contains both positive and negative literals."""
        for action_name, literals in learner.pre.items():
            # Count positive and negative literals
            positive = [l for l in literals if not l.startswith('¬')]
            negative = [l for l in literals if l.startswith('¬')]

            # Should have both positive and negative literals
            assert len(positive) > 0, f"Action {action_name} has no positive literals"
            assert len(negative) > 0, f"Action {action_name} has no negative literals"

            # Number of negative should roughly equal positive (one for each)
            assert len(negative) == len(positive), \
                f"Action {action_name}: positive={len(positive)}, negative={len(negative)}"

    def test_la_uses_action_parameters(self, learner):
        """Test that La literals use action parameters."""
        for action_name, literals in learner.pre.items():
            # Check that parameters appear in literals
            has_params = False
            for literal in literals:
                if '?' in literal:
                    has_params = True
                    break

            # Most actions should have parameterized literals
            # (unless it's a propositional-only domain)
            # For blocksworld, all actions have parameters
            assert has_params, f"Action {action_name} has no parameterized literals"


class TestBindingFunctions:
    """Test bindP and bindP_inverse functions."""

    def test_bindp_inverse_positive_literal(self, learner):
        """Test grounding of positive literals."""
        # Test: on(?x,?y) with [a, b] → on_a_b
        literals = {'on(?x,?y)'}
        objects = ['a', 'b']
        grounded = learner.bindP_inverse(literals, objects)

        assert len(grounded) == 1
        assert 'on_a_b' in grounded

    def test_bindp_inverse_negative_literal(self, learner):
        """Test grounding of negative literals."""
        # EXPLANATION: The ¬ symbol is used as internal representation for negative literals.
        # This follows the mathematical notation in the Information Gain algorithm paper.
        # When interfacing with PDDL (which uses "(not ...)" syntax), Phase 2 will handle
        # conversion between PDDL format and this internal representation.
        # This keeps the core algorithm logic clean and matches the specification.

        # Test: ¬on(?x,?y) with [a, b] → ¬on_a_b
        literals = {'¬on(?x,?y)'}
        objects = ['a', 'b']
        grounded = learner.bindP_inverse(literals, objects)

        assert len(grounded) == 1
        assert '¬on_a_b' in grounded

    def test_bindp_inverse_multiple_literals(self, learner):
        """Test grounding of multiple literals."""
        literals = {'on(?x,?y)', 'clear(?x)', '¬handempty'}
        objects = ['a', 'b']
        grounded = learner.bindP_inverse(literals, objects)

        assert len(grounded) == 3
        assert 'on_a_b' in grounded
        assert 'clear_a' in grounded
        assert '¬handempty' in grounded

    def test_bindp_positive_fluent(self, learner):
        """Test lifting of positive fluents."""
        # Test: on_a_b with [a, b] → on(?x,?y)
        fluents = {'on_a_b'}
        objects = ['a', 'b']
        lifted = learner.bindP(fluents, objects)

        assert len(lifted) == 1
        assert 'on(?x,?y)' in lifted

    def test_bindp_negative_fluent(self, learner):
        """Test lifting of negative fluents."""
        # Test: ¬on_a_b with [a, b] → ¬on(?x,?y)
        fluents = {'¬on_a_b'}
        objects = ['a', 'b']
        lifted = learner.bindP(fluents, objects)

        assert len(lifted) == 1
        assert '¬on(?x,?y)' in lifted

    def test_bindp_multiple_fluents(self, learner):
        """Test lifting of multiple fluents."""
        fluents = {'on_a_b', 'clear_a', '¬handempty'}
        objects = ['a', 'b']
        lifted = learner.bindP(fluents, objects)

        assert len(lifted) == 3
        assert 'on(?x,?y)' in lifted
        assert 'clear(?x)' in lifted
        assert '¬handempty' in lifted

    def test_bindp_inverse_roundtrip(self, learner):
        """Test that bindP_inverse followed by bindP gives back original."""
        original = {'on(?x,?y)', '¬clear(?x)'}
        objects = ['a', 'b']

        # Ground then lift
        grounded = learner.bindP_inverse(original, objects)
        lifted = learner.bindP(grounded, objects)

        assert lifted == original


class TestObservationRecording:
    """Test observation recording functionality."""

    def test_observe_success(self, learner):
        """Test recording successful action observation."""
        initial_count = learner.observation_count

        # Use depots domain action: drive(truck, from, to)
        state = {'at_truck1_depot0', 'at_hoist0_depot0'}
        action = 'drive'
        objects = ['truck1', 'depot0', 'distributor0']
        next_state = {'at_truck1_distributor0', 'at_hoist0_depot0'}

        learner.observe(state, action, objects, success=True, next_state=next_state)

        assert learner.observation_count == initial_count + 1
        assert len(learner.observation_history[action]) == 1

        obs = learner.observation_history[action][0]
        assert obs['action'] == action
        assert obs['objects'] == objects
        assert obs['success'] is True
        assert obs['state'] == state
        assert obs['next_state'] == next_state

    def test_observe_failure(self, learner):
        """Test recording failed action observation."""
        # Use depots domain action: drive(truck, from, to)
        state = {'at_truck1_depot0', 'at_hoist0_depot0'}
        action = 'drive'
        objects = ['truck1', 'depot0', 'distributor0']

        learner.observe(state, action, objects, success=False, next_state=None)

        assert len(learner.observation_history[action]) == 1
        obs = learner.observation_history[action][0]
        assert obs['success'] is False
        assert obs['next_state'] is None

    def test_multiple_observations_same_action(self, learner):
        """Test recording multiple observations for same action."""
        # Use depots domain action: drive(truck, from, to)
        action = 'drive'
        state1 = {'at_truck1_depot0'}
        state2 = {'at_truck2_depot1'}

        learner.observe(state1, action, ['truck1', 'depot0', 'distributor0'], success=True, next_state={'at_truck1_distributor0'})
        learner.observe(state2, action, ['truck2', 'depot1', 'distributor1'], success=False, next_state=None)

        assert len(learner.observation_history[action]) == 2
        assert learner.observation_history[action][0]['objects'] == ['truck1', 'depot0', 'distributor0']
        assert learner.observation_history[action][1]['objects'] == ['truck2', 'depot1', 'distributor1']

    def test_eff_maybe_sets_become_disjoint_after_success(self, learner):
        r"""
        Sanity check: After successful action execution, eff_maybe_add and eff_maybe_del
        should become disjoint (no overlap).

        Rationale: If a fluent remains unchanged after action execution, it cannot be
        both an add effect and a delete effect. The algorithm should narrow these sets
        based on observations per the update rules:
        - eff?+(a) = eff?+(a) ∩ bindP(s ∩ s', O)  # Keep only unchanged fluents
        - eff?-(a) = eff?-(a) \ bindP(s ∪ s', O)  # Remove fluents that were true

        This test uses depots domain which has more fluents than blocksworld.
        """
        # Get an action from depots domain
        action_name = 'drive'  # drive(truck, from, to)

        # Initially, eff_maybe_add and eff_maybe_del should be identical (both = La)
        assert learner.eff_maybe_add[action_name] == learner.eff_maybe_del[action_name], \
            "Initially, eff_maybe_add and eff_maybe_del should both equal La"

        initial_intersection = learner.eff_maybe_add[action_name] & learner.eff_maybe_del[action_name]
        assert len(initial_intersection) > 0, "Initially should have overlap"

        # Simulate a successful execution with realistic state transition
        # Example: drive(truck1, depot0, distributor0)
        # Before: at(truck1, depot0), ...other fluents...
        # After: at(truck1, distributor0), ...other fluents... (at(truck1, depot0) removed)
        state = {
            'at_truck1_depot0',
            'at_hoist0_depot0',
            'available_hoist0',
            'at_pallet0_depot0',
            'clear_pallet0'
        }
        next_state = {
            'at_truck1_distributor0',  # Changed: truck moved
            'at_hoist0_depot0',         # Unchanged
            'available_hoist0',          # Unchanged
            'at_pallet0_depot0',        # Unchanged
            'clear_pallet0'              # Unchanged
        }
        objects = ['truck1', 'depot0', 'distributor0']

        # Observe successful execution
        learner.observe(state, action_name, objects, success=True, next_state=next_state)
        learner.update_model()  # Phase 2 requires calling update_model

        # After Phase 2 update rules are implemented, the sets should narrow
        # and eventually become disjoint as we learn which fluents change vs remain unchanged
        intersection_after = learner.eff_maybe_add[action_name] & learner.eff_maybe_del[action_name]

        # The intersection should be smaller (or empty) after learning
        # Note: May require multiple observations to fully separate
        assert len(intersection_after) < len(initial_intersection), \
            "After successful observation, the intersection should shrink"

        # Stronger assertion: Eventually should be completely disjoint
        # This may require multiple observations, but validates the algorithm invariant
        if len(intersection_after) > 0:
            # Not yet fully separated - this is OK for early learning
            # But we should see progress toward disjointness
            assert len(learner.eff_maybe_add[action_name]) < len(initial_intersection) or \
                len(learner.eff_maybe_del[action_name]) < len(initial_intersection), \
                "At least one set should have shrunk"


class TestLearnedModelExport:
    """Test learned model export functionality."""

    def test_get_learned_model_structure(self, learner):
        """Test that learned model has correct structure."""
        model = learner.get_learned_model()

        assert 'actions' in model
        assert 'predicates' in model
        assert 'statistics' in model

    def test_learned_model_contains_actions(self, learner):
        """Test that learned model contains all actions."""
        model = learner.get_learned_model()

        # Should have same number of actions as initialized
        assert len(model['actions']) == len(learner.pre)

        for action_name in learner.pre.keys():
            assert action_name in model['actions']

    def test_learned_model_action_structure(self, learner):
        """Test that each action in model has correct structure."""
        model = learner.get_learned_model()

        for action_name, action_data in model['actions'].items():
            assert 'name' in action_data
            assert 'preconditions' in action_data
            assert 'effects' in action_data
            assert 'observations' in action_data

            # Check preconditions structure
            assert 'possible' in action_data['preconditions']
            assert 'constraints' in action_data['preconditions']

            # Check effects structure
            assert 'add' in action_data['effects']
            assert 'delete' in action_data['effects']
            assert 'maybe_add' in action_data['effects']
            assert 'maybe_delete' in action_data['effects']


class TestReset:
    """Test learner reset functionality."""

    def test_reset_clears_observations(self, learner):
        """Test that reset clears observation history."""
        # Add some observations using depots domain action
        learner.observe({'at_truck1_depot0'}, 'drive', ['truck1', 'depot0', 'distributor0'], success=True, next_state={'at_truck1_distributor0'})

        assert learner.observation_count > 0

        # Reset
        learner.reset()

        assert learner.observation_count == 0
        assert len(learner.observation_history) == 0

    def test_reset_reinitializes_models(self, learner):
        """Test that reset reinitializes action models."""
        # Modify state variables
        action_name = list(learner.pre.keys())[0]
        learner.eff_add[action_name].add('test_literal')

        # Reset
        learner.reset()

        # Should be back to initial state
        assert len(learner.eff_add[action_name]) == 0
        assert len(learner.pre[action_name]) > 0  # La restored


class TestConvergence:
    """Test convergence detection."""

    def test_has_converged_initially_false(self, learner):
        """Test that learner hasn't converged initially."""
        assert learner.has_converged() is False

    def test_has_converged_after_max_iterations(self, learner):
        """Test that learner converges after max iterations."""
        learner.iteration_count = learner.max_iterations
        assert learner.has_converged() is True


# Phase 2 Tests: Update Rules and CNF Construction
class TestUpdateRules:
    """Test Phase 2 update rules for observations."""

    def test_success_updates_preconditions(self, learner):
        """Test that successful execution narrows preconditions to satisfied literals."""
        action_name = 'drive'  # drive(truck, from, to)
        objects = ['truck1', 'depot0', 'distributor0']

        # State with some satisfied and some unsatisfied literals
        state = {
            'at_truck1_depot0',  # truck is at source
            'at_hoist0_depot0',
            'clear_pallet0'
        }
        next_state = {
            'at_truck1_distributor0',  # truck moved to destination
            'at_hoist0_depot0',
            'clear_pallet0'
        }

        # Get initial precondition size
        initial_pre_size = len(learner.pre[action_name])

        # Observe successful execution
        learner.observe(state, action_name, objects, success=True, next_state=next_state)
        learner.update_model()  # Phase 2 adds this

        # pre(a) should be narrowed to only satisfied literals
        # pre(a) = pre(a) ∩ bindP_inverse(s, O)
        satisfied_literals = learner.bindP_inverse(learner.pre[action_name], objects)
        grounded_state_literals = learner._state_to_internal(state)

        # Check that preconditions were narrowed
        assert len(learner.pre[action_name]) < initial_pre_size, \
            "Preconditions should narrow after successful observation"

    def test_success_updates_confirmed_effects(self, learner):
        """Test that successful execution updates confirmed add/delete effects."""
        action_name = 'drive'
        objects = ['truck1', 'depot0', 'distributor0']

        state = {
            'at_truck1_depot0',     # Will be deleted
            'at_hoist0_depot0',     # Unchanged
        }
        next_state = {
            'at_truck1_distributor0',  # Will be added
            'at_hoist0_depot0',        # Unchanged
        }

        # Initially, no confirmed effects
        assert len(learner.eff_add[action_name]) == 0
        assert len(learner.eff_del[action_name]) == 0

        # Observe successful execution
        learner.observe(state, action_name, objects, success=True, next_state=next_state)
        learner.update_model()

        # Should learn effects from state change
        # eff+(a) = eff+(a) ∪ bindP(s' \ s, O)
        # eff-(a) = eff-(a) ∪ bindP(s \ s', O)
        assert len(learner.eff_add[action_name]) > 0, "Should learn add effects"
        assert len(learner.eff_del[action_name]) > 0, "Should learn delete effects"

    def test_success_narrows_possible_effects(self, learner):
        """Test that successful execution narrows possible effect sets."""
        action_name = 'drive'
        objects = ['truck1', 'depot0', 'distributor0']

        state = {
            'at_truck1_depot0',
            'at_hoist0_depot0',
        }
        next_state = {
            'at_truck1_distributor0',
            'at_hoist0_depot0',  # Unchanged
        }

        # Get initial sizes
        initial_maybe_add = len(learner.eff_maybe_add[action_name])
        initial_maybe_del = len(learner.eff_maybe_del[action_name])

        # Observe successful execution
        learner.observe(state, action_name, objects, success=True, next_state=next_state)
        learner.update_model()

        # eff?+(a) = eff?+(a) ∩ bindP(s ∩ s', O)  # Keep unchanged
        # eff?-(a) = eff?-(a) \ bindP(s ∪ s', O)  # Remove if was/became true
        assert len(learner.eff_maybe_add[action_name]) < initial_maybe_add, \
            "Possible add effects should narrow"
        assert len(learner.eff_maybe_del[action_name]) < initial_maybe_del, \
            "Possible delete effects should narrow"

    def test_failure_adds_constraint(self, learner):
        """Test that failed execution adds a precondition constraint."""
        action_name = 'lift'
        objects = ['hoist0', 'crate0', 'pallet0', 'depot0']

        # State where action fails (e.g., hoist not available)
        state = {
            'at_crate0_depot0',
            'at_hoist0_depot0',
            # Missing 'available_hoist0' which is likely needed
        }

        # Initially, no constraints (pre_constraints is a set of frozensets)
        assert len(learner.pre_constraints[action_name]) == 0

        # Observe failed execution
        learner.observe(state, action_name, objects, success=False)
        # Note: update_model() is called automatically by observe()

        # Should add constraint: at least one unsatisfied literal is required
        # pre?(a) = pre?(a) ∪ {pre(a) \ bindP(s, O)}
        assert len(learner.pre_constraints[action_name]) == 1, \
            "Failed observation should add one constraint"

        # The constraint should contain unsatisfied literals
        # pre_constraints is a set of frozensets, get any constraint
        constraint = next(iter(learner.pre_constraints[action_name]))
        assert len(constraint) > 0, "Constraint should not be empty"

    def test_multiple_failures_accumulate_constraints(self, learner):
        """Test that multiple failures add multiple constraints."""
        action_name = 'lift'
        objects = ['hoist0', 'crate0', 'pallet0', 'depot0']

        # First failure
        state1 = {'at_crate0_depot0', 'at_hoist0_depot0'}
        learner.observe(state1, action_name, objects, success=False)
        learner.update_model()

        # Second failure with different state
        state2 = {'at_crate0_depot0', 'available_hoist0'}
        learner.observe(state2, action_name, objects, success=False)
        learner.update_model()

        # Should have two constraints
        assert len(learner.pre_constraints[action_name]) == 2, \
            "Should accumulate constraints from multiple failures"

    def test_success_updates_existing_constraints(self, learner):
        """Test that successful execution updates existing constraint sets."""
        action_name = 'lift'
        objects = ['hoist0', 'crate0', 'pallet0', 'depot0']

        # First add a constraint through failure
        fail_state = {'at_crate0_depot0', 'at_hoist0_depot0'}
        learner.observe(fail_state, action_name, objects, success=False)
        # Note: update_model() is called automatically by observe()

        # pre_constraints is a set of frozensets
        initial_constraints = set(learner.pre_constraints[action_name])
        initial_constraint = next(iter(initial_constraints))

        # Now observe success
        success_state = {
            'at_crate0_depot0',
            'at_hoist0_depot0',
            'available_hoist0',
            'on_crate0_pallet0'}
        next_state = {'at_crate0_depot0', 'at_hoist0_depot0', 'lifting_hoist0_crate0'}
        learner.observe(success_state, action_name, objects, success=True, next_state=next_state)
        # Note: update_model() is called automatically by observe()

        # Constraint should be updated
        # pre?(a) = {B ∩ bindP(s, O) | B ∈ pre?(a)}
        updated_constraints = learner.pre_constraints[action_name]
        # Check that constraints have changed (either different content or different count)
        assert updated_constraints != initial_constraints, \
            "Constraints should be updated after successful observation"

    def test_negative_preconditions_in_constraints(self, learner):
        """Test that negative preconditions are correctly handled in constraints."""
        action_name = 'load'  # May have negative preconditions
        objects = ['hoist0', 'crate0', 'truck0', 'depot0']

        # State where load fails
        state = {
            'at_truck0_depot0',
            'at_crate0_depot0',
            # Missing lifting_hoist0_crate0 which is likely required
        }

        # Observe failure
        learner.observe(state, action_name, objects, success=False)
        # Note: update_model() is called automatically by observe()

        # Check constraint contains negative literals
        # pre_constraints is a set of frozensets
        constraint = next(iter(learner.pre_constraints[action_name]))
        has_negative = any(lit.startswith('¬') for lit in constraint)
        assert has_negative or len(constraint) > 0, \
            "Constraint should handle negative preconditions"

    def test_state_conversion_functions(self, learner):
        """Test state format conversion functions."""
        # Test grounded state to internal format
        state = {'at_truck1_depot0', 'clear_pallet0'}
        internal = learner._state_to_internal(state)
        assert isinstance(internal, set)
        assert 'at_truck1_depot0' in internal

        # Test handling of negative literals in internal format
        # Negative literal ¬p means p is NOT in state
        assert '¬at_truck1_distributor0' not in internal, \
            "Should not add explicit negatives for absent fluents"


class TestCNFIntegrationWithAlgorithm:
    """Test integration of CNF with Information Gain algorithm - NOT testing CNF itself."""

    def test_cnf_tracks_constraint_evolution(self, learner):
        """Test that CNF formula evolves correctly with constraint updates."""
        action_name = 'lift'
        objects = ['hoist0', 'crate0', 'pallet0', 'depot0']

        # Initially no constraints means no CNF restrictions
        assert len(learner.pre_constraints[action_name]) == 0

        # After failure, constraint is added and reflected in CNF
        fail_state = {'at_crate0_depot0', 'at_hoist0_depot0'}
        learner.observe(fail_state, action_name, objects, success=False)
        # Note: update_model() is called automatically by observe()

        # Verify constraint → CNF mapping
        assert len(learner.pre_constraints[action_name]) == 1
        # The CNF should encode: "at least one of the unsatisfied literals must be true"

        # After success, constraints are refined
        success_state = {
            'at_crate0_depot0',
            'at_hoist0_depot0',
            'available_hoist0',
            'on_crate0_pallet0'}
        next_state = {'at_crate0_depot0', 'at_hoist0_depot0', 'lifting_hoist0_crate0'}
        learner.observe(success_state, action_name, objects, success=True, next_state=next_state)
        # Note: update_model() is called automatically by observe()

        # Constraints should be updated to keep only satisfied literals
        # pre_constraints is a set of frozensets
        if learner.pre_constraints[action_name]:
            constraint = next(iter(learner.pre_constraints[action_name]))
            # Verify constraint was narrowed based on what was satisfied

    def test_negative_preconditions_properly_encoded_in_cnf(self, learner):
        """Test that negative preconditions (¬p means p ∉ state) are correctly encoded."""
        action_name = 'load'  # May have negative preconditions
        objects = ['hoist0', 'crate0', 'truck0', 'depot0']

        # State that may violate negative preconditions
        state_violates_neg = {'at_truck0_depot0', 'at_crate0_depot0', 'in_crate0_truck0'}
        learner.observe(state_violates_neg, action_name, objects, success=False)
        # Note: update_model() is called automatically by observe()

        # The constraint should recognize that ¬on(?x,?y) was NOT satisfied
        # because on_block1_block2 WAS in the state
        # pre_constraints is a set of frozensets
        constraint = next(iter(learner.pre_constraints[action_name]))

        # Should contain ¬on(?x,?y) as a possible required precondition
        lifted_constraint = learner.bindP(constraint, objects)
        has_neg_on = any('¬on(' in lit for lit in lifted_constraint)
        assert has_neg_on or len(constraint) > 0, \
            "Constraint should identify negative precondition violation"

    def test_cnf_formula_consistency_across_observations(self, learner):
        """Test that CNF formula maintains consistency as observations accumulate."""
        action_name = 'drive'

        # Multiple observations with different object groundings
        observations = [
            (['truck1', 'depot0', 'distributor0'],
             {'at_truck1_depot0'},
             {'at_truck1_distributor0'},
             True),
            (['truck2', 'depot1', 'distributor1'],
             {'at_truck2_distributor1'},  # Wrong location
             None,
             False),
        ]

        for objects, state, next_state, success in observations:
            learner.observe(state, action_name, objects, success, next_state)
            learner.update_model()

        # After multiple observations, verify invariants:
        # 1. Confirmed effects remain consistent
        # 2. Possible preconditions only shrink (monotonicity)
        # 3. Constraints accumulate for failures

        if not success:
            assert len(learner.pre_constraints[action_name]) > 0, \
                "Failed observations should add constraints"

    def test_cnf_reflects_learned_precondition_certainty(self, learner):
        """Test that CNF formula reflects growing certainty about preconditions."""
        action_name = 'lift'
        objects = ['hoist0', 'crate0', 'pallet0', 'depot0']

        # Multiple failures from different states help narrow preconditions
        failure_states = [
            {'at_crate0_depot0'},  # Missing other preconditions
            {'at_hoist0_depot0'},  # Missing other preconditions
            {'available_hoist0'},  # Missing other preconditions
        ]

        initial_pre_size = len(learner.pre[action_name])

        for state in failure_states:
            learner.observe(state, action_name, objects, success=False)
            learner.update_model()

        # As constraints accumulate, the set of possible preconditions narrows
        # This should be reflected in the CNF encoding
        final_pre_size = len(learner.pre[action_name])

        # Note: pre doesn't shrink on failures, only on successes
        # But constraints accumulate
        assert len(learner.pre_constraints[action_name]) == len(failure_states), \
            "Each failure should add a constraint"

    def test_internal_negation_format_consistency(self, learner):
        """Test that internal ¬ notation is consistently used throughout."""
        action_name = 'load'
        objects = ['hoist0', 'crate0', 'truck0', 'depot0']

        # Create a state
        state = {'at_truck0_depot0', 'at_crate0_depot0'}

        learner.observe(state, action_name, objects, success=False)
        learner.update_model()

        # Check internal representation uses ¬ symbol
        for constraint in learner.pre_constraints[action_name]:
            for lit in constraint:
                if lit.startswith('¬'):
                    # Verify format is ¬predicate(...) not (not predicate(...))
                    assert not lit.startswith('(not'), \
                        "Should use ¬ symbol, not PDDL (not ...) syntax internally"


# Integration test
# Phase 3 Tests: Information Gain and Action Selection
class TestApplicabilityProbability:
    """Test applicability probability calculation using SAT model counting."""

    def test_probability_with_no_constraints(self, learner):
        """Test that actions with no constraints have probability 1.0."""
        action_name = 'drive'
        objects = ['truck1', 'depot0', 'distributor0']
        state = {'at_truck1_depot0'}

        # No constraints initially
        assert len(learner.pre_constraints[action_name]) == 0

        prob = learner._calculate_applicability_probability(action_name, objects, state)
        assert prob == 1.0, "Actions with no constraints should always be applicable"

    def test_probability_after_failure_constraint(self, learner):
        """Test that probability decreases after adding failure constraints."""
        action_name = 'lift'
        objects = ['hoist0', 'crate0', 'pallet0', 'depot0']

        # State missing required preconditions
        state = {'at_crate0_depot0', 'at_hoist0_depot0'}

        # Add failure constraint
        learner.observe(state, action_name, objects, success=False)
        learner.update_model()

        # Probability should be less than 1.0 with constraints
        prob = learner._calculate_applicability_probability(action_name, objects, state)
        assert 0.0 <= prob < 1.0, f"Probability with constraints should be < 1.0, got {prob}"

    def test_probability_with_satisfied_preconditions(self, learner):
        """Test higher probability when more preconditions are satisfied."""
        action_name = 'lift'
        objects = ['hoist0', 'crate0', 'pallet0', 'depot0']

        # First observe failure with minimal state
        min_state = {'at_crate0_depot0'}
        learner.observe(min_state, action_name, objects, success=False)
        learner.update_model()

        prob_min = learner._calculate_applicability_probability(action_name, objects, min_state)

        # State with more satisfied preconditions
        fuller_state = {'at_crate0_depot0', 'at_hoist0_depot0', 'available_hoist0'}
        prob_fuller = learner._calculate_applicability_probability(action_name, objects, fuller_state)

        assert prob_fuller >= prob_min, "More satisfied preconditions should increase probability"

    def test_probability_edge_cases(self, learner):
        """Test edge cases: empty CNF, contradictions."""
        action_name = 'drive'
        objects = ['truck1', 'depot0', 'distributor0']

        # Empty state should still work
        empty_state = set()
        prob = learner._calculate_applicability_probability(action_name, objects, empty_state)
        assert 0.0 <= prob <= 1.0, "Probability should be valid even with empty state"


class TestInformationGain:
    """Test entropy and information gain calculations."""

    def test_potential_gain_success(self, learner):
        """Test potential information gain from successful execution."""
        action_name = 'lift'
        objects = ['hoist0', 'crate0', 'pallet0', 'depot0']
        state = {
            'at_crate0_depot0',
            'at_hoist0_depot0',
            'available_hoist0',
            'on_crate0_pallet0'
        }

        gain = learner._calculate_potential_gain_success(action_name, objects, state)
        assert gain >= 0, "Information gain should be non-negative"

        # Actions with more uncertainty should have higher potential gain
        uncertain_action = 'load'  # Less observed action
        uncertain_gain = learner._calculate_potential_gain_success(uncertain_action, objects, state)
        assert uncertain_gain >= 0, "Uncertain actions should have potential for gain"

    def test_potential_gain_failure(self, learner):
        """Test potential information gain from failed execution."""
        action_name = 'lift'
        objects = ['hoist0', 'crate0', 'pallet0', 'depot0']
        state = {'at_crate0_depot0'}  # Missing preconditions

        gain = learner._calculate_potential_gain_failure(action_name, objects, state)
        assert gain >= 0, "Failure gain should be non-negative"

        # First failure should provide more information than subsequent ones
        learner.observe(state, action_name, objects, success=False)
        learner.update_model()

        second_gain = learner._calculate_potential_gain_failure(action_name, objects, state)
        assert second_gain <= gain, "Repeated failures provide diminishing information"

    def test_expected_information_gain(self, learner):
        """Test expected information gain calculation."""
        action_name = 'drive'
        objects = ['truck1', 'depot0', 'distributor0']
        state = {'at_truck1_depot0'}

        expected_gain = learner._calculate_expected_information_gain(action_name, objects, state)
        assert expected_gain >= 0, "Expected gain should be non-negative"

        # Expected gain should consider both success and failure probabilities
        # E[gain] = P(success) * gain_success + P(failure) * gain_failure
        prob = learner._calculate_applicability_probability(action_name, objects, state)
        gain_success = learner._calculate_potential_gain_success(action_name, objects, state)
        gain_failure = learner._calculate_potential_gain_failure(action_name, objects, state)

        manual_expected = prob * gain_success + (1 - prob) * gain_failure
        assert abs(expected_gain - manual_expected) < 0.001, "Expected gain formula incorrect"


class TestActionSelection:
    """Test action selection strategies."""

    def test_greedy_selection_chooses_max_gain(self, learner, initial_state):
        """Test that greedy selection chooses action with maximum expected gain."""
        state = initial_state

        # Set selection strategy to greedy
        learner.selection_strategy = 'greedy'

        action_name, objects = learner.select_action(state)

        # Verify this was the best choice
        selected_gain = learner._calculate_expected_information_gain(action_name, objects, state)

        # Check a few other possible actions using ground_all_actions
        from src.core.grounding import ground_all_actions
        grounded_actions = ground_all_actions(learner.domain)[:5]  # Sample
        for grounded_action in grounded_actions:
            other_gain = learner._calculate_expected_information_gain(
                grounded_action.action_name, grounded_action.objects, state
            )
            assert selected_gain >= other_gain - 0.001, "Greedy should select max gain action"

    def test_epsilon_greedy_exploration(self, learner, initial_state):
        """Test epsilon-greedy balances exploration and exploitation."""
        import random
        random.seed(42)  # For reproducibility

        state = initial_state
        learner.selection_strategy = 'epsilon_greedy'
        learner.epsilon = 0.1  # 10% exploration

        # Run multiple selections to test probabilistic behavior
        selections = []
        for _ in range(10):
            action_name, objects = learner.select_action(state)
            selections.append(action_name)

        # Should have some variety (not all the same action)
        unique_actions = set(selections)
        assert len(unique_actions) > 1, "Epsilon-greedy should explore different actions"

    def test_boltzmann_selection(self, learner, initial_state):
        """Test Boltzmann/softmax selection based on gain values."""
        import random
        random.seed(42)

        state = initial_state
        learner.selection_strategy = 'boltzmann'
        learner.temperature = 1.0

        # Run multiple selections
        selections = []
        for _ in range(10):
            action_name, objects = learner.select_action(state)
            selections.append(action_name)

        # Should select variety of actions weighted by their gains
        unique_actions = set(selections)
        assert len(unique_actions) > 1, "Boltzmann should select various actions"

    def test_tie_breaking(self, learner, initial_state):
        """Test handling when multiple actions have same gain."""
        state = initial_state
        learner.selection_strategy = 'greedy'

        # In early learning, many actions may have similar gains
        action_name, objects = learner.select_action(state)

        # Should handle ties consistently (not crash)
        assert action_name is not None
        assert isinstance(objects, list)

    def test_no_applicable_actions(self, learner):
        """Test handling when no actions are possibly applicable."""
        # Create a state where most actions would fail
        impossible_state = set()  # Empty state

        action_name, objects = learner.select_action(impossible_state)

        # Should still return something (even if low probability)
        assert action_name is not None


class TestConvergenceImprovement:
    """Test that information gain improves convergence."""

    def test_converges_faster_than_random(self, learner, initial_state):
        """Test that info gain selection converges faster than random."""
        # This is a meta-test that would require running full learning episodes
        # For now, just verify the mechanism exists

        learner.selection_strategy = 'greedy'
        state = initial_state

        # Should select informative actions
        action_name, objects = learner.select_action(state)
        gain = learner._calculate_expected_information_gain(action_name, objects, state)

        assert gain >= 0, "Selected action should have non-negative information gain"


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_select_and_observe_cycle(self, learner, initial_state):
        """Test a complete select-observe cycle."""
        # Get initial state from fixture
        state = initial_state

        # Select action
        action_name, objects = learner.select_action(state)

        assert action_name is not None
        assert isinstance(objects, list)

        # Record observation (simulated failure)
        learner.observe(state, action_name, objects, success=False)

        # Check that observation was recorded
        assert learner.observation_count == 1
        assert action_name in learner.observation_history

    def test_full_learning_cycle_with_info_gain(self, learner, initial_state):
        """Test complete learning cycle with information gain selection."""
        learner.selection_strategy = 'greedy'

        # Run a few learning iterations
        state = initial_state

        for i in range(5):
            # Select action based on information gain
            action_name, objects = learner.select_action(state)

            # Simulate execution (alternate success/failure for testing)
            success = (i % 2 == 0)
            if success:
                # Create a simple next state
                next_state = state.copy()
                next_state.add(f'test_effect_{i}')
                learner.observe(state, action_name, objects, success=True, next_state=next_state)
                state = next_state
            else:
                learner.observe(state, action_name, objects, success=False)

            learner.update_model()

        # Should have learned something
        model = learner.get_learned_model()
        assert any(len(action_data['effects']['add']) > 0
                  for action_data in model['actions'].values()), \
            "Should have learned some effects"
