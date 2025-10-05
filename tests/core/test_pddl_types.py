"""
Tests for PDDL type-safe data classes.
These tests are written BEFORE implementation (TDD).
"""

import pytest
from unified_planning.shortcuts import UserType
from unified_planning.model import Object
from src.core.pddl_types import (
    ParameterBinding,
    ParameterBoundLiteral,
    GroundedFluent,
    GroundedAction
)


class TestParameterBinding:
    """Test ParameterBinding class."""

    def test_initialization_with_dict(self):
        """Test creating ParameterBinding from dict."""
        block_type = UserType('block')
        obj_a = Object('a', block_type)
        bindings = {'x': obj_a}

        pb = ParameterBinding(bindings)

        assert pb.get_object('x') == obj_a
        assert pb.object_names() == ['a']

    def test_object_names_preserves_order(self):
        """Test object_names() preserves parameter order."""
        block_type = UserType('block')
        obj_a = Object('a', block_type)
        obj_b = Object('b', block_type)
        bindings = {'x': obj_a, 'y': obj_b}

        pb = ParameterBinding(bindings)

        # Should preserve order from parameter names
        assert pb.object_names() == ['a', 'b']

    def test_to_dict(self):
        """Test converting back to dict."""
        block_type = UserType('block')
        obj_a = Object('a', block_type)
        bindings = {'x': obj_a}

        pb = ParameterBinding(bindings)

        assert pb.to_dict() == bindings


class TestParameterBoundLiteral:
    """Test ParameterBoundLiteral class."""

    def test_initialization_positive_literal(self):
        """Test creating positive parameter-bound literal."""
        lit = ParameterBoundLiteral('clear', ['?x'], is_negative=False)

        assert lit.predicate == 'clear'
        assert lit.parameters == ['?x']
        assert lit.is_negative is False

    def test_to_string_positive(self):
        """Test converting positive literal to string."""
        lit = ParameterBoundLiteral('clear', ['?x'])

        assert lit.to_string() == 'clear(?x)'

    def test_to_string_negative(self):
        """Test converting negative literal to string."""
        lit = ParameterBoundLiteral('clear', ['?x'], is_negative=True)

        assert lit.to_string() == '¬clear(?x)'

    def test_to_string_multi_parameter(self):
        """Test multi-parameter literal."""
        lit = ParameterBoundLiteral('on', ['?x', '?y'])

        assert lit.to_string() == 'on(?x,?y)'

    def test_to_string_propositional(self):
        """Test propositional literal (no parameters)."""
        lit = ParameterBoundLiteral('handempty', [])

        assert lit.to_string() == 'handempty'

    def test_from_string_positive(self):
        """Test parsing positive literal from string."""
        lit = ParameterBoundLiteral.from_string('clear(?x)')

        assert lit.predicate == 'clear'
        assert lit.parameters == ['?x']
        assert lit.is_negative is False

    def test_from_string_negative(self):
        """Test parsing negative literal from string."""
        lit = ParameterBoundLiteral.from_string('¬clear(?x)')

        assert lit.predicate == 'clear'
        assert lit.parameters == ['?x']
        assert lit.is_negative is True

    def test_from_string_multi_parameter(self):
        """Test parsing multi-parameter literal."""
        lit = ParameterBoundLiteral.from_string('on(?x,?y)')

        assert lit.predicate == 'on'
        assert lit.parameters == ['?x', '?y']

    def test_from_string_propositional(self):
        """Test parsing propositional literal."""
        lit = ParameterBoundLiteral.from_string('handempty')

        assert lit.predicate == 'handempty'
        assert lit.parameters == []


class TestGroundedFluent:
    """Test GroundedFluent class."""

    def test_initialization(self):
        """Test creating grounded fluent."""
        fluent = GroundedFluent('clear', ['a'])

        assert fluent.predicate == 'clear'
        assert fluent.objects == ['a']

    def test_to_string_single_object(self):
        """Test converting to string with single object."""
        fluent = GroundedFluent('clear', ['a'])

        assert fluent.to_string() == 'clear_a'

    def test_to_string_multiple_objects(self):
        """Test converting to string with multiple objects."""
        fluent = GroundedFluent('on', ['a', 'b'])

        assert fluent.to_string() == 'on_a_b'

    def test_to_string_propositional(self):
        """Test propositional fluent."""
        fluent = GroundedFluent('handempty', [])

        assert fluent.to_string() == 'handempty'

    def test_from_string_single_object(self):
        """Test parsing from string."""
        fluent = GroundedFluent.from_string('clear_a')

        assert fluent.predicate == 'clear'
        assert fluent.objects == ['a']

    def test_from_string_multiple_objects(self):
        """Test parsing multi-object fluent."""
        fluent = GroundedFluent.from_string('on_a_b')

        assert fluent.predicate == 'on'
        assert fluent.objects == ['a', 'b']

    def test_from_string_propositional(self):
        """Test parsing propositional fluent."""
        fluent = GroundedFluent.from_string('handempty')

        assert fluent.predicate == 'handempty'
        assert fluent.objects == []


class TestGroundedAction:
    """Test GroundedAction class."""

    @pytest.fixture
    def simple_action(self):
        """Create simple action for testing."""
        from unified_planning.shortcuts import UserType
        from unified_planning.model import InstantaneousAction

        block_type = UserType("block")
        pickup = InstantaneousAction("pick-up", x=block_type)
        return pickup

    def test_initialization(self, simple_action):
        """Test creating grounded action."""
        from unified_planning.shortcuts import UserType
        block_type = UserType("block")
        obj_a = Object('a', block_type)
        binding = ParameterBinding({'x': obj_a})

        grounded = GroundedAction(simple_action, binding)

        assert grounded.action == simple_action
        assert grounded.binding == binding

    def test_object_names_single_param(self, simple_action):
        """Test getting object names from binding."""
        from unified_planning.shortcuts import UserType
        block_type = UserType("block")
        obj_a = Object('a', block_type)
        binding = ParameterBinding({'x': obj_a})
        grounded = GroundedAction(simple_action, binding)

        assert grounded.object_names() == ['a']

    def test_object_names_multi_param(self):
        """Test multi-parameter action."""
        from unified_planning.shortcuts import UserType
        from unified_planning.model import InstantaneousAction

        block_type = UserType("block")
        stack = InstantaneousAction("stack", x=block_type, y=block_type)

        obj_a = Object('a', block_type)
        obj_b = Object('b', block_type)
        binding = ParameterBinding({'x': obj_a, 'y': obj_b})
        grounded = GroundedAction(stack, binding)

        assert grounded.object_names() == ['a', 'b']

    def test_to_string_single_param(self, simple_action):
        """Test converting to string representation."""
        from unified_planning.shortcuts import UserType
        block_type = UserType("block")
        obj_a = Object('a', block_type)
        binding = ParameterBinding({'x': obj_a})
        grounded = GroundedAction(simple_action, binding)

        assert grounded.to_string() == 'pick-up_a'

    def test_to_string_multi_param(self):
        """Test multi-parameter action string."""
        from unified_planning.shortcuts import UserType
        from unified_planning.model import InstantaneousAction

        block_type = UserType("block")
        stack = InstantaneousAction("stack", x=block_type, y=block_type)

        obj_a = Object('a', block_type)
        obj_b = Object('b', block_type)
        binding = ParameterBinding({'x': obj_a, 'y': obj_b})
        grounded = GroundedAction(stack, binding)

        assert grounded.to_string() == 'stack_a_b'

    def test_to_string_no_params(self):
        """Test propositional action (no parameters)."""
        from unified_planning.model import InstantaneousAction

        noop = InstantaneousAction("noop")
        binding = ParameterBinding({})
        grounded = GroundedAction(noop, binding)

        assert grounded.to_string() == 'noop'

    def test_from_components(self, simple_action):
        """Test creating from action and dict binding."""
        from unified_planning.shortcuts import UserType
        block_type = UserType("block")
        obj_a = Object('a', block_type)
        dict_binding = {'x': obj_a}

        # Should accept dict and convert to ParameterBinding
        grounded = GroundedAction.from_components(simple_action, dict_binding)

        assert grounded.action == simple_action
        assert isinstance(grounded.binding, ParameterBinding)
        assert grounded.object_names() == ['a']

    def test_to_tuple_backward_compat(self, simple_action):
        """Test backward compatibility with tuple representation."""
        from unified_planning.shortcuts import UserType
        block_type = UserType("block")
        obj_a = Object('a', block_type)
        binding = ParameterBinding({'x': obj_a})
        grounded = GroundedAction(simple_action, binding)

        action, dict_binding = grounded.to_tuple()

        assert action == simple_action
        assert dict_binding == {'x': obj_a}
        assert isinstance(dict_binding, dict)

    def test_consistency_with_grounded_fluent_pattern(self, simple_action):
        """Test that GroundedAction follows same pattern as GroundedFluent."""
        from unified_planning.shortcuts import UserType

        # GroundedFluent pattern
        fluent = GroundedFluent('clear', ['a'])
        fluent_str = fluent.to_string()  # "clear_a"

        # GroundedAction should follow same pattern
        block_type = UserType("block")
        obj_a = Object('a', block_type)
        binding = ParameterBinding({'x': obj_a})
        action = GroundedAction(simple_action, binding)
        action_str = action.to_string()  # "pick-up_a"

        # Both use underscore separation
        assert '_' in action_str
        assert '_' in fluent_str


class TestCriticalSemantics:
    """Test critical semantic distinctions."""

    def test_parameter_bound_uses_action_param_names(self):
        """CRITICAL: Parameter-bound literals use action's parameter names.

        For action pick-up(?x), precondition clear(?x) uses ?x, NOT ?y or ?a.
        """
        # This represents clear(?x) for pick-up(?x)
        lit = ParameterBoundLiteral('clear', ['?x'])

        # Must preserve parameter name from action
        assert lit.to_string() == 'clear(?x)'
        assert '?x' in lit.to_string()  # Specific param name

    def test_different_actions_different_param_names(self):
        """Different actions can have different parameter names."""
        # pick-up(?x) has precondition clear(?x)
        lit1 = ParameterBoundLiteral('clear', ['?x'])

        # stack(?x,?y) has precondition clear(?y)
        lit2 = ParameterBoundLiteral('clear', ['?y'])

        # Different parameter names for different actions
        assert lit1.to_string() == 'clear(?x)'
        assert lit2.to_string() == 'clear(?y)'
        assert lit1.to_string() != lit2.to_string()
