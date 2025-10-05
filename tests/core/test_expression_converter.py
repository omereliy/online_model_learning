"""
Tests for FNode expression conversion logic.
These tests are written BEFORE implementation (TDD).
"""

import pytest
from unified_planning.model import Fluent, Problem, Object
from unified_planning.shortcuts import BoolType, UserType, InstantaneousAction
from src.core.expression_converter import ExpressionConverter
from src.core.pddl_types import ParameterBinding


class TestExpressionConverter:
    """Test ExpressionConverter class."""

    @pytest.fixture
    def simple_problem(self):
        """Create simple blocksworld problem for testing."""
        problem = Problem("test")

        # Create types (no need to add explicitly to problem)
        block_type = UserType("block")

        # Add objects
        a = Object("a", block_type)
        b = Object("b", block_type)
        problem.add_object(a)
        problem.add_object(b)

        # Add fluents
        clear = Fluent("clear", BoolType(), block=block_type)
        on = Fluent("on", BoolType(), block1=block_type, block2=block_type)
        handempty = Fluent("handempty", BoolType())

        problem.add_fluent(clear, default_initial_value=False)
        problem.add_fluent(on, default_initial_value=False)
        problem.add_fluent(handempty, default_initial_value=True)

        # Add action
        pickup = InstantaneousAction("pick-up", x=block_type)
        x = pickup.parameter("x")

        # Preconditions: clear(x) AND ontable(x) AND handempty
        problem.add_action(pickup)

        return problem, pickup, clear, on, handempty, a, b

    def test_to_parameter_bound_string_single_param(self, simple_problem):
        """Test converting fluent expression to parameter-bound string."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        x = pickup.parameter("x")
        expr = clear(x)  # FNode for clear(?x)

        result = ExpressionConverter.to_parameter_bound_string(expr, pickup.parameters)

        # Must use action's parameter name (?x from pick-up(?x))
        assert result == "clear(?x)"
        assert "?x" in result  # Verify parameter name preserved

    def test_to_parameter_bound_string_multi_param(self, simple_problem):
        """Test multi-parameter fluent."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        # Create stack action with two parameters
        block_type = UserType("block")
        stack = InstantaneousAction("stack", x=block_type, y=block_type)
        x = stack.parameter("x")
        y = stack.parameter("y")

        expr = on(x, y)  # FNode for on(?x, ?y)

        result = ExpressionConverter.to_parameter_bound_string(expr, stack.parameters)

        assert result == "on(?x,?y)"
        assert "?x" in result and "?y" in result

    def test_to_parameter_bound_string_propositional(self, simple_problem):
        """Test propositional fluent (no parameters)."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        expr = handempty()  # FNode for handempty

        result = ExpressionConverter.to_parameter_bound_string(expr, pickup.parameters)

        assert result == "handempty"

    def test_to_grounded_string_single_param(self, simple_problem):
        """Test converting to grounded string with binding."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        x = pickup.parameter("x")
        expr = clear(x)  # FNode for clear(?x)

        binding = ParameterBinding({"x": a})
        result = ExpressionConverter.to_grounded_string(expr, binding)

        assert result == "clear_a"

    def test_to_grounded_string_multi_param(self, simple_problem):
        """Test grounding multi-parameter fluent."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        block_type = UserType("block")
        stack = InstantaneousAction("stack", x=block_type, y=block_type)
        x = stack.parameter("x")
        y = stack.parameter("y")

        expr = on(x, y)

        binding = ParameterBinding({"x": a, "y": b})
        result = ExpressionConverter.to_grounded_string(expr, binding)

        assert result == "on_a_b"

    def test_to_grounded_string_propositional(self, simple_problem):
        """Test grounding propositional fluent."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        expr = handempty()

        binding = ParameterBinding({})
        result = ExpressionConverter.to_grounded_string(expr, binding)

        assert result == "handempty"

    def test_to_cnf_clauses_single_literal(self, simple_problem):
        """Test converting single literal to CNF."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        x = pickup.parameter("x")
        expr = clear(x)

        result = ExpressionConverter.to_cnf_clauses(expr, pickup.parameters)

        # Single literal becomes single-element clause
        assert result == [["clear(?x)"]]

    def test_to_cnf_clauses_and_expression(self, simple_problem):
        """Test AND expression becomes multiple clauses."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        # This test requires creating an AND expression
        # Will be implemented based on UP's expression builder
        pass  # TODO: Implement when we have AND expression

    def test_preserves_action_parameter_names(self, simple_problem):
        """CRITICAL: Verify parameter names from action are preserved."""
        problem, pickup, clear, on, handempty, a, b = simple_problem

        x = pickup.parameter("x")
        expr = clear(x)

        # pick-up has parameter ?x
        result = ExpressionConverter.to_parameter_bound_string(expr, pickup.parameters)

        # Must use ?x, not ?y or any other name
        assert result == "clear(?x)"
        assert result != "clear(?y)"
        assert result != "clear(?a)"
