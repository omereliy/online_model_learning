"""
Tests for grounding and lifting operations (bindP, bindP⁻¹).
These tests are written BEFORE implementation (TDD).
"""

import pytest
from src.core.binding_operations import FluentBinder
from src.core.pddl_handler import PDDLHandler


class TestFluentBinder:
    """Test FluentBinder class for bindP/bindP⁻¹ operations."""

    @pytest.fixture
    def pddl_handler(self, temp_dir, blocksworld_domain, blocksworld_problem):
        """Create PDDLHandler for testing."""
        domain_file = temp_dir / "domain.pddl"
        problem_file = temp_dir / "problem.pddl"

        domain_file.write_text(blocksworld_domain)
        problem_file.write_text(blocksworld_problem)

        handler = PDDLHandler()
        handler.parse_domain_and_problem(str(domain_file), str(problem_file))

        return handler

    @pytest.fixture
    def binder(self, pddl_handler):
        """Create FluentBinder for testing."""
        return FluentBinder(pddl_handler)

    def test_ground_literal_single_param(self, binder):
        """Test grounding parameter-bound literal with single parameter."""
        literal = "clear(?x)"
        objects = ["a"]

        result = binder.ground_literal(literal, objects)

        assert result == "clear_a"

    def test_ground_literal_multi_param(self, binder):
        """Test grounding multi-parameter literal."""
        literal = "on(?x,?y)"
        objects = ["a", "b"]

        result = binder.ground_literal(literal, objects)

        assert result == "on_a_b"

    def test_ground_literal_propositional(self, binder):
        """Test grounding propositional literal (no params)."""
        literal = "handempty"
        objects = []

        result = binder.ground_literal(literal, objects)

        assert result == "handempty"

    def test_ground_literal_negative(self, binder):
        """Test grounding negative literal."""
        literal = "¬clear(?x)"
        objects = ["a"]

        result = binder.ground_literal(literal, objects)

        assert result == "¬clear_a"

    def test_lift_fluent_single_param(self, binder):
        """Test lifting grounded fluent to parameter-bound."""
        fluent = "clear_a"
        objects = ["a"]

        result = binder.lift_fluent(fluent, objects)

        assert result == "clear(?x)"

    def test_lift_fluent_multi_param(self, binder):
        """Test lifting multi-parameter fluent."""
        fluent = "on_a_b"
        objects = ["a", "b"]

        result = binder.lift_fluent(fluent, objects)

        assert result == "on(?x,?y)"

    def test_lift_fluent_propositional(self, binder):
        """Test lifting propositional fluent."""
        fluent = "handempty"
        objects = []

        result = binder.lift_fluent(fluent, objects)

        assert result == "handempty"

    def test_lift_fluent_negative(self, binder):
        """Test lifting negative fluent."""
        fluent = "¬clear_a"
        objects = ["a"]

        result = binder.lift_fluent(fluent, objects)

        assert result == "¬clear(?x)"

    def test_ground_literals_batch(self, binder):
        """Test batch grounding (bindP⁻¹ operation)."""
        literals = {"clear(?x)", "ontable(?x)", "handempty"}
        objects = ["a"]

        result = binder.ground_literals(literals, objects)

        assert result == {"clear_a", "ontable_a", "handempty"}

    def test_lift_fluents_batch(self, binder):
        """Test batch lifting (bindP operation)."""
        fluents = {"clear_a", "ontable_a", "handempty"}
        objects = ["a"]

        result = binder.lift_fluents(fluents, objects)

        assert result == {"clear(?x)", "ontable(?x)", "handempty"}

    def test_ground_then_lift_inverse(self, binder):
        """Test that lift(ground(x)) = x (inverse operations)."""
        original = {"clear(?x)", "on(?x,?y)"}
        objects = ["a", "b"]

        grounded = binder.ground_literals(original, objects)
        lifted = binder.lift_fluents(grounded, objects)

        assert lifted == original

    def test_parameter_index_mapping(self, binder):
        """Test parameter index correctly maps to objects."""
        # ?x → index 0 → 'a'
        # ?y → index 1 → 'b'
        literal = "on(?x,?y)"
        objects = ["a", "b"]

        result = binder.ground_literal(literal, objects)

        assert result == "on_a_b"
        # Verify order: first param (?x) → first object (a)
        assert "_a_" in result

    def test_ground_literal_with_negation_complex(self, binder):
        """Test grounding complex negative multi-parameter literal."""
        literal = "¬on(?x,?y)"
        objects = ["a", "b"]

        result = binder.ground_literal(literal, objects)

        assert result == "¬on_a_b"

    def test_lift_fluent_with_negation_complex(self, binder):
        """Test lifting complex negative multi-parameter fluent."""
        fluent = "¬on_a_b"
        objects = ["a", "b"]

        result = binder.lift_fluent(fluent, objects)

        assert result == "¬on(?x,?y)"

    def test_ground_literals_with_mixed_negation(self, binder):
        """Test batch grounding with mixed positive and negative literals."""
        literals = {"clear(?x)", "¬ontable(?x)", "handempty"}
        objects = ["a"]

        result = binder.ground_literals(literals, objects)

        assert result == {"clear_a", "¬ontable_a", "handempty"}

    def test_lift_fluents_with_mixed_negation(self, binder):
        """Test batch lifting with mixed positive and negative fluents."""
        fluents = {"clear_a", "¬ontable_a", "handempty"}
        objects = ["a"]

        result = binder.lift_fluents(fluents, objects)

        assert result == {"clear(?x)", "¬ontable(?x)", "handempty"}
