"""
Tests for PDDL type-safe data classes.
"""

import pytest
from unified_planning.shortcuts import UserType
from unified_planning.model import Object
from information_gain_aml.core.expression_converter import ParameterBinding


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
