"""
UPAdapter: Bidirectional converter between Unified Planning types and project types.

This module provides stateless conversion functions between:
- UP State ↔ Set[str] of grounded fluents
- UP Action ↔ Project type representations
- UP FNode expressions ↔ String representations

Design principles:
- Stateless: all methods are static or pure functions
- No domain knowledge stored
- Clear conversion semantics
"""

import logging
from typing import Set, Dict, Any, List, Optional
from unified_planning.model import Problem, State
from unified_planning.shortcuts import *

logger = logging.getLogger(__name__)


class UPAdapter:
    """
    Bidirectional adapter between UP and project types.

    All methods are static - this is a stateless converter.
    """

    # ========== UP → Project Conversions ==========

    @staticmethod
    def up_state_to_fluent_set(state: Any, problem: Problem) -> Set[str]:
        """
        Convert UP state to set of grounded fluent strings.

        Handles both:
        - UP State objects (with get_value method)
        - UP initial_values dict

        Args:
            state: UP state object or initial_values dict
            problem: UP Problem for fluent definitions

        Returns:
            Set of grounded fluent strings (e.g., {'clear_a', 'on_a_b'})

        Example:
            state = simulator.get_initial_state()
            fluents = UPAdapter.up_state_to_fluent_set(state, problem)
            # → {'clear_a', 'on_b_c', 'handempty'}
        """
        if isinstance(state, dict):
            return UPAdapter._fluents_from_initial_values(state, problem)
        else:
            return UPAdapter._fluents_from_state_object(state, problem)

    @staticmethod
    def _fluents_from_initial_values(initial_values: Dict, problem: Problem) -> Set[str]:
        """
        Extract true fluents from UP initial_values dict.

        Args:
            initial_values: Problem.initial_values dict
            problem: UP Problem

        Returns:
            Set of true grounded fluent strings
        """
        true_fluents = set()

        for fluent_expr, value in initial_values.items():
            # Check if value is true
            if value.bool_constant_value():
                # Convert fluent expression to string
                fluent_str = UPAdapter._fluent_expression_to_string(fluent_expr)
                if fluent_str:
                    true_fluents.add(fluent_str)

        return true_fluents

    @staticmethod
    def _fluents_from_state_object(state: State, problem: Problem) -> Set[str]:
        """
        Extract true fluents from UP State object.

        Args:
            state: UP State object with get_value method
            problem: UP Problem for fluent definitions

        Returns:
            Set of true grounded fluent strings
        """
        true_fluents = set()

        # Iterate through all possible fluents and check values
        for fluent in problem.fluents:
            if fluent.arity == 0:
                # Propositional fluent (no parameters)
                if state.get_value(fluent()).bool_constant_value():
                    true_fluents.add(fluent.name)
            else:
                # Relational fluent - check all groundings
                for combo in UPAdapter._get_object_combinations(fluent, problem):
                    fluent_expr = fluent(*combo)
                    if state.get_value(fluent_expr).bool_constant_value():
                        # Convert to string: on(a, b) → on_a_b
                        parts = [fluent.name] + [obj.name for obj in combo]
                        grounded_name = '_'.join(parts)
                        true_fluents.add(grounded_name)

        return true_fluents

    @staticmethod
    def _fluent_expression_to_string(fluent_expr) -> Optional[str]:
        """
        Convert UP fluent expression to string format.

        Args:
            fluent_expr: UP fluent expression (FNode)

        Returns:
            Grounded fluent string (e.g., 'on_a_b') or None
        """
        try:
            if hasattr(fluent_expr, 'fluent'):
                # It's a fluent application
                fluent_name = fluent_expr.fluent().name
                if fluent_expr.args:
                    # Has parameters - extract object names
                    param_names = [str(arg).replace("'", "") for arg in fluent_expr.args]
                    return '_'.join([fluent_name] + param_names)
                else:
                    # No parameters (propositional)
                    return fluent_name
            else:
                # Try direct conversion
                return str(fluent_expr)
        except Exception as e:
            logger.warning(f"Could not convert fluent expression: {fluent_expr}, error: {e}")
            return None

    @staticmethod
    def _get_object_combinations(fluent, problem: Problem) -> List[List]:
        """
        Get all valid object combinations for a fluent's signature.

        Args:
            fluent: UP Fluent
            problem: UP Problem

        Returns:
            List of object combinations matching fluent signature
        """
        import itertools

        # Get parameter types from fluent signature
        param_types = [param.type for param in fluent.signature]

        # Get matching objects for each parameter
        object_lists = []
        for req_type in param_types:
            matching = [
                obj for obj in problem.all_objects
                if UPAdapter._type_matches(obj.type, req_type)
            ]
            object_lists.append(matching)

        # Generate all combinations
        return list(itertools.product(*object_lists))

    @staticmethod
    def _type_matches(obj_type, param_type) -> bool:
        """
        Check if object type matches parameter type.

        Args:
            obj_type: Object's type
            param_type: Parameter's required type

        Returns:
            True if compatible
        """
        try:
            return obj_type == param_type or param_type.is_compatible(obj_type)
        except AttributeError:
            return obj_type == param_type

    @staticmethod
    def fluent_set_to_up_state(fluents: Set[str], problem: Problem) -> Dict:
        """
        Convert set of grounded fluent strings to UP state dict format.

        Args:
            fluents: Set of true grounded fluents
            problem: UP Problem

        Returns:
            State dictionary (all fluents with boolean values)

        Example:
            fluents = {'clear_a', 'on_b_c'}
            state_dict = UPAdapter.fluent_set_to_up_state(fluents, problem)
            # → {'clear_a': True, 'clear_b': False, 'on_a_b': False, ...}
        """
        state_dict = {}

        # Generate all possible grounded fluents and mark which are true
        for fluent in problem.fluents:
            if fluent.arity == 0:
                # Propositional fluent
                state_dict[fluent.name] = fluent.name in fluents
            else:
                # Relational fluent - check all groundings
                for combo in UPAdapter._get_object_combinations(fluent, problem):
                    parts = [fluent.name] + [obj.name for obj in combo]
                    grounded_name = '_'.join(parts)
                    state_dict[grounded_name] = grounded_name in fluents

        return state_dict

    @staticmethod
    def get_all_grounded_fluents(problem: Problem) -> List[str]:
        """
        Get list of all possible grounded fluent strings for a problem.

        Args:
            problem: UP Problem

        Returns:
            List of all grounded fluent strings

        Example:
            fluents = UPAdapter.get_all_grounded_fluents(problem)
            # → ['clear_a', 'clear_b', 'on_a_b', 'on_b_a', ...]
        """
        grounded = []

        for fluent in problem.fluents:
            if fluent.arity == 0:
                # Propositional fluent
                grounded.append(fluent.name)
            else:
                # Generate all groundings
                for combo in UPAdapter._get_object_combinations(fluent, problem):
                    parts = [fluent.name] + [obj.name for obj in combo]
                    grounded_name = '_'.join(parts)
                    grounded.append(grounded_name)

        return grounded

    # ========== Parameter-bound Literal Conversions ==========

    @staticmethod
    def up_expression_to_parameter_bound(expr, parameters: List) -> Optional[str]:
        """
        Convert UP expression to parameter-bound string.

        Uses the action's parameters to generate standard variable names.

        Args:
            expr: UP FNode expression
            parameters: Action parameters list

        Returns:
            Parameter-bound string (e.g., 'on(?x,?y)') or None

        Example:
            expr = <on(x, y) FNode>
            params = [Parameter('x', 'block'), Parameter('y', 'block')]
            result = UPAdapter.up_expression_to_parameter_bound(expr, params)
            # → 'on(?x,?y)'
        """
        from src.core.expression_converter import ExpressionConverter
        return ExpressionConverter.to_parameter_bound_string(expr, parameters)

    @staticmethod
    def up_expression_to_grounded(expr, binding: Dict) -> Optional[str]:
        """
        Convert UP expression to grounded string with parameter binding.

        Args:
            expr: UP FNode expression
            binding: Dict mapping parameter names to objects

        Returns:
            Grounded string (e.g., 'on_a_b') or None

        Example:
            expr = <on(x, y) FNode>
            binding = {'x': Object('a'), 'y': Object('b')}
            result = UPAdapter.up_expression_to_grounded(expr, binding)
            # → 'on_a_b'
        """
        from src.core.expression_converter import ExpressionConverter
        from src.core.pddl_types import ParameterBinding

        pb = ParameterBinding(binding)
        return ExpressionConverter.to_grounded_string(expr, pb)

    # ========== Utility Methods ==========

    @staticmethod
    def get_initial_state_as_fluent_set(problem: Problem) -> Set[str]:
        """
        Get initial state from UP problem as fluent set.

        Convenience method that extracts and converts initial state.

        Args:
            problem: UP Problem

        Returns:
            Set of true fluents in initial state
        """
        return UPAdapter.up_state_to_fluent_set(problem.initial_values, problem)
