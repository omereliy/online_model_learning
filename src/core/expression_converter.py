"""
FNode expression converter for PDDL representations.

This module centralizes conversion logic from Unified Planning FNode
expressions to various string representations.

CRITICAL: Parameter-bound conversions preserve action's parameter names.
"""

from typing import List, Optional, Tuple
from unified_planning.model.fnode import FNode
from unified_planning.model import Parameter
from src.core.pddl_types import ParameterBinding


class ExpressionConverter:
    """Converts UP FNode expressions to string representations.

    This class provides static methods for converting FNode expressions
    to different string formats while preserving semantic distinctions
    between parameter-bound and grounded representations.
    """

    @staticmethod
    def to_parameter_bound_string(expr: FNode,
                                   action_parameters: List[Parameter]) -> Optional[str]:
        """Convert FNode to parameter-bound literal string.

        CRITICAL: Uses action's parameter names, not arbitrary variables.

        Args:
            expr: FNode expression (typically fluent expression)
            action_parameters: Action's Parameter objects (preserves names)

        Returns:
            String like "clear(?x)" using action's parameter name

        Example:
            For action pick-up(?x):
            - expr = clear(x) FNode
            - action_parameters = [Parameter('x', block_type)]
            - Returns: "clear(?x)" ← Uses ?x from action
        """
        if not hasattr(expr, 'fluent'):
            return None

        fluent = expr.fluent()

        if expr.args:
            # Has parameters - create parameter-bound representation
            param_strs = []
            for arg in expr.args:
                # Find matching parameter in action's parameters
                param_name = None
                for param in action_parameters:
                    if str(arg) == param.name:
                        param_name = f"?{param.name}"
                        break

                if param_name:
                    param_strs.append(param_name)
                else:
                    # Not a parameter, use object name (shouldn't happen in normal case)
                    param_strs.append(str(arg))

            return f"{fluent.name}({','.join(param_strs)})"
        else:
            # Propositional fluent
            return fluent.name

    @staticmethod
    def to_grounded_string(expr: FNode,
                          binding: ParameterBinding) -> Optional[str]:
        """Convert FNode to grounded fluent string.

        Args:
            expr: FNode expression
            binding: Parameter binding (param → object mapping)

        Returns:
            String like "clear_a" or "on_a_b"
        """
        # Handle AND expressions recursively
        if hasattr(expr, 'is_and') and expr.is_and():
            # Shouldn't happen in normal precondition extraction
            return None

        # Handle OR expressions recursively
        if hasattr(expr, 'is_or') and expr.is_or():
            return None

        # Handle NOT expressions
        if hasattr(expr, 'is_not') and expr.is_not():
            inner = expr.args[0] if expr.args else expr
            inner_str = ExpressionConverter.to_grounded_string(inner, binding)
            if inner_str:
                return f"-{inner_str}"  # Negative prefix for CNF
            return None

        # Handle fluent expressions
        if hasattr(expr, 'is_fluent_exp') and expr.is_fluent_exp():
            fluent = expr.fluent()

            if expr.args:
                # Has parameters - ground them
                grounded_args = []
                for arg in expr.args:
                    param_name = str(arg)
                    if param_name in binding.bindings:
                        grounded_args.append(binding.bindings[param_name].name)
                    else:
                        # Try without quotes
                        param_name_clean = param_name.replace("'", "")
                        if param_name_clean in binding.bindings:
                            grounded_args.append(binding.bindings[param_name_clean].name)
                        else:
                            grounded_args.append(param_name_clean)

                return f"{fluent.name}_{'_'.join(grounded_args)}"
            else:
                # Propositional
                return fluent.name

        return None

    @staticmethod
    def to_cnf_clauses(expr: FNode,
                       action_parameters: List[Parameter]) -> List[List[str]]:
        """Extract CNF clauses from FNode expression.

        Args:
            expr: FNode expression (can be AND/OR/NOT/FLUENT)
            action_parameters: Action's parameters (for parameter-bound literals)

        Returns:
            List of CNF clauses (each clause is list of literals)
        """
        clauses = []

        # Handle different expression types
        if hasattr(expr, 'is_and') and expr.is_and():
            # AND: each operand becomes a separate clause
            for arg in expr.args:
                sub_clauses = ExpressionConverter.to_cnf_clauses(arg, action_parameters)
                clauses.extend(sub_clauses)

        elif hasattr(expr, 'is_or') and expr.is_or():
            # OR: combine operands into single clause
            clause = []
            for arg in expr.args:
                sub_clauses = ExpressionConverter.to_cnf_clauses(arg, action_parameters)
                if sub_clauses and sub_clauses[0]:
                    clause.extend(sub_clauses[0])
            if clause:
                clauses.append(clause)

        elif hasattr(expr, 'is_not') and expr.is_not():
            # NOT: negate the inner expression
            inner = expr.args[0] if expr.args else expr
            param_bound_str = ExpressionConverter.to_parameter_bound_string(
                inner, action_parameters
            )
            if param_bound_str:
                clauses.append([f"-{param_bound_str}"])

        else:
            # Base case: fluent expression
            param_bound_str = ExpressionConverter.to_parameter_bound_string(
                expr, action_parameters
            )
            if param_bound_str:
                clauses.append([param_bound_str])

        return clauses
