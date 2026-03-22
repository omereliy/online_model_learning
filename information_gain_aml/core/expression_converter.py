"""
FNode expression converter for PDDL representations.

This module centralizes conversion logic from Unified Planning FNode
expressions to various string representations.

CRITICAL: Parameter-bound conversions preserve action's parameter names.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from unified_planning.model.fnode import FNode
from unified_planning.model import Object, Parameter


@dataclass
class ParameterBinding:
    """Type-safe wrapper for parameter -> object mappings.

    Represents binding of action parameters to concrete objects.

    Attributes:
        bindings: Dict mapping parameter names (without ?) to Object instances
    """
    bindings: Dict[str, Object]

    def get_object(self, param_name: str) -> Object:
        """Get object bound to parameter."""
        return self.bindings[param_name]

    def object_names(self) -> List[str]:
        """Get list of object names in parameter order."""
        return [obj.name for obj in self.bindings.values()]

    def to_dict(self) -> Dict[str, Object]:
        """Convert to plain dict."""
        return self.bindings


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
            # Propositional fluent (no parameters)
            return f"{fluent.name}()"

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
