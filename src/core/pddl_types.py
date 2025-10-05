"""
Type-safe data classes for PDDL representations.

This module provides domain-specific types to replace primitive types
throughout the codebase, improving type safety and code clarity.

Critical Semantic Distinction:
- ParameterBoundLiteral: Uses action's specific parameter names (e.g., clear(?x))
- GroundedFluent: Fully instantiated with objects (e.g., clear_a)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from unified_planning.model import Object


@dataclass
class ParameterBinding:
    """Type-safe wrapper for parameter → object mappings.

    Represents binding of action parameters to concrete objects.

    Example:
        For action pick-up(?x) with object 'a':
        ParameterBinding({'x': Object('a', 'block')})

    Attributes:
        bindings: Dict mapping parameter names (without ?) to Object instances
    """
    bindings: Dict[str, Object]

    def get_object(self, param_name: str) -> Object:
        """Get object bound to parameter.

        Args:
            param_name: Parameter name (without '?')

        Returns:
            Object bound to this parameter

        Raises:
            KeyError: If parameter not in bindings
        """
        return self.bindings[param_name]

    def object_names(self) -> List[str]:
        """Get list of object names in parameter order.

        Returns:
            List of object names (e.g., ['a', 'b'])
        """
        return [obj.name for obj in self.bindings.values()]

    def to_dict(self) -> Dict[str, Object]:
        """Convert to plain dict.

        Returns:
            Dictionary mapping parameter names to Objects
        """
        return self.bindings


@dataclass
class ParameterBoundLiteral:
    """Literal using action's parameter names.

    CRITICAL: These literals use the action's specific parameter names,
    not arbitrary variable names.

    Example:
        For action pick-up(?x):
        - Correct: ParameterBoundLiteral('clear', ['?x'])
        - WRONG: ParameterBoundLiteral('clear', ['?a']) or ['?obj']

    Attributes:
        predicate: Predicate name (e.g., 'clear', 'on')
        parameters: List of parameter names (e.g., ['?x', '?y'])
        is_negative: True if negated literal (¬)
    """
    predicate: str
    parameters: List[str]
    is_negative: bool = False

    def to_string(self) -> str:
        """Convert to string representation.

        Returns:
            String like "clear(?x)" or "¬on(?x,?y)"
        """
        if self.parameters:
            param_str = ','.join(self.parameters)
            base = f"{self.predicate}({param_str})"
        else:
            base = self.predicate

        return f"¬{base}" if self.is_negative else base

    @classmethod
    def from_string(cls, s: str) -> 'ParameterBoundLiteral':
        """Parse parameter-bound literal from string.

        Args:
            s: String like "clear(?x)" or "¬on(?x,?y)"

        Returns:
            ParameterBoundLiteral instance
        """
        # Check for negation
        is_negative = s.startswith('¬')
        if is_negative:
            s = s[1:]  # Remove negation symbol

        # Parse predicate and parameters
        if '(' in s:
            # Has parameters
            predicate = s[:s.index('(')]
            param_str = s[s.index('(') + 1:s.rindex(')')]
            parameters = [p.strip() for p in param_str.split(',')]
        else:
            # Propositional
            predicate = s
            parameters = []

        return cls(predicate, parameters, is_negative)


@dataclass
class GroundedFluent:
    """Fully grounded fluent with concrete objects.

    Example:
        GroundedFluent('clear', ['a']) → "clear_a"
        GroundedFluent('on', ['a', 'b']) → "on_a_b"

    Attributes:
        predicate: Predicate name
        objects: List of object names
    """
    predicate: str
    objects: List[str]

    def to_string(self) -> str:
        """Convert to string representation.

        Returns:
            String like "clear_a" or "on_a_b"
        """
        if self.objects:
            return f"{self.predicate}_{'_'.join(self.objects)}"
        else:
            return self.predicate

    @classmethod
    def from_string(cls, s: str) -> 'GroundedFluent':
        """Parse grounded fluent from string.

        Args:
            s: String like "clear_a" or "on_a_b"

        Returns:
            GroundedFluent instance
        """
        parts = s.split('_')

        if len(parts) == 1:
            # Propositional
            return cls(parts[0], [])
        else:
            # Has objects
            return cls(parts[0], parts[1:])
