"""
Type-safe data classes for PDDL representations.

This module provides domain-specific types to replace primitive types
throughout the codebase, improving type safety and code clarity.
"""

from dataclasses import dataclass
from typing import Dict, List
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
