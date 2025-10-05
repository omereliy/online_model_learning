"""
Grounding and lifting operations for PDDL literals and fluents.

This module implements the bindP and bindP⁻¹ operations from the
Information Gain algorithm, converting between parameter-bound and
grounded representations.

Algorithm Context:
- bindP⁻¹(F, O): Ground parameter-bound literals with objects
- bindP(f, O): Lift grounded fluents to parameter-bound form
"""

from typing import Set, List
import logging
from src.core.pddl_handler import PDDLHandler

logger = logging.getLogger(__name__)


class FluentBinder:
    """Handles conversion between parameter-bound and grounded representations.

    This class implements the binding operations from the Information Gain
    algorithm, providing clear semantics for grounding and lifting.

    Operations:
    - ground_literal: Convert clear(?x) + [a] → clear_a
    - lift_fluent: Convert clear_a + [a] → clear(?x)
    - ground_literals: Batch bindP⁻¹ operation
    - lift_fluents: Batch bindP operation
    """

    def __init__(self, pddl_handler: PDDLHandler):
        """Initialize FluentBinder.

        Args:
            pddl_handler: PDDLHandler instance for parameter name generation
        """
        self.pddl_handler = pddl_handler

    def ground_literal(self, param_bound_literal: str, objects: List[str]) -> str:
        """Ground a parameter-bound literal with concrete objects.

        Implements single-literal version of bindP⁻¹.

        Args:
            param_bound_literal: e.g., "clear(?x)" or "¬on(?x,?y)"
            objects: Ordered list matching parameters, e.g., ['a'] or ['a', 'b']

        Returns:
            Grounded fluent, e.g., "clear_a" or "¬on_a_b"

        Example:
            ground_literal("clear(?x)", ["a"]) → "clear_a"
            ground_literal("on(?x,?y)", ["a", "b"]) → "on_a_b"
        """
        # Handle negation
        is_negative = param_bound_literal.startswith('¬')
        if is_negative:
            param_bound_literal = param_bound_literal[1:]

        # Parse literal: predicate(param1,param2,...)
        if '(' not in param_bound_literal:
            # Propositional literal
            grounded = param_bound_literal
        else:
            predicate = param_bound_literal[:param_bound_literal.index('(')]
            params_str = param_bound_literal[param_bound_literal.index('(') + 1:param_bound_literal.rindex(')')]

            if not params_str:
                # No parameters
                grounded = predicate
            else:
                params = [p.strip() for p in params_str.split(',')]

                # Replace each parameter with corresponding object
                grounded_params = []
                for param in params:
                    if param.startswith('?'):
                        # Extract parameter index from name
                        param_idx = PDDLHandler.parameter_index_from_name(param)
                        if param_idx < len(objects):
                            grounded_params.append(objects[param_idx])
                        else:
                            logger.warning(f"Parameter {param} index {param_idx} out of bounds for objects {objects}")
                            grounded_params.append(param)
                    else:
                        # Already grounded
                        grounded_params.append(param)

                # Create grounded fluent string
                grounded = '_'.join([predicate] + grounded_params)

        # Add back negation if needed
        return f"¬{grounded}" if is_negative else grounded

    def lift_fluent(self, grounded_fluent: str, objects: List[str]) -> str:
        """Lift grounded fluent to parameter-bound form.

        Implements single-fluent version of bindP.

        Args:
            grounded_fluent: e.g., "clear_a" or "¬on_a_b"
            objects: Ordered list used in grounding, e.g., ['a'] or ['a', 'b']

        Returns:
            Parameter-bound literal, e.g., "clear(?x)" or "¬on(?x,?y)"

        Example:
            lift_fluent("clear_a", ["a"]) → "clear(?x)"
            lift_fluent("on_a_b", ["a", "b"]) → "on(?x,?y)"
        """
        # Handle negation
        is_negative = grounded_fluent.startswith('¬')
        if is_negative:
            grounded_fluent = grounded_fluent[1:]

        # Parse fluent: predicate_obj1_obj2_...
        parts = grounded_fluent.split('_')

        if len(parts) == 1:
            # Propositional fluent
            lifted = parts[0]
        else:
            # First part is predicate, rest are object names
            predicate = parts[0]
            obj_names = parts[1:]

            # Replace each object with its parameter
            params = []
            for obj_name in obj_names:
                try:
                    obj_idx = objects.index(obj_name)
                    param_name = PDDLHandler.generate_parameter_names(obj_idx + 1)[obj_idx]
                    params.append(param_name)
                except ValueError:
                    logger.warning(f"Object {obj_name} not found in objects list {objects}")
                    params.append(obj_name)

            # Create lifted literal
            if params:
                lifted = f"{predicate}({','.join(params)})"
            else:
                lifted = predicate

        # Add back negation if needed
        return f"¬{lifted}" if is_negative else lifted

    def ground_literals(self, literals: Set[str], objects: List[str]) -> Set[str]:
        """Ground parameter-bound literals with concrete objects.

        Implements bindP⁻¹(F, O) operation from Information Gain algorithm.

        Args:
            literals: Set of parameter-bound literals (e.g., {'clear(?x)', '¬on(?x,?y)'})
            objects: Ordered list of objects (e.g., ['a', 'b'])

        Returns:
            Set of grounded literals (e.g., {'clear_a', '¬on_a_b'})

        Example:
            ground_literals({'clear(?x)', 'handempty'}, ['a'])
            → {'clear_a', 'handempty'}
        """
        grounded = set()

        for literal in literals:
            grounded_literal = self.ground_literal(literal, objects)
            grounded.add(grounded_literal)

        return grounded

    def lift_fluents(self, fluents: Set[str], objects: List[str]) -> Set[str]:
        """Lift grounded fluents to parameter-bound literals.

        Implements bindP(f, O) operation from Information Gain algorithm.

        Args:
            fluents: Set of grounded fluent strings (e.g., {'clear_a', '¬on_a_b'})
            objects: Ordered list of objects (e.g., ['a', 'b'])

        Returns:
            Set of parameter-bound literals (e.g., {'clear(?x)', '¬on(?x,?y)'})

        Example:
            lift_fluents({'clear_a', 'handempty'}, ['a'])
            → {'clear(?x)', 'handempty'}
        """
        lifted = set()

        for fluent in fluents:
            lifted_literal = self.lift_fluent(fluent, objects)
            lifted.add(lifted_literal)

        return lifted
