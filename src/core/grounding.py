"""
Grounding utilities: Functional operations for grounding and lifting.

Pure functions that convert between lifted and grounded representations.
All functions are stateless - they receive domain knowledge and objects
as parameters and return results.

Key operations:
- Ground actions: LiftedAction + objects → GroundedAction
- Ground literals: parameter-bound string + objects → grounded string
- Lift fluents: grounded string + objects → parameter-bound string
- Parse grounded action strings
"""

import logging
import itertools
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Tuple

from src.core.lifted_domain import LiftedDomainKnowledge, LiftedAction, ObjectInfo

logger = logging.getLogger(__name__)


# ========== Data Structures ==========

@dataclass
class GroundedAction:
    """
    Grounded action with specific object bindings.

    Attributes:
        action_name: Action schema name
        objects: Ordered list of bound objects
        grounded_preconditions: Set of grounded precondition strings
        grounded_add_effects: Set of grounded add effect strings
        grounded_del_effects: Set of grounded delete effect strings
    """
    action_name: str
    objects: List[str]
    grounded_preconditions: Set[str]
    grounded_add_effects: Set[str]
    grounded_del_effects: Set[str]

    def to_string(self) -> str:
        """
        Convert to string representation.

        Returns:
            String like 'pick-up_a' or 'stack_a_b'
        """
        if self.objects:
            return f"{self.action_name}_{'_'.join(self.objects)}"
        return self.action_name

    def __str__(self) -> str:
        return self.to_string()


# ========== Grounding Functions ==========

def ground_action(lifted_action: LiftedAction, objects: List[str]) -> GroundedAction:
    """
    Ground a lifted action with specific objects.

    Args:
        lifted_action: Lifted action schema
        objects: Ordered list matching action parameters

    Returns:
        GroundedAction with grounded preconditions and effects

    Raises:
        ValueError: If number of objects doesn't match parameters

    Example:
        action = domain.get_action('stack')  # stack(?x, ?y)
        grounded = ground_action(action, ['a', 'b'])
        # → GroundedAction(action_name='stack', objects=['a', 'b'],
        #                  grounded_preconditions={'holding_a', 'clear_b'}, ...)
    """
    if len(objects) != lifted_action.arity:
        raise ValueError(
            f"Action {lifted_action.name} requires {lifted_action.arity} objects, "
            f"got {len(objects)}"
        )

    # Ground preconditions
    grounded_preconds = {
        ground_parameter_bound_literal(lit, objects)
        for lit in lifted_action.preconditions
    }

    # Ground add effects
    grounded_adds = {
        ground_parameter_bound_literal(lit, objects)
        for lit in lifted_action.add_effects
    }

    # Ground delete effects
    grounded_dels = {
        ground_parameter_bound_literal(lit, objects)
        for lit in lifted_action.del_effects
    }

    return GroundedAction(
        action_name=lifted_action.name,
        objects=objects,
        grounded_preconditions=grounded_preconds,
        grounded_add_effects=grounded_adds,
        grounded_del_effects=grounded_dels
    )


def ground_all_actions(domain: LiftedDomainKnowledge,
                      require_injective: bool = False) -> List[GroundedAction]:
    """
    Generate all possible grounded actions for domain.

    Args:
        domain: Lifted domain knowledge
        require_injective: If True, skip bindings where same object appears multiple times
                          (e.g., stack(a, a) would be skipped)

    Returns:
        List of all grounded actions

    Example:
        actions = ground_all_actions(domain, require_injective=True)
        # → [GroundedAction('pick-up', ['a'], ...), GroundedAction('pick-up', ['b'], ...),
        #     GroundedAction('stack', ['a', 'b'], ...), ...]
    """
    grounded_actions = []

    for action_name, lifted_action in domain.lifted_actions.items():
        # Get all valid object combinations for this action's parameters
        bindings = _get_parameter_bindings(lifted_action, domain, require_injective)

        for obj_list in bindings:
            grounded = ground_action(lifted_action, obj_list)
            grounded_actions.append(grounded)

    logger.debug(f"Generated {len(grounded_actions)} grounded actions "
                f"(injective={require_injective})")
    return grounded_actions


def _get_parameter_bindings(lifted_action: LiftedAction,
                            domain: LiftedDomainKnowledge,
                            require_injective: bool) -> List[List[str]]:
    """
    Get all valid object bindings for action parameters.

    Args:
        lifted_action: Lifted action
        domain: Domain knowledge for objects and types
        require_injective: Skip non-injective bindings

    Returns:
        List of object lists
    """
    if lifted_action.arity == 0:
        return [[]]  # No parameters

    # Get valid objects for each parameter
    param_options = []
    for param in lifted_action.parameters:
        # Get objects matching parameter type (including subtypes)
        matching_objects = domain.get_objects_of_type(param.type, include_subtypes=True)
        param_options.append([obj.name for obj in matching_objects])

    # Generate all combinations
    bindings = []
    for combo in itertools.product(*param_options):
        # Check for injective requirement
        if require_injective and len(set(combo)) != len(combo):
            # Skip - same object appears multiple times
            continue

        bindings.append(list(combo))

    return bindings


def ground_parameter_bound_literal(literal: str, objects: List[str]) -> str:
    """
    Ground a parameter-bound literal with concrete objects.

    Implements bindP⁻¹ operation from Information Gain algorithm.

    Args:
        literal: Parameter-bound literal (e.g., 'on(?x,?y)' or '¬clear(?x)')
        objects: Ordered list of objects (e.g., ['a', 'b'])

    Returns:
        Grounded literal (e.g., 'on_a_b' or '¬clear_a')

    Example:
        ground_parameter_bound_literal('on(?x,?y)', ['a', 'b']) → 'on_a_b'
        ground_parameter_bound_literal('¬clear(?x)', ['a']) → '¬clear_a'
        ground_parameter_bound_literal('handempty', []) → 'handempty'
    """
    # Handle negation
    is_negative = literal.startswith('¬')
    if is_negative:
        literal = literal[1:]

    # Parse literal: predicate(param1,param2,...) or predicate
    if '(' not in literal:
        # Propositional literal (no parameters)
        grounded = literal
    else:
        predicate = literal[:literal.index('(')]
        params_str = literal[literal.index('(') + 1:literal.rindex(')')]

        if not params_str:
            # No parameters (empty parentheses)
            grounded = predicate
        else:
            params = [p.strip() for p in params_str.split(',')]

            # Replace each parameter with corresponding object
            grounded_params = []
            for param in params:
                if param.startswith('?'):
                    # Extract parameter index from name (?x→0, ?y→1, ...)
                    param_idx = _parameter_index_from_name(param)
                    if param_idx < len(objects):
                        grounded_params.append(objects[param_idx])
                    else:
                        logger.warning(f"Parameter {param} index out of bounds for objects {objects}")
                        grounded_params.append(param)  # Keep as-is
                else:
                    # Already grounded (constant)
                    grounded_params.append(param)

            # Create grounded fluent string
            grounded = '_'.join([predicate] + grounded_params)

    # Add back negation if needed
    return f"¬{grounded}" if is_negative else grounded


def lift_grounded_fluent(fluent: str, objects: List[str], domain: LiftedDomainKnowledge) -> str:
    """
    Lift grounded fluent to parameter-bound form.

    Implements bindP operation from Information Gain algorithm.

    Args:
        fluent: Grounded fluent (e.g., 'on_a_b' or '¬clear_a')
        objects: Ordered list of objects (e.g., ['a', 'b'])
        domain: Domain knowledge for predicate signatures

    Returns:
        Parameter-bound literal (e.g., 'on(?x,?y)' or '¬clear(?x)')

    Example:
        lift_grounded_fluent('on_a_b', ['a', 'b'], domain) → 'on(?x,?y)'
        lift_grounded_fluent('¬clear_a', ['a'], domain) → '¬clear(?x)'
    """
    # Handle negation
    is_negative = fluent.startswith('¬')
    if is_negative:
        fluent = fluent[1:]

    # Find predicate by matching against domain predicates
    predicate_name = None
    param_objects = []

    # Try to find matching predicate
    for pred_name in domain.predicates.keys():
        if fluent == pred_name:
            # Propositional predicate (exact match)
            predicate_name = pred_name
            param_objects = []
            break
        elif fluent.startswith(pred_name + '_'):
            # Relational predicate - extract parameters
            rest = fluent[len(pred_name) + 1:]
            parts = rest.split('_')

            # Check if parts match objects from binding
            if _parts_match_objects(parts, objects):
                predicate_name = pred_name
                param_objects = parts
                break

    if predicate_name is None:
        # Fallback: treat first part as predicate
        parts = fluent.split('_')
        predicate_name = parts[0]
        param_objects = parts[1:] if len(parts) > 1 else []

    # Create parameter-bound literal
    if not param_objects:
        lifted = predicate_name
    else:
        # Map objects to parameter variables
        param_vars = []
        for obj in param_objects:
            try:
                obj_idx = objects.index(obj)
                param_name = _parameter_name_from_index(obj_idx)
                param_vars.append(param_name)
            except ValueError:
                # Object not in binding - keep as constant
                param_vars.append(obj)

        lifted = f"{predicate_name}({','.join(param_vars)})"

    # Add back negation if needed
    return f"¬{lifted}" if is_negative else lifted


def _parts_match_objects(parts: List[str], objects: List[str]) -> bool:
    """Check if fluent parts could be parameters from objects."""
    return all(part in objects for part in parts)


# ========== Action String Parsing ==========

def parse_grounded_action_string(action_str: str,
                                domain: LiftedDomainKnowledge) -> Optional[GroundedAction]:
    """
    Parse grounded action string to GroundedAction.

    Args:
        action_str: Grounded action string (e.g., 'pick-up_a' or 'stack_a_b')
        domain: Domain knowledge for action schemas

    Returns:
        GroundedAction or None if parsing fails

    Example:
        grounded = parse_grounded_action_string('stack_a_b', domain)
        # → GroundedAction(action_name='stack', objects=['a', 'b'], ...)
    """
    parts = action_str.split('_')
    if not parts:
        return None

    # Try to match against known actions
    for action_name, lifted_action in domain.lifted_actions.items():
        # Try exact match for zero-parameter actions
        if lifted_action.arity == 0 and action_name == action_str:
            return ground_action(lifted_action, [])

        # Try matching with parameters
        if lifted_action.arity > 0:
            # Expected format: action_name_obj1_obj2_...
            if action_str.startswith(action_name + '_'):
                rest = action_str[len(action_name) + 1:]
                obj_parts = rest.split('_')

                if len(obj_parts) == lifted_action.arity:
                    # Validate objects exist in domain
                    if all(obj in domain.objects for obj in obj_parts):
                        return ground_action(lifted_action, obj_parts)

    logger.warning(f"Could not parse grounded action string: {action_str}")
    return None


# ========== Batch Operations ==========

def ground_literal_set(literals: Set[str], objects: List[str]) -> Set[str]:
    """
    Ground a set of parameter-bound literals.

    Args:
        literals: Set of parameter-bound literals
        objects: Object binding

    Returns:
        Set of grounded literals

    Example:
        literals = {'on(?x,?y)', '¬clear(?x)'}
        grounded = ground_literal_set(literals, ['a', 'b'])
        # → {'on_a_b', '¬clear_a'}
    """
    return {ground_parameter_bound_literal(lit, objects) for lit in literals}


def lift_fluent_set(fluents: Set[str], objects: List[str], domain: LiftedDomainKnowledge) -> Set[str]:
    """
    Lift a set of grounded fluents.

    Args:
        fluents: Set of grounded fluents
        objects: Object binding
        domain: Domain knowledge

    Returns:
        Set of parameter-bound literals

    Example:
        fluents = {'on_a_b', '¬clear_a'}
        lifted = lift_fluent_set(fluents, ['a', 'b'], domain)
        # → {'on(?x,?y)', '¬clear(?x)'}
    """
    return {lift_grounded_fluent(f, objects, domain) for f in fluents}


# ========== Helper Functions ==========

def _parameter_index_from_name(param_name: str) -> int:
    """
    Get parameter index from name.

    Args:
        param_name: Parameter name (e.g., '?x', '?y', '?p10')

    Returns:
        Index (e.g., ?x→0, ?y→1, ?p10→10)
    """
    param_names = 'xyzuvwpqrst'

    if param_name.startswith('?') and len(param_name) == 2:
        letter = param_name[1].lower()
        if letter in param_names:
            return param_names.index(letter)

    # Fallback: try to extract number
    import re
    match = re.search(r'\d+', param_name)
    if match:
        return int(match.group())

    return 0  # Default


def _parameter_name_from_index(index: int) -> str:
    """
    Get parameter name from index.

    Args:
        index: Parameter index

    Returns:
        Parameter name (e.g., 0→'?x', 1→'?y', 10→'?p10')
    """
    param_names = 'xyzuvwpqrst'
    if index < len(param_names):
        return f"?{param_names[index]}"
    return f"?p{index}"


# ========== Validation ==========

def validate_grounded_action(grounded: GroundedAction, domain: LiftedDomainKnowledge) -> bool:
    """
    Validate that grounded action is valid for domain.

    Args:
        grounded: Grounded action to validate
        domain: Domain knowledge

    Returns:
        True if valid

    Checks:
    - Action exists in domain
    - Objects exist in domain
    - Object count matches action arity
    """
    # Check action exists
    lifted_action = domain.get_action(grounded.action_name)
    if not lifted_action:
        logger.warning(f"Action {grounded.action_name} not found in domain")
        return False

    # Check arity
    if len(grounded.objects) != lifted_action.arity:
        logger.warning(f"Action {grounded.action_name} arity mismatch: "
                      f"expected {lifted_action.arity}, got {len(grounded.objects)}")
        return False

    # Check objects exist
    for obj in grounded.objects:
        if obj not in domain.objects:
            logger.warning(f"Object {obj} not found in domain")
            return False

    return True
