"""
PDDL Handler using Unified Planning Framework
Handles PDDL parsing, problem representation, and model export.
"""

import logging
from typing import Set, Tuple, List, Dict, Optional, Any

from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.shortcuts import *

logger = logging.getLogger(__name__)


class PDDLHandler:
    """Handles PDDL operations using Unified Planning Framework."""

    def __init__(self, require_injective_bindings: bool = False):
        """Initialize PDDL handler.

        Args:
            require_injective_bindings: If True, filter out parameter bindings where
                                       the same object appears multiple times (for OLAM compatibility)
        """
        self.reader = PDDLReader()
        self.writer = None  # Will be initialized when needed
        self.problem: Optional[Problem] = None
        self.domain_file: Optional[str] = None
        self.problem_file: Optional[str] = None
        self._fluent_map: Dict[str, Fluent] = {}
        self._object_map: Dict[str, Object] = {}
        self._grounded_actions: List[Tuple[Action, Dict[str, Object]]] = []
        self.require_injective_bindings = require_injective_bindings
        # Support for lifted representations
        self._lifted_actions: Dict[str, Action] = {}  # action_name -> Action
        self._lifted_predicates: Dict[str, Tuple[str, List[str]]] = {}  # pred_name -> (name, param_types)
        self._type_hierarchy: Dict[str, Set[str]] = {}  # type -> subtypes

    def parse_domain_and_problem(self, domain_file: str, problem_file: str) -> Problem:
        """
        Parse PDDL domain and problem files.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file

        Returns:
            Unified Planning Problem object
        """
        self.domain_file = domain_file
        self.problem_file = problem_file

        # Parse using UP reader
        self.problem = self.reader.parse_problem(domain_file, problem_file)

        # Build internal mappings
        self._build_mappings()
        self._ground_actions(self.require_injective_bindings)

        logger.info(f"Parsed problem: {self.problem.name}")
        logger.info(f"Fluents: {len(self._fluent_map)}")
        logger.info(f"Objects: {len(self._object_map)}")
        logger.info(f"Grounded actions: {len(self._grounded_actions)}")

        return self.problem

    def _build_mappings(self):
        """Build internal mappings for fluents and objects."""
        self._fluent_map.clear()
        self._object_map.clear()
        self._lifted_predicates.clear()
        self._type_hierarchy.clear()

        # Map fluents and track lifted predicates
        for fluent in self.problem.fluents:
            self._fluent_map[fluent.name] = fluent
            # Track lifted predicate structure
            if fluent.arity > 0:
                param_types = [str(param.type) for param in fluent.signature]
                self._lifted_predicates[fluent.name] = (fluent.name, param_types)

        # Map objects
        for obj in self.problem.all_objects:
            self._object_map[obj.name] = obj

        # Build type hierarchy
        for user_type in self.problem.user_types:
            self._build_type_hierarchy(user_type)

    def _build_type_hierarchy(self, type_obj):
        """Build type hierarchy for type checking."""
        # Store child -> parent relationship
        type_name = str(type_obj.name) if hasattr(type_obj, 'name') else str(type_obj)

        if type_obj.father:
            parent_name = str(type_obj.father.name) if hasattr(type_obj.father, 'name') else str(type_obj.father)
        else:
            # No explicit parent means 'object' is the parent in PDDL
            parent_name = 'object'

        # Store parent -> children mapping
        if parent_name not in self._type_hierarchy:
            self._type_hierarchy[parent_name] = set()
        self._type_hierarchy[parent_name].add(type_name)

    def _ground_actions(self, require_injective=False):
        """Ground all actions with all possible parameter bindings.

        Args:
            require_injective: If True, skip bindings where same object appears multiple times
        """
        self._grounded_actions.clear()
        self._lifted_actions.clear()
        self.require_injective = require_injective

        for action in self.problem.actions:
            # Store lifted action
            self._lifted_actions[action.name] = action

            # Get all possible parameter bindings
            param_bindings = self._get_parameter_bindings(action, require_injective)

            for binding in param_bindings:
                self._grounded_actions.append((action, binding))

    def _get_parameter_bindings(self, action: Action, require_injective: bool = False) -> List[Dict[str, Object]]:
        """
        Get all possible parameter bindings for an action.

        Args:
            action: UP Action object
            require_injective: If True, skip non-injective bindings (same object in multiple positions)

        Returns:
            List of parameter binding dictionaries
        """
        import itertools

        if not action.parameters:
            return [{}]

        # Get objects for each parameter type
        param_options = []
        for param in action.parameters:
            # Get objects matching parameter type
            matching_objects = [
                obj for obj in self.problem.all_objects
                if self._type_matches(obj.type, param.type)
            ]
            param_options.append(matching_objects)

        # Generate all combinations
        bindings = []
        for combo in itertools.product(*param_options):
            # Check for injective binding if required
            if require_injective and len(set(combo)) != len(combo):
                # Skip this combination - same object appears multiple times
                continue

            binding = {}
            for param, obj in zip(action.parameters, combo):
                binding[param.name] = obj
            bindings.append(binding)

        return bindings

    def _type_matches(self, obj_type: Type, param_type: Type) -> bool:
        """
        Check if object type matches parameter type.

        Args:
            obj_type: Object's type
            param_type: Parameter's required type

        Returns:
            True if types match
        """
        # Simple type matching - extend for type hierarchies
        try:
            return obj_type == param_type or param_type.is_compatible(obj_type)
        except AttributeError:
            # Handle case where is_compatible method doesn't exist
            return obj_type == param_type

    def get_grounded_fluents(self) -> List[str]:
        """
        Get all grounded fluents as strings.

        Returns:
            List of grounded fluent strings (e.g., 'on_a_b', 'clear_c')
        """
        grounded = []

        for fluent in self.problem.fluents:
            if fluent.arity == 0:
                # Propositional fluent
                grounded.append(fluent.name)
            else:
                # Generate all groundings
                param_types = [param.type for param in fluent.signature]
                object_combos = self._get_object_combinations(param_types)

                for combo in object_combos:
                    # Create grounded name
                    parts = [fluent.name] + [obj.name for obj in combo]
                    grounded_name = '_'.join(parts)
                    grounded.append(grounded_name)

        return grounded

    def _get_object_combinations(self, types: List[Type]) -> List[List[Object]]:
        """
        Get all object combinations for given types.

        Args:
            types: List of required types

        Returns:
            List of object combinations
        """
        import itertools

        object_lists = []
        for req_type in types:
            matching = [
                obj for obj in self.problem.all_objects
                if self._type_matches(obj.type, req_type)
            ]
            object_lists.append(matching)

        return list(itertools.product(*object_lists))

    def state_to_fluent_set(self, state: Any) -> Set[str]:
        """
        Convert UP state to set of true fluent strings.

        Args:
            state: UP state object or initial values dict

        Returns:
            Set of true fluent strings
        """
        true_fluents = set()

        # Handle both state objects and initial_values dict
        if isinstance(state, dict):
            # Initial values are a dict mapping fluent expressions to values
            for fluent_expr, value in state.items():
                if value.bool_constant_value():
                    # Extract fluent name and parameters from expression
                    if hasattr(fluent_expr, 'fluent'):
                        # It's a fluent application
                        fluent_name = fluent_expr.fluent().name
                        if fluent_expr.args:
                            # Has parameters
                            param_names = [str(arg).replace("'", "") for arg in fluent_expr.args]
                            grounded_name = '_'.join([fluent_name] + param_names)
                            true_fluents.add(grounded_name)
                        else:
                            # No parameters
                            true_fluents.add(fluent_name)
                    else:
                        # Direct fluent
                        true_fluents.add(str(fluent_expr))
        else:
            # State object with get_value method
            for fluent in self.problem.fluents:
                if fluent.arity == 0:
                    # Propositional fluent
                    if state.get_value(fluent()).bool_constant_value():
                        true_fluents.add(fluent.name)
                else:
                    # Check all groundings
                    param_types = [param.type for param in fluent.signature]
                    object_combos = self._get_object_combinations(param_types)

                    for combo in object_combos:
                        fluent_expr = fluent(*combo)
                        if state.get_value(fluent_expr).bool_constant_value():
                            parts = [fluent.name] + [obj.name for obj in combo]
                            grounded_name = '_'.join(parts)
                            true_fluents.add(grounded_name)

        return true_fluents

    def fluent_set_to_state(self, fluent_set: Set[str]) -> Dict:
        """
        Convert fluent set to state representation.

        Args:
            fluent_set: Set of true fluent strings

        Returns:
            State dictionary representation
        """
        state_dict = {}

        for fluent in self.problem.fluents:
            if fluent.arity == 0:
                # Propositional fluent
                state_dict[fluent.name] = fluent.name in fluent_set
            else:
                # Check all groundings
                param_types = [param.type for param in fluent.signature]
                object_combos = self._get_object_combinations(param_types)

                for combo in object_combos:
                    parts = [fluent.name] + [obj.name for obj in combo]
                    grounded_name = '_'.join(parts)
                    state_dict[grounded_name] = grounded_name in fluent_set

        return state_dict

    def get_grounded_actions_list(self) -> List[str]:
        """
        Get list of all grounded action names.

        Returns:
            List of grounded action strings
        """
        grounded = []

        for action, binding in self._grounded_actions:
            if not binding:
                # No parameters
                grounded.append(action.name)
            else:
                # Include parameter values
                parts = [action.name] + [obj.name for obj in binding.values()]
                grounded_name = '_'.join(parts)
                grounded.append(grounded_name)

        return grounded

    def parse_grounded_action(self, action_str: str) -> Tuple[Optional[Action], Optional[Dict[str, Object]]]:
        """
        Parse grounded action string to action and parameter binding.

        Args:
            action_str: Grounded action string (e.g., 'pick_a')

        Returns:
            Tuple of (Action, parameter_binding) or (None, None) if not found
        """
        parts = action_str.split('_')
        if not parts:
            return None, None

        # Try to match with grounded actions
        for action, binding in self._grounded_actions:
            # Check if this matches
            if not binding and action.name == action_str:
                return action, binding

            if binding:
                expected_parts = [action.name] + [obj.name for obj in binding.values()]
                expected_name = '_'.join(expected_parts)
                if expected_name == action_str:
                    return action, binding

        return None, None

    def export_to_pddl(self, learned_problem: Problem, output_domain: str, output_problem: str):
        """
        Export learned model to PDDL files.

        Args:
            learned_problem: Problem with learned action models
            output_domain: Output path for domain file
            output_problem: Output path for problem file
        """
        # Initialize writer if needed
        if self.writer is None:
            self.writer = PDDLWriter(learned_problem)

        # Write domain and problem
        self.writer.write_domain(output_domain)
        self.writer.write_problem(output_problem)

        logger.info(f"Exported PDDL domain to: {output_domain}")
        logger.info(f"Exported PDDL problem to: {output_problem}")

    def create_modified_problem(self,
                              action_updates: Dict[str, Dict[str, Any]]) -> Problem:
        """
        Create modified problem with updated action definitions.

        Args:
            action_updates: Dictionary mapping action names to updates
                          Format: {action_name: {'preconditions': Set, 'effects': Set}}

        Returns:
            New Problem object with modifications
        """
        # Create new problem
        new_problem = Problem(self.problem.name + "_learned")

        # Copy types
        for user_type in self.problem.user_types:
            new_problem.add_type(user_type.name, user_type.father)

        # Copy objects
        for obj in self.problem.all_objects:
            new_problem.add_object(obj)

        # Copy fluents
        for fluent in self.problem.fluents:
            new_problem.add_fluent(fluent)

        # Copy and modify actions
        for action in self.problem.actions:
            if action.name in action_updates:
                # Create modified action
                new_action = self._create_modified_action(action, action_updates[action.name])
                new_problem.add_action(new_action)
            else:
                # Copy original action
                new_problem.add_action(action)

        # Copy initial state and goals
        new_problem.set_initial_value(self.problem.initial_values)
        for goal in self.problem.goals:
            new_problem.add_goal(goal)

        return new_problem

    def _create_modified_action(self, original: Action, updates: Dict[str, Any]) -> Action:
        """
        Create modified version of action with updates.

        Args:
            original: Original action
            updates: Updates to apply

        Returns:
            Modified action
        """
        # This is simplified - full implementation would handle complex updates
        new_action = Action(original.name, **{p.name: p.type for p in original.parameters})

        # Apply precondition updates if provided
        if 'preconditions' in updates:
            # Convert set-based preconditions to UP format
            # This would need proper implementation based on format
            for precond in updates['preconditions']:
                # Add precondition (simplified)
                pass
        else:
            # Copy original preconditions
            for precond in original.preconditions:
                new_action.add_precondition(precond)

        # Apply effect updates if provided
        if 'effects' in updates:
            # Convert set-based effects to UP format
            for effect in updates['effects']:
                # Add effect (simplified)
                pass
        else:
            # Copy original effects
            for effect in original.effects:
                new_action.add_effect(
                    fluent=effect.fluent,
                    value=effect.value,
                    condition=effect.condition
                )

        return new_action

    def validate_action_applicable(self, action_str: str, state: Set[str]) -> bool:
        """
        Check if grounded action is applicable in given state.

        Args:
            action_str: Grounded action string
            state: Set of true fluents

        Returns:
            True if action is applicable
        """
        action, binding = self.parse_grounded_action(action_str)
        if not action:
            return False

        # Check each precondition
        for precond in action.preconditions:
            # Substitute parameters with binding
            grounded_precond = self._ground_expression(precond, binding)
            # Check if satisfied in state
            if not self._evaluate_in_state(grounded_precond, state):
                return False

        return True

    def _expression_to_lifted_string(self, expr: Any, parameters: List) -> Optional[str]:
        """Convert UP expression to lifted string representation."""
        if hasattr(expr, 'fluent'):
            fluent = expr.fluent()
            if expr.args:
                # Has parameters - create lifted representation
                param_strs = []
                for arg in expr.args:
                    # Find parameter name
                    for param in parameters:
                        if str(arg) == str(param):
                            param_strs.append(f"?{param.name}")
                            break
                    else:
                        # Not a parameter, use object name
                        param_strs.append(str(arg))
                return f"{fluent.name}({','.join(param_strs)})"
            else:
                return fluent.name
        return None

    def _ground_expression_to_string(self, expr: Any, binding: Dict[str, Object]) -> Optional[str]:
        """Convert UP expression to grounded string with binding."""
        # Handle AND expressions recursively
        if hasattr(expr, 'is_and') and expr.is_and():
            # For AND expressions, we need to traverse each operand
            # This shouldn't happen in normal precondition extraction
            # but we handle it for robustness
            return None

        # Handle OR expressions recursively
        if hasattr(expr, 'is_or') and expr.is_or():
            # For OR expressions, similar handling
            return None

        # Handle NOT expressions
        if hasattr(expr, 'is_not') and expr.is_not():
            inner = expr.args[0] if expr.args else expr
            inner_str = self._ground_expression_to_string(inner, binding)
            if inner_str:
                return f"-{inner_str}"
            return None

        # Handle fluent expressions
        if hasattr(expr, 'is_fluent_exp') and expr.is_fluent_exp():
            fluent = expr.fluent()
            if expr.args:
                # Has parameters - ground them
                grounded_args = []
                for arg in expr.args:
                    if str(arg) in binding:
                        grounded_args.append(binding[str(arg)].name)
                    else:
                        grounded_args.append(str(arg).replace("'", ""))
                return f"{fluent.name}_{'_'.join(grounded_args)}"
            else:
                return fluent.name
        return None

    def _ground_expression(self, expr: Any, binding: Dict[str, Object]) -> Any:
        """Ground expression with parameter binding."""
        # This would need full UP expression substitution
        # For now, returning original expression
        return expr

    def _evaluate_in_state(self, expr: Any, state: Set[str]) -> bool:
        """Evaluate expression in state."""
        # Convert expression to grounded string and check if in state
        grounded_str = self._ground_expression_to_string(expr, {})
        if grounded_str:
            # Handle negation
            if hasattr(expr, 'is_not') and expr.is_not():
                return grounded_str not in state
            else:
                return grounded_str in state
        return True

    def get_lifted_action(self, action_name: str) -> Optional[Action]:
        """
        Get lifted action by name.

        Args:
            action_name: Name of the action

        Returns:
            Lifted Action object or None
        """
        return self._lifted_actions.get(action_name)

    def get_lifted_predicate_structure(self, predicate_name: str) -> Optional[Tuple[str, List[str]]]:
        """
        Get structure of a lifted predicate.

        Args:
            predicate_name: Name of the predicate

        Returns:
            Tuple of (name, param_types) or None
        """
        return self._lifted_predicates.get(predicate_name)

    def create_lifted_fluent_string(self, predicate: str, params: List[str]) -> str:
        """
        Create lifted fluent string representation.

        Args:
            predicate: Predicate name
            params: Parameter names (e.g., ['?x', '?y'])

        Returns:
            Lifted fluent string (e.g., 'on(?x,?y)')
        """
        return f"{predicate}({','.join(params)})"

    def create_lifted_action_string(self, action_name: str, params: List[str]) -> str:
        """
        Create lifted action string representation.

        Args:
            action_name: Action name
            params: Parameter names

        Returns:
            Lifted action string (e.g., 'pick(?x)')
        """
        return f"{action_name}({','.join(params)})"

    def get_action_preconditions(self, action_str: str, lifted: bool = False) -> Set[str]:
        """
        Get preconditions of action as a fluent set.

        Args:
            action_str: Action string (grounded or lifted)
            lifted: If True, return lifted representation

        Returns:
            Set of fluent strings (grounded or lifted)
        """
        if lifted and action_str in self._lifted_actions:
            # Return lifted preconditions
            action = self._lifted_actions[action_str]
            preconditions = set()
            for precond in action.preconditions:
                # Convert to lift fluent string representation
                lifted_str = self._expression_to_lifted_string(precond, action.parameters)
                if lifted_str:
                    preconditions.add(lifted_str)
            return preconditions
        else:
            # Grounded version
            action, binding = self.parse_grounded_action(action_str)
            if not action:
                return set()

            preconditions = set()
            for precond in action.preconditions:
                # Check if it's an AND expression and extract individual fluents
                if hasattr(precond, 'is_and') and precond.is_and():
                    for arg in precond.args:
                        grounded_str = self._ground_expression_to_string(arg, binding)
                        if grounded_str:
                            preconditions.add(grounded_str)
                else:
                    grounded_str = self._ground_expression_to_string(precond, binding)
                    if grounded_str:
                        preconditions.add(grounded_str)

            return preconditions

    def get_action_effects(self, action_str: str, lifted: bool = False) -> Tuple[Set[str], Set[str]]:
        """
        Get effects of action.

        Args:
            action_str: Action string (grounded or lifted)
            lifted: If True, return lifted representation

        Returns:
            Tuple of (add_effects, delete_effects) as fluent sets
        """
        if lifted and action_str in self._lifted_actions:
            # Return lifted effects
            action = self._lifted_actions[action_str]
            add_effects = set()
            delete_effects = set()

            for effect in action.effects:
                lifted_str = self._expression_to_lifted_string(effect.fluent, action.parameters)
                if lifted_str:
                    if effect.value.bool_constant_value():
                        add_effects.add(lifted_str)
                    else:
                        delete_effects.add(lifted_str)

            return add_effects, delete_effects
        else:
            # Grounded version
            action, binding = self.parse_grounded_action(action_str)
            if not action:
                return set(), set()

            add_effects = set()
            delete_effects = set()

            for effect in action.effects:
                grounded_str = self._ground_expression_to_string(effect.fluent, binding)
                if grounded_str:
                    if effect.value.bool_constant_value():
                        add_effects.add(grounded_str)
                    else:
                        delete_effects.add(grounded_str)

            return add_effects, delete_effects

    def get_initial_state(self) -> Set[str]:
        """
        Get initial state as fluent set.

        Returns:
            Set of true fluents in initial state
        """
        if not self.problem:
            return set()

        return self.state_to_fluent_set(self.problem.initial_values)

    def get_goal_state(self) -> Set[str]:
        """
        Get goal conditions as fluent set.

        Returns:
            Set of fluents that must be true in goal
        """
        goal_fluents = set()

        # Extract fluents from goal conditions
        for goal in self.problem.goals:
            # Parse goal expression to extract required fluents
            # This is simplified - would need proper goal parsing
            pass

        return goal_fluents

    def extract_lifted_preconditions_cnf(self, action_name: str) -> List[List[str]]:
        """
        Extract action preconditions as CNF clauses with lifted fluents.

        Args:
            action_name: Name of the action

        Returns:
            List of CNF clauses in lifted form
        """
        if action_name not in self._lifted_actions:
            return []

        action = self._lifted_actions[action_name]
        cnf_clauses = []

        # Convert each precondition to CNF clause
        for precond in action.preconditions:
            clauses = self._extract_clauses_from_expression(precond, action.parameters)
            cnf_clauses.extend(clauses)

        return cnf_clauses

    def _extract_clauses_from_expression(self, expr: Any, parameters: List) -> List[List[str]]:
        """Extract CNF clauses from a UP expression."""
        clauses = []

        # Handle different expression types
        if hasattr(expr, 'is_and') and expr.is_and():
            # AND: each operand becomes a separate clause
            for arg in expr.args:
                sub_clauses = self._extract_clauses_from_expression(arg, parameters)
                clauses.extend(sub_clauses)
        elif hasattr(expr, 'is_or') and expr.is_or():
            # OR: combine operands into single clause
            clause = []
            for arg in expr.args:
                sub_clauses = self._extract_clauses_from_expression(arg, parameters)
                if sub_clauses and sub_clauses[0]:
                    clause.extend(sub_clauses[0])
            if clause:
                clauses.append(clause)
        elif hasattr(expr, 'is_not') and expr.is_not():
            # NOT: negate the inner expression
            inner = expr.args[0] if expr.args else expr
            lifted_str = self._expression_to_lifted_string(inner, parameters)
            if lifted_str:
                clauses.append([f"-{lifted_str}"])
        else:
            # Base case: fluent expression
            lifted_str = self._expression_to_lifted_string(expr, parameters)
            if lifted_str:
                clauses.append([lifted_str])

        return clauses

    def supports_negative_preconditions(self) -> bool:
        """
        Check if domain uses negative preconditions.

        Returns:
            True if negative preconditions are present
        """
        for action in self.problem.actions:
            for precond in action.preconditions:
                # Check if any precondition is negated
                if hasattr(precond, 'is_not') and precond.is_not():
                    return True
        return False

    def get_type_hierarchy(self) -> Dict[str, Set[str]]:
        """
        Get the type hierarchy as a parent -> children mapping.

        Returns:
            Dictionary mapping parent types to sets of child types
        """
        return self._type_hierarchy.copy()

    def get_type_ancestors(self, type_name: str) -> List[str]:
        """
        Get all ancestors of a type up to 'object'.

        Args:
            type_name: Name of the type

        Returns:
            List of ancestor type names from immediate parent to root
        """
        ancestors = []

        # Find the type object
        type_obj = None
        for user_type in self.problem.user_types:
            if user_type.name == type_name:
                type_obj = user_type
                break

        if not type_obj:
            return ancestors

        # Traverse up the hierarchy
        current = type_obj.father
        while current is not None:
            parent_name = str(current.name) if hasattr(current, 'name') else str(current)
            ancestors.append(parent_name)
            if hasattr(current, 'father'):
                current = current.father
            else:
                break

        # If no explicit ancestors, 'object' is implicit parent
        if not ancestors and type_obj.father is None:
            ancestors.append('object')

        return ancestors

    def is_subtype_of(self, child_type: str, parent_type: str) -> bool:
        """
        Check if child_type is a subtype of parent_type.

        Args:
            child_type: Name of potential child type
            parent_type: Name of potential parent type

        Returns:
            True if child_type is a subtype of parent_type
        """
        if child_type == parent_type:
            return True

        # Special case: everything is a subtype of 'object'
        if parent_type == 'object':
            return True

        ancestors = self.get_type_ancestors(child_type)
        return parent_type in ancestors

    def get_parameter_bound_literals(self, action_name: str) -> Set[str]:
        """
        Get all parameter-bound literals (La) for an action.

        This includes all possible lifted fluents that can be formed using
        the action's parameters, including both positive and negative literals.

        Args:
            action_name: Name of the action

        Returns:
            Set of parameter-bound literal strings (e.g., 'on(?x,?y)', '¬clear(?x)')
        """
        action = self.get_lifted_action(action_name)
        if not action:
            logger.warning(f"Action {action_name} not found in domain")
            return set()

        La = set()

        # Get parameter names using standard naming convention
        num_params = len(action.parameters)
        param_letters = 'xyzuvwpqrst'
        param_names = [f"?{param_letters[i]}" if i < len(param_letters) else f"?p{i}"
                      for i in range(num_params)]

        # For each predicate in domain, generate all valid lifted literals
        import itertools
        for fluent in self.problem.fluents:
            pred_name = fluent.name
            pred_arity = fluent.arity

            if pred_arity == 0:
                # Propositional fluent
                La.add(pred_name)
                La.add(f"¬{pred_name}")
            else:
                # Generate all parameter combinations of the right arity
                for combo_length in range(1, min(pred_arity + 1, num_params + 1)):
                    for combo in itertools.combinations_with_replacement(
                            param_names[:num_params], combo_length):
                        if len(combo) == pred_arity:
                            # Create positive literal
                            literal = f"{pred_name}({','.join(combo)})"
                            La.add(literal)
                            # Create negative literal
                            La.add(f"¬{literal}")

        return La

    def ground_literals(self, literals: Set[str], objects: List[str]) -> Set[str]:
        """
        Ground parameter-bound literals with concrete objects.

        Implements bindP⁻¹(F, O) - converts lifted literals to grounded form.

        Args:
            literals: Set of parameter-bound literals (e.g., {'on(?x,?y)', '¬clear(?x)'})
            objects: Ordered list of objects (e.g., ['a', 'b'])

        Returns:
            Set of grounded literals (e.g., {'on_a_b', '¬clear_a'})

        Example:
            ground_literals({'on(?x,?y)'}, ['a', 'b']) → {'on_a_b'}
            ground_literals({'¬on(?x,?y)'}, ['a', 'b']) → {'¬on_a_b'}
        """
        grounded = set()

        for literal in literals:
            # Handle negative literals
            is_negative = literal.startswith('¬')
            if is_negative:
                literal = literal[1:]  # Remove negation symbol

            # Ground the literal
            grounded_literal = self._ground_lifted_literal_internal(literal, objects)

            # Add back negation if needed
            if is_negative:
                grounded_literal = f"¬{grounded_literal}"

            grounded.add(grounded_literal)

        return grounded

    def lift_fluents(self, fluents: Set[str], objects: List[str]) -> Set[str]:
        """
        Lift grounded fluents to parameter-bound literals.

        Implements bindP(f, O) - converts grounded fluents to lifted form.

        Args:
            fluents: Set of grounded fluent strings (e.g., {'on_a_b', '¬clear_a'})
            objects: Ordered list of objects (e.g., ['a', 'b'])

        Returns:
            Set of parameter-bound literals (e.g., {'on(?x,?y)', '¬clear(?x)'})

        Example:
            lift_fluents({'on_a_b'}, ['a', 'b']) → {'on(?x,?y)'}
            lift_fluents({'¬on_a_b'}, ['a', 'b']) → {'¬on(?x,?y)'}
        """
        lifted = set()

        for fluent in fluents:
            # Handle negative fluents
            is_negative = fluent.startswith('¬')
            if is_negative:
                fluent = fluent[1:]  # Remove negation symbol

            # Lift the fluent
            lifted_literal = self._lift_grounded_fluent_internal(fluent, objects)

            # Add back negation if needed
            if is_negative:
                lifted_literal = f"¬{lifted_literal}"

            lifted.add(lifted_literal)

        return lifted

    def _ground_lifted_literal_internal(self, literal: str, objects: List[str]) -> str:
        """
        Ground a single lifted literal with concrete objects.

        Args:
            literal: Lifted literal (e.g., 'on(?x,?y)' or 'clear(?x)')
            objects: Ordered list of objects

        Returns:
            Grounded fluent string (e.g., 'on_a_b' or 'clear_a')
        """
        # Parse literal: predicate(param1,param2,...)
        if '(' not in literal:
            # Propositional literal
            return literal

        predicate = literal[:literal.index('(')]
        params_str = literal[literal.index('(') + 1:literal.rindex(')')]

        if not params_str:
            # No parameters
            return predicate

        params = [p.strip() for p in params_str.split(',')]

        # Replace each parameter with corresponding object
        grounded_params = []
        for param in params:
            if param.startswith('?'):
                # Extract parameter index from name
                param_idx = self._get_parameter_index_internal(param, objects)
                if param_idx < len(objects):
                    grounded_params.append(objects[param_idx])
                else:
                    logger.warning(f"Parameter {param} index {param_idx} out of bounds for objects {objects}")
                    grounded_params.append(param)  # Keep original if error
            else:
                # Already grounded
                grounded_params.append(param)

        # Create grounded fluent string
        result = '_'.join([predicate] + grounded_params)
        return result

    def _lift_grounded_fluent_internal(self, fluent: str, objects: List[str]) -> str:
        """
        Lift a grounded fluent to parameter-bound literal.

        Args:
            fluent: Grounded fluent (e.g., 'on_a_b' or 'clear_a')
            objects: Ordered list of objects used in grounding

        Returns:
            Lifted literal (e.g., 'on(?x,?y)' or 'clear(?x)')
        """
        # Parse fluent: predicate_obj1_obj2_...
        parts = fluent.split('_')

        if len(parts) == 1:
            # Propositional fluent
            return parts[0]

        # First part is predicate, rest are object names
        predicate = parts[0]
        obj_names = parts[1:]

        # Replace each object with its parameter
        params = []
        for obj_name in obj_names:
            try:
                obj_idx = objects.index(obj_name)
                params.append(self._get_parameter_name_internal(obj_idx))
            except ValueError:
                logger.warning(f"Object {obj_name} not found in objects list {objects}")
                params.append(obj_name)  # Keep original if not found

        # Create lifted literal
        if params:
            result = f"{predicate}({','.join(params)})"
        else:
            result = predicate

        return result

    def _get_parameter_index_internal(self, param_name: str, objects: List[str]) -> int:
        """
        Get index of parameter in action's parameter list.

        Args:
            param_name: Parameter name (e.g., '?x', '?y')
            objects: Object list (used for validation)

        Returns:
            Parameter index
        """
        # Simple heuristic: ?x → 0, ?y → 1, ?z → 2, etc.
        param_letters = 'xyzuvwpqrst'
        if param_name.startswith('?') and len(param_name) == 2:
            letter = param_name[1].lower()
            if letter in param_letters:
                return param_letters.index(letter)

        # Fallback: try to parse number from name
        import re
        match = re.search(r'\d+', param_name)
        if match:
            return int(match.group())

        return 0  # Default to first parameter

    def _get_parameter_name_internal(self, index: int) -> str:
        """
        Get parameter name for given index.

        Args:
            index: Parameter index

        Returns:
            Parameter name (e.g., '?x', '?y')
        """
        param_letters = 'xyzuvwpqrst'
        if index < len(param_letters):
            return f"?{param_letters[index]}"
        else:
            return f"?p{index}"

    def extract_predicate_name(self, literal: str) -> Optional[str]:
        """
        Extract predicate name from literal.

        Args:
            literal: Literal string (e.g., 'on(?x,?y)' or '¬clear(?x)')

        Returns:
            Predicate name or None
        """
        # Remove negation if present
        if literal.startswith('¬'):
            literal = literal[1:]

        # Extract predicate name
        if '(' in literal:
            return literal[:literal.index('(')]
        else:
            return literal

    def __str__(self) -> str:
        """String representation."""
        if self.problem:
            return f"PDDLHandler({self.problem.name})"
        return "PDDLHandler(no problem loaded)"

    def __repr__(self) -> str:
        """Detailed representation."""
        if self.problem:
            return f"PDDLHandler(problem={self.problem.name}, fluents={len(self._fluent_map)}, actions={len(self._grounded_actions)})"
        return "PDDLHandler(empty)"