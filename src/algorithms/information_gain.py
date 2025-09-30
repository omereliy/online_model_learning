"""
Information Gain-based Online Action Model Learning.

Implements a CNF/SAT-based information-theoretic approach to learning action models
using expected information gain for action selection.
"""

import logging
from typing import Tuple, List, Dict, Optional, Any, Set
from collections import defaultdict
from pathlib import Path

from .base_learner import BaseActionModelLearner
from src.core.pddl_handler import PDDLHandler

logger = logging.getLogger(__name__)


class InformationGainLearner(BaseActionModelLearner):
    """
    Information Gain-based action model learner.

    Uses CNF formulas and SAT solving to represent uncertainty about action models
    and selects actions that maximize expected information gain.

    Phase 1 implementation: Core data structures and binding functions only.
    CNF formulas and action selection will be added in later phases.
    """

    def __init__(self,
                 domain_file: str,
                 problem_file: str,
                 max_iterations: int = 1000,
                 **kwargs):
        """
        Initialize Information Gain learner.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            max_iterations: Maximum learning iterations
            **kwargs: Additional parameters
        """
        super().__init__(domain_file, problem_file, **kwargs)

        self.max_iterations = max_iterations

        # Initialize PDDL handler for lifted action/fluent support
        self.pddl_handler = PDDLHandler()
        self.pddl_handler.parse_domain_and_problem(domain_file, problem_file)

        # Action model state variables (per action schema)
        # Structure: Dict[action_name, data]
        self.pre: Dict[str, Set[str]] = {}          # Possible preconditions (not ruled out)
        self.pre_constraints: Dict[str, List[Set[str]]] = {}  # Constraint sets (pre?)
        self.eff_add: Dict[str, Set[str]] = {}      # Confirmed add effects
        self.eff_del: Dict[str, Set[str]] = {}      # Confirmed delete effects
        self.eff_maybe_add: Dict[str, Set[str]] = {}  # Possible add effects (not determined)
        self.eff_maybe_del: Dict[str, Set[str]] = {}  # Possible delete effects (not determined)

        # Track observations for each action schema
        self.observation_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Initialize action models
        self._initialize_action_models()

        logger.info(f"Initialized Information Gain learner with {len(self.pre)} actions")

    def _initialize_action_models(self):
        """Initialize action model state variables for all actions."""
        # Get all lifted actions from domain
        for action_name, action in self.pddl_handler._lifted_actions.items():
            # Get La: all parameter-bound literals for this action
            La = self._get_parameter_bound_literals(action_name)

            # Initialize state variables according to algorithm
            self.pre[action_name] = La.copy()  # Initially all literals possible
            self.pre_constraints[action_name] = []  # Empty constraint set
            self.eff_add[action_name] = set()  # No confirmed add effects
            self.eff_del[action_name] = set()  # No confirmed delete effects
            self.eff_maybe_add[action_name] = La.copy()  # All possible add effects
            self.eff_maybe_del[action_name] = La.copy()  # All possible delete effects

            logger.debug(f"Initialized action {action_name} with |La|={len(La)}")

    def _get_parameter_bound_literals(self, action_name: str) -> Set[str]:
        """
        Get all parameter-bound literals (La) for an action.

        This includes all possible lifted fluents that can be formed using
        the action's parameters, including both positive and negative literals.

        Args:
            action_name: Name of the action

        Returns:
            Set of parameter-bound literal strings (e.g., 'on(?x,?y)', '¬clear(?x)')
        """
        action = self.pddl_handler.get_lifted_action(action_name)
        if not action:
            logger.warning(f"Action {action_name} not found in domain")
            return set()

        La = set()

        # Get parameter names - use standard naming ?x, ?y, ?z, etc.
        num_params = len(action.parameters)
        param_letters = 'xyzuvwpqrst'
        param_names = [f"?{param_letters[i]}" if i < len(param_letters) else f"?p{i}"
                      for i in range(num_params)]

        # For each predicate in the domain, generate all valid lifted literals
        for fluent in self.pddl_handler.problem.fluents:
            pred_name = fluent.name
            pred_arity = fluent.arity

            if pred_arity == 0:
                # Propositional fluent
                La.add(pred_name)
                La.add(f"¬{pred_name}")
            else:
                # Generate all parameter combinations of the right arity
                # For simplicity in Phase 1, generate all combinations up to action's arity
                import itertools

                # Generate combinations with repetition allowed
                for combo_length in range(1, min(pred_arity + 1, num_params + 1)):
                    for combo in itertools.combinations_with_replacement(param_names[:num_params], combo_length):
                        if len(combo) == pred_arity:
                            # Create positive literal
                            literal = f"{pred_name}({','.join(combo)})"
                            La.add(literal)
                            # Create negative literal
                            La.add(f"¬{literal}")

        return La

    def bindP_inverse(self, literals: Set[str], objects: List[str]) -> Set[str]:
        """
        Ground parameter-bound literals with concrete objects.

        bindP⁻¹(F, O) returns groundings of parameter-bound literals F
        with respect to object order O.

        Args:
            literals: Set of parameter-bound literals (e.g., {'on(?x,?y)', '¬clear(?x)'})
            objects: Ordered list of objects (e.g., ['a', 'b'])

        Returns:
            Set of grounded literals (e.g., {'on_a_b', '¬clear_a'})

        Example:
            bindP_inverse({'on(?x,?y)'}, ['a', 'b']) → {'on_a_b'}
            bindP_inverse({'¬on(?x,?y)'}, ['a', 'b']) → {'¬on_a_b'}
        """
        grounded = set()

        for literal in literals:
            # Handle negative literals
            is_negative = literal.startswith('¬')
            if is_negative:
                literal = literal[1:]  # Remove negation symbol

            # Ground the literal
            grounded_literal = self._ground_lifted_literal(literal, objects)

            # Add back negation if needed
            if is_negative:
                grounded_literal = f"¬{grounded_literal}"

            grounded.add(grounded_literal)

        return grounded

    def bindP(self, fluents: Set[str], objects: List[str]) -> Set[str]:
        """
        Lift grounded fluents to parameter-bound literals.

        bindP(f, O) returns parameter-bound literals from grounded literals f
        with respect to object order O.

        Args:
            fluents: Set of grounded fluent strings (e.g., {'on_a_b', '¬clear_a'})
            objects: Ordered list of objects (e.g., ['a', 'b'])

        Returns:
            Set of parameter-bound literals (e.g., {'on(?x,?y)', '¬clear(?x)'})

        Example:
            bindP({'on_a_b'}, ['a', 'b']) → {'on(?x,?y)'}
            bindP({'¬on_a_b'}, ['a', 'b']) → {'¬on(?x,?y)'}
        """
        lifted = set()

        for fluent in fluents:
            # Handle negative fluents
            is_negative = fluent.startswith('¬')
            if is_negative:
                fluent = fluent[1:]  # Remove negation symbol

            # Lift the fluent
            lifted_literal = self._lift_grounded_fluent(fluent, objects)

            # Add back negation if needed
            if is_negative:
                lifted_literal = f"¬{lifted_literal}"

            lifted.add(lifted_literal)

        return lifted

    def _ground_lifted_literal(self, literal: str, objects: List[str]) -> str:
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
                # Extract parameter index from name (e.g., ?x → 0, ?y → 1)
                # This assumes parameters are in order as they appear in action definition
                # For robust handling, we need to track parameter order
                param_idx = self._get_parameter_index(param, objects)
                if param_idx < len(objects):
                    grounded_params.append(objects[param_idx])
                else:
                    logger.warning(f"Parameter {param} index {param_idx} out of bounds for objects {objects}")
                    grounded_params.append(param)  # Keep original if error
            else:
                # Already grounded
                grounded_params.append(param)

        # Create grounded fluent string
        return '_'.join([predicate] + grounded_params)

    def _lift_grounded_fluent(self, fluent: str, objects: List[str]) -> str:
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
                params.append(self._get_parameter_name(obj_idx))
            except ValueError:
                logger.warning(f"Object {obj_name} not found in objects list {objects}")
                params.append(obj_name)  # Keep original if not found

        # Create lifted literal
        if params:
            return f"{predicate}({','.join(params)})"
        else:
            return predicate

    def _get_parameter_index(self, param_name: str, objects: List[str]) -> int:
        """
        Get index of parameter in action's parameter list.

        For now, uses simple heuristic based on parameter name.
        In full implementation, would track action parameter order.

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

    def _get_parameter_name(self, index: int) -> str:
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

    def select_action(self, state: Any) -> Tuple[str, List[str]]:
        """
        Select next action to execute.

        Phase 1: Returns first action (placeholder).
        Phase 3 will implement information gain-based selection.

        Args:
            state: Current state

        Returns:
            Tuple of (action_name, objects)
        """
        self.iteration_count += 1

        # Placeholder: return first grounded action
        if self.pddl_handler._grounded_actions:
            action, binding = self.pddl_handler._grounded_actions[0]
            objects = [obj.name for obj in binding.values()] if binding else []
            return action.name, objects

        # Fallback
        return "no_action", []

    def observe(self,
                state: Any,
                action: str,
                objects: List[str],
                success: bool,
                next_state: Optional[Any] = None) -> None:
        """
        Observe action execution result.

        Phase 1: Records observation.
        Phase 2 will implement model update logic.

        Args:
            state: State before action execution
            action: Action name that was executed
            objects: Objects involved in the action
            success: Whether the action succeeded
            next_state: State after execution (if successful)
        """
        self.observation_count += 1

        # Record observation
        observation = {
            'iteration': self.iteration_count,
            'action': action,
            'objects': objects,
            'success': success,
            'state': state,
            'next_state': next_state
        }
        self.observation_history[action].append(observation)

        logger.debug(f"Recorded observation {self.observation_count}: {action}({','.join(objects)}) - {'success' if success else 'failure'}")

    def get_learned_model(self) -> Dict[str, Any]:
        """
        Export the current learned model.

        Returns:
            Dictionary containing the learned model
        """
        model = {
            'actions': {},
            'predicates': set(),
            'statistics': self.get_statistics()
        }

        # Export learned action models
        for action_name in self.pre.keys():
            model['actions'][action_name] = {
                'name': action_name,
                'preconditions': {
                    'possible': list(self.pre[action_name]),
                    'constraints': [list(c) for c in self.pre_constraints[action_name]]
                },
                'effects': {
                    'add': list(self.eff_add[action_name]),
                    'delete': list(self.eff_del[action_name]),
                    'maybe_add': list(self.eff_maybe_add[action_name]),
                    'maybe_delete': list(self.eff_maybe_del[action_name])
                },
                'observations': len(self.observation_history[action_name])
            }

            # Extract predicates from literals
            for literal in self.pre[action_name]:
                pred_name = self._extract_predicate_name(literal)
                if pred_name:
                    model['predicates'].add(pred_name)

        return model

    def _extract_predicate_name(self, literal: str) -> Optional[str]:
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

    def has_converged(self) -> bool:
        """
        Check if learning has converged.

        Returns:
            True if model has converged, False otherwise
        """
        # Check iteration limit
        if self.iteration_count >= self.max_iterations:
            self._converged = True
            return True

        # Future: Add convergence criteria based on model uncertainty

        return self._converged

    def reset(self) -> None:
        """Reset the learner to initial state."""
        super().reset()

        # Clear all state variables
        self.pre.clear()
        self.pre_constraints.clear()
        self.eff_add.clear()
        self.eff_del.clear()
        self.eff_maybe_add.clear()
        self.eff_maybe_del.clear()
        self.observation_history.clear()

        # Reinitialize
        self._initialize_action_models()