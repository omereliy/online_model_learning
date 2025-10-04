"""
Information Gain-based Online Action Model Learning.

Implements a CNF/SAT-based information-theoretic approach to learning action models
using expected information gain for action selection.
"""

import logging
import math
import random
from typing import Tuple, List, Dict, Optional, Any, Set
from collections import defaultdict
from pathlib import Path

from .base_learner import BaseActionModelLearner
from src.core.pddl_handler import PDDLHandler
from src.core.cnf_manager import CNFManager

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
        logger.info(
            f"Initializing Information Gain learner: domain={domain_file}, problem={problem_file}")
        logger.debug(f"Configuration: max_iterations={max_iterations}, kwargs={kwargs}")

        super().__init__(domain_file, problem_file, **kwargs)

        self.max_iterations = max_iterations

        # Initialize PDDL handler for lifted action/fluent support
        logger.debug("Parsing PDDL domain and problem files")
        self.pddl_handler = PDDLHandler()
        self.pddl_handler.parse_domain_and_problem(domain_file, problem_file)
        logger.debug(
            f"PDDL parsing complete: {len(self.pddl_handler._lifted_actions)} lifted actions, "
            f"{len(self.pddl_handler.problem.fluents)} fluents")

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

        # CNF managers for each action (Phase 2)
        self.cnf_managers: Dict[str, CNFManager] = {}

        # Phase 3: Action selection strategy
        self.selection_strategy = kwargs.get('selection_strategy', 'greedy')  # 'greedy', 'epsilon_greedy', 'boltzmann'
        self.epsilon = kwargs.get('epsilon', 0.1)  # For epsilon-greedy
        self.temperature = kwargs.get('temperature', 1.0)  # For Boltzmann

        # Initialize action models
        logger.debug("Initializing action models")
        self._initialize_action_models()

        logger.info(f"Initialization complete: {len(self.pre)} actions initialized")

    def _initialize_action_models(self):
        """Initialize action model state variables for all actions."""
        logger.debug(f"Initializing models for {len(self.pddl_handler._lifted_actions)} actions")

        # Get all lifted actions from domain
        for action_name, action in self.pddl_handler._lifted_actions.items():
            logger.debug(
                f"Processing action: {action_name}, parameters: {[p.name for p in action.parameters]}")

            # Get La: all parameter-bound literals for this action
            La = self._get_parameter_bound_literals(action_name)
            logger.debug(f"Generated {len(La)} parameter-bound literals for {action_name}")

            # Initialize state variables according to algorithm
            self.pre[action_name] = La.copy()  # Initially all literals possible
            self.pre_constraints[action_name] = []  # Empty constraint set
            self.eff_add[action_name] = set()  # No confirmed add effects
            self.eff_del[action_name] = set()  # No confirmed delete effects
            self.eff_maybe_add[action_name] = La.copy()  # All possible add effects
            self.eff_maybe_del[action_name] = La.copy()  # All possible delete effects

            # Initialize CNF manager for this action (Phase 2)
            self.cnf_managers[action_name] = CNFManager()

            logger.info(
                f"Initialized action '{action_name}': |La|={len(La)}, |pre|={len(self.pre[action_name])}")

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
                    for combo in itertools.combinations_with_replacement(
                            param_names[:num_params], combo_length):
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
        logger.debug(f"bindP_inverse: Grounding {len(literals)} literals with objects {objects}")
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

        logger.debug(f"bindP_inverse: Produced {len(grounded)} grounded literals")
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
        logger.debug(f"bindP: Lifting {len(fluents)} fluents with objects {objects}")
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

        logger.debug(f"bindP: Produced {len(lifted)} lifted literals")
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
            logger.debug(f"_ground_lifted_literal: '{literal}' is propositional")
            return literal

        predicate = literal[:literal.index('(')]
        params_str = literal[literal.index('(') + 1:literal.rindex(')')]

        if not params_str:
            # No parameters
            logger.debug(f"_ground_lifted_literal: '{literal}' has no parameters")
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
                    logger.warning(
                        f"Parameter {param} index {param_idx} out of bounds for objects {objects}")
                    grounded_params.append(param)  # Keep original if error
            else:
                # Already grounded
                grounded_params.append(param)

        # Create grounded fluent string
        result = '_'.join([predicate] + grounded_params)
        logger.debug(f"_ground_lifted_literal: '{literal}' + {objects} → '{result}'")
        return result

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
            logger.debug(f"_lift_grounded_fluent: '{fluent}' is propositional")
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
            result = f"{predicate}({','.join(params)})"
        else:
            result = predicate

        logger.debug(f"_lift_grounded_fluent: '{fluent}' + {objects} → '{result}'")
        return result

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

    def _calculate_applicability_probability(self, action: str, objects: List[str], state: Set[str]) -> float:
        """
        Calculate probability that action is applicable in current state.

        According to algorithm:
        pr(app(a, O, s) = 1) = {
            1,                                             if pre?(a) = ∅
            |SAT(cnf_pre?(a,O,s))| / |SAT(cnf_pre?(a))|,  otherwise
        }

        Args:
            action: Action name
            objects: Object binding
            state: Current state

        Returns:
            Probability between 0 and 1
        """
        # If no constraints, action is always applicable
        if len(self.pre_constraints[action]) == 0:
            logger.debug(f"Action {action} has no constraints, probability = 1.0")
            return 1.0

        # Build CNF formula if needed
        if len(self.cnf_managers[action].cnf.clauses) == 0:
            self._build_cnf_formula(action)

        cnf = self.cnf_managers[action]

        # If CNF is still empty after building, no real constraints
        if len(cnf.cnf.clauses) == 0:
            logger.debug(f"Action {action} has empty CNF, probability = 1.0")
            return 1.0

        # Count satisfying models for unrestricted formula
        total_models = cnf.count_solutions()
        if total_models == 0:
            # Contradiction in formula
            logger.warning(f"Action {action} has contradictory constraints")
            return 0.0

        # Now add constraints for unsatisfied literals in current state
        # cnf_pre?(a,O,s) = cnf_pre?(a) ∧ (¬xl) for l ∈ (⋃pre?(a)) \ bindP(s, O)
        state_internal = self._state_to_internal(state)
        satisfied_literals = self._get_satisfied_literals(action, state_internal, objects)

        # Create a copy of CNF to add state constraints
        cnf_with_state = CNFManager()
        cnf_with_state.cnf.clauses = cnf.cnf.clauses.copy()
        cnf_with_state.fluent_to_var = cnf.fluent_to_var.copy()
        cnf_with_state.var_to_fluent = cnf.var_to_fluent.copy()
        cnf_with_state.next_var = cnf.next_var

        # Add unit clauses for unsatisfied literals
        all_constraint_literals = set()
        for constraint_set in self.pre_constraints[action]:
            all_constraint_literals.update(constraint_set)

        unsatisfied = all_constraint_literals - satisfied_literals
        for literal in unsatisfied:
            # Add unit clause asserting this literal is false
            if literal.startswith('¬'):
                # Negative literal is unsatisfied, so positive must be true
                positive = literal[1:]
                cnf_with_state.add_clause([positive])
            else:
                # Positive literal is unsatisfied, so it must be false
                cnf_with_state.add_clause(['-' + literal])

        # Count satisfying models with state constraints
        state_models = cnf_with_state.count_solutions()

        probability = state_models / total_models if total_models > 0 else 0.0
        logger.debug(f"Applicability probability for {action}: {state_models}/{total_models} = {probability:.3f}")
        return probability

    def _calculate_entropy(self, action: str) -> float:
        """
        Calculate entropy of action model to measure uncertainty.

        Higher entropy means more uncertainty about the action's preconditions and effects.

        Args:
            action: Action name

        Returns:
            Entropy value (non-negative)
        """
        # Entropy based on size of uncertain sets
        # H = -Σ p(x) * log(p(x))

        # Calculate entropy from possible preconditions
        la_size = len(self.pre[action])
        if la_size == 0:
            pre_entropy = 0.0
        else:
            # Uncertain preconditions contribute to entropy
            uncertain_pre = len(self.pre[action]) - len(self.eff_add[action]) - len(self.eff_del[action])
            p_uncertain = uncertain_pre / la_size if la_size > 0 else 0
            p_certain = 1.0 - p_uncertain

            pre_entropy = 0.0
            if p_uncertain > 0:
                pre_entropy -= p_uncertain * math.log2(p_uncertain)
            if p_certain > 0:
                pre_entropy -= p_certain * math.log2(p_certain)

        # Calculate entropy from possible effects
        eff_entropy = 0.0
        for eff_set in [self.eff_maybe_add[action], self.eff_maybe_del[action]]:
            if len(eff_set) > 0:
                # Each undetermined effect adds to entropy
                p = len(eff_set) / la_size if la_size > 0 else 0
                if p > 0:
                    eff_entropy -= p * math.log2(p)

        total_entropy = pre_entropy + eff_entropy
        logger.debug(f"Entropy for {action}: {total_entropy:.3f} (pre: {pre_entropy:.3f}, eff: {eff_entropy:.3f})")
        return total_entropy

    def _calculate_potential_gain_success(self, action: str, objects: List[str], state: Set[str]) -> float:
        """
        Calculate potential information gain from successful execution.

        According to algorithm:
        - preAppPotential(a, O, s) = |pre(a) \ bindP(s, O)|
        - effPotential(a, O, s) = eff+Potential + eff-Potential
        - sucPotential(a, O, s) = effPotential + preAppPotential

        Args:
            action: Action name
            objects: Object binding
            state: Current state

        Returns:
            Potential information gain (normalized)
        """
        state_internal = self._state_to_internal(state)
        satisfied_literals = self._get_satisfied_literals(action, state_internal, objects)

        # Precondition knowledge gain: literals that would be ruled out
        pre_gain = len(self.pre[action] - satisfied_literals)

        # Effect knowledge gain
        # eff+Potential(a, O, s) = |eff?+(a) \ bindP(s, O)| / |La|
        # eff-Potential(a, O, s) = |eff?-(a) ∩ bindP(s, O)| / |La|

        # Get lifted versions of state fluents
        lifted_state = self.bindP(state_internal, objects)

        # Add effects we can rule out (not in unchanged fluents)
        eff_add_gain = len(self.eff_maybe_add[action] - lifted_state)

        # Delete effects we can confirm (are in current state)
        eff_del_gain = len(self.eff_maybe_del[action].intersection(lifted_state))

        # Normalize by La size
        la_size = len(self._get_parameter_bound_literals(action))
        if la_size == 0:
            return 0.0

        normalized_gain = (pre_gain + eff_add_gain + eff_del_gain) / la_size
        logger.debug(f"Success potential for {action}: {normalized_gain:.3f} "
                    f"(pre: {pre_gain}, eff+: {eff_add_gain}, eff-: {eff_del_gain})")
        return normalized_gain

    def _calculate_potential_gain_failure(self, action: str, objects: List[str], state: Set[str]) -> float:
        """
        Calculate potential information gain from failed execution.

        According to algorithm:
        preFailPotential(a, O, s) = 1 - (|SAT(cnf_pre?(a))| - |SAT(cnf'_pre?(a))|) / 2^|La|

        Args:
            action: Action name
            objects: Object binding
            state: Current state

        Returns:
            Potential information gain (normalized)
        """
        # If no constraints yet, failure would provide maximum information
        if len(self.pre_constraints[action]) == 0:
            return 1.0

        state_internal = self._state_to_internal(state)
        satisfied_literals = self._get_satisfied_literals(action, state_internal, objects)

        # Build CNF formula if needed
        if len(self.cnf_managers[action].cnf.clauses) == 0:
            self._build_cnf_formula(action)

        # Current CNF model count
        cnf = self.cnf_managers[action]

        # If CNF is empty, use maximum possible models
        if len(cnf.cnf.clauses) == 0:
            la_size = len(self._get_parameter_bound_literals(action))
            current_models = 2 ** la_size if la_size > 0 else 1
        else:
            current_models = cnf.count_solutions()

        # Simulate adding failure constraint
        unsatisfied = self.pre[action] - satisfied_literals
        if len(unsatisfied) == 0:
            # All preconditions satisfied but action failed - shouldn't happen
            return 0.0

        # Create temporary CNF with new constraint
        temp_cnf = CNFManager()
        temp_cnf.cnf.clauses = cnf.cnf.clauses.copy()
        temp_cnf.fluent_to_var = cnf.fluent_to_var.copy()
        temp_cnf.var_to_fluent = cnf.var_to_fluent.copy()
        temp_cnf.next_var = cnf.next_var

        # Add the new constraint as a clause
        clause = []
        for literal in unsatisfied:
            if literal.startswith('¬'):
                positive = literal[1:]
                clause.append('-' + positive)
            else:
                clause.append(literal)
        temp_cnf.add_clause(clause)

        # Count models with new constraint
        new_models = temp_cnf.count_solutions()

        # Calculate information gain
        la_size = len(self._get_parameter_bound_literals(action))
        max_models = 2 ** la_size if la_size > 0 else 1

        model_reduction = current_models - new_models
        normalized_gain = 1.0 - (model_reduction / max_models)

        logger.debug(f"Failure potential for {action}: {normalized_gain:.3f} "
                    f"(models: {current_models} → {new_models})")
        return max(0.0, normalized_gain)  # Ensure non-negative

    def _calculate_expected_information_gain(self, action: str, objects: List[str], state: Set[str]) -> float:
        """
        Calculate expected information gain for an action in current state.

        E[X(a,O,s)] = pr(app(a,O,s)=1) * sucPotential(a,O,s) +
                      pr(app(a,O,s)=0) * preFailPotential(a,O,s)

        Args:
            action: Action name
            objects: Object binding
            state: Current state

        Returns:
            Expected information gain
        """
        prob_success = self._calculate_applicability_probability(action, objects, state)
        gain_success = self._calculate_potential_gain_success(action, objects, state)
        gain_failure = self._calculate_potential_gain_failure(action, objects, state)

        expected_gain = prob_success * gain_success + (1.0 - prob_success) * gain_failure

        logger.debug(f"Expected gain for {action}({','.join(objects)}): {expected_gain:.3f} "
                    f"(P={prob_success:.2f}, S={gain_success:.2f}, F={gain_failure:.2f})")
        return expected_gain

    def select_action(self, state: Any) -> Tuple[str, List[str]]:
        """
        Select next action to execute based on expected information gain.

        Phase 3: Implements information gain-based selection with multiple strategies:
        - greedy: Always select action with maximum expected gain
        - epsilon_greedy: Explore with probability epsilon, exploit otherwise
        - boltzmann: Probabilistic selection based on gain values

        Args:
            state: Current state

        Returns:
            Tuple of (action_name, objects)
        """
        self.iteration_count += 1
        logger.debug(f"Selecting action for iteration {self.iteration_count}, strategy: {self.selection_strategy}")

        # Convert state to set if needed
        if not isinstance(state, set):
            state = set(state) if state else set()

        # Get all grounded actions
        # Use the existing grounded actions from initialization
        grounded_actions = self.pddl_handler._grounded_actions
        if not grounded_actions:
            logger.warning(f"No grounded actions available at iteration {self.iteration_count}")
            return "no_action", []

        # Calculate expected information gain for each action
        action_gains = []
        for action, binding in grounded_actions:
            objects = [obj.name for obj in binding.values()] if binding else []

            try:
                expected_gain = self._calculate_expected_information_gain(action.name, objects, state)
                action_gains.append((action.name, objects, expected_gain))
            except Exception as e:
                logger.warning(f"Error calculating gain for {action.name}: {e}")
                # Add with zero gain as fallback
                action_gains.append((action.name, objects, 0.0))

        # Sort by gain (highest first)
        action_gains.sort(key=lambda x: x[2], reverse=True)

        # Select action based on strategy
        selected_action, selected_objects = self._select_by_strategy(action_gains)

        logger.info(f"Selected action: {selected_action}({','.join(selected_objects)}) "
                   f"[iteration {self.iteration_count}, gain: {action_gains[0][2]:.3f}]")
        return selected_action, selected_objects

    def _select_by_strategy(self, action_gains: List[Tuple[str, List[str], float]]) -> Tuple[str, List[str]]:
        """
        Select action based on configured strategy.

        Args:
            action_gains: List of (action_name, objects, expected_gain) tuples sorted by gain

        Returns:
            Tuple of (action_name, objects)
        """
        if not action_gains:
            return "no_action", []

        if self.selection_strategy == 'greedy':
            # Always select action with highest gain
            return action_gains[0][0], action_gains[0][1]

        elif self.selection_strategy == 'epsilon_greedy':
            # Explore with probability epsilon
            if random.random() < self.epsilon:
                # Random exploration
                idx = random.randint(0, len(action_gains) - 1)
                logger.debug(f"Epsilon-greedy: Exploring (selected index {idx})")
                return action_gains[idx][0], action_gains[idx][1]
            else:
                # Exploit: select best
                logger.debug("Epsilon-greedy: Exploiting (selected best)")
                return action_gains[0][0], action_gains[0][1]

        elif self.selection_strategy == 'boltzmann':
            # Probabilistic selection based on gains
            # Convert gains to probabilities using softmax with temperature
            gains = [gain for _, _, gain in action_gains]

            # Handle case where all gains are zero
            if max(gains) == 0:
                # Uniform random selection
                idx = random.randint(0, len(action_gains) - 1)
                return action_gains[idx][0], action_gains[idx][1]

            # Compute softmax probabilities
            # Scale gains by temperature
            scaled_gains = [g / self.temperature for g in gains]

            # Subtract max for numerical stability
            max_gain = max(scaled_gains)
            exp_gains = [math.exp(g - max_gain) for g in scaled_gains]
            sum_exp = sum(exp_gains)

            if sum_exp == 0:
                # Fallback to uniform
                idx = random.randint(0, len(action_gains) - 1)
                return action_gains[idx][0], action_gains[idx][1]

            probabilities = [e / sum_exp for e in exp_gains]

            # Sample from distribution
            r = random.random()
            cumsum = 0.0
            for i, prob in enumerate(probabilities):
                cumsum += prob
                if r <= cumsum:
                    logger.debug(f"Boltzmann: Selected index {i} with probability {prob:.3f}")
                    return action_gains[i][0], action_gains[i][1]

            # Fallback to last action
            return action_gains[-1][0], action_gains[-1][1]

        else:
            # Unknown strategy, default to greedy
            logger.warning(f"Unknown selection strategy '{self.selection_strategy}', defaulting to greedy")
            return action_gains[0][0], action_gains[0][1]

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

        logger.info(
            f"Observation {self.observation_count}: {action}({','.join(objects)}) - {'SUCCESS' if success else 'FAILURE'}")
        logger.debug(f"  State size: {len(state) if isinstance(state, (set, list)) else 'N/A'}")
        if success and next_state:
            logger.debug(
                f"  Next state size: {len(next_state) if isinstance(next_state, (set, list)) else 'N/A'}")

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

        logger.debug(f"Total observations for '{action}': {len(self.observation_history[action])}")

    def update_model(self) -> None:
        """
        Update action models based on the latest observation (Phase 2).

        This method processes the most recent observation and updates the
        action model according to the Information Gain algorithm update rules.
        """
        logger.debug("Starting model update")

        # Get the most recent observation across all actions
        latest_action = None
        latest_obs = None
        latest_time = -1

        for action_name, obs_list in self.observation_history.items():
            if obs_list and obs_list[-1]['iteration'] > latest_time:
                latest_time = obs_list[-1]['iteration']
                latest_action = action_name
                latest_obs = obs_list[-1]

        if not latest_obs:
            logger.debug("No observations to process")
            return

        # Extract observation details
        action = latest_action
        objects = latest_obs['objects']
        success = latest_obs['success']
        state = latest_obs['state']
        next_state = latest_obs.get('next_state')

        logger.info(
            f"Updating model for action '{action}' based on {'success' if success else 'failure'} observation")

        if success:
            self._update_success(action, objects, state, next_state)
        else:
            self._update_failure(action, objects, state)

        # Rebuild CNF formula after updates
        logger.debug(f"Rebuilding CNF formula for '{action}'")
        self._build_cnf_formula(action)
        logger.info(f"Model update complete for '{action}'")

    def _update_success(self, action: str, objects: List[str],
                        state: Set[str], next_state: Set[str]) -> None:
        r"""
        Update model after successful action execution.

        According to algorithm:
        - pre(a) = pre(a) ∩ bindP⁻¹(s, O)
        - eff+(a) = eff+(a) ∪ bindP(s' \ s, O)
        - eff-(a) = eff-(a) ∪ bindP(s \ s', O)
        - eff?+(a) = eff?+(a) ∩ bindP(s ∩ s', O)
        - eff?-(a) = eff?-(a) \ bindP(s ∪ s', O)
        - pre?(a) = {B ∩ bindP(s, O) | B ∈ pre?(a)}
        """
        logger.debug(f"_update_success: Processing {action} with objects {objects}")

        # Convert state to internal format
        state_internal = self._state_to_internal(state)
        next_state_internal = self._state_to_internal(next_state)
        logger.debug(
            f"  State: {len(state_internal)} fluents, Next state: {len(next_state_internal)} fluents")

        # Get satisfied literals in state (considering negative literals)
        satisfied_in_state = self._get_satisfied_literals(action, state_internal, objects)
        logger.debug(f"  Satisfied literals: {len(satisfied_in_state)}/{len(self.pre[action])}")

        # Update preconditions: keep only satisfied literals
        # pre(a) = pre(a) ∩ bindP⁻¹(s, O)
        pre_before = len(self.pre[action])
        self.pre[action] = self.pre[action].intersection(satisfied_in_state)
        logger.debug(f"  Preconditions reduced: {pre_before} → {len(self.pre[action])}")

        # Update effects based on state changes
        # eff+(a) = eff+(a) ∪ bindP(s' \ s, O)
        added_fluents = next_state_internal - state_internal
        if added_fluents:
            logger.debug(f"  Added fluents: {len(added_fluents)}")
            lifted_adds = self.bindP(added_fluents, objects)
            eff_add_before = len(self.eff_add[action])
            self.eff_add[action] = self.eff_add[action].union(lifted_adds)
            logger.debug(f"  Add effects updated: {eff_add_before} → {len(self.eff_add[action])}")

        # eff-(a) = eff-(a) ∪ bindP(s \ s', O)
        deleted_fluents = state_internal - next_state_internal
        if deleted_fluents:
            logger.debug(f"  Deleted fluents: {len(deleted_fluents)}")
            lifted_dels = self.bindP(deleted_fluents, objects)
            eff_del_before = len(self.eff_del[action])
            self.eff_del[action] = self.eff_del[action].union(lifted_dels)
            logger.debug(
                f"  Delete effects updated: {eff_del_before} → {len(self.eff_del[action])}")

        # Update possible effects
        # eff?+(a) = eff?+(a) ∩ bindP(s ∩ s', O)
        unchanged_fluents = state_internal.intersection(next_state_internal)
        if unchanged_fluents:
            lifted_unchanged = self.bindP(unchanged_fluents, objects)
            # Possible add effects: must be in unchanged fluents
            # But we need to intersect with lifted space, not grounded
            # So we keep literals that COULD produce these unchanged fluents
            possible_adds = set()
            for lit in self.eff_maybe_add[action]:
                # Check if this literal could produce an unchanged fluent
                grounded = self.bindP_inverse({lit}, objects)
                if any(g in unchanged_fluents for g in grounded):
                    possible_adds.add(lit)
            self.eff_maybe_add[action] = possible_adds
        else:
            # No unchanged fluents means nothing can be a conditional add
            self.eff_maybe_add[action] = set()

        # eff?-(a) = eff?-(a) \ bindP(s ∪ s', O)
        all_true_fluents = state_internal.union(next_state_internal)
        if all_true_fluents:
            lifted_all_true = self.bindP(all_true_fluents, objects)
            # Remove from possible deletes anything that was true before or after
            self.eff_maybe_del[action] = self.eff_maybe_del[action] - lifted_all_true

        # Update constraint sets
        # pre?(a) = {B ∩ bindP(s, O) | B ∈ pre?(a)}
        constraints_before = len(self.pre_constraints[action])
        updated_constraints = []
        for constraint in self.pre_constraints[action]:
            # Keep only literals from constraint that were satisfied
            updated = constraint.intersection(satisfied_in_state)
            if updated:  # Don't add empty constraints
                updated_constraints.append(updated)
        self.pre_constraints[action] = updated_constraints
        logger.debug(
            f"  Constraints updated: {constraints_before} → {len(self.pre_constraints[action])}")

        logger.info(f"Success update complete for {action}: |pre|={len(self.pre[action])}, "
                    f"|eff+|={len(self.eff_add[action])}, |eff-|={len(self.eff_del[action])}")

    def _update_failure(self, action: str, objects: List[str], state: Set[str]) -> None:
        r"""
        Update model after failed action execution.

        According to algorithm:
        - pre?(a) = pre?(a) ∪ {pre(a) \ bindP(s, O)}
        """
        logger.debug(f"_update_failure: Processing {action} with objects {objects}")

        # Convert state to internal format
        state_internal = self._state_to_internal(state)
        logger.debug(f"  State: {len(state_internal)} fluents")

        # Get satisfied literals in state
        satisfied_in_state = self._get_satisfied_literals(action, state_internal, objects)
        logger.debug(f"  Satisfied literals: {len(satisfied_in_state)}/{len(self.pre[action])}")

        # Add new constraint: unsatisfied literals from pre(a)
        # pre?(a) = pre?(a) ∪ {pre(a) \ bindP(s, O)}
        unsatisfied = self.pre[action] - satisfied_in_state

        if unsatisfied:
            constraints_before = len(self.pre_constraints[action])
            self.pre_constraints[action].append(unsatisfied)
            logger.info(
                f"Failure update for {action}: Added constraint with {len(unsatisfied)} unsatisfied literals "
                f"(total constraints: {constraints_before} → {len(self.pre_constraints[action])})")
        else:
            # This shouldn't happen - if all preconditions were satisfied, action should succeed
            logger.warning(
                f"Failed action {action} had all preconditions satisfied - possible environment issue")

    def _state_to_internal(self, state: Set[str]) -> Set[str]:
        """
        Convert state to internal format.

        The state contains grounded fluents that are true.
        Internal format uses the same representation (no conversion needed for positive fluents).
        Negative literals (¬p) are NOT explicitly added for absent fluents.

        Args:
            state: Set of grounded fluents that are true

        Returns:
            Set of fluents in internal format
        """
        # For now, internal format is the same as input
        # We don't add explicit negatives for absent fluents
        return state.copy()

    def _get_satisfied_literals(self, action: str, state: Set[str], objects: List[str]) -> Set[str]:
        """
        Get all literals from pre(a) that are satisfied in the given state.

        A literal is satisfied if:
        - Positive literal p: p ∈ state
        - Negative literal ¬p: p ∉ state

        Args:
            action: Action name
            state: Set of true grounded fluents
            objects: Object binding for the action

        Returns:
            Set of satisfied lifted literals
        """

        satisfied = set()

        for literal in self.pre[action]:
            # Check if literal is negative
            if literal.startswith('¬'):
                # Negative literal: ¬p
                # Remove negation symbol and ground it
                positive_literal = literal[1:]
                grounded = self.bindP_inverse({positive_literal}, objects)

                # Satisfied if NONE of the grounded versions are in state
                if not any(g in state for g in grounded):
                    satisfied.add(literal)
            else:
                # Positive literal: p
                grounded = self.bindP_inverse({literal}, objects)

                # Satisfied if ANY of the grounded versions are in state
                if any(g in state for g in grounded):
                    satisfied.add(literal)

        return satisfied

    def _build_cnf_formula(self, action: str) -> CNFManager:
        """
        Build CNF formula from constraint sets for an action.

        According to algorithm:
        cnf_pre?(a) = ⋀(⋁xl) for B ∈ pre?(a), l ∈ B

        Each constraint set becomes a clause (disjunction).

        Args:
            action: Action name

        Returns:
            CNF manager with the formula
        """
        logger.debug(f"_build_cnf_formula: Building CNF for action '{action}'")
        cnf = self.cnf_managers[action]

        # Clear existing clauses
        clauses_before = len(cnf.cnf.clauses)
        cnf.cnf.clauses = []
        cnf.fluent_to_var = {}
        cnf.var_to_fluent = {}
        cnf.next_var = 1
        logger.debug(f"  Cleared {clauses_before} existing clauses")

        # Build CNF from constraints
        logger.debug(f"  Processing {len(self.pre_constraints[action])} constraint sets")
        clauses_added = 0
        for i, constraint_set in enumerate(self.pre_constraints[action]):
            if not constraint_set:
                logger.debug(f"  Constraint {i}: empty, skipping")
                continue

            # Each constraint set becomes a clause
            clause = []
            for literal in constraint_set:
                if literal.startswith('¬'):
                    # Negative literal: add as negated variable
                    positive = literal[1:]
                    clause.append('-' + positive)
                else:
                    # Positive literal
                    clause.append(literal)

            if clause:
                cnf.add_clause(clause)
                clauses_added += 1
                logger.debug(f"  Constraint {i}: added clause with {len(clause)} literals")

        logger.info(f"CNF formula built for '{action}': {clauses_added} clauses, "
                    f"{len(cnf.fluent_to_var)} unique variables")
        return cnf

    def get_learned_model(self) -> Dict[str, Any]:
        """
        Export the current learned model.

        Returns:
            Dictionary containing the learned model
        """
        logger.debug("Exporting learned model")

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

            logger.debug(
                f"Exported action '{action_name}': {len(self.pre[action_name])} preconditions, "
                f"{len(self.eff_add[action_name])} add effects, "
                f"{len(self.eff_del[action_name])} delete effects, "
                f"{len(self.observation_history[action_name])} observations")

            # Extract predicates from literals
            for literal in self.pre[action_name]:
                pred_name = self._extract_predicate_name(literal)
                if pred_name:
                    model['predicates'].add(pred_name)

        logger.info(f"Model export complete: {len(model['actions'])} actions, "
                    f"{len(model['predicates'])} predicates")
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
            logger.info(f"Convergence: Reached max iterations ({self.max_iterations})")
            self._converged = True
            return True

        # Future: Add convergence criteria based on model uncertainty

        if self._converged:
            logger.info("Convergence: Model has converged")

        return self._converged

    def reset(self) -> None:
        """Reset the learner to initial state."""
        logger.info("Resetting learner to initial state")

        super().reset()

        # Clear all state variables
        num_actions = len(self.pre)
        num_observations = sum(len(obs) for obs in self.observation_history.values())

        self.pre.clear()
        self.pre_constraints.clear()
        self.eff_add.clear()
        self.eff_del.clear()
        self.eff_maybe_add.clear()
        self.eff_maybe_del.clear()
        self.observation_history.clear()
        self.cnf_managers.clear()

        logger.debug(f"Cleared {num_actions} action models and {num_observations} observations")

        # Reinitialize
        self._initialize_action_models()
        logger.info("Reset complete")
