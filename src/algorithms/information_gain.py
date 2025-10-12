"""
Information Gain-based Online Action Model Learning.

Implements a CNF/SAT-based information-theoretic approach to learning action models
using expected information gain for action selection.
"""

import logging
import math
import random
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Any, Set

from src.core import grounding
from src.core.cnf_manager import CNFManager
from src.core.pddl_io import PDDLReader
from .base_learner import BaseActionModelLearner

logger = logging.getLogger(__name__)


class InformationGainLearner(BaseActionModelLearner):
    """
    Information Gain-based action model learner.

    Uses CNF formulas and SAT solving to represent uncertainty about action models
    and selects actions that maximize expected information gain.

    Phase 1 implementation: Core data structures and binding functions only.
    CNF formulas and action selection will be added in later phases.
    """

    # Configuration constants
    PARAMETER_VARIABLE_NAMES = 'xyzuvwpqrst'  # Standard parameter naming: ?x, ?y, ?z, ...
    FLOAT_COMPARISON_EPSILON = 1e-6  # Tolerance for float comparisons
    DEFAULT_MAX_ITERATIONS = 1000

    # Convergence detection parameters (defaults - conservative for statistical validity)
    DEFAULT_MODEL_STABILITY_WINDOW = 50  # Number of iterations to check for model stability
    DEFAULT_INFO_GAIN_EPSILON = 0.001  # Threshold for low information gain
    DEFAULT_SUCCESS_RATE_THRESHOLD = 0.98  # 98% success rate threshold
    DEFAULT_SUCCESS_RATE_WINDOW = 50  # Window size for success rate calculation

    def __init__(self,
                 domain_file: str,
                 problem_file: str,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 model_stability_window: Optional[int] = None,
                 info_gain_epsilon: Optional[float] = None,
                 success_rate_threshold: Optional[float] = None,
                 success_rate_window: Optional[int] = None,
                 **kwargs):
        """
        Initialize Information Gain learner.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            max_iterations: Maximum learning iterations
            model_stability_window: Iterations to check for model stability (default: 50)
            info_gain_epsilon: Threshold for low information gain (default: 0.001)
            success_rate_threshold: Success rate threshold for convergence (default: 0.98)
            success_rate_window: Window size for success rate calculation (default: 50)
            **kwargs: Additional parameters (selection_strategy, epsilon, temperature)

        Raises:
            FileNotFoundError: If domain or problem file doesn't exist
            ValueError: If max_iterations is not positive
        """
        from pathlib import Path

        # Validate inputs
        if not Path(domain_file).exists():
            raise FileNotFoundError(f"Domain file not found: {domain_file}")
        if not Path(problem_file).exists():
            raise FileNotFoundError(f"Problem file not found: {problem_file}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        logger.info(
            f"Initializing Information Gain learner: domain={domain_file}, problem={problem_file}")
        logger.debug(f"Configuration: max_iterations={max_iterations}, kwargs={kwargs}")

        super().__init__(domain_file, problem_file, **kwargs)

        self.max_iterations = max_iterations

        # Set convergence parameters (use provided or defaults)
        self.MODEL_STABILITY_WINDOW = model_stability_window if model_stability_window is not None else self.DEFAULT_MODEL_STABILITY_WINDOW
        self.INFO_GAIN_EPSILON = info_gain_epsilon if info_gain_epsilon is not None else self.DEFAULT_INFO_GAIN_EPSILON
        self.SUCCESS_RATE_THRESHOLD = success_rate_threshold if success_rate_threshold is not None else self.DEFAULT_SUCCESS_RATE_THRESHOLD
        self.SUCCESS_RATE_WINDOW = success_rate_window if success_rate_window is not None else self.DEFAULT_SUCCESS_RATE_WINDOW

        logger.debug(
            f"Convergence parameters: model_stability_window={self.MODEL_STABILITY_WINDOW}, "
            f"info_gain_epsilon={self.INFO_GAIN_EPSILON}, "
            f"success_rate_threshold={self.SUCCESS_RATE_THRESHOLD}, "
            f"success_rate_window={self.SUCCESS_RATE_WINDOW}"
        )

        # Initialize domain knowledge using new architecture
        logger.debug("Parsing PDDL domain and problem files")
        reader = PDDLReader()
        self.domain, _ = reader.parse_domain_and_problem(domain_file, problem_file)
        logger.debug(
            f"PDDL parsing complete: {len(self.domain.lifted_actions)} lifted actions, "
            f"{len(self.domain.predicates)} predicates")

        # Action model state variables (per action schema)
        # Structure: Dict[action_name, data]
        self.pre: Dict[str, Set[str]] = {}          # Possible preconditions (not ruled out)
        self.pre_constraints: Dict[str, Set[frozenset[str]]] = {}  # Constraint sets (pre?)
        self.eff_add: Dict[str, Set[str]] = {}      # Confirmed add effects
        self.eff_del: Dict[str, Set[str]] = {}      # Confirmed delete effects
        self.eff_maybe_add: Dict[str, Set[str]] = {}  # Possible add effects (not determined)
        self.eff_maybe_del: Dict[str, Set[str]] = {}  # Possible delete effects (not determined)

        # Track observations for each action schema
        self.observation_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # CNF managers for each action (Phase 2)
        self.cnf_managers: Dict[str, CNFManager] = {}

        # Performance optimization: cache for constraint literals
        self._constraint_literals_cache: Dict[str, Set[str]] = {}

        # Performance optimization: cache base CNF model counts (Phase 1 enhancement)
        # Invalidated when CNF formula changes (after observe() → update_model())
        self._base_cnf_count_cache: Dict[str, int] = {}

        # Phase 3: Action selection strategy
        self.selection_strategy = kwargs.get('selection_strategy', 'greedy')  # 'greedy', 'epsilon_greedy', 'boltzmann'
        self.epsilon = kwargs.get('epsilon', 0.1)  # For epsilon-greedy
        self.temperature = kwargs.get('temperature', 1.0)  # For Boltzmann

        # Convergence tracking
        self._model_snapshot_history: List[Dict[str, Set[str]]] = []  # History of pre(a) snapshots
        self._success_history: List[bool] = []  # History of action successes/failures
        self._last_max_gain: float = float('inf')  # Track maximum information gain

        # Initialize action models
        logger.debug("Initializing action models")
        self._initialize_action_models()

        logger.info(f"Initialization complete: {len(self.pre)} actions initialized")

    def _initialize_action_models(self):
        """Initialize action model state variables for all actions."""
        logger.debug(f"Initializing models for {len(self.domain.lifted_actions)} actions")

        # Get all lifted actions from domain
        for action_name, action in self.domain.lifted_actions.items():
            logger.debug(
                f"Processing action: {action_name}, parameters: {[p.name for p in action.parameters]}")

            # Get La: all parameter-bound literals for this action
            La = self._get_parameter_bound_literals(action_name)
            logger.debug(f"Generated {len(La)} parameter-bound literals for {action_name}")

            # Initialize state variables according to algorithm
            self.pre[action_name] = La.copy()  # Initially all literals possible
            self.pre_constraints[action_name] = set()  # Empty constraint set
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

        Uses LiftedDomainKnowledge for lifted representations.

        Args:
            action_name: Name of the action

        Returns:
            Set of parameter-bound literal strings (e.g., 'on(?x,?y)', '¬clear(?x)')
        """
        return self.domain.get_parameter_bound_literals(action_name)

    def bindP_inverse(self, literals: Set[str], objects: List[str]) -> Set[str]:
        """
        Ground parameter-bound literals with concrete objects.

        Uses functional grounding utilities.

        Args:
            literals: Set of parameter-bound literals (e.g., {'on(?x,?y)', '¬clear(?x)'})
            objects: Ordered list of objects (e.g., ['a', 'b'])

        Returns:
            Set of grounded literals (e.g., {'on_a_b', '¬clear_a'})
        """
        logger.debug(f"bindP_inverse: Grounding {len(literals)} literals")
        return grounding.ground_literal_set(literals, objects)

    def bindP(self, fluents: Set[str], objects: List[str]) -> Set[str]:
        """
        Lift grounded fluents to parameter-bound literals.

        Uses functional grounding utilities.

        Args:
            fluents: Set of grounded fluent strings (e.g., {'on_a_b', '¬clear_a'})
            objects: Ordered list of objects (e.g., ['a', 'b'])

        Returns:
            Set of parameter-bound literals (e.g., {'on(?x,?y)', '¬clear(?x)'})
        """
        logger.debug(f"bindP: Lifting {len(fluents)} fluents")
        return grounding.lift_fluent_set(fluents, objects, self.domain)

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
        if not self.cnf_managers[action].has_clauses():
            self._build_cnf_formula(action)

        cnf = self.cnf_managers[action]

        # If CNF is still empty after building, no real constraints
        if not cnf.has_clauses():
            logger.debug(f"Action {action} has empty CNF, probability = 1.0")
            return 1.0

        # Count satisfying models for unrestricted formula (cached)
        total_models = self._get_base_model_count(action)
        if total_models == 0:
            # Contradiction in formula
            logger.warning(f"Action {action} has contradictory constraints")
            return 0.0

        # Now add constraints for unsatisfied literals in current state
        # cnf_pre?(a,O,s) = cnf_pre?(a) ∧ (¬xl) for l ∈ (⋃pre?(a)) \ bindP(s, O)
        state_internal = self._state_to_internal(state)
        satisfied_literals = self._get_satisfied_literals(action, state_internal, objects)

        # Get all constraint literals (cached for performance)
        all_constraint_literals = self._get_all_constraint_literals(action)

        unsatisfied = all_constraint_literals - satisfied_literals

        # Build state constraints dict for CNF manager
        state_constraints = {}
        for literal in unsatisfied:
            if literal.startswith('¬'):
                # Negative literal is unsatisfied, so positive must be true
                positive = literal[1:]
                state_constraints[positive] = True
            else:
                # Positive literal is unsatisfied, so it must be false
                state_constraints[literal] = False

        # Count models with state constraints using assumptions (Phase 2 enhancement - no deep copy!)
        assumptions = cnf.state_constraints_to_assumptions(state_constraints)
        state_models = cnf.count_models_with_assumptions(assumptions)

        probability = state_models / total_models if total_models > 0 else 0.0
        logger.debug(f"Applicability probability for {action}: {state_models}/{total_models} = {probability:.3f}")
        return probability

    def _calculate_entropy(self, action: str) -> float:
        """
        Calculate entropy of action model to measure uncertainty.

        Entropy is measured as the log2 of the number of satisfying models in the
        CNF hypothesis space. This directly measures our uncertainty about which
        precondition model is correct.

        H = log2(|SAT(cnf_pre?(a))|)

        Higher entropy means more uncertainty (more possible models).

        Args:
            action: Action name

        Returns:
            Entropy value (non-negative, in bits)
        """
        # Build CNF formula if needed
        if not self.cnf_managers[action].has_clauses():
            self._build_cnf_formula(action)

        cnf = self.cnf_managers[action]

        # Get model count (cached)
        num_models = self._get_base_model_count(action)

        # No uncertainty if 0 or 1 model
        if num_models <= 1:
            entropy = 0.0
        else:
            # Entropy is log2 of model count
            entropy = math.log2(num_models)

        logger.debug(f"Entropy for {action}: {entropy:.3f} ({num_models} models)")
        return entropy

    def _calculate_potential_gain_success(self, action: str, objects: List[str], state: Set[str]) -> float:
        r"""
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

        # Check if constraint already exists - if so, no information gain from failure
        unsatisfied = frozenset(self.pre[action] - satisfied_literals)
        if unsatisfied in self.pre_constraints[action]:
            logger.debug(f"Failure constraint for {action} already exists, no gain")
            return 0.0

        # Build CNF formula if needed
        if not self.cnf_managers[action].has_clauses():
            self._build_cnf_formula(action)

        # Current CNF model count
        cnf = self.cnf_managers[action]

        # Get current model count (cached)
        current_models = self._get_base_model_count(action)

        # Check if no unsatisfied literals (all preconditions satisfied)
        if len(unsatisfied) == 0:
            # All preconditions satisfied meaning itr will not fail so return 0
            return 0.0

        # Count models with temporary constraint (Phase 2 enhancement - no deep copy!)
        new_models = cnf.count_models_with_temporary_clause(unsatisfied)

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

        # Ensure state is in set format
        state = self._ensure_state_is_set(state)

        # Log current state (detailed logging for debugging)
        logger.debug(f"\n{'='*80}")
        logger.debug(f"ITERATION {self.iteration_count}")
        logger.debug(f"{'='*80}")
        logger.debug(f"State: {len(state)} fluents")
        logger.debug(f"State fluents: {sorted(list(state)[:5])}..." if len(state) > 5 else f"State fluents: {sorted(state)}")

        # Calculate expected information gains for all actions (with logging inside)
        action_gains = self._calculate_all_action_gains(state)

        if not action_gains:
            logger.warning(f"No grounded actions available at iteration {self.iteration_count}")
            return "no_action", []

        # Track maximum information gain for convergence detection
        self._last_max_gain = action_gains[0][2] if action_gains else 0.0

        # If maximum gain is effectively zero, no action provides information - signal convergence
        if self._last_max_gain < self.FLOAT_COMPARISON_EPSILON:
            logger.info(f"\nNo action provides information gain (max_gain={self._last_max_gain:.6f}), stopping")
            return "no_action", []

        # Select action based on strategy
        selected_action, selected_objects = self._select_by_strategy(action_gains)

        logger.info(f"Selected action: {selected_action}({','.join(selected_objects)}) "
                   f"[iteration {self.iteration_count}, gain: {action_gains[0][2]:.3f}]")

        # Log action model state before execution
        self._log_action_model_state(selected_action, selected_objects, state)

        return selected_action, selected_objects

    def _ensure_state_is_set(self, state: Any) -> Set[str]:
        """
        Convert state to set format if needed.

        Args:
            state: State in any format (set, list, tuple, etc.)

        Returns:
            State as a set of strings
        """
        if not isinstance(state, set):
            return set(state) if state else set()
        return state

    def _calculate_all_action_gains(self, state: Set[str]) -> List[Tuple[str, List[str], float]]:
        """
        Calculate expected information gain for all grounded actions.

        Args:
            state: Current state as set of fluent strings

        Returns:
            List of (action_name, objects, expected_gain) tuples, sorted by gain (highest first)
        """
        # Use new grounding utilities
        grounded_actions = grounding.ground_all_actions(self.domain, require_injective=False)
        if not grounded_actions:
            return []

        # Log detailed action selection breakdown (debug level)
        logger.debug("\nAction Selection (sorted by expected gain):")

        action_gains = []
        for i, grounded_action in enumerate(grounded_actions):
            try:
                # Calculate all components for logging
                prob_success = self._calculate_applicability_probability(
                    grounded_action.action_name,
                    grounded_action.objects,
                    state
                )
                gain_success = self._calculate_potential_gain_success(
                    grounded_action.action_name,
                    grounded_action.objects,
                    state
                )
                gain_failure = self._calculate_potential_gain_failure(
                    grounded_action.action_name,
                    grounded_action.objects,
                    state
                )
                expected_gain = prob_success * gain_success + (1.0 - prob_success) * gain_failure

                action_gains.append((grounded_action.action_name, grounded_action.objects, expected_gain))

                # Log detailed breakdown for first 10 and last action
                if i < 10 or i == len(grounded_actions) - 1:
                    action_str = f"{grounded_action.action_name}({','.join(grounded_action.objects)})"
                    logger.debug(f"  {action_str:30s} E[gain]={expected_gain:.6f} "
                               f"[P(success)={prob_success:.3f}, gain_success={gain_success:.3f}, gain_failure={gain_failure:.3f}]")
                elif i == 10:
                    logger.debug(f"  ... ({len(grounded_actions) - 11} more actions)")

            except Exception as e:
                logger.warning(f"Error calculating gain for {grounded_action.action_name}: {e}")
                # Add with zero gain as fallback
                action_gains.append((grounded_action.action_name, grounded_action.objects, 0.0))

        # Sort by gain (highest first)
        action_gains.sort(key=lambda x: x[2], reverse=True)

        # Log top action after sorting
        if action_gains:
            top_action, top_objects, top_gain = action_gains[0]
            logger.debug(f"\nTop action after sorting: {top_action}({','.join(top_objects)}) with E[gain]={top_gain:.6f}")

        return action_gains

    def _log_action_model_state(self, action: str, objects: List[str], state: Set[str]) -> None:
        """
        Log the current action model state before execution (debug level for detailed diagnostics).

        Args:
            action: Action name
            objects: Object binding
            state: Current state
        """
        logger.debug(f"\nApplying: {action}({','.join(objects)})")

        # Log preconditions
        if self.pre[action]:
            logger.debug(f"  Preconditions pre(a): {{{', '.join(sorted(list(self.pre[action])[:10]))}{'...' if len(self.pre[action]) > 10 else ''}}}")
            logger.debug(f"    Total: {len(self.pre[action])} literals")
        else:
            logger.debug(f"  Preconditions pre(a): ∅ (empty)")

        # Log constraints
        if self.pre_constraints[action]:
            logger.debug(f"  Constraints pre?(a): {len(self.pre_constraints[action])} constraint sets")
            for i, constraint in enumerate(list(self.pre_constraints[action])[:3], 1):
                logger.debug(f"    {i}. {{{', '.join(sorted(list(constraint))[:5])}{' ...' if len(constraint) > 5 else ''}}} ({len(constraint)} literals)")
            if len(self.pre_constraints[action]) > 3:
                logger.debug(f"    ... ({len(self.pre_constraints[action]) - 3} more constraints)")
        else:
            logger.debug(f"  Constraints pre?(a): ∅ (no constraints yet)")

        # Log CNF statistics
        cnf = self.cnf_managers[action]
        if cnf.has_clauses():
            num_clauses = len(cnf.cnf.clauses)
            num_vars = len(cnf.fluent_to_var)
            num_models = cnf.count_solutions()
            logger.debug(f"  CNF formula: {num_clauses} clauses, {num_vars} variables, {num_models} satisfying models")
        else:
            logger.debug(f"  CNF formula: empty (no constraints)")

        # Log confirmed effects
        if self.eff_add[action] or self.eff_del[action]:
            logger.debug(f"  Confirmed effects:")
            if self.eff_add[action]:
                logger.debug(f"    Add: {{{', '.join(sorted(list(self.eff_add[action])[:5]))}{'...' if len(self.eff_add[action]) > 5 else ''}}} ({len(self.eff_add[action])} total)")
            else:
                logger.debug(f"    Add: ∅")
            if self.eff_del[action]:
                logger.debug(f"    Del: {{{', '.join(sorted(list(self.eff_del[action])[:5]))}{'...' if len(self.eff_del[action]) > 5 else ''}}} ({len(self.eff_del[action])} total)")
            else:
                logger.debug(f"    Del: ∅")
        else:
            logger.debug(f"  Confirmed effects: None yet")

        # Log uncertain effects summary
        maybe_add_count = len(self.eff_maybe_add[action])
        maybe_del_count = len(self.eff_maybe_del[action])
        logger.debug(f"  Uncertain effects: {maybe_add_count} maybe-add, {maybe_del_count} maybe-del")

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
        Observe action execution result and update model.

        Records the observation and immediately processes it to update
        the action model according to Information Gain algorithm rules.

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

        # Track success/failure for convergence detection
        self._success_history.append(success)

        logger.debug(f"Total observations for '{action}': {len(self.observation_history[action])}")

        # Automatically update model after recording observation
        # This ensures the model learns immediately from each observation
        self.update_model()

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
        cnf_before_clauses = len(self.cnf_managers[action].cnf.clauses) if self.cnf_managers[action].has_clauses() else 0
        self._build_cnf_formula(action)
        cnf_after_clauses = len(self.cnf_managers[action].cnf.clauses) if self.cnf_managers[action].has_clauses() else 0

        # Invalidate base CNF count cache after formula changes (Phase 1 enhancement)
        self._base_cnf_count_cache.pop(action, None)

        if cnf_before_clauses != cnf_after_clauses:
            logger.debug(f"  CNF clauses: {cnf_before_clauses} → {cnf_after_clauses}")

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
        # Keep only literals that could be conditional add effects
        # (i.e., fluents that remained unchanged - were true before and after)
        unchanged_fluents = state_internal.intersection(next_state_internal)
        lifted_unchanged = self.bindP(unchanged_fluents, objects)
        self.eff_maybe_add[action] = self.eff_maybe_add[action].intersection(lifted_unchanged)

        # eff?-(a) = eff?-(a) \ bindP(s ∪ s', O)
        # Remove literals that were true before OR after
        # (if a fluent was ever true, it can't be a delete effect)
        all_true_fluents = state_internal.union(next_state_internal)
        lifted_all_true = self.bindP(all_true_fluents, objects)
        self.eff_maybe_del[action] = self.eff_maybe_del[action] - lifted_all_true

        # Update constraint sets
        # pre?(a) = {B ∩ bindP(s, O) | B ∈ pre?(a)}
        constraints_before = len(self.pre_constraints[action])
        updated_constraints = set()
        for constraint in self.pre_constraints[action]:
            # Keep only literals from constraint that were satisfied
            updated = constraint.intersection(satisfied_in_state)
            if updated:  # Don't add empty constraints
                updated_constraints.add(frozenset(updated))
        self.pre_constraints[action] = updated_constraints
        # Invalidate cache after modifying constraints
        self._invalidate_constraint_cache(action)
        logger.debug(
            f"  Constraints updated: {constraints_before} → {len(self.pre_constraints[action])}")

        logger.info(f"Success update complete for {action}: |pre|={len(self.pre[action])}, "
                    f"|eff+|={len(self.eff_add[action])}, |eff-|={len(self.eff_del[action])}")

        # Detailed logging of changes (debug level)
        logger.debug(f"\nResult: SUCCESS")
        logger.debug(f"Updated model for {action}:")
        logger.debug(f"  Preconditions refined: {pre_before} → {len(self.pre[action])} literals")
        if added_fluents:
            logger.debug(f"  Add effects confirmed: {eff_add_before} → {len(self.eff_add[action])}")
        if deleted_fluents:
            logger.debug(f"  Del effects confirmed: {eff_del_before} → {len(self.eff_del[action])}")
        logger.debug(f"  Constraints refined: {constraints_before} → {len(self.pre_constraints[action])}")

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
        unsatisfied = frozenset(self.pre[action] - satisfied_in_state)

        if unsatisfied:
            constraints_before = len(self.pre_constraints[action])
            self.pre_constraints[action].add(unsatisfied)
            # Invalidate cache after modifying constraints
            self._invalidate_constraint_cache(action)
            new_count = len(self.pre_constraints[action])
            if new_count > constraints_before:
                logger.info(
                    f"Failure update for {action}: Added constraint with {len(unsatisfied)} unsatisfied literals "
                    f"(total constraints: {constraints_before} → {new_count})")
            else:
                logger.debug(
                    f"Failure update for {action}: Constraint already exists (no new information gained)")
        else:
            # This shouldn't happen - if all preconditions were satisfied, action should succeed
            logger.warning(
                f"Failed action {action} had all preconditions satisfied - possible environment issue")

        # Detailed logging of changes (debug level)
        logger.debug(f"\nResult: FAILURE")
        if unsatisfied and new_count > constraints_before:
            logger.debug(f"Updated model for {action}:")
            logger.debug(f"  Added constraint: {{{', '.join(sorted(list(unsatisfied))[:5])}{' ...' if len(unsatisfied) > 5 else ''}}} ({len(unsatisfied)} literals)")
            logger.debug(f"  Total constraints: {constraints_before} → {new_count}")
        elif unsatisfied:
            logger.debug(f"No new information for {action}: constraint already exists")

    def _state_to_internal(self, state: Set[str]) -> Set[str]:
        """
        Convert state to internal format.

        The algorithm represents states as sets of true grounded fluents.
        Negative literals (¬p) are handled implicitly: if p is not in the set, ¬p is true.
        This is standard STRIPS-style representation with closed-world assumption.

        Args:
            state: Set of grounded fluents that are true

        Returns:
            Set of fluents in internal format (copy for safety)
        """
        # Internal format matches input format (closed-world assumption)
        # Copy to prevent external modifications
        return state.copy()

    def _get_satisfied_literals(self, action: str, state: Set[str], objects: List[str]) -> Set[str]:
        """
        Get all literals from pre(a) that are satisfied in the given state.

        A literal is satisfied if:
        - Positive literal p: ALL groundings of p are in state
        - Negative literal ¬p: ALL groundings of p are NOT in state (i.e., p ∉ state)

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

                # Satisfied if ALL grounded versions are NOT in state
                # (i.e., none of them are in state)
                if all(g not in state for g in grounded):
                    satisfied.add(literal)
            else:
                # Positive literal: p
                grounded = self.bindP_inverse({literal}, objects)

                # Satisfied if ALL grounded versions ARE in state
                if all(g in state for g in grounded):
                    satisfied.add(literal)

        return satisfied

    def _get_all_constraint_literals(self, action: str) -> Set[str]:
        """
        Get all literals from all constraint sets for an action (cached for performance).

        Args:
            action: Action name

        Returns:
            Set of all literals appearing in any constraint set
        """
        if action not in self._constraint_literals_cache:
            all_literals = set()
            for constraint_set in self.pre_constraints[action]:
                all_literals.update(constraint_set)
            self._constraint_literals_cache[action] = all_literals

        return self._constraint_literals_cache[action]

    def _invalidate_constraint_cache(self, action: str):
        """
        Invalidate the constraint literals cache for an action.

        Should be called when pre_constraints[action] is modified.

        Args:
            action: Action name
        """
        self._constraint_literals_cache.pop(action, None)

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

        # Use CNF manager method to build from constraint sets
        cnf.build_from_constraint_sets(self.pre_constraints[action])

        logger.info(f"CNF formula built for '{action}': {len(cnf.cnf.clauses)} clauses, "
                    f"{len(cnf.fluent_to_var)} unique variables")
        return cnf

    def _get_base_model_count(self, action: str) -> int:
        """
        Get cached base CNF model count for action (Phase 1 performance enhancement).

        Caches the model count to avoid redundant expensive SAT solving calls
        within the same iteration. Cache is invalidated after CNF formula changes.

        Args:
            action: Action name

        Returns:
            Number of satisfying models for base CNF formula
        """
        # Check cache first
        if action in self._base_cnf_count_cache:
            return self._base_cnf_count_cache[action]

        # Build CNF if needed
        if not self.cnf_managers[action].has_clauses():
            self._build_cnf_formula(action)

        cnf = self.cnf_managers[action]

        # If still empty, use maximum possible models
        if not cnf.has_clauses():
            la_size = len(self._get_parameter_bound_literals(action))
            count = 2 ** la_size if la_size > 0 else 1
        else:
            # Perform expensive count and cache it
            count = cnf.count_solutions()

        # Cache the result
        self._base_cnf_count_cache[action] = count
        logger.debug(f"Cached base model count for {action}: {count}")
        return count

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

        Simple parsing logic - no need for delegation.

        Args:
            literal: Literal string (e.g., 'on(?x,?y)' or '¬clear(?x)')

        Returns:
            Predicate name or None
        """
        # Remove negation if present
        if literal.startswith('¬'):
            literal = literal[1:]

        # Extract predicate name (before '(' or whole string if no parens)
        if '(' in literal:
            return literal[:literal.index('(')]
        return literal

    def has_converged(self) -> bool:
        """
        Check if learning has converged.

        Convergence is determined by multiple criteria:
        1. Max iterations reached (forced convergence)
        2. Model stability: no precondition changes for MODEL_STABILITY_WINDOW iterations
        3. Low information gain: max gain < INFO_GAIN_EPSILON
        4. High success rate: >95% success in last SUCCESS_RATE_WINDOW actions

        Returns:
            True if model has converged, False otherwise
        """
        # Check iteration limit (forced convergence)
        if self.iteration_count >= self.max_iterations:
            logger.info(f"Convergence: Reached max iterations ({self.max_iterations})")
            self._converged = True
            return True

        # Need minimum observations before checking other criteria
        if self.iteration_count < self.MODEL_STABILITY_WINDOW:
            return False

        # Criterion 1: Model stability check
        # Check if preconditions (pre(a)) haven't changed for N iterations
        model_stable = self._check_model_stability()

        # Criterion 2: Low information gain check
        # Check if maximum expected gain is below epsilon threshold
        low_info_gain = self._check_low_information_gain()

        # Criterion 3: High success rate check
        # Check if success rate is above threshold in recent window
        high_success_rate = self._check_high_success_rate()

        # Converge only if ALL three criteria are met (conservative for statistical validity)
        # Previous aggressive logic (ANY 2 of 3) caused premature convergence
        all_criteria_met = model_stable and low_info_gain and high_success_rate

        if all_criteria_met:
            if not self._converged:
                logger.info(
                    f"Convergence: All criteria met "
                    f"(stable={model_stable}, low_gain={low_info_gain}, high_success={high_success_rate})"
                )
            self._converged = True
            return True

        return False

    def _check_model_stability(self) -> bool:
        """
        Check if model has been stable (no precondition changes) for N iterations.

        Returns:
            True if model is stable, False otherwise
        """
        # Take snapshot of current preconditions
        current_snapshot = {action: pre_set.copy() for action, pre_set in self.pre.items()}

        # Add to history
        self._model_snapshot_history.append(current_snapshot)

        # Keep only last MODEL_STABILITY_WINDOW snapshots
        if len(self._model_snapshot_history) > self.MODEL_STABILITY_WINDOW:
            self._model_snapshot_history.pop(0)

        # Need full window to check stability
        if len(self._model_snapshot_history) < self.MODEL_STABILITY_WINDOW:
            return False

        # Check if all snapshots in window are identical
        first_snapshot = self._model_snapshot_history[0]
        stable = all(
            snapshot == first_snapshot
            for snapshot in self._model_snapshot_history[1:]
        )

        if stable:
            logger.debug(
                f"Model stability: STABLE (no changes for {self.MODEL_STABILITY_WINDOW} iterations)"
            )
        else:
            logger.debug("Model stability: UNSTABLE (recent changes detected)")

        return stable

    def _check_low_information_gain(self) -> bool:
        """
        Check if maximum expected information gain is below epsilon threshold.

        Returns:
            True if information gain is low, False otherwise
        """
        # If no observations yet, gain is high
        if self.observation_count == 0:
            return False

        # Maximum gain is tracked during action selection
        # Check if it's below threshold
        low_gain = self._last_max_gain < self.INFO_GAIN_EPSILON

        if low_gain:
            logger.debug(
                f"Information gain: LOW (max_gain={self._last_max_gain:.4f} < ε={self.INFO_GAIN_EPSILON})"
            )
        else:
            logger.debug(
                f"Information gain: HIGH (max_gain={self._last_max_gain:.4f} >= ε={self.INFO_GAIN_EPSILON})"
            )

        return low_gain

    def _check_high_success_rate(self) -> bool:
        """
        Check if success rate in recent window is above threshold.

        Returns:
            True if success rate is high, False otherwise
        """
        # Need minimum history
        if len(self._success_history) < self.SUCCESS_RATE_WINDOW:
            return False

        # Calculate success rate over recent window
        recent_successes = self._success_history[-self.SUCCESS_RATE_WINDOW:]
        success_rate = sum(recent_successes) / len(recent_successes)

        high_rate = success_rate >= self.SUCCESS_RATE_THRESHOLD

        if high_rate:
            logger.debug(
                f"Success rate: HIGH ({success_rate:.2%} >= {self.SUCCESS_RATE_THRESHOLD:.0%})"
            )
        else:
            logger.debug(
                f"Success rate: LOW ({success_rate:.2%} < {self.SUCCESS_RATE_THRESHOLD:.0%})"
            )

        return high_rate

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

        # Clear convergence tracking
        self._model_snapshot_history.clear()
        self._success_history.clear()
        self._last_max_gain = float('inf')

        logger.debug(f"Cleared {num_actions} action models and {num_observations} observations")

        # Reinitialize
        self._initialize_action_models()
        logger.info("Reset complete")
