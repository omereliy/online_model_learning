"""
Information Gain-based Online Action Model Learning.

Implements a CNF/SAT-based information-theoretic approach to learning action models
using expected information gain for action selection.
"""

import json
import logging
import math
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Set

from src.core import grounding
from src.core.grounding import is_injective_binding
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

    def __init__(self,
                 domain_file: str,
                 problem_file: str,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 use_object_subset: bool = True,
                 spare_objects_per_type: int = 2,
                 max_iterations_per_subset: int = DEFAULT_MAX_ITERATIONS,
                 num_workers: Optional[int] = None,
                 parallel_threshold: int = 3000,
                 **kwargs):
        """
        Initialize Information Gain learner.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            max_iterations: Maximum learning iterations
            use_object_subset: Enable object subset selection for reduced grounding (default: True).
                              Uses state-aware selection to prioritize objects from current state.
            spare_objects_per_type: Extra objects per type beyond minimum requirement (default: 2)
            max_iterations_per_subset: Max iterations before rotating to new subset (default: 100)
            num_workers: Number of worker processes for parallel gain computation.
                        None = auto (cpu_count), 0 = disabled (sequential only)
            parallel_threshold: Minimum number of grounded actions to enable parallel computation.
                        Default is 5000 because process creation overhead is significant.
                        Parallelization is only beneficial for very large action spaces (>5000 actions)
                        or single long-running computations. For typical domains, sequential is faster.
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
        self._last_max_gain: float = float('inf')  # Track maximum information gain

        # Object subset selection (for reduced grounding space)
        self.use_object_subset = use_object_subset
        self._original_use_object_subset = use_object_subset  # Store for reset()
        self.spare_objects_per_type = spare_objects_per_type
        self.max_iterations_per_subset = max_iterations_per_subset
        self.subset_manager: Optional['ObjectSubsetManager'] = None
        self._subset_iteration_count = 0

        if self.use_object_subset:
            from src.core.object_subset_manager import ObjectSubsetManager
            self.subset_manager = ObjectSubsetManager(
                domain=self.domain,
                spare_objects_per_type=self.spare_objects_per_type,
                random_seed=kwargs.get('seed'),
                defer_initial_selection=True  # Wait for state to select subset
            )
            logger.info(f"[SUBSET] Object subset selection enabled (deferred until first state): {self.subset_manager.get_status()}")

        # Parallel computation settings (Phase C optimization)
        self.num_workers = num_workers
        self.parallel_threshold = parallel_threshold

        # Persistent process pool for parallel computation (avoids per-iteration spawn overhead)
        self._pool: Optional['ProcessPoolExecutor'] = None
        self._pool_workers: int = 0

        # Approximate counting configuration (for large formulas)
        # Note: Threshold lowered from 15 to 8 to avoid exponential enumeration
        # when model count is large despite few variables (e.g., blocksworld with weak constraints)
        self.use_approximate_counting = kwargs.get('use_approximate_counting', True)
        self.approximate_threshold_vars = kwargs.get('approximate_threshold_vars', 8)
        self.approximate_epsilon = kwargs.get('approximate_epsilon', 0.3)
        self.approximate_delta = kwargs.get('approximate_delta', 0.05)

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
        # logger.debug(f"bindP_inverse: Grounding {len(literals)} literals")
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
        # For action to be applicable, unsatisfied literals must NOT be preconditions
        # (If an unsatisfied literal IS a precondition, action is inapplicable)
        # Literals are variable names (¬ is part of name, not CNF negation)
        state_constraints = {}
        for literal in unsatisfied:
            state_constraints[literal] = False  # var_literal = FALSE (not a precondition)

        # Count models with state constraints using assumptions (Phase 2 enhancement - no deep copy!)
        assumptions = cnf.state_constraints_to_assumptions(state_constraints)
        state_models = cnf.count_models_with_assumptions(assumptions)

        probability = state_models / total_models if total_models > 0 else 0.0
        logger.debug(f"Applicability probability for {action}: {state_models}/{total_models} = {probability:.3f}")
        return probability

    def _calculate_potential_gain_success(self, action: str, objects: List[str], state: Set[str]) -> float:
        r"""
        Calculate potential information gain from successful execution.

        According to algorithm:
        - preAppPotential(a, O, s) = (|SAT(cnf_pre?(a))| - |SAT(cnf'_pre?(a))|) / 3^|fluents|
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

        # Calculate normalization factor: 3^|fluents| where |fluents| = |La| / 2
        la_size = len(self._get_parameter_bound_literals(action))
        num_fluents = la_size // 2  # La contains both p and ¬p for each fluent
        total_hypotheses = 3 ** num_fluents if num_fluents > 0 else 1

        # Calculate precondition knowledge gain using SAT count difference
        # preAppPotential(a, O, s) = (|SAT(cnf_pre?(a))| - |SAT(cnf'_pre?(a))|) / 3^|fluents|
        if not self.pre_constraints[action]:
            # No CNF constraints yet - no precondition gain from success
            pre_gain = 0
        else:
            # Build CNF formula if needed
            if not self.cnf_managers[action].has_clauses():
                self._build_cnf_formula(action)

            cnf = self.cnf_managers[action]

            # Get current SAT count
            current_models = self._get_base_model_count(action)

            # Simulate adding unit clauses for satisfied literals
            # Each satisfied literal l was satisfied in state, so l is NOT required as a precondition
            # Use get_variable() to avoid creating new variables as side effect
            unit_clause_assumptions = []
            for literal in satisfied_literals:
                if literal in self.pre[action]:
                    var_id = cnf.get_variable(literal)
                    if var_id is not None:
                        unit_clause_assumptions.append(-var_id)  # literal is NOT a precondition

            # Count models after adding unit clause assumptions
            if unit_clause_assumptions and cnf.has_clauses():
                new_models = cnf.count_models_with_assumptions(unit_clause_assumptions)
            else:
                new_models = current_models

            # Calculate precondition gain as model reduction
            pre_gain = current_models - new_models

        # Effect knowledge gain
        # eff+Potential(a, O, s) = |eff?+(a) \ bindP(s, O)| / |La|
        # eff-Potential(a, O, s) = |eff?-(a) ∩ bindP(s, O)| / |La|

        # Get lifted versions of state fluents
        lifted_state = self.bindP(state_internal, objects)

        # Add effects we can rule out (not in unchanged fluents)
        eff_add_gain = len(self.eff_maybe_add[action] - lifted_state)

        # Delete effects we can confirm (are in current state)
        eff_del_gain = len(self.eff_maybe_del[action].intersection(lifted_state))

        if total_hypotheses == 0:
            return 0.0

        normalized_gain = (pre_gain + eff_add_gain + eff_del_gain) / total_hypotheses
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
        # La contains both p and ¬p for each fluent, so num_fluents = |La| / 2
        # Each fluent has 3 possible states: positive precondition, negative precondition, or not a precondition
        la_size = len(self._get_parameter_bound_literals(action))
        num_fluents = la_size // 2  # La contains both p and ¬p for each fluent
        total_hypotheses = 3 ** num_fluents if num_fluents > 0 else 1

        model_reduction = current_models - new_models
        normalized_gain = model_reduction / total_hypotheses

        logger.debug(f"Failure potential for {action}: {normalized_gain:.3f} "
                    f"(models: {current_models} → {new_models}, reduction: {model_reduction})")
        return max(0.0, min(1.0, normalized_gain))  # Clamp to [0, 1]

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
        self._subset_iteration_count += 1

        # Ensure state is in set format
        state = self._ensure_state_is_set(state)

        # Handle object subset selection with state awareness
        if self.use_object_subset and self.subset_manager:
            # First iteration: select initial subset using state information
            if self.subset_manager.subset_rotation_count == 0:
                logger.info(f"[SUBSET] Selecting initial state-aware subset (max possible subsets: {self.subset_manager.get_max_possible_subsets()})")
                self.subset_manager.select_state_aware_subset(state)
            # Check for subset rotation
            elif self._should_rotate_subset():
                if not self.subset_manager.rotate_state_aware(state):
                    # All subsets exhausted - switch to full object set for final validation
                    # This ensures we don't get stuck with limited grounding space
                    # and can validate/refine the learned model on all objects
                    logger.info(
                        f"[SUBSET] All {self.subset_manager.get_max_possible_subsets()} subsets exhausted "
                        f"- switching to full object set for final validation"
                    )
                    self.use_object_subset = False
                    self._subset_iteration_count = 0
                else:
                    self._subset_iteration_count = 0  # Reset counter after rotation
            else:
                # Note: augment_with_state_objects removed - it was adding ALL objects
                # from state fluents, defeating subset selection. Initial state-aware
                # selection already prioritizes objects from the current state.
                pass

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

        # If maximum gain is zero or less, check lifted-level learning potential
        # Grounded info gain can be zero even when there's still uncertainty at lifted level
        if self._last_max_gain <= 0:
            logger.info(f"\nNo grounded information gain (max_gain={self._last_max_gain:.6f})")

            # Check if there's still learning potential at the lifted level
            if self._has_lifted_learning_potential():
                logger.info("Lifted-level uncertainty exists - continuing exploration")
                uncertainty = self._get_lifted_uncertainty_summary()
                total_uncertain = sum(
                    u['uncertain_pre'] + u['uncertain_add'] + u['uncertain_del']
                    for u in uncertainty.values()
                )
                logger.debug(f"Total lifted uncertainty: {total_uncertain} literals across {len(uncertainty)} actions")

                # Find applicable actions to continue exploring
                applicable_actions = self._filter_applicable_actions(action_gains, state)

                if applicable_actions:
                    # Select random applicable action to continue exploring
                    selected_action, selected_objects = random.choice(applicable_actions)
                    logger.info(f"Selecting random applicable action to continue exploring: "
                               f"{selected_action}({','.join(selected_objects)})")
                else:
                    # No applicable actions in current state - need state change
                    # Select any action (even if likely to fail) to potentially learn from failure
                    selected_action, selected_objects = random.choice(
                        [(a, o) for a, o, _ in action_gains]
                    )
                    logger.info(f"No applicable actions - selecting random action for failure learning: "
                               f"{selected_action}({','.join(selected_objects)})")
            else:
                # No lifted-level uncertainty - truly converged
                logger.info("No lifted-level learning potential - model fully learned")
                return "no_action", []
        else:
            # Select action based on strategy (normal information gain-based selection)
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
        # Use subset-aware grounding if enabled, otherwise use full grounding
        if self.use_object_subset and self.subset_manager:
            from src.core.grounding import ground_all_actions_with_subset
            active_objects = self.subset_manager.get_active_object_names()
            grounded_actions = ground_all_actions_with_subset(
                self.domain, active_objects, require_injective=False
            )
            logger.debug(f"[SUBSET] Using object subset with {len(active_objects)} objects")
        else:
            grounded_actions = grounding.ground_all_actions(self.domain, require_injective=False)

        if not grounded_actions:
            return []

        # NOTE: Removed cache pre-population loop that caused exponential enumeration hang
        # The loop called get_all_solutions() which enumerates ALL models - extremely slow
        # when model count is large (e.g., 2000+ in blocksworld) even with few variables.
        # Model counting is now done on-demand with count_solutions() which is more efficient.

        # Determine whether to use parallel computation
        num_actions = len(grounded_actions)
        use_parallel = self._should_use_parallel(num_actions)

        if use_parallel:
            return self._calculate_all_action_gains_parallel(grounded_actions, state)
        else:
            return self._calculate_all_action_gains_sequential(grounded_actions, state)

    def _calculate_all_action_gains_sequential(self, grounded_actions: List, state: Set[str]) -> List[Tuple[str, List[str], float]]:
        """
        Sequential implementation of action gain computation.

        Args:
            grounded_actions: List of GroundedAction objects
            state: Current state as set of fluent strings

        Returns:
            List of (action_name, objects, expected_gain) tuples, sorted by gain (highest first)
        """
        # Log detailed action selection breakdown (debug level)
        logger.debug("\nAction Selection (sorted by expected gain):")

        action_gains = []
        for i, grounded_action in enumerate(grounded_actions):
            try:
                action_name = grounded_action.action_name
                objects = grounded_action.objects

                # Early termination: Check if action is fully learned (no uncertainty remains)
                # This avoids expensive SAT queries for converged actions
                has_constraints = bool(self.pre_constraints[action_name])
                has_pre_uncertainty = len(self.pre[action_name]) > len(self._get_certain_preconditions(action_name))
                has_add_uncertainty = bool(self.eff_maybe_add[action_name] - self.eff_add[action_name])
                has_del_uncertainty = bool(self.eff_maybe_del[action_name] - self.eff_del[action_name])

                if not has_constraints and not has_pre_uncertainty and not has_add_uncertainty and not has_del_uncertainty:
                    # Action is fully learned - no information gain possible
                    action_gains.append((action_name, objects, 0.0))
                    continue

                # Calculate applicability probability and success gain
                prob_success = self._calculate_applicability_probability(action_name, objects, state)
                gain_success = self._calculate_potential_gain_success(action_name, objects, state)

                # Early termination for failure gain: Check if constraint already exists
                # If so, failure would add no new information
                state_internal = self._state_to_internal(state)
                satisfied = self._get_satisfied_literals(action_name, state_internal, objects)
                unsatisfied = frozenset(self.pre[action_name] - satisfied)

                if not unsatisfied or unsatisfied in self.pre_constraints[action_name]:
                    # Failure constraint already exists or all preconditions satisfied
                    gain_failure = 0.0
                else:
                    gain_failure = self._calculate_potential_gain_failure(action_name, objects, state)

                expected_gain = prob_success * gain_success + (1.0 - prob_success) * gain_failure

                action_gains.append((action_name, objects, expected_gain))

                # Log detailed breakdown for first 10 and last action
                if i < 10 or i == len(grounded_actions) - 1:
                    action_str = f"{action_name}({','.join(objects)})"
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

        # Filter/prioritize injective bindings based on action execution history
        # Actions without successful observations: filter out non-injective (strict)
        # Actions with successful observations: prioritize injective (lenient)
        filtered_gains = []
        for action_name, objects, gain in action_gains:
            is_injective = is_injective_binding(objects)
            has_success = self._has_successful_observation(action_name)

            if has_success:
                # Action already executed successfully - prioritize injective but allow non-injective
                filtered_gains.append((action_name, objects, gain, is_injective))
            else:
                # Action not yet executed - filter out non-injective (unless no alternative)
                if is_injective:
                    filtered_gains.append((action_name, objects, gain, is_injective))
                # Non-injective for unexecuted action: defer (handled in fallback below)

        # Separate by injectivity, preserving gain ordering within each group
        injective_actions = [(a, o, g) for a, o, g, inj in filtered_gains if inj]
        non_injective_actions = [(a, o, g) for a, o, g, inj in filtered_gains if not inj]

        # Fallback: For action types with NO injective bindings (unexecuted), add non-injective
        actions_with_injective = {a for a, _, _, inj in filtered_gains if inj}
        for action_name, objects, gain in action_gains:
            if not is_injective_binding(objects) and not self._has_successful_observation(action_name):
                # Check if this action type has any injective options
                if action_name not in actions_with_injective:
                    non_injective_actions.append((action_name, objects, gain))
                    # TODO: use ESAM effects logic on non-injective bindings

        if non_injective_actions:
            logger.debug(f"Injective binding filter: {len(injective_actions)} injective, "
                         f"{len(non_injective_actions)} non-injective actions")

        action_gains = injective_actions + non_injective_actions

        # Log top action after sorting
        if action_gains:
            top_action, top_objects, top_gain = action_gains[0]
            logger.debug(f"\nTop action after sorting: {top_action}({','.join(top_objects)}) with E[gain]={top_gain:.6f}")

        return action_gains

    def _should_use_parallel(self, num_actions: int) -> bool:
        """
        Determine whether parallel computation should be used.

        Always uses parallel unless explicitly disabled (num_workers=0)
        or only 1 worker is available.

        Args:
            num_actions: Number of grounded actions (unused, kept for API compatibility)

        Returns:
            True if parallel computation should be used
        """
        # Explicitly disabled
        if self.num_workers == 0:
            return False

        # Must have more than 1 worker to benefit from parallelism
        actual_workers = self.num_workers
        if actual_workers is None:
            actual_workers = os.cpu_count() or 4
        if actual_workers <= 1:
            return False

        return True

    def _get_or_create_pool(self, num_workers: int) -> ProcessPoolExecutor:
        """
        Get existing pool or create new one if worker count changed.

        Maintains a persistent pool across iterations to avoid per-iteration
        spawn overhead (~50-100ms per pool creation).

        Args:
            num_workers: Desired number of worker processes

        Returns:
            ProcessPoolExecutor ready for task submission
        """
        if self._pool is None or self._pool_workers != num_workers:
            self._cleanup_pool()
            self._pool = ProcessPoolExecutor(max_workers=num_workers)
            self._pool_workers = num_workers
            logger.debug(f"[PARALLEL] Created new process pool with {num_workers} workers")
        return self._pool

    def _cleanup_pool(self):
        """
        Shutdown existing pool if any.

        Called when worker count changes or when learner is done.
        Uses wait=False for faster shutdown since we don't need results.
        """
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False)
                logger.debug(f"[PARALLEL] Shut down process pool ({self._pool_workers} workers)")
            except Exception as e:
                logger.warning(f"[PARALLEL] Error shutting down pool: {e}")
            self._pool = None
            self._pool_workers = 0

    def __del__(self):
        """Destructor to ensure process pool cleanup on object destruction."""
        self._cleanup_pool()

    def _calculate_all_action_gains_parallel(self, grounded_actions: List, state: Set[str]) -> List[Tuple[str, List[str], float]]:
        """
        Parallel implementation of action gain computation.

        Uses persistent ProcessPoolExecutor to distribute gain computation across
        multiple workers. Each worker reconstructs its own CNFManagers from serialized state.

        Args:
            grounded_actions: List of GroundedAction objects
            state: Current state as set of fluent strings

        Returns:
            List of (action_name, objects, expected_gain) tuples, sorted by gain (highest first)
        """
        from src.algorithms.parallel_gain import (
            ActionGainTask, _compute_action_gains_chunk_with_context
        )

        # Determine worker count
        num_workers = self.num_workers
        if num_workers is None:
            num_workers = min(os.cpu_count() or 4, len(grounded_actions))
        num_workers = max(1, min(num_workers, len(grounded_actions)))

        logger.info(f"[PARALLEL] Action gain computation: {len(grounded_actions)} actions, {num_workers} workers")

        # Create context for workers
        context = self._create_parallel_context(state)

        # Create and chunk tasks (sort by action_name for better CNF cache reuse in workers)
        tasks = [ActionGainTask(ga.action_name, list(ga.objects)) for ga in grounded_actions]
        tasks.sort(key=lambda t: t.action_name)
        chunk_size = max(1, (len(tasks) + num_workers - 1) // num_workers)
        chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

        # Execute parallel computation using persistent pool
        action_gains = []
        try:
            executor = self._get_or_create_pool(num_workers)
            # Submit all chunks with context (context passed per-submission for persistence)
            futures = [
                executor.submit(_compute_action_gains_chunk_with_context, chunk, context)
                for chunk in chunks
            ]
            for future in futures:
                try:
                    results = future.result(timeout=120)  # 2 min timeout per chunk
                    for r in results:
                        if r.error:
                            logger.debug(f"[PARALLEL] Worker error for {r.action_name}: {r.error}")
                        action_gains.append((r.action_name, r.objects, r.expected_gain))
                except Exception as e:
                    logger.error(f"[PARALLEL] Chunk failed: {e}")
        except Exception as e:
            logger.error(f"[PARALLEL] Execution failed, falling back to sequential: {e}")
            self._cleanup_pool()  # Clean up on error
            return self._calculate_all_action_gains_sequential(grounded_actions, state)

        # Sort by gain (highest first)
        action_gains.sort(key=lambda x: x[2], reverse=True)

        # Filter/prioritize injective bindings based on action execution history
        # Actions without successful observations: filter out non-injective (strict)
        # Actions with successful observations: prioritize injective (lenient)
        filtered_gains = []
        for action_name, objects, gain in action_gains:
            is_injective = is_injective_binding(objects)
            has_success = self._has_successful_observation(action_name)

            if has_success:
                # Action already executed successfully - prioritize injective but allow non-injective
                filtered_gains.append((action_name, objects, gain, is_injective))
            else:
                # Action not yet executed - filter out non-injective (unless no alternative)
                if is_injective:
                    filtered_gains.append((action_name, objects, gain, is_injective))
                # Non-injective for unexecuted action: defer (handled in fallback below)

        # Separate by injectivity, preserving gain ordering within each group
        injective_actions = [(a, o, g) for a, o, g, inj in filtered_gains if inj]
        non_injective_actions = [(a, o, g) for a, o, g, inj in filtered_gains if not inj]

        # Fallback: For action types with NO injective bindings (unexecuted), add non-injective
        actions_with_injective = {a for a, _, _, inj in filtered_gains if inj}
        for action_name, objects, gain in action_gains:
            if not is_injective_binding(objects) and not self._has_successful_observation(action_name):
                # Check if this action type has any injective options
                if action_name not in actions_with_injective:
                    non_injective_actions.append((action_name, objects, gain))
                    # TODO: use ESAM effects logic on non-injective bindings

        if non_injective_actions:
            logger.debug(f"[PARALLEL] Injective binding filter: {len(injective_actions)} injective, "
                         f"{len(non_injective_actions)} non-injective actions")

        action_gains = injective_actions + non_injective_actions

        # Log top action after sorting
        if action_gains:
            top_action, top_objects, top_gain = action_gains[0]
            logger.debug(f"[PARALLEL] Top action after sorting: {top_action}({','.join(top_objects)}) with E[gain]={top_gain:.6f}")

        return action_gains

    def _create_parallel_context(self, state: Set[str]) -> 'ActionGainContext':
        """
        Create serializable context for parallel workers.

        Packages all necessary state for workers to compute gains independently.

        Args:
            state: Current state as set of fluent strings

        Returns:
            ActionGainContext with all necessary state for workers
        """
        from src.algorithms.parallel_gain import ActionGainContext

        cnf_clauses = {}
        cnf_fluent_to_var = {}
        cnf_var_to_fluent = {}
        cnf_next_var = {}
        cnf_solution_cache = {}
        parameter_bound_literals = {}
        base_model_counts = {}

        for action_name in self.pre.keys():
            cnf = self.cnf_managers[action_name]
            cnf_clauses[action_name] = [list(c) for c in cnf.cnf.clauses]
            cnf_fluent_to_var[action_name] = dict(cnf.fluent_to_var)
            # Convert int keys to str for JSON-safe serialization
            cnf_var_to_fluent[action_name] = {k: v for k, v in cnf.var_to_fluent.items()}
            cnf_next_var[action_name] = cnf.next_var
            # Copy solution cache ONLY if it's valid (invalidated by clause additions)
            if cnf._cache_valid and cnf._solution_cache is not None:
                cnf_solution_cache[action_name] = [set(s) for s in cnf._solution_cache]
            else:
                cnf_solution_cache[action_name] = None
            parameter_bound_literals[action_name] = set(self._get_parameter_bound_literals(action_name))
            # Pre-compute base model count (avoids recalculation in workers)
            base_model_counts[action_name] = self._get_base_model_count(action_name)

        return ActionGainContext(
            pre={k: set(v) for k, v in self.pre.items()},
            pre_constraints={k: set(v) for k, v in self.pre_constraints.items()},
            eff_add={k: set(v) for k, v in self.eff_add.items()},
            eff_del={k: set(v) for k, v in self.eff_del.items()},
            eff_maybe_add={k: set(v) for k, v in self.eff_maybe_add.items()},
            eff_maybe_del={k: set(v) for k, v in self.eff_maybe_del.items()},
            cnf_clauses=cnf_clauses,
            cnf_fluent_to_var=cnf_fluent_to_var,
            cnf_var_to_fluent=cnf_var_to_fluent,
            cnf_next_var=cnf_next_var,
            cnf_solution_cache=cnf_solution_cache,
            parameter_bound_literals=parameter_bound_literals,
            state=set(state),
            base_model_counts=base_model_counts,
            use_approximate_counting=self.use_approximate_counting,
            approximate_threshold_vars=self.approximate_threshold_vars,
            approximate_epsilon=self.approximate_epsilon,
            approximate_delta=self.approximate_delta
        )

    def _filter_applicable_actions(self, action_gains: List[Tuple[str, List[str], float]],
                                   state: Set[str]) -> List[Tuple[str, List[str]]]:
        """
        Find applicable actions when information gain is zero.

        An action is considered applicable if it has positive applicability probability
        based on learned preconditions.

        Args:
            action_gains: List of (action_name, objects, expected_gain) tuples
            state: Current state

        Returns:
            List of (action_name, objects) tuples for applicable actions
        """
        applicable = []
        for action_name, objects, _ in action_gains:
            prob = self._calculate_applicability_probability(action_name, objects, state)
            if prob > 0:  # Any positive applicability
                applicable.append((action_name, objects))

        logger.debug(f"Found {len(applicable)}/{len(action_gains)} applicable actions")
        return applicable

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

        # Session 2: Use incremental CNF updates instead of full rebuild
        # The updates are now done directly in _update_success() and _update_failure()
        # Only do full rebuild if CNF is empty but we have constraints (initial build case)
        cnf = self.cnf_managers[action]
        if not cnf.has_clauses() and self.pre_constraints[action]:
            logger.debug(f"Building initial CNF formula for '{action}'")
            self._build_cnf_formula(action)
            logger.debug(f"  CNF built: {len(cnf.cnf.clauses)} clauses")

        # Invalidate base CNF count cache after formula changes (Phase 1 enhancement)
        self._base_cnf_count_cache.pop(action, None)

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

        # Capture unsatisfied literals BEFORE intersection (for unit clause constraints)
        # These literals cannot be preconditions since action succeeded without them
        unsatisfied_literals = self.pre[action] - satisfied_in_state

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

        # Remove contradictions: if l is confirmed as add/delete, ¬l cannot be in maybe sets
        self._remove_contradictions(action, self.eff_add[action], self.eff_del[action])

        # Update constraint sets
        # pre?(a) = {B ∩ bindP(s, O) | B ∈ pre?(a)} ∪ {⋀{¬xl} | l ∈ (pre(a) ∩ bindP⁻¹(s, O))}
        constraints_before = len(self.pre_constraints[action])
        updated_constraints = set()
        for constraint in self.pre_constraints[action]:
            # Keep only literals from constraint that were satisfied
            updated = constraint.intersection(satisfied_in_state)
            if updated:  # Don't add empty constraints
                updated_constraints.add(frozenset(updated))

        # Add unit clauses for UNSATISFIED literals to exclude them from being preconditions
        # {⋀{¬xl} | l ∈ (pre(a) \ bindP⁻¹(s, O))}
        # Each unsatisfied literal gets a unit clause saying it cannot be a precondition
        # (action succeeded without them being satisfied, so they can't be required)
        for literal in unsatisfied_literals:
            # Add negated unit clause - if literal is "p(?x)", add "¬p(?x)"
            # If literal is "¬p(?x)", add "p(?x)" (double negation)
            if literal.startswith('¬'):
                negated = literal[1:]  # Remove negation
            else:
                negated = '¬' + literal  # Add negation
            unit_clause = frozenset({negated})
            updated_constraints.add(unit_clause)

        self.pre_constraints[action] = updated_constraints
        # Invalidate cache after modifying constraints
        self._invalidate_constraint_cache(action)
        logger.debug(
            f"  Constraints updated: {constraints_before} → {len(self.pre_constraints[action])}")

        # Session 2: Incremental CNF update - refine clauses in-place instead of full rebuild
        if self.cnf_managers[action].has_clauses():
            modified = self.cnf_managers[action].refine_clauses_by_intersection(satisfied_in_state)
            if modified > 0:
                logger.debug(f"  Refined {modified} CNF clauses directly (incremental update)")

        # Add unit clauses to CNF for UNSATISFIED literals (reduces model count)
        # These correspond to the {⋀{¬xl} | l ∈ (pre(a) \ bindP⁻¹(s, O))} constraint
        # Unsatisfied literals cannot be preconditions since action succeeded without them
        for literal in unsatisfied_literals:
            # Always prefix with '-' to say "literal is NOT a precondition"
            # CNF Manager treats 'p(?x)' and '¬p(?x)' as different variables
            # So '-p(?x)' means "p(?x) var = FALSE" and '-¬p(?x)' means "¬p(?x) var = FALSE"
            clause = ['-' + literal]
            self.cnf_managers[action].add_clause_with_subsumption(clause)

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

                # Session 2: Incremental CNF update - add clause directly instead of full rebuild
                # Failure constraint: "at least one of these IS a precondition"
                # Literals are variable names (¬ is part of the name, not CNF negation)
                clause = list(unsatisfied)
                if clause:
                    self.cnf_managers[action].add_clause_with_subsumption(clause)
                    logger.debug(f"  Added clause to CNF directly (incremental update)")
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

        Also adds mutual exclusion constraints: p and ¬p can't both be preconditions.
        This reduces model space from 4 to 3 states per fluent pair.

        Args:
            action: Action name

        Returns:
            CNF manager with the formula
        """
        logger.debug(f"_build_cnf_formula: Building CNF for action '{action}'")
        cnf = self.cnf_managers[action]

        # Use CNF manager method to build from constraint sets
        cnf.build_from_constraint_sets(self.pre_constraints[action])

        # Add mutual exclusion constraints: ¬(p ∧ ¬p) for each fluent pair
        # This means p and ¬p can't both be preconditions (contradiction)
        # CNF form: (¬X_p ∨ ¬X_¬p) = [-var_p, -var_¬p]
        self._add_mutual_exclusion_constraints(action, cnf)

        logger.info(f"CNF formula built for '{action}': {len(cnf.cnf.clauses)} clauses, "
                    f"{len(cnf.fluent_to_var)} unique variables")
        return cnf

    def _add_mutual_exclusion_constraints(self, action: str, cnf: 'CNFManager') -> None:
        """
        Add mutual exclusion constraints: p and ¬p can't both be preconditions.

        For each fluent f in the action's parameter-bound literals:
        - If both f and ¬f exist as variables, add clause [-var_f, -var_¬f]
        - This reduces model space from 4 to 3 states per fluent pair

        Args:
            action: Action name
            cnf: CNF manager to add constraints to
        """
        # Get all positive fluents (without ¬ prefix)
        positive_fluents = set()
        for fluent in cnf.fluent_to_var.keys():
            if fluent.startswith('¬'):
                positive_fluents.add(fluent[1:])
            else:
                positive_fluents.add(fluent)

        # For each positive fluent, check if both p and ¬p have variables
        mutex_count = 0
        for fluent in positive_fluents:
            neg_fluent = '¬' + fluent

            var_p = cnf.fluent_to_var.get(fluent)
            var_neg_p = cnf.fluent_to_var.get(neg_fluent)

            if var_p is not None and var_neg_p is not None:
                # Add mutual exclusion: ¬(p ∧ ¬p) = (¬p ∨ ¬¬p) in hypothesis space
                # CNF: [-var_p, -var_neg_p]
                cnf.add_var_clause([-var_p, -var_neg_p])
                mutex_count += 1

        if mutex_count > 0:
            logger.debug(f"Added {mutex_count} mutual exclusion constraints for {action}")

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
            # Perform count using adaptive method (exact for small, approximate for large)
            count = cnf.count_solutions_adaptive(
                threshold=self.approximate_threshold_vars,
                use_approximate=self.use_approximate_counting,
                epsilon=self.approximate_epsilon,
                delta=self.approximate_delta
            )

        # Cache the result
        self._base_cnf_count_cache[action] = count
        logger.debug(f"Cached base model count for {action}: {count}")
        return count

    def _has_lifted_learning_potential(self) -> bool:
        """
        Check if there's learning potential at the lifted level.

        Returns True if ANY action has:
        - Uncertain preconditions (possible but not certain)
        - Uncertain add effects (maybe_add that aren't confirmed)
        - Uncertain delete effects (maybe_del that aren't confirmed)

        This is independent of the current state/grounding and provides
        a true measure of whether the model is fully learned.

        Returns:
            True if there's still uncertainty to resolve
        """
        for action_name in self.pre.keys():
            # Check uncertain preconditions
            # Certain preconditions are singleton constraint sets
            certain_pre = self._get_certain_preconditions(action_name)
            uncertain_pre = self.pre[action_name] - certain_pre
            if uncertain_pre:
                logger.debug(f"Action {action_name} has {len(uncertain_pre)} uncertain preconditions")
                return True

            # Check uncertain add effects (maybe_add that aren't in confirmed add)
            uncertain_add = self.eff_maybe_add[action_name] - self.eff_add[action_name]
            if uncertain_add:
                logger.debug(f"Action {action_name} has {len(uncertain_add)} uncertain add effects")
                return True

            # Check uncertain delete effects (maybe_del that aren't in confirmed del)
            uncertain_del = self.eff_maybe_del[action_name] - self.eff_del[action_name]
            if uncertain_del:
                logger.debug(f"Action {action_name} has {len(uncertain_del)} uncertain delete effects")
                return True

        logger.info("No lifted-level learning potential - all actions fully learned")
        return False

    def _get_lifted_uncertainty_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Get summary of uncertainty at lifted level for all actions.

        Returns:
            Dict mapping action names to uncertainty counts
        """
        summary = {}
        for action_name in self.pre.keys():
            certain_pre = self._get_certain_preconditions(action_name)
            summary[action_name] = {
                'uncertain_pre': len(self.pre[action_name] - certain_pre),
                'uncertain_add': len(self.eff_maybe_add[action_name] - self.eff_add[action_name]),
                'uncertain_del': len(self.eff_maybe_del[action_name] - self.eff_del[action_name]),
            }
        return summary

    def get_action_model_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed learning metrics for each action showing what has been learned.

        For each action, computes:
        - Certain components (definitively in the model)
        - Excluded components (definitively NOT in the model)
        - Uncertain components (still unsure)

        This is a standard metric in action model learning papers.

        Returns:
            Dict[action_name -> metrics] where metrics contains counts and percentages
        """
        action_metrics = {}

        for action_name in self.pre.keys():
            # Get La: all possible parameter-bound literals for this action
            La = self._get_parameter_bound_literals(action_name)
            la_size = len(La)

            # === PRECONDITIONS ===
            # Certain preconditions: literals that MUST be preconditions
            # These are in the intersection of all satisfying models of the CNF
            # For practical tracking: literals that appear in all constraint sets
            # (In the absence of explicit model intersection, we use pre(a) as "possible")
            certain_pre = set()
            if self.pre_constraints[action_name]:
                # Literals that appear in ALL constraints are required preconditions
                # Convert frozensets to sets for intersection operation
                constraint_sets = [set(c) for c in self.pre_constraints[action_name]]
                certain_pre = set.intersection(*constraint_sets) if constraint_sets else set()

            # Excluded preconditions: literals that are definitely NOT preconditions
            # These are La \ pre(a) - ruled out by successful actions
            excluded_pre = La - self.pre[action_name]

            # Uncertain preconditions: literals that might be preconditions
            # These are pre(a) \ certain_pre
            uncertain_pre = self.pre[action_name] - certain_pre

            # === ADD EFFECTS ===
            # Certain add effects: eff_add[action] - confirmed by observations
            certain_eff_add = self.eff_add[action_name]

            # Excluded add effects: literals that are definitely NOT add effects
            # These are La \ (eff_add ∪ eff_maybe_add)
            excluded_eff_add = La - (self.eff_add[action_name] | self.eff_maybe_add[action_name])

            # Uncertain add effects: eff_maybe_add[action]
            uncertain_eff_add = self.eff_maybe_add[action_name]

            # === DELETE EFFECTS ===
            # Certain delete effects: eff_del[action] - confirmed by observations
            certain_eff_del = self.eff_del[action_name]

            # Excluded delete effects: literals that are definitely NOT delete effects
            # These are La \ (eff_del ∪ eff_maybe_del)
            excluded_eff_del = La - (self.eff_del[action_name] | self.eff_maybe_del[action_name])

            # Uncertain delete effects: eff_maybe_del[action]
            uncertain_eff_del = self.eff_maybe_del[action_name]

            # Compute metrics
            action_metrics[action_name] = {
                # Counts
                'La_size': la_size,
                'observations': len(self.observation_history[action_name]),

                # Preconditions
                'preconditions': {
                    'certain_count': len(certain_pre),
                    'excluded_count': len(excluded_pre),
                    'uncertain_count': len(uncertain_pre),
                    'certain_percent': (len(certain_pre) / la_size * 100) if la_size > 0 else 0,
                    'excluded_percent': (len(excluded_pre) / la_size * 100) if la_size > 0 else 0,
                    'uncertain_percent': (len(uncertain_pre) / la_size * 100) if la_size > 0 else 0,
                },

                # Add effects
                'add_effects': {
                    'certain_count': len(certain_eff_add),
                    'excluded_count': len(excluded_eff_add),
                    'uncertain_count': len(uncertain_eff_add),
                    'certain_percent': (len(certain_eff_add) / la_size * 100) if la_size > 0 else 0,
                    'excluded_percent': (len(excluded_eff_add) / la_size * 100) if la_size > 0 else 0,
                    'uncertain_percent': (len(uncertain_eff_add) / la_size * 100) if la_size > 0 else 0,
                },

                # Delete effects
                'delete_effects': {
                    'certain_count': len(certain_eff_del),
                    'excluded_count': len(excluded_eff_del),
                    'uncertain_count': len(uncertain_eff_del),
                    'certain_percent': (len(certain_eff_del) / la_size * 100) if la_size > 0 else 0,
                    'excluded_percent': (len(excluded_eff_del) / la_size * 100) if la_size > 0 else 0,
                    'uncertain_percent': (len(uncertain_eff_del) / la_size * 100) if la_size > 0 else 0,
                },

                # Overall learning progress
                'learning_progress': {
                    # Total certain knowledge (preconditions + effects)
                    'total_certain': len(certain_pre) + len(certain_eff_add) + len(certain_eff_del),
                    # Total excluded knowledge
                    'total_excluded': len(excluded_pre) + len(excluded_eff_add) + len(excluded_eff_del),
                    # Total uncertain
                    'total_uncertain': len(uncertain_pre) + len(uncertain_eff_add) + len(uncertain_eff_del),
                    # Percentage of model space explored (certain + excluded)
                    'explored_percent': ((len(certain_pre) + len(excluded_pre) +
                                        len(certain_eff_add) + len(excluded_eff_add) +
                                        len(certain_eff_del) + len(excluded_eff_del)) / (3 * la_size) * 100) if la_size > 0 else 0,
                }
            }

        return action_metrics

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get learning statistics including detailed action model metrics.

        Returns:
            Dictionary with learning statistics and per-action metrics
        """
        # Get base statistics
        stats = super().get_statistics()

        # Add information gain specific stats
        stats['max_information_gain'] = self._last_max_gain

        # Add detailed action model metrics
        stats['action_model_metrics'] = self.get_action_model_metrics()

        return stats

    def get_learned_model(self) -> Dict[str, Any]:
        """
        Export the current learned model.

        Returns:
            Dictionary containing the learned model
        """
        logger.debug("Exporting learned model")

        predicates_set = set()  # Use set for deduplication
        actions_dict = {}

        # Export learned action models
        for action_name in self.pre.keys():
            actions_dict[action_name] = {
                'name': action_name,
                'preconditions': {
                    'possible': sorted(list(self.pre[action_name])),
                    'constraints': [sorted(list(c)) for c in self.pre_constraints[action_name]]
                },
                'effects': {
                    'add': sorted(list(self.eff_add[action_name])),
                    'delete': sorted(list(self.eff_del[action_name])),
                    'maybe_add': sorted(list(self.eff_maybe_add[action_name])),
                    'maybe_delete': sorted(list(self.eff_maybe_del[action_name]))
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
                    predicates_set.add(pred_name)

        model = {
            'actions': actions_dict,
            'predicates': sorted(list(predicates_set)),  # Convert set to sorted list for JSON
            'statistics': self.get_statistics()
        }

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

    def _negate_literal(self, literal: str) -> str:
        """
        Get the negation of a literal.

        Args:
            literal: Literal string (e.g., 'on(?x,?y)' or '¬clear(?x)')

        Returns:
            Negated literal (e.g., '¬on(?x,?y)' or 'clear(?x)')
        """
        if literal.startswith('¬'):
            # Already negative, remove negation
            return literal[1:]
        else:
            # Positive, add negation
            return '¬' + literal

    def _remove_contradictions(self, action: str, confirmed_adds: Set[str], confirmed_dels: Set[str]) -> None:
        """
        Remove contradictory negative literals from maybe sets after confirming effects.

        When a literal l is confirmed as an add effect, ¬l cannot be an effect.
        When a literal l is confirmed as a delete effect, ¬l cannot be an effect.

        Args:
            action: Action name
            confirmed_adds: Literals confirmed as add effects
            confirmed_dels: Literals confirmed as delete effects
        """
        # For each confirmed add effect, remove its negation from both maybe sets
        for literal in confirmed_adds:
            negated = self._negate_literal(literal)
            self.eff_maybe_add[action].discard(negated)
            self.eff_maybe_del[action].discard(negated)

        # For each confirmed delete effect, remove its negation from both maybe sets
        for literal in confirmed_dels:
            negated = self._negate_literal(literal)
            self.eff_maybe_add[action].discard(negated)
            self.eff_maybe_del[action].discard(negated)

    def has_converged(self) -> bool:
        """
        Check if learning has converged.

        Convergence occurs when:
        1. Max iterations reached (forced convergence), OR
        2. No lifted-level learning potential (all actions fully learned)

        Note: Zero grounded information gain alone does NOT cause convergence.
        We check lifted-level uncertainty to avoid premature convergence.

        With object subset selection enabled:
        - Converge only when ALL objects exhausted AND no lifted-level potential
        - Subset rotation is handled separately in select_action()

        Returns:
            True if model has converged, False otherwise
        """
        # Check global iteration limit (forced convergence)
        if self.iteration_count >= self.max_iterations:
            logger.info(f"Convergence: Reached max iterations ({self.max_iterations})")
            self._converged = True
            return True

        # Need at least one observation to check convergence
        if self.observation_count == 0:
            return False

        # With object subset selection: check exhaustion first
        if self.use_object_subset and self.subset_manager:
            if not self.subset_manager.all_objects_exhausted():
                # Can still rotate - not converged
                return False

        # Check lifted-level learning potential (the true convergence criterion)
        if self._has_lifted_learning_potential():
            # Still have uncertainty to resolve
            logger.debug(
                f"Not converged: lifted-level uncertainty exists, "
                f"iterations={self.iteration_count}/{self.max_iterations}"
            )
            return False

        # No lifted-level uncertainty - truly converged
        if not self._converged:
            logger.info(
                f"Convergence: No lifted-level learning potential "
                f"(all actions fully learned at iteration {self.iteration_count})"
            )
        self._converged = True
        return True

    def _has_successful_observation(self, action_name: str) -> bool:
        """
        Check if action has been successfully executed at least once.

        Used to determine injective binding filtering strategy:
        - Actions without successful observations: filter non-injective (strict)
        - Actions with successful observations: prioritize injective (lenient)

        Args:
            action_name: Name of the lifted action

        Returns:
            True if at least one successful execution recorded
        """
        return any(obs['success'] for obs in self.observation_history.get(action_name, []))

    def _should_rotate_subset(self) -> bool:
        """
        Check if we should rotate to a new object subset.

        Rotation triggers:
        1. Max iterations per subset reached, OR
        2. Information gain converged to zero for current subset

        Returns:
            True if rotation should occur, False otherwise
        """
        if not self.use_object_subset or not self.subset_manager:
            return False

        if self.subset_manager.all_objects_exhausted():
            return False

        # Check iteration limit for this subset
        if self._subset_iteration_count >= self.max_iterations_per_subset:
            logger.info(
                f"Subset rotation: Iteration limit reached "
                f"({self._subset_iteration_count} >= {self.max_iterations_per_subset})"
            )
            return True

        # Check if model has converged on current subset (zero info gain)
        if self._last_max_gain < self.FLOAT_COMPARISON_EPSILON:
            logger.info(
                f"Subset rotation: Zero information gain "
                f"(max_gain={self._last_max_gain:.6f})"
            )
            return True

        return False

    def _rotate_to_new_subset(self) -> bool:
        """
        Rotate to a new object subset.

        Returns:
            True if rotation succeeded, False if all objects exhausted
        """
        if not self.subset_manager:
            return False

        if self.subset_manager.rotate_subset():
            self._subset_iteration_count = 0  # Reset subset iteration counter
            logger.info(f"[SUBSET] Rotated to new subset: {self.subset_manager.get_status()}")
            return True
        else:
            logger.info("[SUBSET] Cannot rotate - all objects exhausted")
            return False

    def export_model_snapshot(self, iteration: int, output_dir: Path) -> None:
        """
        Export model snapshot at checkpoint.

        Exports the current state of knowledge sets for all actions to a JSON file.
        This includes preconditions (certain/uncertain), effects (confirmed/possible),
        and constraint sets for post-processing analysis.

        Args:
            iteration: Current iteration number
            output_dir: Directory to export the model snapshot to
        """
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Extract domain and problem names from file paths
        domain_name = Path(self.domain_file).stem
        problem_name = Path(self.problem_file).stem

        snapshot = {
            "iteration": iteration,
            "algorithm": "information_gain",
            "actions": {},
            "metadata": {
                "domain": domain_name,
                "problem": problem_name,
                "export_timestamp": datetime.now().isoformat()
            }
        }

        # Export knowledge for each action
        for action_name in self.pre.keys():
            # Extract knowledge sets
            possible_precs = self._get_possible_preconditions(action_name)
            certain_precs = self._get_certain_preconditions(action_name)
            uncertain_precs = possible_precs - certain_precs

            # Get action parameters
            action = self.domain.lifted_actions.get(action_name)
            parameters = [p.name for p in action.parameters] if action else []

            snapshot["actions"][action_name] = {
                "parameters": parameters,
                "possible_preconditions": sorted(list(possible_precs)),
                "certain_preconditions": sorted(list(certain_precs)),
                "uncertain_preconditions": sorted(list(uncertain_precs)),
                "confirmed_add_effects": sorted(list(self.eff_add.get(action_name, set()))),
                "confirmed_del_effects": sorted(list(self.eff_del.get(action_name, set()))),
                "possible_add_effects": sorted(list(self.eff_maybe_add.get(action_name, set()))),
                "possible_del_effects": sorted(list(self.eff_maybe_del.get(action_name, set()))),
                "constraint_sets": [sorted(list(cs)) for cs in self.pre_constraints.get(action_name, set())]
            }

        # Write to file
        output_path = models_dir / f"model_iter_{iteration:03d}.json"
        with open(output_path, 'w') as f:
            json.dump(snapshot, f, indent=2)

        logger.debug(f"Exported model snapshot at iteration {iteration} to {output_path}")

    def _get_certain_preconditions(self, action_name: str) -> Set[str]:
        """
        Extract certain preconditions (singleton constraint sets).

        A precondition is certain if it appears as a singleton constraint set,
        meaning it must be true for the action to succeed.

        Args:
            action_name: Name of the action

        Returns:
            Set of certain precondition literals
        """
        certain = set()
        for constraint_set in self.pre_constraints.get(action_name, set()):
            if len(constraint_set) == 1:
                # Singleton constraint set means this literal is certain
                literal = next(iter(constraint_set))
                certain.add(literal)
        return certain

    def _get_possible_preconditions(self, action_name: str) -> Set[str]:
        """
        Get all possible preconditions (not yet ruled out).

        Args:
            action_name: Name of the action

        Returns:
            Set of possible precondition literals
        """
        return set(self.pre.get(action_name, set()))

    def reset(self) -> None:
        """Reset the learner to initial state."""
        logger.info("Resetting learner to initial state")

        # Clean up process pool to prevent orphaned workers
        self._cleanup_pool()

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
        self._last_max_gain = float('inf')

        # Reset object subset selection to original setting
        self.use_object_subset = self._original_use_object_subset
        self._subset_iteration_count = 0
        if self._original_use_object_subset and self.subset_manager:
            self.subset_manager.reset()

        logger.debug(f"Cleared {num_actions} action models and {num_observations} observations")

        # Reinitialize
        self._initialize_action_models()
        logger.info("Reset complete")
