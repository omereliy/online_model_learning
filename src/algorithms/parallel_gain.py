"""
Parallel action gain computation for Information Gain algorithm.

This module provides infrastructure for computing action gains in parallel using
ProcessPoolExecutor. It handles data serialization, CNFManager reconstruction,
and simplified gain computation functions that work without shared state.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, FrozenSet, Optional, Any

logger = logging.getLogger(__name__)


# ========== Data Classes ==========

@dataclass
class ActionGainContext:
    """
    Serializable context for parallel action gain computation.

    Contains all state needed by workers to compute gains without
    access to the original InformationGainLearner instance.
    """
    # Per-action type model state
    pre: Dict[str, Set[str]]
    pre_constraints: Dict[str, Set[FrozenSet[str]]]
    eff_add: Dict[str, Set[str]]
    eff_del: Dict[str, Set[str]]
    eff_maybe_add: Dict[str, Set[str]]
    eff_maybe_del: Dict[str, Set[str]]

    # CNF state for reconstruction (keyed by action_name)
    cnf_clauses: Dict[str, List[List[int]]]
    cnf_fluent_to_var: Dict[str, Dict[str, int]]
    cnf_var_to_fluent: Dict[str, Dict[int, str]]
    cnf_next_var: Dict[str, int]
    cnf_solution_cache: Dict[str, Optional[List[Set[str]]]]

    # Pre-computed domain info
    parameter_bound_literals: Dict[str, Set[str]]  # La per action

    # Current state
    state: Set[str]

    # Pre-computed base model counts (avoids recalculation in workers)
    base_model_counts: Dict[str, int] = field(default_factory=dict)

    # Approximate counting configuration
    use_approximate_counting: bool = True
    approximate_threshold_vars: int = 15
    approximate_epsilon: float = 0.3
    approximate_delta: float = 0.05

    # Cached constraint literals (computed on access)
    _constraint_literals_cache: Dict[str, Set[str]] = field(default_factory=dict)


@dataclass
class ActionGainTask:
    """Single action gain computation task."""
    action_name: str
    objects: List[str]


@dataclass
class ActionGainResult:
    """Result of action gain computation."""
    action_name: str
    objects: List[str]
    expected_gain: float
    error: Optional[str] = None


# ========== Worker State ==========

# Global worker cache (set during worker initialization)
_worker_cache: Optional['WorkerCNFCache'] = None


class WorkerCNFCache:
    """
    Per-worker cache of reconstructed CNFManagers.

    Created once per worker process via initializer function.
    Caches CNFManagers by action type to avoid repeated reconstruction
    for multiple groundings of the same lifted action.

    For persistent pools, use update_context() to refresh with new context
    while preserving CNFManagers that haven't changed.
    """

    def __init__(self, context: ActionGainContext):
        self.context = context
        self._managers: Dict[str, 'CNFManager'] = {}

    def update_context(self, context: ActionGainContext):
        """
        Update context for new iteration.

        Clears cached CNFManagers since they may be stale.
        Called when using persistent process pools.

        Args:
            context: Fresh ActionGainContext for this iteration
        """
        self.context = context
        # Clear cached managers - they may have stale CNF state
        # Future optimization: compare CNF versions and only clear changed ones
        self._managers.clear()

    def get_cnf(self, action_name: str) -> 'CNFManager':
        """Get or create CNFManager for action type."""
        if action_name not in self._managers:
            self._managers[action_name] = self._reconstruct(action_name)
        return self._managers[action_name]

    def _reconstruct(self, action_name: str) -> 'CNFManager':
        """Reconstruct CNFManager from serialized state."""
        from src.core.cnf_manager import CNFManager
        from pysat.formula import CNF

        mgr = CNFManager()

        # Restore variable mappings
        mgr.fluent_to_var = dict(self.context.cnf_fluent_to_var.get(action_name, {}))
        # Convert string keys back to int (JSON serialization converts int keys to strings)
        mgr.var_to_fluent = {
            int(k): v for k, v in self.context.cnf_var_to_fluent.get(action_name, {}).items()
        }
        mgr.next_var = self.context.cnf_next_var.get(action_name, 1)

        # Restore CNF clauses
        mgr.cnf = CNF()
        for clause in self.context.cnf_clauses.get(action_name, []):
            mgr.cnf.append(list(clause))

        # Restore solution cache for fast filtering (only if explicitly provided and valid)
        cached_solutions = self.context.cnf_solution_cache.get(action_name)
        if cached_solutions is not None:
            mgr._solution_cache = [set(s) for s in cached_solutions]
            mgr._cache_valid = True
        else:
            # Explicitly set invalid cache when no valid cache provided
            mgr._solution_cache = None
            mgr._cache_valid = False

        return mgr


# ========== Worker Functions ==========

def _worker_init(context: ActionGainContext):
    """
    Initialize worker process with shared context.

    Called once per worker when ProcessPoolExecutor starts.
    Sets up global _worker_cache with reconstructed CNFManagers.
    """
    global _worker_cache
    _worker_cache = WorkerCNFCache(context)


def _compute_action_gains_chunk(tasks: List[ActionGainTask]) -> List[ActionGainResult]:
    """
    Compute gains for a chunk of actions (main worker function).

    Requires worker to be initialized via _worker_init().

    Args:
        tasks: List of action gain tasks to compute

    Returns:
        List of results with computed gains (or errors)
    """
    global _worker_cache

    if _worker_cache is None:
        return [
            ActionGainResult(t.action_name, t.objects, 0.0, "Worker not initialized")
            for t in tasks
        ]

    results = []
    for task in tasks:
        try:
            gain = _compute_single_action_gain(
                task.action_name,
                task.objects,
                _worker_cache
            )
            results.append(ActionGainResult(
                action_name=task.action_name,
                objects=task.objects,
                expected_gain=gain
            ))
        except Exception as e:
            logger.debug(f"[PARALLEL] Error computing gain for {task.action_name}({task.objects}): {e}")
            results.append(ActionGainResult(
                action_name=task.action_name,
                objects=task.objects,
                expected_gain=0.0,
                error=str(e)
            ))

    return results


def _compute_action_gains_chunk_with_context(
    tasks: List[ActionGainTask],
    context: ActionGainContext
) -> List[ActionGainResult]:
    """
    Compute gains for a chunk of actions with fresh context.

    Used by persistent process pools where context changes between iterations.
    Updates worker cache with new context before processing.

    Args:
        tasks: List of action gain tasks to compute
        context: Fresh ActionGainContext for this iteration

    Returns:
        List of results with computed gains (or errors)
    """
    global _worker_cache

    # Update worker cache with fresh context
    if _worker_cache is None:
        _worker_cache = WorkerCNFCache(context)
    else:
        _worker_cache.update_context(context)

    results = []
    for task in tasks:
        try:
            gain = _compute_single_action_gain(
                task.action_name,
                task.objects,
                _worker_cache
            )
            results.append(ActionGainResult(
                action_name=task.action_name,
                objects=task.objects,
                expected_gain=gain
            ))
        except Exception as e:
            logger.debug(f"[PARALLEL] Error computing gain for {task.action_name}({task.objects}): {e}")
            results.append(ActionGainResult(
                action_name=task.action_name,
                objects=task.objects,
                expected_gain=0.0,
                error=str(e)
            ))

    return results


def _compute_single_action_gain(
    action_name: str,
    objects: List[str],
    cache: WorkerCNFCache
) -> float:
    """
    Compute expected information gain for a single grounded action.

    Simplified version of InformationGainLearner._calculate_expected_information_gain
    that works with serialized context.
    """
    ctx = cache.context

    # Early termination: Check if action is fully learned
    has_constraints = bool(ctx.pre_constraints.get(action_name))
    has_pre_uncertainty = len(ctx.pre.get(action_name, set())) > _get_certain_preconditions_count(action_name, ctx)
    has_add_uncertainty = bool(ctx.eff_maybe_add.get(action_name, set()) - ctx.eff_add.get(action_name, set()))
    has_del_uncertainty = bool(ctx.eff_maybe_del.get(action_name, set()) - ctx.eff_del.get(action_name, set()))

    if not has_constraints and not has_pre_uncertainty and not has_add_uncertainty and not has_del_uncertainty:
        return 0.0

    # Get CNFManager for this action type
    cnf = cache.get_cnf(action_name)

    # Calculate probability of success and gain components
    prob_success = _calculate_applicability_probability(action_name, objects, ctx, cnf)
    gain_success = _calculate_potential_gain_success(action_name, objects, ctx, cnf)

    # Check for early termination on failure gain
    state_internal = ctx.state.copy()
    satisfied = _get_satisfied_literals(action_name, state_internal, objects, ctx)
    unsatisfied = frozenset(ctx.pre.get(action_name, set()) - satisfied)

    if not unsatisfied or unsatisfied in ctx.pre_constraints.get(action_name, set()):
        gain_failure = 0.0
    else:
        gain_failure = _calculate_potential_gain_failure(action_name, objects, ctx, cnf, unsatisfied)

    expected_gain = prob_success * gain_success + (1.0 - prob_success) * gain_failure
    return expected_gain


# ========== Helper Functions ==========

def _get_certain_preconditions_count(action_name: str, ctx: ActionGainContext) -> int:
    """Count certain preconditions (singleton constraint sets)."""
    count = 0
    for constraint_set in ctx.pre_constraints.get(action_name, set()):
        if len(constraint_set) == 1:
            count += 1
    return count


def _get_all_constraint_literals(action_name: str, ctx: ActionGainContext) -> Set[str]:
    """Get all literals from all constraint sets (cached)."""
    if action_name not in ctx._constraint_literals_cache:
        all_literals = set()
        for constraint_set in ctx.pre_constraints.get(action_name, set()):
            all_literals.update(constraint_set)
        ctx._constraint_literals_cache[action_name] = all_literals
    return ctx._constraint_literals_cache[action_name]


def _get_satisfied_literals(
    action_name: str,
    state: Set[str],
    objects: List[str],
    ctx: ActionGainContext
) -> Set[str]:
    """
    Get all literals from pre(a) that are satisfied in the given state.

    Simplified version that uses ground_literal_set directly.
    """
    from src.core.grounding import ground_literal_set

    satisfied = set()
    pre_literals = ctx.pre.get(action_name, set())

    for literal in pre_literals:
        if literal.startswith('¬'):
            # Negative literal: satisfied if grounded version NOT in state
            positive_literal = literal[1:]
            grounded = ground_literal_set({positive_literal}, objects)
            if all(g not in state for g in grounded):
                satisfied.add(literal)
        else:
            # Positive literal: satisfied if grounded version IS in state
            grounded = ground_literal_set({literal}, objects)
            if all(g in state for g in grounded):
                satisfied.add(literal)

    return satisfied


def _calculate_applicability_probability(
    action_name: str,
    objects: List[str],
    ctx: ActionGainContext,
    cnf: 'CNFManager'
) -> float:
    """
    Calculate probability that action is applicable.

    pr(app(a,O,s)=1) = SAT(cnf_pre?(a,O,s)) / SAT(cnf_pre?(a))
    """
    from src.core.grounding import ground_literal_set

    # No constraints = always applicable
    if not ctx.pre_constraints.get(action_name):
        return 1.0

    # Empty CNF after building = no real constraints
    if not cnf.has_clauses():
        return 1.0

    # Get total models (base count)
    total_models = _get_base_model_count(action_name, ctx, cnf)
    if total_models == 0:
        return 0.0

    # Calculate state constraints for unsatisfied literals
    state_internal = ctx.state.copy()
    satisfied = _get_satisfied_literals(action_name, state_internal, objects, ctx)
    all_constraint_literals = _get_all_constraint_literals(action_name, ctx)
    unsatisfied = all_constraint_literals - satisfied

    # Build state constraints dict for CNF manager
    # For action to be applicable, unsatisfied literals must NOT be preconditions
    # (If an unsatisfied literal IS a precondition, action is inapplicable)
    # Literals are variable names (¬ is part of name, not CNF negation)
    state_constraints = {}
    for literal in unsatisfied:
        state_constraints[literal] = False  # var_literal = FALSE (not a precondition)

    # Count models with assumptions
    assumptions = cnf.state_constraints_to_assumptions(state_constraints)
    state_models = cnf.count_models_with_assumptions(assumptions)

    return state_models / total_models if total_models > 0 else 0.0


def _get_base_model_count(action_name: str, ctx: ActionGainContext, cnf: 'CNFManager') -> int:
    """
    Get base model count for action, using adaptive counting for complex formulas.

    Uses pre-computed base model counts from context if available,
    otherwise falls back to adaptive counting.
    """
    # Use pre-computed count if available (avoids expensive recalculation)
    if ctx.base_model_counts and action_name in ctx.base_model_counts:
        return ctx.base_model_counts[action_name]

    # Fallback: calculate (for backwards compatibility with older contexts)
    if not cnf.has_clauses():
        la_size = len(ctx.parameter_bound_literals.get(action_name, set()))
        return 2 ** la_size if la_size > 0 else 1

    # Use adaptive counting based on configuration
    num_vars = len(cnf.fluent_to_var)
    if num_vars > ctx.approximate_threshold_vars and ctx.use_approximate_counting:
        try:
            return cnf.count_solutions_approximate(ctx.approximate_epsilon, ctx.approximate_delta)
        except ImportError:
            # pyapproxmc not available - use upper bound estimate
            logger.warning(f"[APPROX] pyapproxmc not available, using upper bound 2^{num_vars}")
            return 2 ** num_vars

    return cnf.count_solutions()


def _calculate_potential_gain_success(
    action_name: str,
    objects: List[str],
    ctx: ActionGainContext,
    cnf: 'CNFManager' = None
) -> float:
    """
    Calculate potential information gain from successful execution.

    preAppPotential(a, O, s) = (|SAT(cnf_pre?(a))| - |SAT(cnf'_pre?(a))|) / 3^|fluents|
    sucPotential = preAppPotential + effPotential
    """
    from src.core.grounding import ground_literal_set

    state_internal = ctx.state.copy()
    satisfied = _get_satisfied_literals(action_name, state_internal, objects, ctx)

    # Calculate normalization factor: 3^|fluents| where |fluents| = |La| / 2
    la_size = len(ctx.parameter_bound_literals.get(action_name, set()))
    num_fluents = la_size // 2  # La contains both p and ¬p for each fluent
    total_hypotheses = 3 ** num_fluents if num_fluents > 0 else 1

    # Calculate precondition knowledge gain using SAT count difference
    # preAppPotential(a, O, s) = (|SAT(cnf_pre?(a))| - |SAT(cnf'_pre?(a))|) / 3^|fluents|
    if not ctx.pre_constraints.get(action_name):
        # No CNF constraints yet - no precondition gain from success
        pre_gain = 0
    else:
        # Get current SAT count
        current_models = _get_base_model_count(action_name, ctx, cnf)

        # Simulate adding unit clauses for satisfied literals
        # Each satisfied literal l was satisfied in state, so l is NOT required as a precondition
        # Assumption: var_l = FALSE for each satisfied literal
        # Literals are variable names (¬ is part of name, not CNF negation)
        unit_clause_assumptions = []
        pre_literals = ctx.pre.get(action_name, set())
        for literal in satisfied:
            if literal in pre_literals:
                # Use get_variable() to avoid creating new variables as side effect
                var_id = cnf.get_variable(literal) if cnf else None
                if var_id is not None:
                    unit_clause_assumptions.append(-var_id)  # literal is NOT a precondition

        # Count models after adding unit clause assumptions
        if unit_clause_assumptions and cnf and cnf.has_clauses():
            new_models = cnf.count_models_with_assumptions(unit_clause_assumptions)
        else:
            new_models = current_models

        # Calculate precondition gain as model reduction
        pre_gain = current_models - new_models

    # Get lifted versions of state fluents
    # We need bindP (lift_fluent_set), but we don't have domain in workers
    # Use a simplified approach: check which maybe effects overlap with current state

    # For eff+: count fluents in eff_maybe_add that are NOT in state
    # (after success, we know they're not add effects)
    eff_maybe_add = ctx.eff_maybe_add.get(action_name, set())
    eff_maybe_del = ctx.eff_maybe_del.get(action_name, set())

    # Ground the maybe effects and check against state
    grounded_maybe_add = ground_literal_set(eff_maybe_add, objects)
    grounded_maybe_del = ground_literal_set(eff_maybe_del, objects)

    # Add effects we can rule out (grounded version not in state after action)
    # Count grounded maybe_add fluents that aren't in state
    eff_add_gain = sum(1 for g in grounded_maybe_add if g not in state_internal)

    # Delete effects we can confirm (grounded version IS in state)
    # Count grounded maybe_del fluents that are in state
    eff_del_gain = sum(1 for g in grounded_maybe_del if g in state_internal)

    if total_hypotheses == 0:
        return 0.0

    return (pre_gain + eff_add_gain + eff_del_gain) / total_hypotheses


def _calculate_potential_gain_failure(
    action_name: str,
    objects: List[str],
    ctx: ActionGainContext,
    cnf: 'CNFManager',
    unsatisfied: FrozenSet[str]
) -> float:
    """
    Calculate potential information gain from failed execution.

    preFailPotential = 1 - (SAT(cnf) - SAT(cnf')) / 2^|La|
    """
    # No constraints = maximum information gain from failure
    if not ctx.pre_constraints.get(action_name):
        return 1.0

    # Get current model count
    current_models = _get_base_model_count(action_name, ctx, cnf)

    # Count models with temporary constraint
    new_models = cnf.count_models_with_temporary_clause(unsatisfied)

    # Calculate normalized gain
    la_size = len(ctx.parameter_bound_literals.get(action_name, set()))
    num_fluents = la_size // 2  # La contains both p and not-p
    total_hypotheses = 3 ** num_fluents if num_fluents > 0 else 1

    model_reduction = current_models - new_models
    normalized_gain = model_reduction / total_hypotheses

    return max(0.0, min(1.0, normalized_gain))
