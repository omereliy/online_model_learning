# CNF Handling Fix Strategy

## Document Information
- **Created**: December 24, 2025 (Updated: December 25, 2025)
- **Companion Document**: `cnf_issues_analysis.md`
- **Scope**: Remediation plan for all identified CNF handling issues

---

## Table of Contents
1. [Important Clarification: Variable Counting](#important-clarification-variable-counting)
2. [Fix Priority Matrix](#fix-priority-matrix)
3. [Phase 1: Core Architecture Fixes](#phase-1-core-architecture-fixes)
4. [Phase 2: Model Counting Fixes](#phase-2-model-counting-fixes)
5. [Phase 3: Algorithm Integration Fixes](#phase-3-algorithm-integration-fixes)
6. [Phase 4: Performance & Polish](#phase-4-performance--polish)
7. [Implementation Details](#implementation-details)
8. [Testing Strategy](#testing-strategy)
9. [Migration Guide](#migration-guide)

---

## Important Clarification: Variable Counting

### All Hypothesis Variables Must Be Counted

For the information gain algorithm, **both** `l_var` (positive literal) **and** `neg_l_var` (negative literal) represent independent hypothesis choices. They are NOT primary/auxiliary - they are ALL hypothesis variables.

| l_var (is `f` precondition?) | neg_l_var (is `¬f` precondition?) | Meaning |
|:---:|:---:|:---|
| T | F | `f` is a precondition |
| F | T | `¬f` is a precondition |
| F | F | Neither is a precondition |
| T | T | ❌ Invalid (mutex prevents) |

**Key insights:**
- You cannot derive `neg_l_var`'s state from `l_var` alone
- The mutex only eliminates the (T,T) state
- **3 valid states** per fluent pair
- **No projection needed** - count ALL satisfying assignments
- **Correct unconstrained count** = 3^n (for n fluent pairs with mutex)

### When Would Projection Be Needed?

Projection is only needed for **encoding auxiliary variables** (e.g., Tseitin transformation, cardinality encodings). The current algorithm doesn't use such encodings, so no projection is required.

---

## Fix Priority Matrix

| Priority | Issue ID | Description | Effort | Impact |
|----------|----------|-------------|--------|--------|
| P0 | CRIT-01 | Literal pair model with mutex | High | Critical |
| P0 | CRIT-02 | Mutex at initialization | Medium | Critical |
| P0 | CRIT-03 | Validate model counting invariants | Low | Critical |
| P0 | CRIT-04 | Remove `refine_clauses_by_intersection` | Low | Critical |
| P1 | CRIT-05 | Fix success update semantics | Medium | High |
| P1 | CRIT-06 | Fix parallel cache | Medium | High |
| P1 | CRIT-07 | Clarify negation handling | Medium | High |
| P2 | MOD-01 | Add symmetry breaking | High | Medium |
| P2 | MOD-02 | Fix approximate counting API | Low | Medium |
| P2 | MOD-03 | Use native assumptions | Low | Medium |
| P3 | MOD-04-08 | Cache and performance | Medium | Low |
| P3 | MIN-01-06 | Minor improvements | Low | Low |

---

## Phase 1: Core Architecture Fixes

### 1.1 Redesign CNFManager with Literal Pairs and Automatic Mutex

**Goal**: Implement proper literal pair model where mutex is ALWAYS added at pair creation.

#### New Data Model

```python
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple, FrozenSet
from pysat.solvers import Glucose4
from pysat.formula import CNF


@dataclass
class LiteralPair:
    """
    Represents a fluent with its positive and negative precondition variables.
    
    For fluent f:
    - l_var: "is f a precondition?"
    - neg_l_var: "is ¬f a precondition?"
    
    BOTH are hypothesis variables - BOTH must be counted.
    
    Valid states: (F,F), (T,F), (F,T) - exactly 3 states
    Invalid state: (T,T) - prevented by mutex clause
    """
    fluent_name: str       # Base fluent name (e.g., 'clear(?x)')
    l_var: int             # Variable for positive literal - COUNTED
    neg_l_var: int         # Variable for negative literal - ALSO COUNTED
    mutex_clause: Tuple[int, int]  # [-l_var, -neg_l_var]
    
    @property
    def positive_name(self) -> str:
        return self.fluent_name
    
    @property
    def negative_name(self) -> str:
        return f'¬{self.fluent_name}'


class CNFManagerV2:
    """
    CNF Manager with proper literal pair model for model counting.
    
    Key design principles:
    1. Each fluent creates a PAIR of variables (l, neg_l)
    2. Mutex constraints added AUTOMATICALLY at pair creation
    3. ALL variables are hypothesis variables - no projection needed
    4. Clear separation of CNF negation (-) and literal negation (¬)
    5. With n fluent pairs and no other constraints: count = 3^n
    """
    
    def __init__(self):
        # Variable management
        self._next_var: int = 1
        self._pairs: Dict[str, LiteralPair] = {}  # fluent_name -> pair
        
        # Variable lookups
        self._var_to_pair: Dict[int, LiteralPair] = {}
        self._literal_to_var: Dict[str, int] = {}  # 'f' or '¬f' -> var
        
        # CNF formula
        self._cnf = CNF()
        self._mutex_clauses: List[List[int]] = []
        self._constraint_clauses: List[List[int]] = []
        self._unit_clauses: List[List[int]] = []
        
        # Solver management
        self._solver: Optional[Glucose4] = None
        self._solver_valid: bool = False
        
        # Cache
        self._cache_valid: bool = False
        self._solution_cache: Optional[List[Tuple[int, ...]]] = None
```

#### Literal Pair Creation with Automatic Mutex

```python
def create_pair(self, fluent_name: str) -> LiteralPair:
    """
    Create a literal pair for a fluent with AUTOMATIC mutex.
    
    This is the ONLY way to create variables - ensures mutex invariant.
    
    Args:
        fluent_name: Base fluent name WITHOUT negation (e.g., 'clear(?x)')
    
    Returns:
        LiteralPair with both variables and mutex clause
    
    Raises:
        ValueError: If fluent_name starts with '¬' (use base name)
    """
    if fluent_name.startswith('¬'):
        raise ValueError(
            f"Use base fluent name without ¬: got '{fluent_name}', "
            f"use '{fluent_name[1:]}'"
        )
    
    if fluent_name in self._pairs:
        return self._pairs[fluent_name]
    
    # Allocate two consecutive variables
    l_var = self._next_var
    neg_l_var = self._next_var + 1
    self._next_var += 2
    
    # Create mutex clause - THIS IS CRITICAL
    mutex = [-l_var, -neg_l_var]
    
    # Create pair
    pair = LiteralPair(
        fluent_name=fluent_name,
        l_var=l_var,
        neg_l_var=neg_l_var,
        mutex_clause=(mutex[0], mutex[1])
    )
    
    # Register in lookups
    self._pairs[fluent_name] = pair
    self._var_to_pair[l_var] = pair
    self._var_to_pair[neg_l_var] = pair
    
    self._literal_to_var[fluent_name] = l_var
    self._literal_to_var[f'¬{fluent_name}'] = neg_l_var
    
    # Add mutex to CNF IMMEDIATELY (invariant: mutex always present)
    self._mutex_clauses.append(mutex)
    self._cnf.append(mutex)
    
    self._invalidate_cache()
    
    return pair


def get_var(self, literal: str) -> Optional[int]:
    """
    Get variable ID for a literal.
    
    Args:
        literal: Literal string, e.g., 'clear(?x)' or '¬clear(?x)'
    
    Returns:
        Variable ID or None if not registered
    """
    return self._literal_to_var.get(literal)


def get_or_create_var(self, literal: str) -> int:
    """
    Get or create variable for a literal.
    
    Args:
        literal: Literal string, e.g., 'clear(?x)' or '¬clear(?x)'
    
    Returns:
        Variable ID (creates pair with mutex if needed)
    """
    if literal in self._literal_to_var:
        return self._literal_to_var[literal]
    
    # Extract base fluent name
    if literal.startswith('¬'):
        base_fluent = literal[1:]
    else:
        base_fluent = literal
    
    # Create pair (includes mutex)
    pair = self.create_pair(base_fluent)
    
    # Return appropriate variable
    if literal.startswith('¬'):
        return pair.neg_l_var
    else:
        return pair.l_var
```

### 1.2 Clause Management with Clear Semantics

```python
def add_coverage_clause(self, literals: List[str]) -> None:
    """
    Add coverage clause: "at least one of these IS a precondition".
    
    This is a disjunction over precondition variables.
    
    Args:
        literals: List of literals (e.g., ['on(?x,?y)', '¬clear(?x)'])
    
    Example:
        add_coverage_clause(['on(?x,?y)', 'clear(?x)'])
        # Adds clause: (var_on ∨ var_clear)
        # Meaning: on(?x,?y) or clear(?x) must be a precondition
    """
    clause = []
    for literal in literals:
        var_id = self.get_or_create_var(literal)
        clause.append(var_id)  # Positive: var = TRUE means IS a precondition
    
    if clause:
        self._constraint_clauses.append(clause)
        self._cnf.append(clause)
        self._invalidate_cache()


def add_exclusion(self, literal: str) -> None:
    """
    Add exclusion: "this literal is NOT a precondition".
    
    Adds unit clause with NEGATIVE variable.
    
    Args:
        literal: Literal to exclude (e.g., 'clear(?x)' or '¬clear(?x)')
    
    Example:
        add_exclusion('clear(?x)')
        # Adds clause: (¬var_clear)  i.e., [-var_clear]
        # Meaning: clear(?x) is NOT a precondition
    """
    var_id = self.get_or_create_var(literal)
    unit = [-var_id]  # Negative: var = FALSE means NOT a precondition
    
    self._unit_clauses.append(unit)
    self._cnf.append(unit)
    self._invalidate_cache()


def add_inclusion(self, literal: str) -> None:
    """
    Add inclusion: "this literal IS a precondition".
    
    Adds unit clause with POSITIVE variable.
    
    Args:
        literal: Literal to include (e.g., 'clear(?x)')
    
    Example:
        add_inclusion('clear(?x)')
        # Adds clause: (var_clear)
        # Meaning: clear(?x) IS a precondition
    """
    var_id = self.get_or_create_var(literal)
    unit = [var_id]  # Positive: var = TRUE means IS a precondition
    
    self._unit_clauses.append(unit)
    self._cnf.append(unit)
    self._invalidate_cache()
```

---

## Phase 2: Model Counting Fixes

### 2.1 Count All Variables (No Projection)

Since ALL variables are hypothesis variables, we count ALL satisfying assignments.

```python
def count_models(self) -> int:
    """
    Count ALL satisfying assignments.
    
    All variables are hypothesis variables - no projection needed.
    With n fluent pairs and mutex constraints only: returns 3^n
    
    Returns:
        Number of satisfying assignments
    """
    if not self._cnf.clauses:
        # No clauses = all assignments valid
        # But we should have mutex clauses for each pair
        return 3 ** len(self._pairs) if self._pairs else 1
    
    solver = Glucose4(bootstrap_with=self._cnf)
    try:
        count = 0
        
        while solver.solve():
            count += 1
            model = solver.get_model()
            
            # Block this complete assignment
            blocking = [-lit for lit in model]
            solver.add_clause(blocking)
        
        return count
    finally:
        solver.delete()


def count_models_with_assumptions(self, assumptions: List[int]) -> int:
    """
    Count models with temporary assumptions.
    
    Uses solver's native assumption mechanism (no CNF modification).
    
    Args:
        assumptions: List of signed variable IDs
                    Positive = must be TRUE, Negative = must be FALSE
    
    Returns:
        Model count under assumptions
    """
    solver = Glucose4(bootstrap_with=self._cnf)
    try:
        count = 0
        
        while solver.solve(assumptions=assumptions):
            count += 1
            model = solver.get_model()
            
            # Block this assignment
            blocking = [-lit for lit in model]
            solver.add_clause(blocking)
        
        return count
    finally:
        solver.delete()
```

### 2.2 Approximate Counting

```python
def count_models_approximate(self) -> int:
    """
    Approximate model count using ApproxMC.
    
    Returns:
        Approximate model count
    """
    try:
        import pyapproxmc
    except ImportError:
        raise ImportError(
            "pyapproxmc required for approximate counting. "
            "Install with: pip install pyapproxmc"
        )
    
    counter = pyapproxmc.Counter()
    
    # Add all clauses
    for clause in self._cnf.clauses:
        counter.add_clause(clause)
    
    # Count all variables (no projection/sampling set needed)
    count = counter.count()
    
    return max(1, count)


def count_models_adaptive(self, threshold: int = 15) -> int:
    """
    Choose exact or approximate counting based on problem size.
    
    Args:
        threshold: Use approximate if num_vars > threshold
    
    Returns:
        Model count (exact or approximate)
    """
    num_vars = self._next_var - 1
    
    if num_vars <= threshold:
        return self.count_models()
    else:
        try:
            return self.count_models_approximate()
        except ImportError:
            # Fallback to exact (may be slow)
            return self.count_models()
```

### 2.3 Invariant Validation

```python
def validate_mutex_invariant(self) -> bool:
    """
    Verify that all fluent pairs have their mutex clause in the CNF.
    
    This should ALWAYS be true if create_pair is used correctly.
    
    Returns:
        True if all mutex clauses are present
    """
    for fluent_name, pair in self._pairs.items():
        expected_mutex = sorted([-pair.l_var, -pair.neg_l_var])
        
        # Check if mutex exists in CNF
        found = False
        for clause in self._cnf.clauses:
            if len(clause) == 2 and sorted(clause) == expected_mutex:
                found = True
                break
        
        if not found:
            return False
    
    return True


def expected_unconstrained_count(self) -> int:
    """
    Return the expected model count with only mutex constraints.
    
    With n fluent pairs, should be 3^n.
    """
    return 3 ** len(self._pairs)
```

---

## Phase 3: Algorithm Integration Fixes

### 3.1 Fix Information Gain Learner Integration

```python
# In information_gain.py

class InformationGainLearner(BaseActionModelLearner):
    
    def _initialize_action_models(self):
        """Initialize action model state variables for all actions."""
        
        for action_name, action in self.domain.lifted_actions.items():
            # Get La: all parameter-bound literals for this action
            La = self._get_parameter_bound_literals(action_name)
            
            # Initialize state variables
            self.pre[action_name] = La.copy()
            self.pre_constraints[action_name] = set()
            self.eff_add[action_name] = set()
            self.eff_del[action_name] = set()
            self.eff_maybe_add[action_name] = La.copy()
            self.eff_maybe_del[action_name] = La.copy()
            
            # Initialize CNF manager with literal pairs
            cnf = CNFManagerV2()
            
            # Create pairs for ALL possible preconditions
            # This ensures mutex constraints are in place from the start
            for literal in La:
                if literal.startswith('¬'):
                    base = literal[1:]
                else:
                    base = literal
                cnf.create_pair(base)  # Creates both vars + mutex
            
            self.cnf_managers[action_name] = cnf
            
            # Verify invariant
            assert cnf.validate_mutex_invariant(), \
                f"Mutex invariant violated for action {action_name}"


def _update_success(self, action: str, objects: List[str],
                    state: Set[str], next_state: Set[str]) -> None:
    """
    Update model after successful action execution.
    
    Key changes from original:
    1. Use add_exclusion() for unsatisfied literals
    2. Don't modify constraint sets incorrectly
    3. No clause refinement (add exclusions instead)
    """
    state_internal = self._state_to_internal(state)
    next_state_internal = self._state_to_internal(next_state)
    
    # Get satisfied literals
    satisfied = self._get_satisfied_literals(action, state_internal, objects)
    unsatisfied = self.pre[action] - satisfied
    
    # Update possible preconditions
    self.pre[action] = self.pre[action].intersection(satisfied)
    
    # Add EXCLUSION constraints for unsatisfied literals
    # "literal was not satisfied, but action succeeded, so literal is NOT a precondition"
    cnf = self.cnf_managers[action]
    for literal in unsatisfied:
        cnf.add_exclusion(literal)  # Adds [-var] to CNF
    
    # Update constraint sets: refine by intersection
    updated_constraints = set()
    for constraint in self.pre_constraints[action]:
        refined = constraint.intersection(satisfied)
        if refined:
            updated_constraints.add(frozenset(refined))
    self.pre_constraints[action] = updated_constraints
    
    # Invalidate caches
    self._base_cnf_count_cache.pop(action, None)


def _update_failure(self, action: str, objects: List[str], state: Set[str]) -> None:
    """
    Update model after failed action execution.
    
    Key change: Use add_coverage_clause() for failure constraint.
    """
    state_internal = self._state_to_internal(state)
    satisfied = self._get_satisfied_literals(action, state_internal, objects)
    unsatisfied = frozenset(self.pre[action] - satisfied)
    
    if not unsatisfied:
        logger.warning("Failed action had all preconditions satisfied")
        return
    
    # Check if constraint already exists
    if unsatisfied in self.pre_constraints[action]:
        return
    
    # Add to constraint sets
    self.pre_constraints[action].add(unsatisfied)
    
    # Add COVERAGE clause to CNF
    # "At least one unsatisfied literal IS a precondition"
    cnf = self.cnf_managers[action]
    cnf.add_coverage_clause(list(unsatisfied))
    
    # Invalidate caches
    self._base_cnf_count_cache.pop(action, None)
```

### 3.2 Fix Parallel Gain Computation

```python
# In parallel_gain.py

@dataclass
class ActionGainContextV2:
    """
    Serializable context for parallel workers.
    
    Key changes:
    1. No cache transfer (workers always recompute)
    2. Include mutex clauses for verification
    """
    # Model state
    pre: Dict[str, Set[str]]
    pre_constraints: Dict[str, Set[FrozenSet[str]]]
    
    # CNF state (minimal for reconstruction)
    cnf_clauses: Dict[str, List[List[int]]]
    literal_to_var: Dict[str, Dict[str, int]]
    mutex_clauses: Dict[str, List[List[int]]]
    pairs: Dict[str, List[Tuple[str, int, int]]]  # [(fluent, l_var, neg_l_var), ...]
    
    # Pre-computed
    base_model_counts: Dict[str, int]
    parameter_bound_literals: Dict[str, Set[str]]
    
    # Current state
    state: Set[str]
    
    # NO solution cache - workers always recompute


class WorkerCNFCacheV2:
    """Per-worker cache with safe reconstruction."""
    
    def __init__(self, context: ActionGainContextV2):
        self.context = context
        self._managers: Dict[str, CNFManagerV2] = {}
    
    def get_cnf(self, action_name: str) -> CNFManagerV2:
        if action_name not in self._managers:
            self._managers[action_name] = self._reconstruct(action_name)
        return self._managers[action_name]
    
    def _reconstruct(self, action_name: str) -> CNFManagerV2:
        """Reconstruct CNFManager WITHOUT cache."""
        mgr = CNFManagerV2()
        
        # Restore pairs (this will add mutex automatically)
        pairs_data = self.context.pairs.get(action_name, [])
        for fluent_name, l_var, neg_l_var in pairs_data:
            # We need to recreate with same variable IDs
            # This requires internal manipulation
            pair = LiteralPair(
                fluent_name=fluent_name,
                l_var=l_var,
                neg_l_var=neg_l_var,
                mutex_clause=(-l_var, -neg_l_var)
            )
            mgr._pairs[fluent_name] = pair
            mgr._var_to_pair[l_var] = pair
            mgr._var_to_pair[neg_l_var] = pair
            mgr._literal_to_var[fluent_name] = l_var
            mgr._literal_to_var[f'¬{fluent_name}'] = neg_l_var
            mgr._next_var = max(mgr._next_var, neg_l_var + 1)
        
        # Restore clauses
        for clause in self.context.cnf_clauses.get(action_name, []):
            mgr._cnf.append(list(clause))
        
        for mutex in self.context.mutex_clauses.get(action_name, []):
            mgr._mutex_clauses.append(list(mutex))
        
        # Verify invariant
        assert mgr.validate_mutex_invariant(), \
            f"Mutex invariant violated during reconstruction for {action_name}"
        
        return mgr
```

---

## Phase 4: Performance & Polish

### 4.1 Symmetry Breaking

```python
def add_symmetry_breakers(self, symmetric_groups: List[List[str]]) -> None:
    """
    Add lexicographic ordering constraints for symmetric literals.
    
    For symmetric entities, we want to count only canonical representatives.
    If literals {l1, l2, l3} are interchangeable, add constraints:
    - l2 → l1 (if l2 is precondition, l1 must be too)
    - l3 → l2 (if l3 is precondition, l2 must be too)
    
    This reduces model count by n! for each symmetric group.
    
    Args:
        symmetric_groups: List of groups, each group is symmetric literals
    """
    for group in symmetric_groups:
        if len(group) <= 1:
            continue
        
        # Sort for canonical ordering
        sorted_group = sorted(group)
        
        # Add ordering constraints
        for i in range(len(sorted_group) - 1):
            lit_i = sorted_group[i]
            lit_j = sorted_group[i + 1]
            
            var_i = self.get_or_create_var(lit_i)
            var_j = self.get_or_create_var(lit_j)
            
            # (l_j → l_i) in CNF: (¬l_j ∨ l_i)
            self._cnf.append([-var_j, var_i])
```

### 4.2 DIMACS Export

```python
def to_dimacs(self) -> str:
    """
    Export CNF in DIMACS format.
    
    Returns:
        DIMACS format string
    """
    lines = []
    
    # Header
    num_vars = self._next_var - 1
    num_clauses = len(self._cnf.clauses)
    lines.append(f"p cnf {num_vars} {num_clauses}")
    
    # Comments for structure
    lines.append("c === MUTEX CLAUSES ===")
    for clause in self._mutex_clauses:
        lines.append(' '.join(str(lit) for lit in clause) + ' 0')
    
    lines.append("c === UNIT CLAUSES ===")
    for clause in self._unit_clauses:
        lines.append(' '.join(str(lit) for lit in clause) + ' 0')
    
    lines.append("c === CONSTRAINT CLAUSES ===")
    for clause in self._constraint_clauses:
        lines.append(' '.join(str(lit) for lit in clause) + ' 0')
    
    return '\n'.join(lines)
```

### 4.3 Validation and Invariant Checking

```python
def validate(self) -> Tuple[bool, List[str]]:
    """
    Validate CNF structure and invariants.
    
    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    
    # INV-1: Every pair has a mutex clause in CNF
    for fluent, pair in self._pairs.items():
        mutex = sorted([-pair.l_var, -pair.neg_l_var])
        found = any(sorted(c) == mutex for c in self._cnf.clauses)
        if not found:
            errors.append(f"Missing mutex for '{fluent}'")
    
    # INV-2: No conflicting unit clauses
    unit_pos = {c[0] for c in self._unit_clauses if len(c) == 1 and c[0] > 0}
    unit_neg = {-c[0] for c in self._unit_clauses if len(c) == 1 and c[0] < 0}
    conflicts = unit_pos & unit_neg
    if conflicts:
        errors.append(f"Conflicting unit clauses for vars: {conflicts}")
    
    # INV-3: All clauses reference valid variables
    all_vars = set()
    for pair in self._pairs.values():
        all_vars.add(pair.l_var)
        all_vars.add(pair.neg_l_var)
    
    for clause in self._cnf.clauses:
        for lit in clause:
            if abs(lit) not in all_vars:
                errors.append(f"Clause references unknown var {abs(lit)}")
    
    # INV-4: Formula is satisfiable (unless explicitly made UNSAT)
    if not self.is_satisfiable():
        errors.append("Formula is UNSAT (may be intentional)")
    
    return len(errors) == 0, errors
```

---

## Implementation Details

### Complete CNFManagerV2 Class

```python
"""
CNF Manager V2: Proper literal pairs with automatic mutex.

All variables are hypothesis variables - count ALL of them.
Expected count for n fluent pairs with no constraints: 3^n
"""

from dataclasses import dataclass
from typing import Dict, Set, List, Optional, Tuple
from pysat.solvers import Glucose4
from pysat.formula import CNF
import logging

logger = logging.getLogger(__name__)


@dataclass
class LiteralPair:
    """Both variables are hypothesis variables - both must be counted."""
    fluent_name: str
    l_var: int       # "Is f a precondition?" - COUNTED
    neg_l_var: int   # "Is ¬f a precondition?" - ALSO COUNTED
    mutex_clause: Tuple[int, int]  # Eliminates (T,T) state
    
    @property
    def positive_name(self) -> str:
        return self.fluent_name
    
    @property
    def negative_name(self) -> str:
        return f'¬{self.fluent_name}'


class CNFManagerV2:
    """CNF Manager with literal pair model for model counting."""
    
    def __init__(self):
        self._next_var: int = 1
        self._pairs: Dict[str, LiteralPair] = {}
        self._var_to_pair: Dict[int, LiteralPair] = {}
        self._literal_to_var: Dict[str, int] = {}
        
        self._cnf = CNF()
        self._mutex_clauses: List[List[int]] = []
        self._constraint_clauses: List[List[int]] = []
        self._unit_clauses: List[List[int]] = []
        
        self._solver: Optional[Glucose4] = None
        self._solver_valid: bool = False
        self._cache_valid: bool = False
        self._solution_cache: Optional[List[Tuple[int, ...]]] = None
    
    # === PAIR MANAGEMENT ===
    
    def create_pair(self, fluent_name: str) -> LiteralPair:
        """Create literal pair with automatic mutex."""
        if fluent_name.startswith('¬'):
            raise ValueError(f"Use base name: '{fluent_name}' → '{fluent_name[1:]}'")
        
        if fluent_name in self._pairs:
            return self._pairs[fluent_name]
        
        l_var = self._next_var
        neg_l_var = self._next_var + 1
        self._next_var += 2
        
        mutex = [-l_var, -neg_l_var]
        
        pair = LiteralPair(fluent_name, l_var, neg_l_var, (mutex[0], mutex[1]))
        
        self._pairs[fluent_name] = pair
        self._var_to_pair[l_var] = pair
        self._var_to_pair[neg_l_var] = pair
        self._literal_to_var[fluent_name] = l_var
        self._literal_to_var[f'¬{fluent_name}'] = neg_l_var
        
        # CRITICAL: Add mutex immediately
        self._mutex_clauses.append(mutex)
        self._cnf.append(mutex)
        self._invalidate_cache()
        
        return pair
    
    def get_var(self, literal: str) -> Optional[int]:
        return self._literal_to_var.get(literal)
    
    def get_or_create_var(self, literal: str) -> int:
        if literal in self._literal_to_var:
            return self._literal_to_var[literal]
        
        base = literal[1:] if literal.startswith('¬') else literal
        pair = self.create_pair(base)
        return pair.neg_l_var if literal.startswith('¬') else pair.l_var
    
    # === CLAUSE MANAGEMENT ===
    
    def add_coverage_clause(self, literals: List[str]) -> None:
        """Add clause: at least one of these IS a precondition."""
        clause = [self.get_or_create_var(lit) for lit in literals]
        if clause:
            self._constraint_clauses.append(clause)
            self._cnf.append(clause)
            self._invalidate_cache()
    
    def add_exclusion(self, literal: str) -> None:
        """Add clause: this literal is NOT a precondition."""
        var_id = self.get_or_create_var(literal)
        unit = [-var_id]
        self._unit_clauses.append(unit)
        self._cnf.append(unit)
        self._invalidate_cache()
    
    def add_inclusion(self, literal: str) -> None:
        """Add clause: this literal IS a precondition."""
        var_id = self.get_or_create_var(literal)
        unit = [var_id]
        self._unit_clauses.append(unit)
        self._cnf.append(unit)
        self._invalidate_cache()
    
    # === MODEL COUNTING ===
    
    def is_satisfiable(self) -> bool:
        solver = Glucose4(bootstrap_with=self._cnf)
        try:
            return solver.solve()
        finally:
            solver.delete()
    
    def count_models(self) -> int:
        """
        Count ALL satisfying assignments.
        
        All variables are hypothesis variables.
        With n fluent pairs and mutex only: returns 3^n
        """
        solver = Glucose4(bootstrap_with=self._cnf)
        try:
            count = 0
            
            while solver.solve():
                count += 1
                model = solver.get_model()
                solver.add_clause([-lit for lit in model])
            
            return count
        finally:
            solver.delete()
    
    def count_models_with_assumptions(self, assumptions: List[int]) -> int:
        """Count with assumptions (no CNF modification)."""
        solver = Glucose4(bootstrap_with=self._cnf)
        try:
            count = 0
            
            while solver.solve(assumptions=assumptions):
                count += 1
                model = solver.get_model()
                solver.add_clause([-lit for lit in model])
            
            return count
        finally:
            solver.delete()
    
    # === VALIDATION ===
    
    def validate_mutex_invariant(self) -> bool:
        """Verify all pairs have mutex in CNF."""
        for pair in self._pairs.values():
            expected = sorted([-pair.l_var, -pair.neg_l_var])
            found = any(sorted(c) == expected for c in self._cnf.clauses)
            if not found:
                return False
        return True
    
    def expected_unconstrained_count(self) -> int:
        """Expected count with only mutex: 3^n."""
        return 3 ** len(self._pairs)
    
    # === INTERNALS ===
    
    def _invalidate_cache(self) -> None:
        self._cache_valid = False
        self._solution_cache = None
        self._solver_valid = False
    
    # === UTILITIES ===
    
    @property
    def num_vars(self) -> int:
        return self._next_var - 1
    
    @property
    def num_clauses(self) -> int:
        return len(self._cnf.clauses)
    
    @property
    def num_pairs(self) -> int:
        return len(self._pairs)
    
    def __str__(self) -> str:
        return f"CNFManagerV2({self.num_pairs} pairs, {self.num_vars} vars, {self.num_clauses} clauses)"
```

---

## Testing Strategy

### Unit Tests

```python
import pytest
from cnf_manager_v2 import CNFManagerV2


class TestLiteralPairs:
    def test_pair_creation(self):
        mgr = CNFManagerV2()
        pair = mgr.create_pair('clear(?x)')
        
        assert pair.fluent_name == 'clear(?x)'
        assert pair.l_var == 1
        assert pair.neg_l_var == 2
    
    def test_mutex_auto_added(self):
        mgr = CNFManagerV2()
        pair = mgr.create_pair('clear(?x)')
        
        # Mutex should be in CNF
        assert [-1, -2] in mgr._cnf.clauses or [-2, -1] in mgr._cnf.clauses
        assert mgr.validate_mutex_invariant()
    
    def test_negation_parsing(self):
        mgr = CNFManagerV2()
        
        var1 = mgr.get_or_create_var('clear(?x)')
        var2 = mgr.get_or_create_var('¬clear(?x)')
        
        assert var1 == 1  # l_var
        assert var2 == 2  # neg_l_var


class TestModelCounting:
    def test_single_pair_count(self):
        mgr = CNFManagerV2()
        mgr.create_pair('a')
        
        # 3 valid states: (F,F), (T,F), (F,T)
        count = mgr.count_models()
        assert count == 3
    
    def test_two_pairs_count(self):
        mgr = CNFManagerV2()
        mgr.create_pair('a')
        mgr.create_pair('b')
        
        # 3^2 = 9 valid states
        count = mgr.count_models()
        assert count == 9
    
    def test_three_pairs_count(self):
        mgr = CNFManagerV2()
        mgr.create_pair('a')
        mgr.create_pair('b')
        mgr.create_pair('c')
        
        # 3^3 = 27 valid states
        count = mgr.count_models()
        assert count == 27
    
    def test_with_exclusion(self):
        mgr = CNFManagerV2()
        mgr.create_pair('a')
        mgr.create_pair('b')
        
        mgr.add_exclusion('a')  # a cannot be precondition
        
        # a is FALSE, b has 3 states
        # But a being FALSE means l_a = F
        # neg_l_a can still be T or F (mutex allows both since l_a = F)
        # So for a: 2 states (FF, FT)
        # For b: 3 states
        # Total: 2 * 3 = 6
        count = mgr.count_models()
        assert count == 6
    
    def test_with_coverage_clause(self):
        mgr = CNFManagerV2()
        mgr.create_pair('a')
        mgr.create_pair('b')
        
        mgr.add_coverage_clause(['a', 'b'])
        
        # At least one of l_a or l_b must be TRUE
        # All 9 states minus the state where both l_a=F and l_b=F
        # But (l_a=F, l_b=F) combined with any neg_l states gives multiple excluded
        # Actually: the clause is (l_a OR l_b), so we need l_a=T or l_b=T
        # States: (l_a, neg_l_a, l_b, neg_l_b)
        # l_a=T: 1 * 1 (neg_l_a must be F) * 3 (b) = 3
        # l_a=F, l_b=T: 2 (neg_l_a can be T or F) * 1 * 1 = 2
        # Total = 3 + 2 = 5? Let me recalculate...
        # 
        # Actually for each pair independently:
        # If l_a=T: neg_l_a=F (mutex), so 1 state for a
        # If l_a=F: neg_l_a can be T or F, so 2 states for a
        # 
        # Coverage clause: l_a OR l_b must be true
        # Total without coverage: 3 * 3 = 9
        # Excluded: l_a=F AND l_b=F, neg_l_a ∈ {T,F}, neg_l_b ∈ {T,F}
        # = 2 * 2 = 4 excluded
        # Total: 9 - 4 = 5
        count = mgr.count_models()
        assert count == 5


class TestAssumptions:
    def test_assumptions_dont_modify_cnf(self):
        mgr = CNFManagerV2()
        mgr.create_pair('a')
        mgr.create_pair('b')
        
        original_clauses = len(mgr._cnf.clauses)
        
        count = mgr.count_models_with_assumptions([1])  # Assume l_a = TRUE
        
        assert len(mgr._cnf.clauses) == original_clauses  # Unchanged
```

---

## Migration Guide

### Step 1: Parallel Development

1. Create `cnf_manager_v2.py` with new implementation
2. Keep `cnf_manager.py` unchanged
3. Add feature flag to switch between them

### Step 2: Incremental Integration

```python
class InformationGainLearner:
    def __init__(self, ..., use_v2_cnf: bool = False):
        self.use_v2_cnf = use_v2_cnf
        
        if use_v2_cnf:
            from src.core.cnf_manager_v2 import CNFManagerV2
            self._cnf_class = CNFManagerV2
        else:
            from src.core.cnf_manager import CNFManager
            self._cnf_class = CNFManager
```

### Step 3: Validation Phase

1. Run both implementations on same problems
2. Compare model counts (V2 should give 3^n, V1 gives 4^n)
3. Verify V2 count equals expected_unconstrained_count()
4. Compare learning outcomes

### Step 4: Full Migration

1. Remove feature flag
2. Rename `cnf_manager_v2.py` → `cnf_manager.py`
3. Update all imports
4. Archive old implementation

---

## Appendix: Quick Reference

### Negation Semantics

| Notation | Meaning | Example |
|----------|---------|---------|
| `f` | Positive fluent | `clear(?x)` |
| `¬f` | Negative fluent (part of name) | `¬clear(?x)` |
| `l_var` | Variable: "Is f a precondition?" | `l_clear = 1` |
| `neg_l_var` | Variable: "Is ¬f a precondition?" | `neg_l_clear = 2` |
| `-var` | CNF negation (var = FALSE) | `-1` means `l_clear = FALSE` |

### Model Count Quick Reference

| Scenario | Count | Formula |
|----------|-------|---------|
| n pairs, mutex only | 3^n | Base hypothesis space |
| 1 pair | 3 | (FF), (TF), (FT) |
| 2 pairs | 9 | 3 × 3 |
| 5 pairs | 243 | 3^5 |
| 10 pairs | 59,049 | 3^10 |

### Clause Type Summary

| Type | Format | Purpose |
|------|--------|---------|
| Mutex | `[-l, -neg_l]` | Prevent (T,T) - auto-added |
| Coverage | `[l1, l2, ...]` | At least one true |
| Exclusion | `[-l]` | Force false |
| Inclusion | `[l]` | Force true |

---

*End of Fix Strategy Document*
