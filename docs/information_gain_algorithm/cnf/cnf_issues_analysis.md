# CNF Handling Issues Analysis

## Document Information
- **Analysis Date**: December 24, 2025 (Updated: December 25, 2025)
- **Files Analyzed**:
  - `src/core/cnf_manager.py`
  - `src/algorithms/information_gain.py`
  - `src/algorithms/parallel_gain.py`
  - `src/core/grounding.py`
- **Reference Specification**: SAT-based Model Counting for Action Model Learning

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Critical Issues](#critical-issues)
3. [Moderate Issues](#moderate-issues)
4. [Minor Issues](#minor-issues)
5. [Specification Compliance Matrix](#specification-compliance-matrix)
6. [Impact Assessment](#impact-assessment)

---

## Executive Summary

The CNF handling implementation has **12 critical issues**, **8 moderate issues**, and **6 minor gaps** when compared against the specification for SAT-based model counting in action model learning.

### Key Finding Categories

| Category | Count | Impact |
|----------|-------|--------|
| Variable Model Errors | 3 | Model count inflation by factor of ~1.33^n |
| Missing Mutex Constraints | 2 | Invalid (T,T) states counted |
| Constraint Handling Bugs | 4 | Incorrect hypothesis space |
| Cache/Concurrency Issues | 3 | Wrong results in parallel execution |
| Missing Optimizations | 5 | Performance degradation |

### Most Critical Problems
1. **No literal pair model** - treats `p` and `¬p` as independent variables without mutex
2. **Mutex constraints incomplete** - invalid (T,T) states included in count, inflating by (4/3)^n
3. **Clause refinement corrupts formula** - `refine_clauses_by_intersection` is broken
4. **Success update semantic confusion** - constraint sets and CNF become inconsistent

---

## Important Clarification: Variable Counting Model

### All Hypothesis Variables Must Be Counted

For the information gain algorithm, **both** `l_var` (positive literal) **and** `neg_l_var` (negative literal) represent independent hypothesis choices that must be counted:

| l_var (is `f` precondition?) | neg_l_var (is `¬f` precondition?) | Meaning |
|:---:|:---:|:---|
| T | F | `f` is a precondition |
| F | T | `¬f` is a precondition |
| F | F | Neither is a precondition |
| T | T | ❌ Invalid (mutex prevents) |

**Key insight**: You cannot derive `neg_l_var`'s state from `l_var` alone. The mutex only eliminates the (T,T) state, leaving 3 valid states per fluent pair.

### When Projection IS Needed

Projection is only needed for **encoding auxiliary variables** (e.g., Tseitin transformation auxiliaries, cardinality encoding auxiliaries). The current implementation doesn't use such encodings, so:

- **All variables are hypothesis variables** → count ALL of them
- **No projection needed** in the current design
- **Correct unconstrained count** = 3^n (for n fluent pairs with mutex)

---

## Critical Issues

### CRIT-01: Incorrect Variable Model - No Literal Pairs with Mutex

**Severity**: 🔴 Critical  
**Category**: Architecture  
**Files Affected**: `cnf_manager.py`, `information_gain.py`

#### Specification Requirement

Each fluent `f` requires TWO hypothesis variables with a mutex constraint:
- `l_var`: "Is `f` a precondition?" (yes/no)
- `neg_l_var`: "Is `¬f` a precondition?" (yes/no)  
- Mutex clause: `[-l_var, -neg_l_var]` (cannot both be TRUE)

Valid states per fluent pair: **(F,F), (T,F), (F,T)** — exactly 3 states.

#### Current Implementation

**Location**: `cnf_manager.py`, lines 47-75

```python
def _literal_to_var(self, literal: str) -> tuple:
    """
    IMPORTANT: p(?x) and ¬p(?x) are treated as TWO SEPARATE VARIABLES.
    This is required by the information gain algorithm which models:
    - "is p(?x) a precondition?" (yes/no)
    - "is ¬p(?x) a precondition?" (yes/no)
    as independent questions in the hypothesis space.
    """
    cnf_negated = False
    fluent = literal

    if fluent.startswith('-'):
        cnf_negated = True
        fluent = fluent[1:]

    # IMPORTANT: Keep '¬' as part of the fluent name!
    var_id = self.add_fluent(fluent)
    return var_id, cnf_negated
```

#### Problem Analysis

The implementation treats `p(?x)` and `¬p(?x)` as completely independent variables **without automatically adding mutex constraints**. This creates **4 possible states** per fluent pair:

| p(?x) var | ¬p(?x) var | Interpretation | Valid? |
|-----------|------------|----------------|--------|
| FALSE | FALSE | Neither is precondition | ✓ |
| TRUE | FALSE | p(?x) is precondition | ✓ |
| FALSE | TRUE | ¬p(?x) is precondition | ✓ |
| TRUE | TRUE | BOTH are preconditions | ✗ |

The specification requires only **3 valid states** per entity pair.

#### Impact

- **Model Count Inflation**: For n fluent pairs, count is inflated by factor of (4/3)^n
- **Information Gain Miscalculation**: All probability and entropy calculations are wrong
- **Example**: With 10 fluent pairs:
  - Correct max models: 3^10 = 59,049
  - Current max models: 4^10 = 1,048,576 (17.7x inflation)

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `cnf_manager.py` | 47-75 | `_literal_to_var` doesn't enforce pairing |
| `cnf_manager.py` | 23-30 | `__init__` has no entity/pair tracking |
| `information_gain.py` | 198-215 | `_initialize_action_models` doesn't create pairs with mutex |

#### Correct Approach

```python
@dataclass
class LiteralPair:
    """Both variables are hypothesis variables - both must be counted."""
    fluent_name: str
    l_var: int       # "Is f a precondition?" - COUNTED
    neg_l_var: int   # "Is ¬f a precondition?" - ALSO COUNTED
    mutex: Tuple[int, int]  # Eliminates (T,T) state

def create_pair(self, fluent_name: str) -> LiteralPair:
    """Create paired variables with automatic mutex."""
    l_var = self._next_var
    neg_l_var = self._next_var + 1
    self._next_var += 2
    
    # CRITICAL: Add mutex immediately
    mutex = [-l_var, -neg_l_var]
    self._cnf.append(mutex)
    
    return LiteralPair(fluent_name, l_var, neg_l_var, tuple(mutex))
```

---

### CRIT-02: Mutex Constraints Added Conditionally and Late

**Severity**: 🔴 Critical  
**Category**: Constraint Integrity  
**Files Affected**: `information_gain.py`, `cnf_manager.py`

#### Specification Requirement

> Mutex Clauses (Binary)
> Purpose: Prevent both l_i and neg_l_i from being true simultaneously.
> ```
> For each entity E_i: (¬l_i ∨ ¬neg_l_i)
> ```
> - Count: n clauses (one per fluent pair)
> - Added: **At initialization (permanent)**

#### Current Implementation

**Location**: `information_gain.py`, lines 742-776

```python
def _add_mutual_exclusion_constraints(self, action: str, cnf: 'CNFManager') -> None:
    """Add mutual exclusion constraints: p and ¬p can't both be preconditions."""
    positive_fluents = set()
    for fluent in cnf.fluent_to_var.keys():
        if fluent.startswith('¬'):
            positive_fluents.add(fluent[1:])
        else:
            positive_fluents.add(fluent)

    mutex_count = 0
    for fluent in positive_fluents:
        neg_fluent = '¬' + fluent
        var_p = cnf.fluent_to_var.get(fluent)
        var_neg_p = cnf.fluent_to_var.get(neg_fluent)

        # PROBLEM: Only adds mutex if BOTH variables exist!
        if var_p is not None and var_neg_p is not None:
            cnf.add_var_clause([-var_p, -var_neg_p])
            mutex_count += 1
```

#### Problem Analysis

1. **Conditional Addition**: Mutex only added if BOTH `p` and `¬p` happen to exist as variables
2. **Late Timing**: Called from `_build_cnf_formula`, not at initialization
3. **Not Maintained**: Incremental updates don't preserve mutex invariant:
   - `refine_clauses_by_intersection` can remove mutex clauses
   - `add_clause_with_subsumption` doesn't check mutex preservation

#### Impact

- Constraint sets that only mention `p` (not `¬p`) have no mutex
- Model count includes invalid (T,T) states
- After clause refinement, mutex may be lost entirely

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `information_gain.py` | 742-776 | Conditional mutex addition |
| `information_gain.py` | 720-740 | `_build_cnf_formula` calls mutex late |
| `cnf_manager.py` | 107-149 | `refine_clauses_by_intersection` can corrupt mutex |

---

### CRIT-03: Model Counting Doesn't Validate Hypothesis Space

**Severity**: 🔴 Critical  
**Category**: Model Counting  
**Files Affected**: `cnf_manager.py`

#### Correct Understanding

For model counting in this algorithm:
- **All hypothesis variables must be counted** (both `l_var` and `neg_l_var`)
- **No projection is needed** since all variables represent hypothesis choices
- **Expected count** for n fluent pairs with mutex = 3^n

#### Current Implementation

**Location**: `cnf_manager.py`, lines 186-209

```python
def count_solutions(self, max_solutions: int = None) -> int:
    """Count all satisfying assignments."""
    solver = Glucose4(bootstrap_with=self.cnf)
    try:
        count = 0
        while solver.solve():
            count += 1
            if max_solutions and count >= max_solutions:
                break
            model = solver.get_model()
            solver.add_clause([-lit for lit in model if abs(lit) < self.next_var])
        return count
    finally:
        solver.delete()
```

#### Problem Analysis

The counting method itself is correct in principle (counts all assignments), but:

1. **Without mutex constraints**, it counts 4^n instead of 3^n
2. **No validation** that mutex constraints are present
3. **No sanity checks** on expected vs actual counts

#### What Should Be Added

```python
def count_models(self) -> int:
    """
    Count ALL satisfying assignments.
    
    All variables are hypothesis variables - no projection needed.
    With n fluent pairs and mutex constraints: returns at most 3^n
    Without constraints: returns exactly 3^n (or 4^n if mutex missing!)
    """
    # Validate invariant
    if not self._verify_mutex_constraints():
        raise ValueError("Mutex constraints missing - count would be incorrect")
    
    # ... existing counting logic ...
```

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `cnf_manager.py` | 186-209 | `count_solutions` has no invariant validation |
| `cnf_manager.py` | — | No `_verify_mutex_constraints` method exists |

---

### CRIT-04: `refine_clauses_by_intersection` Corrupts Formula

**Severity**: 🔴 Critical  
**Category**: Constraint Integrity  
**Files Affected**: `cnf_manager.py`

#### Specification Context

The algorithm requires refinement after successful action:
> pre?(a) = {B ∩ bindP(s, O) | B ∈ pre?(a)}

This should **intersect constraint sets**, not modify CNF clauses directly.

#### Current Implementation

**Location**: `cnf_manager.py`, lines 107-167

```python
def refine_clauses_by_intersection(self, satisfied_literals: Set[str]) -> int:
    for clause in self.cnf.clauses:
        # Skip unit negative clauses
        if len(clause) == 1 and clause[0] < 0:
            new_clauses.append(clause)
            continue

        # PROBLEM 1: Only considers POSITIVE CNF literals
        clause_literals = set()
        for lit in clause:
            if lit > 0:  # Ignores negative literals!
                fluent = self.var_to_fluent.get(lit)
                if fluent is not None:
                    clause_literals.add(fluent)

        # PROBLEM 2: Intersects with string names, loses structure
        refined = clause_literals.intersection(satisfied_literals)

        if len(refined) < len(clause_literals):
            modified += 1

        if refined:
            # PROBLEM 3: Converts back, may lose literals
            new_clause = []
            for lit_str in refined:
                var_id = self.fluent_to_var.get(lit_str)
                if var_id:
                    new_clause.append(var_id)
            if new_clause:
                new_clauses.append(new_clause)
        else:
            # PROBLEM 4: Empty clause = UNSAT, but continues anyway
            logger.warning("Empty clause after refinement...")
```

#### Problem Analysis

| Problem | Description | Impact |
|---------|-------------|--------|
| Ignores negative literals | `if lit > 0` skips all negative CNF literals | Loses exclusion constraints |
| String-based intersection | Converts to names, intersects, converts back | Lossy transformation |
| Creates empty clauses | Empty result logged but no UNSAT handling | Invalid formula state |
| Loses mutex clauses | Binary mutex clauses may be "refined" away | Model count corrupted |

#### Concrete Example

Original clause: `[-3, 5, -7]` (¬var3 ∨ var5 ∨ ¬var7)

After refinement (assuming var5's fluent is satisfied):
- Extracts only positive: `{fluent_5}`
- Intersects with satisfied: `{fluent_5}` ∩ `{fluent_5}` = `{fluent_5}`
- Result: `[5]` — **lost the negative literals entirely!**

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `cnf_manager.py` | 107-167 | Entire method is problematic |
| `information_gain.py` | 596-598 | Calls this broken method |

---

### CRIT-05: Success Update Has Semantic Confusion

**Severity**: 🔴 Critical  
**Category**: Algorithm Correctness  
**Files Affected**: `information_gain.py`

#### Specification Requirement

After successful action, unsatisfied literals cannot be preconditions:
> Add unit clauses: {¬x_l | l ∈ unsatisfied}
> 
> (In CNF terms: for each unsatisfied literal l, add [-var_l])

#### Current Implementation

**Location**: `information_gain.py`, lines 548-570

```python
def _update_success(self, action: str, objects: List[str],
                    state: Set[str], next_state: Set[str]) -> None:
    # ...
    unsatisfied_literals = self.pre[action] - satisfied_in_state

    # BLOCK 1: Add to constraint sets (CONFUSED)
    for literal in unsatisfied_literals:
        if literal.startswith('¬'):
            negated = literal[1:]  # Remove negation: '¬p' → 'p'
        else:
            negated = '¬' + literal  # Add negation: 'p' → '¬p'
        unit_clause = frozenset({negated})  # Adds '¬literal' as constraint
        updated_constraints.add(unit_clause)

    # ...

    # BLOCK 2: Add to CNF (CORRECT)
    for literal in unsatisfied_literals:
        clause = ['-' + literal]  # '-p' means var_p = FALSE
        self.cnf_managers[action].add_clause_with_subsumption(clause)
```

#### Problem Analysis

**Block 1** (constraint sets) is semantically confused:
- For unsatisfied `p(?x)`: adds `¬p(?x)` to constraints
- This creates a NEW constraint saying "¬p(?x) might be a precondition"
- But we wanted to say "p(?x) is NOT a precondition"

**Block 2** (CNF) is correct:
- `-p(?x)` means "variable for p(?x) = FALSE"
- Correctly excludes p(?x) from being a precondition

The constraint sets and CNF are now **inconsistent**.

#### Impact

- `pre_constraints` contains wrong information
- Future constraint-based calculations are wrong
- Information gain calculations use wrong constraint data

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `information_gain.py` | 548-570 | Constraint set logic confused |
| `information_gain.py` | 589-594 | CNF logic correct but duplicative |

---

### CRIT-06: Parallel Context Cache Validity Not Verified

**Severity**: 🔴 Critical  
**Category**: Concurrency  
**Files Affected**: `parallel_gain.py`

#### Current Implementation

**Location**: `parallel_gain.py`, lines 82-112

```python
class WorkerCNFCache:
    def _reconstruct(self, action_name: str) -> 'CNFManager':
        # ...
        # Restore solution cache for fast filtering
        cached_solutions = self.context.cnf_solution_cache.get(action_name)
        if cached_solutions is not None:
            mgr._solution_cache = [set(s) for s in cached_solutions]
            mgr._cache_valid = True  # PROBLEM: Blindly trusts parent's cache
        else:
            mgr._solution_cache = None
            mgr._cache_valid = False
        return mgr
```

**Location**: `parallel_gain.py`, lines 68-80

```python
def update_context(self, context: ActionGainContext):
    """Update context for new iteration."""
    self.context = context
    # Clear cached managers - they may be stale
    self._managers.clear()
    # NOTE: Does not verify cache validity of new context
```

#### Problem Analysis

1. **Parent Cache May Be Stale**: Parent invalidates cache on some operations but not all
2. **Workers Modify CNF**: `count_models_with_temporary_clause` adds/removes clauses
3. **No Verification**: Workers trust `_cache_valid` without checking CNF state
4. **Cross-Iteration Staleness**: Persistent pools reuse workers with old cache

#### Impact

- Workers may use stale solution cache
- Model counts differ between parallel and sequential execution
- Race conditions in cache state

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `parallel_gain.py` | 82-112 | Blind cache trust in reconstruction |
| `parallel_gain.py` | 68-80 | No cache verification on context update |
| `information_gain.py` | 455-475 | Parent creates context with potentially stale cache |

---

### CRIT-07: Double Negation Handling Ambiguous

**Severity**: 🔴 Critical  
**Category**: Semantics  
**Files Affected**: `cnf_manager.py`, `information_gain.py`

#### The Problem

The code uses TWO different negation systems that interact confusingly:

| Symbol | Meaning | Context |
|--------|---------|---------|
| `¬` (Unicode) | Part of literal name | `¬clear(?x)` is a distinct variable |
| `-` (ASCII) | CNF negation | `-var` means variable is FALSE |

#### Ambiguous Cases

```python
# Case 1: Negative literal, exclude from preconditions
literal = '¬clear(?x)'
clause = ['-' + literal]  # '-¬clear(?x)'

# In _literal_to_var:
# - Sees '-', sets cnf_negated = True
# - Remaining: '¬clear(?x)' (kept as variable name)
# - Result: var_id for '¬clear(?x)', negated = True
# - Meaning: "¬clear(?x) variable = FALSE"
```

This IS correct, but extremely confusing. The mental model required:
- `¬clear(?x)` = variable representing "is ¬clear(?x) a precondition?"
- `-¬clear(?x)` = "the answer is NO, ¬clear(?x) is NOT a precondition"

#### Problem Areas

1. **No Documentation**: The dual-negation system isn't clearly documented
2. **Easy to Misuse**: Contributors will likely create bugs
3. **Validation Missing**: No checks for malformed literals like `--p` or `¬¬p`

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `cnf_manager.py` | 47-75 | `_literal_to_var` dual negation handling |
| `information_gain.py` | 548-570 | Mixes both negation styles |
| `information_gain.py` | 589-594 | Correct but confusing `-` + literal |

---

## Moderate Issues

### MOD-01: No Symmetry Breaking

**Severity**: 🟠 Moderate  
**Category**: Optimization  
**Files Affected**: All

#### Specification Requirement

> Symmetry Breaking: Eliminate equivalent solutions via lexicographic ordering.
> ```
> For symmetric entities E_i, E_j (i < j):
>   (¬l_j ∨ l_i)  # If j selected, i must be selected first
> ```

#### Current State

**Not implemented.** No symmetry detection or breaking anywhere in the codebase.

#### Impact

For problems with symmetric entities (e.g., interchangeable blocks):
- n! symmetric solutions counted instead of 1
- With 5 symmetric entities: 120x overcounting

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `cnf_manager.py` | — | No symmetry-related code |
| `information_gain.py` | — | No symmetry detection |

---

### MOD-02: `count_solutions_approximate` Uses Wrong API

**Severity**: 🟠 Moderate  
**Category**: Implementation  
**Files Affected**: `cnf_manager.py`

#### Current Implementation

**Location**: `cnf_manager.py`, lines 296-321

```python
def count_solutions_approximate(self, epsilon: float = 0.3, delta: float = 0.05) -> int:
    import pyapproxmc

    counter = pyapproxmc.Counter(epsilon=epsilon, delta=delta)
    for clause in self.cnf.clauses:
        counter.add_clause(clause)

    cells, hashes = counter.count()
    return max(1, cells * (2 ** hashes))
```

#### Problems

1. **Wrong Constructor**: `pyapproxmc.Counter()` doesn't take epsilon/delta in constructor
2. **Wrong Return Parsing**: Return format differs by pyapproxmc version

#### Correct API

```python
import pyapproxmc

counter = pyapproxmc.Counter()
for clause in self.cnf.clauses:
    counter.add_clause(clause)

# Count (returns integer directly in newer versions)
count = counter.count()
```

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `cnf_manager.py` | 296-321 | Wrong pyapproxmc API usage |

---

### MOD-03: Assumptions Implemented via CNF Modification

**Severity**: 🟠 Moderate  
**Category**: Implementation  
**Files Affected**: `cnf_manager.py`

#### Specification Recommendation

> **Assumptions** = temporary literal assignments for a single `solve()` call
> - Not permanently added to formula
> - Automatic retraction after solve

#### Current Implementation

**Location**: `cnf_manager.py`, lines 399-420

```python
def count_models_with_assumptions(self, assumptions: List[int], use_cache: bool = True) -> int:
    # Add assumptions as temporary unit clauses
    num_added = 0
    for assumption in assumptions:
        self.cnf.clauses.append([assumption])  # MODIFIES CNF!
        num_added += 1

    try:
        count = self.count_solutions_adaptive()
        return count
    finally:
        # Remove temporary unit clauses
        for _ in range(num_added):
            self.cnf.clauses.pop()
```

#### Problems

1. **Modifies CNF Directly**: Should use solver's assumption mechanism
2. **Not Thread-Safe**: Multiple callers could interfere
3. **Loses Learned Clauses**: Solver doesn't know these are temporary

#### Correct Approach

```python
def count_models_with_assumptions(self, assumptions: List[int]) -> int:
    solver = Glucose4(bootstrap_with=self.cnf)
    try:
        count = 0
        while solver.solve(assumptions=assumptions):  # Native support!
            count += 1
            model = solver.get_model()
            solver.add_clause([-lit for lit in model])
        return count
    finally:
        solver.delete()
```

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `cnf_manager.py` | 399-420 | `count_models_with_assumptions` modifies CNF |
| `cnf_manager.py` | 454-478 | `count_models_with_temporary_clause` same issue |

---

### MOD-04: Inconsistent Cache Invalidation

**Severity**: 🟠 Moderate  
**Category**: State Management  
**Files Affected**: `cnf_manager.py`

#### Current Implementation

**Location**: `cnf_manager.py`, lines 282-288

```python
def _invalidate_cache(self):
    """Invalidate solution cache and solver when formula changes."""
    self._cache_valid = False
    self._solution_cache = None
    self._cleanup_solver()
```

#### Places That Modify CNF Without Invalidating

| Method | Modification | Invalidates? |
|--------|--------------|--------------|
| `add_clause` | Appends clause | ✓ Yes |
| `add_var_clause` | Appends clause | ✓ Yes |
| `count_models_with_assumptions` | Appends/removes | ✗ No |
| `count_models_with_temporary_clause` | Appends/removes | ✗ No |
| `refine_clauses_by_intersection` | Modifies clauses | ✓ Yes |

#### Impact

- Solver may retain learned clauses from temporary modifications
- Cache may be marked valid when it's actually stale
- Unpredictable behavior in incremental solving

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `cnf_manager.py` | 399-420 | No invalidation around temp clauses |
| `cnf_manager.py` | 454-478 | Same issue |

---

### MOD-05: Missing Subsumption in Primary `add_clause`

**Severity**: 🟠 Moderate  
**Category**: Performance  
**Files Affected**: `cnf_manager.py`

#### Current Implementation

**Location**: `cnf_manager.py`, lines 77-93

```python
def add_clause(self, clause: List[str]):
    """Add clause with fluent strings."""
    var_clause = []
    for lit in clause:
        var_id, is_negated = self._literal_to_var(lit)
        var_clause.append(-var_id if is_negated else var_id)

    self.cnf.append(var_clause)  # No subsumption check!
    self._invalidate_cache()
```

vs.

**Location**: `cnf_manager.py`, lines 95-131

```python
def add_clause_with_subsumption(self, clause: List[str]) -> bool:
    """Add clause with subsumption checking to keep CNF minimal."""
    # ... full subsumption logic
```

#### Problem

Two separate methods exist, and `add_clause` (the simpler one) is often used when subsumption would be beneficial.

#### Impact

- CNF grows with redundant clauses
- SAT solving becomes slower
- More memory usage

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `cnf_manager.py` | 77-93 | No subsumption in `add_clause` |
| `information_gain.py` | various | Inconsistent use of both methods |

---

### MOD-06: Base Model Count Cache Can Be Stale

**Severity**: 🟠 Moderate  
**Category**: State Management  
**Files Affected**: `information_gain.py`

#### Current Implementation

**Location**: `information_gain.py`, lines 778-810

```python
def _get_base_model_count(self, action: str) -> int:
    # Check cache first
    if action in self._base_cnf_count_cache:
        return self._base_cnf_count_cache[action]  # May be stale!

    # ... compute count ...
    
    self._base_cnf_count_cache[action] = count
    return count
```

#### Invalidation

**Location**: `information_gain.py`, lines 504-506

```python
# Invalidate base CNF count cache after formula changes
self._base_cnf_count_cache.pop(action, None)
```

#### Problem

Cache is only invalidated in `update_model()`, but CNF can change in:
- `_build_cnf_formula()` — adds mutex constraints
- `_add_mutual_exclusion_constraints()` — adds clauses
- Direct CNF manipulation in edge cases

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `information_gain.py` | 778-810 | Stale cache usage |
| `information_gain.py` | 504-506 | Only invalidation point |

---

### MOD-07: Parallel Workers Don't Share Learned Clauses

**Severity**: 🟠 Moderate  
**Category**: Performance  
**Files Affected**: `parallel_gain.py`

#### Current Implementation

Each worker creates its own solver instance:

```python
def _reconstruct(self, action_name: str) -> 'CNFManager':
    mgr = CNFManager()
    # ... restore state ...
    # Each worker has independent solver
```

#### Problem

CDCL solvers learn clauses during search. With independent solvers:
- Same conflicts discovered repeatedly
- No learning transfer between workers
- Redundant computation

#### Impact

For complex formulas, parallel execution may be slower than sequential due to:
- Process spawn overhead
- No shared learning
- Memory duplication

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `parallel_gain.py` | 82-112 | Independent solver per worker |
| `information_gain.py` | 390-430 | No clause sharing mechanism |

---

### MOD-08: Hypothesis Space Size Calculation Assumes Perfect Pairing

**Severity**: 🟠 Moderate  
**Category**: Algorithm Correctness  
**Files Affected**: `information_gain.py`, `parallel_gain.py`

#### Current Implementation

**Location**: `information_gain.py`, lines 270-275

```python
la_size = len(self._get_parameter_bound_literals(action))
num_fluents = la_size // 2  # Assumes La contains both p and ¬p
total_hypotheses = 3 ** num_fluents if num_fluents > 0 else 1
```

#### Problem

Assumes `La` always contains exactly pairs `(p, ¬p)`. But:

1. **Propositional predicates** (no parameters) might only have one form
2. **Constraints** might only mention `p` without `¬p`
3. **Integer division** loses odd counts: `La` = 5 → num_fluents = 2

#### Impact

- Normalization factor is wrong
- Information gain values are skewed
- Comparison between actions is unfair

#### Code References

| File | Line(s) | Issue |
|------|---------|-------|
| `information_gain.py` | 270-275 | `_calculate_potential_gain_success` |
| `information_gain.py` | 310-315 | `_calculate_potential_gain_failure` |
| `parallel_gain.py` | 200-205 | Same assumption |

---

## Minor Issues

### MIN-01: No DIMACS Output Support

**Severity**: 🟡 Minor  
**Category**: Interoperability  

#### Current State

`CNFManager.to_string()` produces human-readable output, not DIMACS format.

No method to export for external model counters (sharpSAT, d4, GANAK).

---

### MIN-02: No Clause Ordering Convention

**Severity**: 🟡 Minor  
**Category**: Performance  

#### Specification Recommendation

> Clause Ordering Convention:
> 1. Unit clauses (positive first, then negative)
> 2. Binary clauses (mutex constraints)
> 3. Large clauses (coverage constraints)

#### Current State

Clauses are added in arbitrary order based on observation sequence.

---

### MIN-03: Solver Reuse Strategy Loses Learning

**Severity**: 🟡 Minor  
**Category**: Performance  

#### Current Implementation

```python
def _invalidate_cache(self):
    self._cleanup_solver()  # Destroys solver and learned clauses!
```

Any modification destroys all learned clauses, even for unrelated changes.

---

### MIN-04: `minimize_qm` and `minimize_espresso` Are Non-Functional

**Severity**: 🟡 Minor  
**Category**: Dead Code  

#### Current State

```python
# TODO: CNF Minimization Methods (minimize_qm, _rebuild_from_solutions, minimize_espresso)
# These methods are currently unused but could potentially improve...
```

These methods exist but:
- Call `get_all_solutions()` which can hang on large formulas
- Are never called from information_gain.py
- Would produce wrong results if called

---

### MIN-05: No Bounded Variable Elimination

**Severity**: 🟡 Minor  
**Category**: Optimization  

#### Specification Mentioned

> Redundancy Elimination:
> - Subsumption (partial)
> - Blocked clause elimination (not implemented)
> - Bounded variable elimination (not implemented)

---

### MIN-06: Mixed Naming Conventions

**Severity**: 🟡 Minor  
**Category**: Code Quality  

#### Examples

| Inconsistency | Locations |
|---------------|-----------|
| `fluent_to_var` vs `var_to_fluent` | Inverse naming |
| `cnf_negated` vs `is_negated` | Boolean naming |
| `-` prefix vs `¬` prefix | String conventions |
| `lit` vs `literal` vs `fluent` | Term confusion |

---

## Specification Compliance Matrix

| Requirement | Status | Notes |
|-------------|--------|-------|
| Literal pair model with mutex | ❌ Not Implemented | Variables created independently, mutex added late |
| Mutex constraints at init | ❌ Partially | Added late, conditional |
| 3 valid states per entity | ❌ No | 4 states possible without proper mutex |
| Count all hypothesis variables | ⚠️ Partial | Counts all, but wrong count due to missing mutex |
| Unit propagation | ⚠️ Partial | PySAT does this internally |
| Symmetry breaking | ❌ Not Implemented | — |
| Coverage clauses | ✅ Implemented | Via constraint sets |
| Assumptions for incremental | ❌ Wrong | Modifies CNF directly |
| DIMACS output | ❌ Not Implemented | — |
| Clause ordering | ❌ Not Implemented | — |
| Cache/solver reuse | ⚠️ Partial | Loses learned clauses |

---

## Impact Assessment

### Quantitative Impact on Model Counting

| Issue | Impact Factor | Example (n=10 fluent pairs) |
|-------|---------------|------------------------|
| Missing mutex constraints | (4/3)^n | 17.7x overcounting |
| Missing symmetry breaking | Up to n! | Variable, problem-dependent |
| **Combined** | **Multiplicative** | **17.7x+ overcounting** |

### Correct vs Incorrect Model Counts

| Scenario | Correct Count | Current Count (No Mutex) |
|----------|---------------|--------------------------|
| 5 fluent pairs, no constraints | 3^5 = 243 | 4^5 = 1,024 |
| 10 fluent pairs, no constraints | 3^10 = 59,049 | 4^10 = 1,048,576 |
| 15 fluent pairs, no constraints | 3^15 = 14,348,907 | 4^15 = 1,073,741,824 |

### Qualitative Impact on Learning

| Issue | Effect |
|-------|--------|
| Wrong model count | Information gain completely wrong |
| Stale caches | Inconsistent results between runs |
| Parallel bugs | Different results parallel vs sequential |
| Corrupted CNF | Formula becomes unsatisfiable incorrectly |

### Risk Assessment

| Risk | Likelihood | Impact | Priority |
|------|------------|--------|----------|
| Learning converges to wrong model | High | Critical | P0 |
| Exponential blowup on large problems | High | Critical | P0 |
| Parallel execution gives wrong results | Medium | High | P1 |
| Performance degradation | High | Medium | P2 |

---

## Appendix: Code Location Quick Reference

### cnf_manager.py

| Lines | Function/Method | Issues |
|-------|-----------------|--------|
| 23-30 | `__init__` | No pair tracking |
| 47-75 | `_literal_to_var` | Dual negation confusion |
| 77-93 | `add_clause` | No subsumption |
| 107-167 | `refine_clauses_by_intersection` | Corrupts formula |
| 186-209 | `count_solutions` | No mutex validation |
| 296-321 | `count_solutions_approximate` | Wrong API |
| 399-420 | `count_models_with_assumptions` | Modifies CNF |

### information_gain.py

| Lines | Function/Method | Issues |
|-------|-----------------|--------|
| 198-215 | `_initialize_action_models` | No pair creation with mutex |
| 270-275 | `_calculate_potential_gain_success` | Wrong hypothesis count |
| 548-570 | `_update_success` | Semantic confusion |
| 720-740 | `_build_cnf_formula` | Late mutex addition |
| 742-776 | `_add_mutual_exclusion_constraints` | Conditional mutex |
| 778-810 | `_get_base_model_count` | Stale cache |

### parallel_gain.py

| Lines | Function/Method | Issues |
|-------|-----------------|--------|
| 68-80 | `update_context` | No cache verification |
| 82-112 | `_reconstruct` | Blind cache trust |
| 200-205 | `_calculate_potential_gain_success` | Wrong hypothesis count |

---

*End of Issues Analysis Document*
