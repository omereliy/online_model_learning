You are designing the model counting module for a SAT-based action model learning system.

## Required Reading

First, read these files in the project:
1. docs/information_gain_algorithm/cnf/cnf_issues_analysis.md - Focus on section CRIT-03
2. docs/information_gain_algorithm/cnf/cnf_fix_strategy.md - Focus on "Phase 2: Model Counting Fixes"

## Important Clarification: No Projection Needed

For this algorithm, **ALL variables are hypothesis variables**. Both `l_var` AND `neg_l_var` represent independent hypothesis choices that must be counted:

| l_var | neg_l_var | Meaning | Valid? |
|:---:|:---:|:---|:---:|
| T | F | `f` is a precondition | ✓ |
| F | T | `¬f` is a precondition | ✓ |
| F | F | Neither is a precondition | ✓ |
| T | T | Both are preconditions | ✗ (mutex) |

**Key insight**: The mutex eliminates (T,T), leaving **3 valid states** per pair. Count = 3^n.

**No projection needed** - count ALL satisfying assignments over ALL variables.

## Problem Summary

The current model counting has issues:
1. Without proper mutex, counts 4^n instead of 3^n
2. No validation that mutex constraints are present
3. Modifies CNF directly for assumptions (should use solver's assumption mechanism)
4. Cache staleness issues in parallel execution

## Your Task

Design a clean model counting module that:
1. Counts ALL variables (no projection)
2. Validates mutex invariants before counting
3. Uses solver's native assumptions (no CNF modification)
4. Is thread-safe for parallel execution

## Requirements

### 1. Count All Variables
- Count over ALL variables (l_var AND neg_l_var)
- No projection/sampling set needed
- Expected count for n pairs with mutex only = 3^n

### 2. Counting Methods
- Exact counting: enumerate all satisfying assignments
- Approximate counting: use ApproxMC (no sampling set needed)
- Adaptive: choose based on problem size

### 3. Assumptions Support
- Temporary constraints for "what-if" queries
- Use solver's native `solve(assumptions=[...])` 
- NO modification of the CNF formula
- Must be thread-safe

### 4. Invariant Validation
- Verify mutex constraints present before counting
- Provide expected_unconstrained_count() = 3^n
- Sanity check: actual count ≤ expected count

### 5. Caching Strategy
- Cache base counts (no assumptions)
- Invalidate on CNF modification
- NO cache sharing between parallel workers

## Deliverables

Provide your response in this format:

### A. Interface Definition

```python
# Define the model counter interface
# Key methods:
#   - count_models() -> int  # Count ALL assignments
#   - count_models_with_assumptions(assumptions) -> int
#   - validate_mutex_invariant() -> bool
#   - expected_unconstrained_count() -> int  # 3^n
```

### B. Exact Counter

```python
# Interface for exact counting
# Should enumerate ALL satisfying assignments
# Block each found model and continue until UNSAT
```

### C. Approximate Counter

```python
# Interface for approximate counting with ApproxMC
# NO sampling set needed (count all variables)
```

### D. Adaptive Strategy

```python
# How to choose between exact and approximate
# Based on number of variables (not pairs)
```

### E. Assumptions Handling

```python
# Interface for counting with temporary assumptions
# MUST use solver's native assumptions parameter
# MUST NOT modify CNF
```

### F. Invariant Validation

Explain how to ensure correct counting:
1. Verify all pairs have mutex before counting
2. Compare actual count to 3^n upper bound
3. Detect missing mutex by count > 3^n

### G. Parallel Safety

Explain how to ensure thread-safety:
1. Fresh solver per count operation
2. No shared mutable state
3. No cache sharing between workers

### H. Usage Examples

```python
# Show common usage patterns:
# 1. Count all models (should be 3^n for n pairs with mutex only)
# 2. Count with assumptions
# 3. Validate invariant before counting
# 4. Detect incorrect count due to missing mutex
```

DO NOT write implementation code. Focus on clean interface design.
