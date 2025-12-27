You are designing the CNF (Conjunctive Normal Form) management module for an action model learning system.

## Required Reading

First, read these files in the project:
1. docs/information_gain_algorithm/cnf/cnf_issues_analysis.md - Focus on sections CRIT-01, CRIT-02, CRIT-03, CRIT-04
2. docs/information_gain_algorithm/cnf/cnf_fix_strategy.md - Focus on "Phase 1: Core Architecture Fixes"

## Problem Summary

The current CNF implementation has critical bugs:
1. Treats p and ¬p as independent variables WITHOUT automatic mutex
2. Mutex constraints added late and conditionally (not at pair creation)
3. Model counting returns 4^n instead of correct 3^n
4. Negation handling is confusing (¬ vs - mixed up)

## Important Clarification: All Variables Are Hypothesis Variables

For the information gain algorithm, **both** `l_var` AND `neg_l_var` must be counted:

| l_var (is f precondition?) | neg_l_var (is ¬f precondition?) | Meaning |
|:---:|:---:|:---|
| T | F | `f` is a precondition |
| F | T | `¬f` is a precondition |
| F | F | Neither is a precondition |
| T | T | ❌ Invalid (mutex prevents) |

**Key insight**: The mutex only eliminates the (T,T) state, leaving **3 valid states** per fluent pair.

**No projection needed** - count ALL satisfying assignments. Expected count = 3^n.

## Your Task

Design a clean CNF module architecture that fixes these issues.

## Requirements

### 1. Literal Pair Model with Automatic Mutex
- Each fluent f creates exactly TWO variables:
  - `l_var`: "is f a precondition?"
  - `neg_l_var`: "is ¬f a precondition?"
- Mutex constraint `[-l_var, -neg_l_var]` auto-added at creation
- Only 3 valid states: (F,F), (T,F), (F,T) - never (T,T)

### 2. All Variables Are Hypothesis Variables
- NO primary/auxiliary distinction needed
- NO projection needed - count ALL variables
- Expected unconstrained count = 3^n for n fluent pairs

### 3. Clause Types
- MUTEX: Binary clauses preventing (T,T) - AUTO-ADDED at pair creation
- COVERAGE: "At least one of these IS a precondition"
- EXCLUSION: "This literal is NOT a precondition"
- INCLUSION: "This literal IS a precondition"

### 4. Clear API
- No ambiguity about what each method does
- Separate string literals from variable IDs
- Explicit negation handling (¬ is part of name, - is CNF negation)

## Deliverables

Provide your response in this format:

### A. Data Structures

```python
# Define the key dataclasses
# Include docstrings explaining each field
# LiteralPair should have: fluent_name, l_var, neg_l_var, mutex_clause
```

### B. CNFManager Interface

```python
# Define the class with method signatures
# Include comprehensive docstrings
# Key methods:
#   - create_pair(fluent_name) -> LiteralPair  # ONLY way to create vars
#   - get_or_create_var(literal) -> int
#   - add_coverage_clause(literals)
#   - add_exclusion(literal)
#   - add_inclusion(literal)
#   - count_models() -> int
#   - validate_mutex_invariant() -> bool
#   - expected_unconstrained_count() -> int  # Should return 3^n
```

### C. Invariants

List the invariants that must ALWAYS hold:
1. Every fluent pair has a mutex clause in CNF
2. Mutex is added at pair creation, never removed
3. count_models() on empty CNF with n pairs = 3^n
4. All variables are hypothesis variables (no projection)

### D. Usage Examples

```python
# Show how the API would be used for common operations:
# 1. Creating a fluent pair (should auto-add mutex)
# 2. Verifying count = 3 for single pair
# 3. Adding a coverage clause
# 4. Adding an exclusion
# 5. Verifying count matches expected
```

### E. Interaction with Model Counting

Explain:
- Why no projection is needed (all vars are hypothesis vars)
- How mutex ensures correct 3^n count
- How to validate the count is correct

DO NOT write implementation code. Focus on clean interface design.
