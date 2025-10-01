# Information Gain Algorithm for Online Action Model Learning

## Overview

This document describes a novel information-theoretic approach to online action model learning that uses CNF formulas and SAT solvers to represent uncertainty and select actions that maximize expected information gain.

## Mathematical Notation Reference

### Set Operations
- `∪` : Union (combine sets)
- `∩` : Intersection (common elements)
- `\` : Set difference (elements in first but not second)
- `∅` : Empty set
- `|S|` : Cardinality (size) of set S

### Logical Operations
- `∧` : Logical AND (conjunction)
- `∨` : Logical OR (disjunction)
- `¬` : Logical NOT (negation)
- `⋀` : Big AND (conjunction over multiple formulas)
- `⋁` : Big OR (disjunction over multiple formulas)

## Core Concepts

### Key Definitions

- **`a`**: A lifted action
- **`O`**: An ordered set of objects
- **`<a, O>`**: A grounding of action `a` if `O` satisfies `param(a)`
- **`La`**: Parameter-bound literals of action `a` (all possible lifted fluents for this action)
- **`s`**: Pre-state (state before action execution)
- **`s'`**: Post-state (state after action execution)
- **`pre*(a)`**: Real (unknown) action model preconditions
- **`eff*+(a)`**: Real (unknown) positive effects (add effects)
- **`eff*-(a)`**: Real (unknown) negative effects (delete effects)

### Binding Functions

- **`bindP⁻¹(F, O)`**: Returns groundings of parameter-bound literals `F` with respect to order `O`
  - Example: `bindP⁻¹({on(?x,?y)}, [a,b])` → `{on(a,b)}`
  - Example with negation: `bindP⁻¹({¬on(?x,?y)}, [a,b])` → `{¬on(a,b)}`
- **`bindP(f, O)`**: Returns parameter-bound literals from grounded literals `f` with respect to order `O`
  - Example: `bindP({on(a,b)}, [a,b])` → `{on(?x,?y)}`
  - Example with negation: `bindP({¬on(a,b)}, [a,b])` → `{¬on(?x,?y)}`

## Negative Preconditions

### Understanding Negative Preconditions

Negative preconditions specify what must NOT be true for an action to be applicable. They are essential for many PDDL domains.

#### Examples:

1. **`put-down(?x)` action**:
   - Positive precondition: `holding(?x)` (robot must be holding x)
   - Negative precondition: `¬handempty` (hand must NOT be empty)
   - In state representation: Both `holding(a)` and `¬handempty` must be satisfied

2. **`stack(?x,?y)` action**:
   - Positive preconditions: `holding(?x)`, `clear(?y)`
   - Negative precondition: `¬on(?x,?y)` (x must NOT already be on y)
   - Prevents redundant stacking

3. **`unload-truck(?pkg,?truck,?loc)` action**:
   - Positive preconditions: `in(?pkg,?truck)`, `at(?truck,?loc)`
   - Negative precondition: `¬at(?pkg,?loc)` (package not already at location)

### How Algorithm Handles Negative Preconditions

When checking if literals are satisfied in state `s`:
- **Positive literal** `p`: Check if `p ∈ s`
- **Negative literal** `¬p`: Check if `p ∉ s` (p is NOT in state)

Example with state `s = {clear(a), on(b,c), handempty}`:
- `clear(a)` is satisfied ✓ (present in state)
- `¬clear(a)` is NOT satisfied ✗ (clear(a) is true)
- `¬on(a,b)` is satisfied ✓ (on(a,b) is not in state)
- `¬handempty` is NOT satisfied ✗ (handempty is true)

## Algorithm State Representation

### State Variables Explanation

For each action `a`, the algorithm maintains six knowledge sets:

1. **`pre(a)`**: Possible preconditions (literals not yet ruled out, including negative literals)
2. **`pre?(a)`**: Constraint sets (each failure adds a constraint)
3. **`eff+(a)`**: Confirmed add effects
4. **`eff-(a)`**: Confirmed delete effects
5. **`eff?+(a)`**: Possible add effects (not yet determined)
6. **`eff?-(a)`**: Possible delete effects (not yet determined)

### Initialization

```python
# For each action a:
pre(a) = La           # All literals not ruled out as preconditions (initially all possible)
pre?(a) = ∅          # Constraint sets: each set contains at least 1 required precondition
eff+(a) = ∅          # Confirmed add effects (fluents that MUST be added)
eff-(a) = ∅          # Confirmed delete effects (fluents that MUST be deleted)
eff?+(a) = La        # Possible add effects (not yet confirmed or ruled out)
eff?-(a) = La        # Possible delete effects (not yet confirmed or ruled out)
```

### Update Rules

#### When Action Succeeds (a applicable in state s → s')

**What happens**: Action execution succeeded, so we learn from the state transition.

```python
# Keep only satisfied preconditions
pre(a) = pre(a) ∩ bindP⁻¹(s, O)
# Example with positive: If 'clear(?x)' is in pre(a) and clear(a) is true in s, keep it
# Example with negative: If '¬on(?x,?y)' is in pre(a) and on(a,b) is false in s, keep it
# Example removal: If 'handempty' is in pre(a) but false in s, remove it
# Example removal: If '¬clear(?x)' is in pre(a) but clear(a) is true in s, remove it

# Add confirmed effects based on state changes
eff+(a) = eff+(a) ∪ bindP(s' \ s, O)     # Fluents that became true (add effects)
eff-(a) = eff-(a) ∪ bindP(s \ s', O)     # Fluents that became false (delete effects)

# Narrow down possible effects
eff?+(a) = eff?+(a) ∩ bindP(s ∩ s', O)   # Fluents unchanged (still possible add effects)
eff?-(a) = eff?-(a) \ bindP(s ∪ s', O)   # Remove fluents that were true or became true

# Update constraint sets
pre?(a) = {B ∩ bindP(s, O) | B ∈ pre?(a)}  # Keep only satisfied literals in each constraint
```

#### When Action Fails (a not applicable in state s)

**What happens**: Action execution failed, so at least one precondition was not satisfied.

```python
# Add new constraint: at least one unsatisfied literal must be a precondition
pre?(a) = pre?(a) ∪ {pre(a) \ bindP(s, O)}
# This constraint contains all literals from pre(a) that were NOT satisfied in s
```

## CNF Formula Construction

### Why CNF?

CNF (Conjunctive Normal Form) allows us to:
- Represent uncertainty about preconditions compactly
- Use SAT solvers for efficient model counting
- Calculate exact probabilities of action applicability

### Precondition CNF Representation

The algorithm builds a CNF formula to represent precondition uncertainty:

```python
# Build CNF from constraint sets
cnf_pre?(a) = ⋀(⋁xl) for B ∈ pre?(a), l ∈ B
# Each constraint set B becomes a clause (disjunction)
# All clauses together form a conjunction (CNF)
```

#### Negative Preconditions in CNF

Negative preconditions are naturally handled in CNF:
- Positive literal `clear(?x)` → variable `x_clear_x`
- Negative literal `¬clear(?x)` → negated variable `¬x_clear_x`

**Example CNF with mixed preconditions**:
```
Action: stack(?x,?y)
Possible preconditions: {holding(?x), clear(?y), ¬on(?x,?y)}
After failures, constraint: At least one of {holding(?x), ¬on(?x,?y)} required

CNF formula: (x_holding_x ∨ ¬x_on_x_y)
This captures that either holding(x) is true OR on(x,y) is false (or both)
```

For action applicability in state s with grounding O:
```python
# Add constraints for unsatisfied literals
cnf_pre?(a,O,s) = cnf_pre?(a) ⋀ (¬xl) for l ∈ (⋃pre?(a)) \ bindP(s, O)
# For each literal NOT satisfied in s, add a unit clause asserting it's false
```

### Applicability Probability Calculation

The probability that action (a, O) is applicable in state s:

```python
pr(app(a, O, s) = 1) = {
    1,                                             if pre?(a) = ∅
    |SAT(cnf_pre?(a,O,s))| / |SAT(cnf_pre?(a))|,  otherwise
}
# SAT() returns the set of satisfying assignments
# Ratio gives probability that action is applicable
```

Where:
- `SAT(cnf)` returns the set of satisfying assignment functions for the CNF formula
- Empty `pre?(a)` means no constraints (action always applicable)
- The ratio represents the fraction of models where action is applicable

## Information Gain Metrics

### Overview

The algorithm measures information gain in three areas:
1. **Precondition certainty** - Learning which literals are required
2. **Add effect certainty** - Learning what the action adds to state
3. **Delete effect certainty** - Learning what the action removes from state

### Precondition Knowledge Gain

#### Success Case
When action succeeds from state s:
```python
preAppPotential(a, O, s) = |pre(a) \ bindP(s, O)|
```

#### Failure Case
When action fails from state s:
```python
# Information gain from learning a precondition constraint
preFailPotential(a, O, s) = 1 - (|SAT(cnf_pre?(a))| - |SAT(cnf'_pre?(a))|) / 2^|La|
# cnf'_pre?(a) is the updated CNF after adding the failure constraint
# Measures reduction in uncertainty
```

### Effects Knowledge Gain

For successful execution:
```python
# Information gain from learning effects
eff+Potential(a, O, s) = |eff?+(a) \ bindP(s, O)| / |La|  # Add effects we can rule out
eff-Potential(a, O, s) = |eff?-(a) ∩ bindP(s, O)| / |La|  # Delete effects we can confirm
effPotential(a, O, s) = eff+Potential(a, O, s) + eff-Potential(a, O, s)
```

### Total Knowledge Gain

```python
sucPotential(a, O, s) = effPotential(a, O, s) + preAppPotential(a, O, s)
```

## Action Selection Strategy

### Expected Information Gain

For each grounded action (a, O) in state s:

```python
E[X(a,O,s)] = pr(app(a,O,s)=1) * sucPotential(a,O,s) +
              pr(app(a,O,s)=0) * preFailPotential(a,O,s)
```

### Selection Methods

#### Option 1: Greedy Selection (Recommended)
Select action that maximizes expected information gain:
```python
argmax_{(a,O)} E[X(a,O,s)]
```

#### Option 2: Probabilistic Selection (Exploration-friendly)
Select actions with probability proportional to expected gain:
```python
P_s(choose(a,O)) = w(a,O,s) / Σ_{all (a',O')} w(a',O',s)
where w(a,O,s) = E[X(a,O,s)]
```

## Implementation Architecture

### CNF/SAT Integration Components

1. **Variable Mapping**: Map fluents to CNF variable IDs
2. **CNF Builder**: Construct CNF formulas from observations
3. **SAT Solver**: Use PySAT (minisat) for model counting
4. **Formula Minimizer**: Optimize large CNF formulas

### Core Algorithm Flow

```python
# Main learning loop
while not fully_learned:
    1. Calculate expected information gain for all actions
    2. Select action based on strategy (greedy or probabilistic)
    3. Execute action in environment
    4. Observe success/failure and resulting state
    5. Update action model (pre, pre?, eff+, eff-, eff?+, eff?-)
    6. Update CNF formulas
    7. Minimize formulas if needed
```

## Key Properties

### Important Invariants

1. **Soundness**: Every literal in `eff+(a)` and `eff-(a)` is correct
2. **Monotonicity**: Sets `pre(a)`, `eff?+(a)`, `eff?-(a)` only shrink
3. **Constraint validity**: Every constraint in `pre?(a)` is satisfied by true model

### Convergence Guarantees

- With enough negative samples, the algorithm converges to certain preconditions where:
  - `∀l ∈ pre(a), {l} ∈ pre?(a)`
  - Each learned precondition is guaranteed to be in the true model

### Learning Completeness

- **Effects fully learned**: `|eff?+(a)| = |eff?-(a)| = 0`
  - All add effects identified: `eff+(a)` complete
  - All delete effects identified: `eff-(a)` complete
- **Preconditions fully certain**: All literals have singleton sets in `pre?(a)`
  - Each required precondition identified with certainty

## Implementation Specifications

### Required Dependencies

- **PySAT**: For CNF manipulation and SAT solving
- **Unified Planning Framework**: For PDDL parsing and planning
- **External Algorithms**: OLAM and ModelLearner for comparison

### Performance Optimizations

1. **Formula Caching**: Cache SAT solver results for repeated queries
2. **Incremental Updates**: Update CNF formulas incrementally
3. **Approximate Counting**: Use approximate model counting for large formulas
4. **Formula Minimization**: Reduce CNF size when exceeding thresholds

### Error Handling

- Handle SAT solver timeouts with approximate methods
- Validate variable mapping consistency
- Check formula size limits before solving

## Algorithm Pseudocode

```python
class InformationGainLearner:
    def __init__(self, domain):
        self.actions = extract_actions(domain)
        self.variable_mapper = VariableMapper()
        self.cnf_builder = CNFBuilder(self.variable_mapper)
        self.sat_solver = SATSolver('minisat')
        self.initialize_action_models()

    def initialize_action_models(self):
        for action in self.actions:
            La = get_parameter_bound_literals(action)
            self.pre[action] = La
            self.pre_uncertain[action] = CNF()  # Empty CNF
            self.eff_add[action] = set()
            self.eff_del[action] = set()
            self.eff_maybe_add[action] = La
            self.eff_maybe_del[action] = La.copy()

    def select_action(self, state):
        best_action = None
        best_gain = -1

        for action, objects in get_grounded_actions(state):
            prob_success = self.calculate_applicability_prob(action, objects, state)
            success_gain = self.calculate_success_gain(action, objects, state)
            fail_gain = self.calculate_fail_gain(action, objects, state)

            expected_gain = prob_success * success_gain + (1-prob_success) * fail_gain

            if expected_gain > best_gain:
                best_gain = expected_gain
                best_action = (action, objects)

        return best_action

    def update_model(self, action, objects, state, success, next_state=None):
        if success:
            self.update_success(action, objects, state, next_state)
        else:
            self.update_failure(action, objects, state)

        # Minimize CNF if too large
        if self.cnf_size(action) > MAX_CLAUSES:
            self.minimize_cnf(action)
```

## Future Work

- Address injective binding problem using equality predicates
- Implement proof of correctness for exported models
- Analyze worst-case learning bounds