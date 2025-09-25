# Information Gain Algorithm for Online Action Model Learning

## Overview

This document describes a novel information-theoretic approach to online action model learning that uses CNF formulas and SAT solvers to represent uncertainty and select actions that maximize expected information gain.

## Core Concepts

### Key Definitions

- **`a`**: A lifted action
- **`O`**: An ordered set of objects
- **`<a, O>`**: A grounding of action `a` if `O` satisfies `param(a)`
- **`La`**: Parameter-bound literals of action `a`
- **`s, s'`**: Pre-state and post-state respectively
- **`pre*(a), eff*-(a), eff*+(a)`**: Real (unknown) action model preconditions and effects

### Binding Functions

- **`bindP⁻¹(F, O)`**: Returns groundings of parameter-bound literals `F` with respect to order `O`
- **`bindP(f, O)`**: Returns parameter-bound literals from grounded literals `f` with respect to order `O`

## Algorithm State Representation

### Initialization

```python
# For each action a:
pre(a) = La           # All literals not ruled out as preconditions
pre?(a) = ∅          # Sets of subsets, each containing at least 1 required precondition
eff+(a) = ∅          # Observed add effects
eff-(a) = ∅          # Observed delete effects
eff?+(a) = La        # Possible add effects (not yet determined)
eff?-(a) = La        # Possible delete effects (not yet determined)
```

### Update Rules

#### When Action Succeeds (a applicable in state s → s')

```python
pre(a) = pre(a) ∩ bindP⁻¹(s, O)                    # Keep only literals true in s
eff+(a) = eff+(a) ∪ bindP(s' \ s, O)              # Add newly true literals
eff-(a) = eff-(a) ∪ bindP(s \ s', O)              # Add newly false literals
eff?+(a) = eff?+(a) ∩ bindP(s ∩ s', O)            # Narrow possible add effects
eff?-(a) = eff?(a) \ bindP(s ∪ s', O)             # Remove impossible delete effects
pre?(a) = {B ∩ bindP(s, O) | B ∈ pre?(a)}         # Update precondition constraints
```

#### When Action Fails (a not applicable in state s)

```python
pre?(a) = pre?(a) ∪ {{pre(a) \ bindP(s, O)}}      # Add new constraint: at least one missing literal is required
```

## CNF Formula Construction

### Precondition CNF Representation

The algorithm builds a CNF formula to represent precondition uncertainty:

```python
cnf_pre?(a) = ⋀(⋁xl) for B ∈ pre?(a), l ∈ B
```

For action applicability in state s with grounding O:
```python
cnf_pre?(a),O,s = cnf_pre?(a) ⋀ (¬xl) for l ∈ (⋃pre?(a)) \ bindP(s, O)
```

### Applicability Probability Calculation

The probability that action (a, O) is applicable in state s:

```python
pr(app(a, O, s) = 1) = {
    1,                                          if pre?(a) = ∅
    |SAT(cnf_pre?(a),O,s)| / |SAT(cnf_pre?(a))|,  otherwise
}
```

Where `SAT(cnf)` returns the set of satisfying assignments for the CNF formula.

## Information Gain Metrics

### Precondition Knowledge Gain

#### Success Case
When action succeeds from state s:
```python
preAppPotential(a, O, s) = |pre(a) \ bindP(s, O)|
```

#### Failure Case
When action fails from state s:
```python
preFailPotential(a, O, s) = 1 - (|SAT(cnf_pre?(a))| - |SAT(cnf'_pre?(a))|) / 2^|La|
```

### Effects Knowledge Gain

For successful execution:
```python
eff+Potential(a, O, s) = |eff?+ \ bindP(s, O)| / |La|
eff-Potential(a, O, s) = |eff?- ∩ bindP(s, O)| / |La|
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

#### Option 1: Greedy Selection
Select action that maximizes expected information gain:
```python
argmax_{(a,O)} E[X(a,O,s)]
```

#### Option 2: Probabilistic Selection
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

### Convergence Guarantees

- With enough negative samples, the algorithm converges to certain preconditions where:
  - `∀l ∈ pre(a), {l} ∈ pre?(a)`
  - Each learned precondition is guaranteed to be in the true model

### Learning Completeness

- Effects are fully learned when `|eff?+(a)| = |eff?-(a)| = 0`
- Preconditions are fully certain when all literals have singleton sets in `pre?(a)`

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
            self.eff_maybe_del[action] = La

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
- Implement proof of consistency for exported models
- Analyze worst-case learning bounds
- Extend to conditional effects and numeric fluents