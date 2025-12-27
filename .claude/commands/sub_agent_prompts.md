# Sub-Agent Prompts for Information Gain Rewrite

## How to Use These Prompts

These are ready-to-use prompts for dispatching to sub-agents. Each sub-agent should:
1. Read the specified documentation
2. Design the interface/architecture (NOT implementation)
3. Return structured deliverables

Copy each prompt block and dispatch to a sub-agent before starting implementation.

---

## Sub-Agent 1: CNF Module Architecture

```
You are designing the CNF (Conjunctive Normal Form) management module for an action model learning system.

## Required Reading

First, read these files in the project:
1. docs/information_gain_algorithm/cnf/cnf_issues_analysis.md - Focus on sections CRIT-01, CRIT-02, CRIT-03, CRIT-04
2. docs/information_gain_algorithm/cnf/cnf_fix_strategy.md - Focus on "Phase 1: Core Architecture Fixes"

## Problem Summary

The current CNF implementation has critical bugs:
1. Treats p and ¬p as independent variables (should be paired with mutex)
2. Doesn't track primary vs auxiliary variables for projection
3. Model counting is unprojected (exponentially wrong)
4. Negation handling is confusing (¬ vs - mixed up)

## Your Task

Design a clean CNF module architecture that fixes these issues.

## Requirements

### 1. Literal Pair Model
- Each fluent f creates exactly TWO variables:
  - `l_var` (primary): "is f a precondition?"
  - `neg_l_var` (auxiliary): "is ¬f a precondition?"
- Mutex constraint `[-l_var, -neg_l_var]` auto-added at creation
- Only 3 valid states: (F,F), (T,F), (F,T) - never (T,T)

### 2. Variable Typing
- PRIMARY variables: count over these (projection set)
- AUXILIARY variables: project out (don't count)
- FIXED variables: propagated, removed from projection

### 3. Clause Types
- MUTEX: Binary clauses preventing (T,T)
- COVERAGE: "At least one of these IS a precondition"
- EXCLUSION: "This literal is NOT a precondition"
- INCLUSION: "This literal IS a precondition"

### 4. Clear API
- No ambiguity about what each method does
- Separate string literals from variable IDs
- Explicit negation handling

## Deliverables

Provide your response in this format:

### A. Data Structures

```python
# Define the key dataclasses and enums
# Include docstrings explaining each field
```

### B. CNFManager Interface

```python
# Define the class with method signatures
# Include comprehensive docstrings
# NO implementation bodies - just `pass` or `...`
```

### C. Invariants

List the invariants that must ALWAYS hold:
1. ...
2. ...

### D. Usage Examples

```python
# Show how the API would be used for common operations:
# 1. Creating a fluent pair
# 2. Adding a coverage clause
# 3. Adding an exclusion
# 4. Getting model count
```

### E. Interaction with Model Counting

Explain how this module interfaces with model counting:
- What does it provide?
- What does it require?

DO NOT write implementation code. Focus on clean interface design.
```

---

## Sub-Agent 2: Model Counting Architecture

```
You are designing the model counting module for a SAT-based action model learning system.

## Required Reading

First, read these files in the project:
1. docs/information_gain_algorithm/cnf/cnf_issues_analysis.md - Focus on section CRIT-03
2. docs/information_gain_algorithm/cnf/cnf_fix_strategy.md - Focus on "Phase 2: Model Counting Fixes"

## Problem Summary

The current model counting has critical bugs:
1. Counts ALL variable assignments (should project onto primary vars)
2. Modifies CNF directly for assumptions (should use solver's assumption mechanism)
3. Cache staleness issues in parallel execution
4. Wrong API usage for pyapproxmc

## Your Task

Design a clean model counting module that fixes these issues.

## Requirements

### 1. Projected Counting
- Count over PRIMARY variables only
- Auxiliary variables are projected out
- Fixed variables are excluded from projection set

### 2. Counting Methods
- Exact counting: enumerate all projected models
- Approximate counting: use ApproxMC with projection
- Adaptive: choose based on problem size

### 3. Assumptions Support
- Temporary constraints for "what-if" queries
- Use solver's native `solve(assumptions=[...])` 
- NO modification of the CNF formula
- Must be thread-safe

### 4. Caching Strategy
- Cache base counts (no assumptions)
- Invalidate on CNF modification
- NO cache sharing between parallel workers

## Deliverables

Provide your response in this format:

### A. Interface Definition

```python
# Define the model counter interface
# Could be abstract class or protocol
```

### B. Exact Counter

```python
# Interface for exact projected counting
```

### C. Approximate Counter

```python
# Interface for approximate counting with ApproxMC
```

### D. Adaptive Strategy

```python
# How to choose between exact and approximate
```

### E. Assumptions Handling

```python
# Interface for counting with temporary assumptions
# Show how it avoids CNF modification
```

### F. Parallel Safety

Explain how to ensure thread-safety:
1. What state is shared?
2. What state is per-worker?
3. How are caches handled?

### G. Usage Examples

```python
# Show common usage patterns:
# 1. Count all models (projected)
# 2. Count with assumptions
# 3. Adaptive counting
```

DO NOT write implementation code. Focus on clean interface design.
```

---

## Sub-Agent 3: Information Gain Calculator Architecture

```
You are designing the information gain calculation module for an action model learning system.

## Required Reading

First, read the current implementation in:
- src/algorithms/information_gain.py - specifically these methods:
  - _calculate_applicability_probability (lines ~220-270)
  - _calculate_potential_gain_success (lines ~272-330)
  - _calculate_potential_gain_failure (lines ~332-380)
  - _calculate_expected_information_gain (lines ~382-410)

Also read the mathematical specification in:
- docs/information_gain_algorithm/cnf/cnf_fix_strategy.md

## Problem Summary

The current gain calculation:
1. Is tightly coupled to CNF internals
2. Has duplicated logic between sequential and parallel
3. Mixes computation with state management
4. Is hard to test in isolation

## Your Task

Design a clean, testable, parallel-friendly gain calculation module.

## Requirements

### 1. Separation of Concerns
- Gain calculation should NOT know about CNF internals
- Use an abstract "model counter" interface
- Pure functions where possible (no side effects)

### 2. Mathematical Formulas
Document the exact formulas being computed:
- P(applicable) = ?
- Gain(success) = ?
- Gain(failure) = ?
- E[Gain] = ?

### 3. Testability
- Each gain component testable independently
- Mock-friendly interfaces (dependency injection)
- Clear input/output contracts

### 4. Parallel-Friendly
- Stateless computation functions
- All context passed as parameters
- No shared mutable state

## Deliverables

Provide your response in this format:

### A. Mathematical Specification

Document the exact formulas:
```
P(app(a,O,s) = 1) = ...
GainSuccess(a,O,s) = ...
GainFailure(a,O,s) = ...
E[Gain(a,O,s)] = ...
```

### B. Context Data Classes

```python
# Define the data needed for gain computation
# This is what gets passed to workers in parallel
```

### C. Counter Interface

```python
# Define the abstract interface for model counting
# That gain calculator depends on
```

### D. Gain Calculator Interface

```python
# Define the gain calculation interface
# Show how it composes the components
```

### E. Individual Components

```python
# Interface for each gain component:
# - ApplicabilityCalculator
# - SuccessGainCalculator  
# - FailureGainCalculator
```

### F. Parallel Execution Strategy

Explain how to parallelize:
1. What is the unit of parallel work?
2. What context needs to be serialized?
3. How are results aggregated?

### G. Testing Strategy

```python
# Show how to unit test each component
# With mock model counter
```

DO NOT write implementation code. Focus on clean interface design.
```

---

## Sub-Agent 4: Learner Orchestration Architecture

```
You are designing the main learner orchestrator for an action model learning system.

## Required Reading

First, read:
1. src/algorithms/information_gain.py - the InformationGainLearner class
2. docs/information_gain_algorithm/cnf/cnf_fix_strategy.md

## Problem Summary

The current InformationGainLearner:
1. Is ~900 lines - way too long
2. Mixes orchestration with computation
3. Has unclear state ownership
4. Is hard to test or extend

## Your Task

Design a thin orchestrator that delegates to specialized components.

## Requirements

### 1. Thin Orchestrator
- Main learner class should be <200 lines
- Only handles high-level flow
- Delegates all computation to components

### 2. Clear State Ownership
- Each piece of mutable state has ONE owner
- Explicit state transitions
- Easy to serialize for checkpointing

### 3. Component Composition
- Clear dependency injection
- Easy to swap implementations
- Mockable for testing

### 4. Learning Loop
The main loop is:
1. Select action (based on information gain)
2. Execute action (via environment)
3. Observe result (success/failure)
4. Update model
5. Check convergence
6. Repeat

## Deliverables

Provide your response in this format:

### A. Component Diagram

List all components and their responsibilities:
```
Learner (orchestrator)
├── ActionSelector: chooses next action
├── GainCalculator: computes information gain
├── ModelUpdater: updates model on observation
├── ConvergenceChecker: detects when to stop
└── ActionModels: per-action learned state
```

### B. Learner Interface

```python
# Main learner class interface
# <200 lines target
```

### C. State Management

```python
# Define what state exists and who owns it
# Show state transitions
```

### D. Dependency Injection

```python
# Show how components are composed
# And how to swap implementations
```

### E. Sequence Diagram

Describe one learning iteration step-by-step:
1. ...
2. ...

### F. Extension Points

How to extend for:
1. New selection strategies (epsilon-greedy, Boltzmann, etc.)
2. Different model counting backends
3. New update rules

### G. Configuration

```python
# Show configuration/options pattern
```

DO NOT write implementation code. Focus on clean interface design.
```

---

## After Sub-Agent Tasks

Once you have all four sub-agent designs:

1. **Review for Consistency**: Do the interfaces fit together?
2. **Identify Gaps**: Any missing components?
3. **Resolve Conflicts**: Any incompatible assumptions?
4. **Create Unified Design**: Combine into one coherent architecture

Then proceed with implementation following the unified design.
