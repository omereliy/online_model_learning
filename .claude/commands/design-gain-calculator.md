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
