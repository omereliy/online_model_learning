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
