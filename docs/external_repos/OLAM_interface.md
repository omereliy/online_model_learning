# OLAM Interface Documentation

## Overview
OLAM (Online Learning of Action Models) - Lamanna et al.'s algorithm for learning STRIPS action models online.

## Key Characteristics
- **Optimistic initialization** - All actions initially assumed applicable
- **Learning from failures** - Refines preconditions when actions fail
- **No negative preconditions** - Domain limitation
- **Injective bindings required** - No repeated objects in parameters

## Repository Location
`/home/omer/projects/OLAM/`

## Main Classes

### `OLAM.Learner`
Primary learning class with action selection and model refinement.

#### Key Methods
- `select_action()` → (action_index, strategy)
- `learn()` → Update model from successful execution
- `learn_failed_action_precondition(simulator)` → Learn from failure
- `compute_not_executable_actionsJAVA()` → Get filtered actions

#### Key Attributes
- `operator_certain_predicates` - Learned certain preconditions
- `operator_uncertain_predicates` - Uncertain preconditions
- `operator_positive_effects` - Positive effects
- `operator_negative_effects` - Negative effects
- `model_convergence` - Convergence flag

### Supporting Classes
- `Util.PddlParser` - PDDL parsing
- `Util.Simulator` - State tracking
- `OLAM.Planner` - Planning with learned model

## Adapter Implementation
See `src/algorithms/olam_adapter.py` for complete integration.

### Key Conversions
- **State**: Set of strings → List of PDDL predicates
- **Action**: (name, objects) → "name(obj1,obj2)"
- **Model**: Operator-level storage (not grounded)

### Java Bypass Mode
When Java unavailable, uses Python fallback:
```python
def compute_not_executable_actionsJAVA():
    # Filter using learned model only
    return filtered_indices
```

## Validated Behaviors
Per Lamanna et al. paper:
1. ✅ Optimistic initialization (0 filtered initially)
2. ✅ Learning from failures (precondition refinement)
3. ✅ Hypothesis space reduction (monotonic)
4. ✅ Online learning without ground truth

## Domain Requirements
- No negative preconditions
- No conditional effects
- No disjunctive preconditions
- Injective parameter bindings

## Usage Example
```python
from src.algorithms.olam_adapter import OLAMAdapter

adapter = OLAMAdapter(domain_file, problem_file, bypass_java=True)
action, objects = adapter.select_action(state)
adapter.observe(state, action, objects, success, next_state)
model = adapter.get_learned_model()
```

## References
Lamanna, L., Saetti, A., Serafini, L., Gerevini, A., & Traverso, P. (2021). "Online Learning of Action Models for PDDL Planning"