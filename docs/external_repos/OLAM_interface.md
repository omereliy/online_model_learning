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

## Experiment Requirements

### Java JDK (REQUIRED)
OLAM requires Oracle JDK 17+ for action filtering via `compute_not_executable_actions.jar`.

**Installation**:
1. Download from: https://www.oracle.com/java/technologies/downloads/
2. Extract to `/home/omer/projects/OLAM/Java/jdk-*/`
3. Verify: `ls /home/omer/projects/OLAM/Java/*/bin/java`

**Note**: No Python bypass available in native OLAM. Our adapter supports bypass mode for testing only.

### External Planners
- FastDownward: Required for planning
- FastForward: Used for action grounding

See [OLAM EXPERIMENT_GUIDE.md](/home/omer/projects/OLAM/EXPERIMENT_GUIDE.md) for full setup.

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

# Production: Use Java (default)
adapter = OLAMAdapter(
    domain_file,
    problem_file,
    bypass_java=False  # Use OLAM's bundled Java
)

# Testing only: Bypass Java
adapter = OLAMAdapter(domain_file, problem_file, bypass_java=True)

action, objects = adapter.select_action(state)
adapter.observe(state, action, objects, success, next_state)
model = adapter.get_learned_model()
```

## Experiment Integration

### Adapter vs Native OLAM
Our adapter integrates OLAM as a library for algorithm comparison research. Key differences from native OLAM experiments:

| Aspect | Native OLAM | Our Adapter |
|--------|-------------|-------------|
| Java | Required | Optional (testing) |
| Workflow | Multi-instance sequential | Single problem |
| Results | Excel + per-instance | CSV/JSON unified |
| PDDL Directory | Persistent | Temporary |
| Transfer Learning | Yes | No |

### Configuration Mapping
**Native OLAM** (`Configuration.py`):
- `TIME_LIMIT_SECONDS`, `MAX_ITER`, `PLANNER_TIME_LIMIT`
- `NEG_EFF_ASSUMPTION`, `OUTPUT_CONSOLE`

**Our Framework** (YAML):
- `stopping_criteria.max_runtime_seconds`
- `algorithm_params.olam.max_iterations`

### For Detailed Analysis
See [OLAM Experiment Review](../reviews/experimenting_OLAM_review.md) for:
- Complete file structure comparison
- Code misalignment analysis
- Configuration differences
- Result format conversions

## References
Lamanna, L., Saetti, A., Serafini, L., Gerevini, A., & Traverso, P. (2021). "Online Learning of Action Models for PDDL Planning"