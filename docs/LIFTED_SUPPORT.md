# Lifted Fluent and Action Support

## Overview

The framework provides comprehensive support for both lifted (parameterized) and grounded representations of fluents and actions using a clean layered architecture.

**Architecture layers**:
1. **UP Layer**: Unified Planning types (FNode, Action, Problem)
2. **Domain Layer**: `LiftedDomainKnowledge` (central representation)
3. **Grounding Layer**: Functional operations in `grounding.py`
4. **Environment Layer**: `ActiveEnvironment` (execution only)

## Core Components

### 1. LiftedDomainKnowledge (`src/core/lifted_domain.py`)

Central domain representation supporting both complete and partial knowledge.

#### Data Structures

```python
from src.core.lifted_domain import (
    LiftedDomainKnowledge,  # Main domain representation
    LiftedAction,           # Action schema with parameters
    PredicateSignature,     # Predicate structure
    Parameter,              # Parameter with type
    ObjectInfo,             # Object with type
    TypeInfo                # Type with hierarchy
)

# Create or load domain
domain = LiftedDomainKnowledge('blocksworld')

# Query actions
action = domain.get_action('stack')  # LiftedAction
print(action.name)          # 'stack'
print(action.parameters)    # [Parameter('?x', 'block'), Parameter('?y', 'block')]
print(action.preconditions) # {'holding(?x)', 'clear(?y)'}
print(action.add_effects)   # {'on(?x,?y)', ...}
```

#### Key Features

**Type-Safe Representations**:
```python
# Parameter-bound literals (uses action's parameter names)
action = domain.get_action('stack')
preconds = action.preconditions  # Set[str]
# → {'holding(?x)', 'clear(?y)'}

# Partial knowledge support for learning
action.uncertain_preconditions = {'holding(?x)', 'clear(?y)'}  # Not yet certain
action.maybe_add_effects = {'on(?x,?y)'}  # Might be an effect
```

**Type Hierarchy Operations**:
```python
# Check subtype relationships
domain.is_subtype('block', 'object')  # True

# Get type ancestors
ancestors = domain.get_type_ancestors('block')  # ['object']

# Get objects of type (including subtypes)
blocks = domain.get_objects_of_type('block', include_subtypes=True)
```

**Parameter-Bound Literals (La)**:
```python
# Get all possible parameter-bound literals for an action
La = domain.get_parameter_bound_literals('stack')
# → {'on(?x,?y)', '¬on(?x,?y)', 'clear(?x)', '¬clear(?x)',
#    'holding(?x)', '¬holding(?x)', ...}

# Used by Information Gain algorithm for hypothesis space
```

### 2. Grounding Utilities (`src/core/grounding.py`)

Functional operations for converting between lifted and grounded representations.

#### Grounding Actions

```python
from src.core.grounding import (
    ground_action,           # Single action grounding
    ground_all_actions,      # Batch grounding
    GroundedAction          # Result type
)

# Ground single action
lifted = domain.get_action('stack')
grounded = ground_action(lifted, ['a', 'b'])

print(grounded.action_name)            # 'stack'
print(grounded.objects)                # ['a', 'b']
print(grounded.to_string())            # 'stack_a_b'
print(grounded.grounded_preconditions) # {'holding_a', 'clear_b'}
print(grounded.grounded_add_effects)   # {'on_a_b', ...}

# Ground all actions in domain
all_grounded = ground_all_actions(domain, require_injective=True)
# require_injective=True: skip stack(a,a) - same object twice
# → [GroundedAction('pick-up', ['a']), GroundedAction('stack', ['a','b']), ...]
```

#### Grounding/Lifting Literals (bindP and bindP⁻¹)

```python
from src.core.grounding import (
    ground_parameter_bound_literal,  # bindP⁻¹: lifted → grounded
    lift_grounded_fluent,           # bindP: grounded → lifted
    ground_literal_set,             # Batch bindP⁻¹
    lift_fluent_set                 # Batch bindP
)

# bindP⁻¹: Ground parameter-bound literals
grounded = ground_parameter_bound_literal('on(?x,?y)', ['a', 'b'])
# → 'on_a_b'

grounded_neg = ground_parameter_bound_literal('¬clear(?x)', ['a'])
# → '¬clear_a'

grounded_prop = ground_parameter_bound_literal('handempty', [])
# → 'handempty' (no change for propositional)

# bindP: Lift grounded fluents
lifted = lift_grounded_fluent('on_a_b', ['a', 'b'], domain)
# → 'on(?x,?y)'

# Batch operations
literals = {'on(?x,?y)', '¬clear(?x)', 'handempty'}
grounded_set = ground_literal_set(literals, ['a', 'b'])
# → {'on_a_b', '¬clear_a', 'handempty'}

fluents = {'on_a_b', '¬clear_a', 'handempty'}
lifted_set = lift_fluent_set(fluents, ['a', 'b'], domain)
# → {'on(?x,?y)', '¬clear(?x)', 'handempty'}

# Inverse property: bindP(bindP⁻¹(F, O), O) = F
assert lift_fluent_set(ground_literal_set(literals, ['a', 'b']), ['a', 'b'], domain) == literals
```

#### Parsing Grounded Actions

```python
from src.core.grounding import parse_grounded_action_string

# Parse from string
grounded = parse_grounded_action_string('stack_a_b', domain)
# → GroundedAction(action_name='stack', objects=['a', 'b'], ...)

# Validation
from src.core.grounding import validate_grounded_action
is_valid = validate_grounded_action(grounded, domain)  # True
```

### 3. UPAdapter (`src/core/up_adapter.py`)

Stateless bidirectional converter between UP types and project types.

```python
from src.core.up_adapter import UPAdapter
from src.core.pddl_io import PDDLReader

# Parse PDDL
problem = PDDLReader.parse_domain_and_problem(domain_file, problem_file)

# Convert UP state to fluent set
fluents = UPAdapter.up_state_to_fluent_set(problem.initial_values, problem)
# → {'clear_a', 'on_b_c', 'handempty'}

# Convert fluent set to UP state
state_dict = UPAdapter.fluent_set_to_up_state(fluents, problem)

# Get all grounded fluents
all_fluents = UPAdapter.get_all_grounded_fluents(problem)
# → ['clear_a', 'clear_b', 'on_a_b', 'on_b_a', ...]

# Get initial state directly
initial = UPAdapter.get_initial_state_as_fluent_set(problem)
```

### 4. PDDL I/O (`src/core/pddl_io.py`)

Read and write PDDL using UP + domain knowledge.

```python
from src.core.pddl_io import PDDLReader, parse_pddl
from src.core.lifted_domain import LiftedDomainKnowledge
from src.core.up_adapter import UPAdapter

# Convenience function
up_problem, initial_state = parse_pddl(domain_file, problem_file)
# → (UP Problem, Set[str] of initial fluents)

# Manual construction
reader = PDDLReader()
up_problem = reader.parse_domain_and_problem(domain_file, problem_file)

# Convert to domain knowledge
domain = LiftedDomainKnowledge.from_up_problem(up_problem, UPAdapter)

# Now use domain for learning/grounding
action = domain.get_action('stack')
grounded_actions = ground_all_actions(domain)
```

## Integration Examples

### Example 1: Complete Learning Pipeline

```python
from src.core.pddl_io import parse_pddl
from src.core.lifted_domain import LiftedDomainKnowledge
from src.core.up_adapter import UPAdapter
from src.core.grounding import ground_all_actions, ground_action
from src.environments.active_environment import ActiveEnvironment

# 1. Load domain
up_problem, initial_state = parse_pddl(domain_file, problem_file)
domain = LiftedDomainKnowledge.from_up_problem(up_problem, UPAdapter)

# 2. Create environment
env = ActiveEnvironment.from_pddl(domain_file, problem_file)

# 3. Initialize learner with partial knowledge
learned_domain = LiftedDomainKnowledge(domain.name)
for action_name, action in domain.lifted_actions.items():
    # Start with optimistic initialization (all preconditions possible)
    learned_action = LiftedAction(
        name=action.name,
        parameters=action.parameters,
        preconditions=set(),  # Start with no known preconditions
        uncertain_preconditions=domain.get_parameter_bound_literals(action_name),
        add_effects=action.add_effects,
        del_effects=action.del_effects
    )
    learned_domain.lifted_actions[action_name] = learned_action

# 4. Learning loop
state = env.get_state()
for iteration in range(max_iterations):
    # Select action using learned model
    action_to_test = select_action_with_info_gain(learned_domain, state)

    # Ground action
    lifted_action = learned_domain.get_action(action_to_test.name)
    grounded = ground_action(lifted_action, action_to_test.objects)

    # Execute
    success, next_state = env.execute(grounded.to_string())

    # Update learned model
    if success:
        # Refine preconditions (these were sufficient)
        update_preconditions(learned_domain, grounded, state)
    else:
        # Learn from failure (at least one precondition was false)
        learn_from_failure(learned_domain, grounded, state)

    state = next_state
```

### Example 2: Information Gain Calculation

```python
from src.core.lifted_domain import LiftedDomainKnowledge
from src.core.grounding import ground_literal_set, lift_fluent_set

def calculate_information_gain(domain: LiftedDomainKnowledge,
                               action_name: str,
                               objects: List[str],
                               state: Set[str]) -> float:
    """Calculate expected information gain from testing an action."""

    # Get action's parameter-bound literals (hypothesis space)
    La = domain.get_parameter_bound_literals(action_name)

    # Ground hypothesis space for this specific binding
    grounded_La = ground_literal_set(La, objects)

    # Calculate which fluents are true/false in current state
    true_fluents = grounded_La & state
    false_fluents = grounded_La - state

    # Estimate probability of success based on learned model
    action = domain.get_action(action_name)
    grounded_preconds = ground_literal_set(action.preconditions, objects)

    # Check if all known preconditions are satisfied
    if grounded_preconds <= state:
        p_success = 0.8  # High probability
    else:
        p_success = 0.2  # Low probability

    # Calculate entropy reduction
    current_entropy = len(action.uncertain_preconditions) if action.uncertain_preconditions else 0
    expected_entropy = p_success * calculate_success_entropy(action) + \
                      (1 - p_success) * calculate_failure_entropy(action)

    return current_entropy - expected_entropy
```

### Example 3: Model Validation

```python
from src.core.model_validator import ModelValidator
from src.core.lifted_domain import LiftedDomainKnowledge

# Load ground truth and learned model
ground_truth_domain = LiftedDomainKnowledge.from_up_problem(up_problem, UPAdapter)
learned_domain = load_learned_model()

# Validate learned model
validator = ModelValidator(domain_file)

for action_name in learned_domain.lifted_actions:
    learned_action = learned_domain.get_action(action_name)

    # Convert to format expected by validator
    learned_model = {
        action_name: {
            'preconditions': learned_action.preconditions,
            'add_effects': learned_action.add_effects,
            'del_effects': learned_action.del_effects
        }
    }

    # Compare against ground truth
    result = validator.compare_learned_model(learned_model)

    print(f"\n{action_name}:")
    print(f"  Precondition F1: {result.precondition_f1:.2f}")
    print(f"  Add Effect F1: {result.add_effect_f1:.2f}")
    print(f"  Overall Accuracy: {result.overall_accuracy:.2f}")
```

## Performance Considerations

### Grounding Strategy
- **Lazy Grounding**: Only ground actions when needed for execution
- **Caching**: Store grounded actions if used repeatedly
- **Injective Requirement**: Use `require_injective=True` to reduce grounding count (e.g., blocksworld with 10 blocks: ~900 vs ~1000 actions)

### Best Practices
1. **Use LiftedDomainKnowledge as Central Store**: Single source of truth
2. **Ground Only for Execution**: Keep reasoning at lifted level when possible
3. **Batch Operations**: Use `ground_all_actions()` instead of individual calls
4. **Type Hierarchy**: Leverage `get_objects_of_type(include_subtypes=True)` for flexibility

## Debugging Tips

### Inspecting Domain Structure

```python
# Summary statistics
stats = domain.summary()
print(stats)
# → {'types': 1, 'objects': 3, 'predicates': 4, 'actions': 4, ...}

# List all actions
for name, action in domain.lifted_actions.items():
    print(f"{action}")  # pick-up(?x), stack(?x, ?y), ...

# Check action details
action = domain.get_action('stack')
print(f"Parameters: {[str(p) for p in action.parameters]}")
print(f"Preconditions: {action.preconditions}")
print(f"Add Effects: {action.add_effects}")
print(f"Has partial knowledge: {action.has_partial_knowledge()}")
```

### Common Issues

**Issue**: Parameter index mismatch in grounding
```python
# Wrong: Objects don't match parameter count
ground_action(action_with_2_params, ['a'])  # ValueError

# Correct: Match parameter count
ground_action(action_with_2_params, ['a', 'b'])
```

**Issue**: Negation not preserved
```python
# Ensure negation is handled
literal = '¬clear(?x)'
grounded = ground_parameter_bound_literal(literal, ['a'])
assert grounded == '¬clear_a'  # Not 'clear_a'
```

**Issue**: Type compatibility
```python
# Check if object type matches parameter type
obj = domain.get_object('a')
param_type = action.parameters[0].type
is_compatible = domain.is_subtype(obj.type, param_type)
```

## Migration from Old Architecture

**Old code** (using PDDLHandler):
```python
# DEPRECATED - old architecture
handler = PDDLHandler()
handler.parse_domain_and_problem(domain_file, problem_file)
lifted = handler.get_lifted_action('stack')
grounded = handler.get_all_grounded_actions_typed()
```

**New code** (layered architecture):
```python
# NEW - clean architecture
from src.core.pddl_io import parse_pddl
from src.core.lifted_domain import LiftedDomainKnowledge
from src.core.up_adapter import UPAdapter
from src.core.grounding import ground_all_actions

up_problem, initial_state = parse_pddl(domain_file, problem_file)
domain = LiftedDomainKnowledge.from_up_problem(up_problem, UPAdapter)
lifted = domain.get_action('stack')
grounded = ground_all_actions(domain)
```

## See Also

- **UPAdapter**: `src/core/up_adapter.py` - UP ↔ Project conversions
- **LiftedDomainKnowledge**: `src/core/lifted_domain.py` - Domain representation
- **Grounding**: `src/core/grounding.py` - Functional grounding operations
- **PDDL I/O**: `src/core/pddl_io.py` - Read/write PDDL
- **ActiveEnvironment**: `src/environments/active_environment.py` - Execution interface
