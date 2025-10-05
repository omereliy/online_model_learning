# Lifted Fluent and Action Support

## Overview
The framework provides comprehensive support for both lifted (parameterized) and grounded representations of fluents and actions, enabling efficient representation and reasoning about action models at different levels of abstraction.

## Type-Safe Representations

The codebase uses type-safe classes from `src/core/pddl_types.py` for all PDDL representations:

```python
from src.core.pddl_types import (
    ParameterBoundLiteral,    # Lifted literals using action's parameter names
    GroundedFluent,           # Fully instantiated fluents
    GroundedAction,           # Grounded actions (replaces tuples)
    ParameterBinding          # Type-safe parameter bindings
)
from unified_planning.model import Action as LiftedAction  # Type alias

# Parameter-bound literal (uses action's specific parameter names)
lit = ParameterBoundLiteral('clear', ['?x'])
lit.to_string()  # "clear(?x)"

# Grounded fluent
fluent = GroundedFluent('clear', ['a'])
fluent.to_string()  # "clear_a"

# Grounded action (type-safe wrapper)
action = GroundedAction(pickup_action, ParameterBinding({'x': obj_a}))
action.to_string()  # "pick-up_a"
action.object_names()  # ['a']

# Lifted action (use UP's Action directly, LiftedAction is just type alias)
lifted_action: LiftedAction = pickup_action
```

**Key Points**:
- `ParameterBoundLiteral` preserves action's specific parameter names (e.g., `?x` from `pick-up(?x)`)
- `GroundedAction` replaces `Tuple[Action, Dict[str, Object]]` for type safety
- `LiftedAction` is a type alias for `unified_planning.model.Action` (no wrapper needed)

## Core Components

### CNF Manager (`src/core/cnf_manager.py`)
Enhanced to handle lifted fluents with variables and their grounded instances.

#### Key Features
- **Lifted Fluent Registration**: Track predicates with parameter types
- **Variable Mapping**: Bidirectional mapping between lifted and grounded forms
- **Clause Instantiation**: Ground lifted clauses with object bindings
- **Unified CNF Representation**: Mix lifted and grounded clauses in same formula

#### API Reference

##### Adding Lifted Fluents
```python
# Register a lifted predicate
cnf.add_lifted_fluent("on", ["?x", "?y"])  # Returns variable ID

# Alternative: Direct string format
cnf.add_fluent("on(?x,?y)", is_lifted=True)
```

##### Grounding Lifted Fluents
```python
# Create grounded instance
cnf.ground_lifted_fluent("on", ["a", "b"])  # Creates "on_a_b"

# Get all groundings of a predicate
groundings = cnf.get_lifted_groundings("on")
# Returns: {'on_a_b', 'on_b_c', ...}
```

##### Working with Lifted Clauses
```python
# Add lifted clause
lifted_clause = ["on(?x,?y)", "-clear(?y)"]
cnf.add_clause(lifted_clause, lifted=True)

# Instantiate with bindings
bindings = {"?x": "block1", "?y": "block2"}
grounded = cnf.instantiate_lifted_clause(lifted_clause, bindings)
# Result: ['on_block1_block2', '-clear_block2']
```

### PDDL Handler (`src/core/pddl_handler.py`)
Extended to work with lifted actions and predicates from PDDL domains.

#### Key Features
- **Lifted Action Storage**: Maintain action schemas with parameters
- **Predicate Structure Tracking**: Store predicate signatures
- **Flexible Precondition/Effect Access**: Return lifted or grounded forms
- **CNF Extraction**: Convert action preconditions to CNF clauses

#### API Reference

##### Accessing Lifted Actions
```python
# Get lifted action
action = handler.get_lifted_action("pick-up")

# Get action preconditions (lifted form)
preconds = handler.get_action_preconditions("pick-up", lifted=True)
# Returns: {'clear(?x)', 'ontable(?x)', 'handempty'}

# Get action effects (lifted form)
add_eff, del_eff = handler.get_action_effects("pick-up", lifted=True)
# add_eff: {'holding(?x)'}
# del_eff: {'clear(?x)', 'ontable(?x)', 'handempty'}
```

##### Working with Predicates
```python
# Get predicate structure
structure = handler.get_lifted_predicate_structure("on")
# Returns: ("on", ["?x", "?y"])

# Create lifted representations
fluent_str = handler.create_lifted_fluent_string("on", ["?x", "?y"])
# Returns: "on(?x,?y)"

action_str = handler.create_lifted_action_string("pick-up", ["?x"])
# Returns: "pick-up(?x)"
```

##### Grounding and Lifting Operations
```python
from src.core.binding_operations import FluentBinder

# Initialize binder
binder = FluentBinder(handler)

# bindP⁻¹: Ground parameter-bound literals with objects
grounded = binder.ground_literals({'clear(?x)', 'on(?x,?y)'}, ['a', 'b'])
# Returns: {'clear_a', 'on_a_b'}

# bindP: Lift grounded fluents to parameter-bound form
lifted = binder.lift_fluents({'clear_a', 'on_a_b'}, ['a', 'b'])
# Returns: {'clear(?x)', 'on(?x,?y)'}

# These operations are inverse:
# bindP(bindP⁻¹(F, O), O) = F
# bindP⁻¹(bindP(f, O), O) = f
```

##### CNF Extraction
```python
# Extract preconditions as CNF clauses
cnf_clauses = handler.extract_lifted_preconditions_cnf("pick-up")
# Returns: [['clear(?x)'], ['ontable(?x)'], ['handempty']]
```

## Integration Examples

### Example 1: Learning Lifted Action Models
```python
from src.core.cnf_manager import CNFManager
from src.core.pddl_handler import PDDLHandler

# Initialize components
cnf = CNFManager()
handler = PDDLHandler()
handler.parse_domain_and_problem(domain_file, problem_file)

# Track uncertainty for lifted action
action_name = "stack"
action = handler.get_lifted_action(action_name)

# Create CNF for precondition uncertainty
for pred_name in ["clear", "holding"]:
    structure = handler.get_lifted_predicate_structure(pred_name)
    if structure:
        cnf.add_lifted_fluent(pred_name, structure[1])

# Add constraints from observations
# Success: stack(?x,?y) succeeded with x=a, y=b
success_clause = ["holding(?x)", "clear(?y)"]
cnf.add_clause(success_clause, lifted=True)

# Failure: stack(?x,?y) failed with x=c, y=d
# At least one precondition was false
failure_bindings = {"?x": "c", "?y": "d"}
failure_clause = cnf.instantiate_lifted_clause(
    ["-holding(?x)", "-clear(?y)"],
    failure_bindings
)
cnf.add_clause(failure_clause)
```

### Example 2: Information Gain Calculation
```python
# Calculate information gain for testing an action
def calculate_info_gain(cnf, action_name, bindings):
    # Get current entropy
    current_entropy = cnf.get_entropy()

    # Create copies for success/failure scenarios
    success_cnf = cnf.copy()
    failure_cnf = cnf.copy()

    # Update with potential observations
    # ... (add appropriate clauses)

    # Calculate expected entropy reduction
    success_entropy = success_cnf.get_entropy()
    failure_entropy = failure_cnf.get_entropy()

    # Weight by probability
    success_prob = 0.5  # Estimated from model
    expected_entropy = (
        success_prob * success_entropy +
        (1 - success_prob) * failure_entropy
    )

    return current_entropy - expected_entropy
```

### Example 3: Mixed Lifted/Grounded Reasoning with Type-Safe Classes
```python
from src.core.cnf_manager import CNFManager
from src.core.pddl_handler import PDDLHandler
from src.core.binding_operations import FluentBinder
from src.core.pddl_types import ParameterBoundLiteral, GroundedFluent

# Start with lifted knowledge
cnf = CNFManager()
handler = PDDLHandler()
binder = FluentBinder(handler)

cnf.add_lifted_fluent("on", ["?x", "?y"])
cnf.add_lifted_fluent("clear", ["?x"])

# General rule: nothing can be on top of something that's clear
cnf.add_clause(["on(?x,?y)", "-clear(?y)"], lifted=True)

# Add specific grounded observation
cnf.add_clause(["clear_a"])  # Block a is clear
cnf.add_clause(["-on_b_a"])   # Block b is not on a

# Check consistency
if cnf.is_satisfiable():
    model = cnf.get_model()
    print(f"Consistent model: {model}")

# Use type-safe classes for manipulation
param_lit = ParameterBoundLiteral('clear', ['?x'])
grounded_lit = GroundedFluent.from_string('clear_a')

# Convert between representations
grounded_set = binder.ground_literals({param_lit.to_string()}, ['a'])
# Returns: {'clear_a'}
```

## Performance Considerations

### Grounding Strategy
- **Lazy Grounding**: Only ground predicates when needed
- **Caching**: Store grounded instances for reuse
- **Incremental Updates**: Update only affected groundings

### CNF Formula Management
- **Variable Reuse**: Share variables across lifted/grounded forms
- **Formula Minimization**: Periodically minimize to reduce size
- **Solution Caching**: Cache satisfiability results

### Best Practices
1. **Use Lifted Representation When Possible**: More compact and general
2. **Ground Only for Specific Tests**: Reduce formula size
3. **Batch Variable Creation**: Add all fluents before clauses
4. **Monitor Formula Growth**: Use `len(cnf.cnf.clauses)` to track size

## Debugging Tips

### Inspecting Lifted Structures
```python
# View all lifted fluents
for pred, params in cnf.lifted_fluents.items():
    print(f"{pred}({','.join(params)})")

# Check groundings
for lifted, grounded in cnf.lifted_to_grounded.items():
    print(f"{lifted} -> {grounded}")

# View CNF in readable form
print(cnf.to_string())
```

### Common Issues and Solutions
1. **Variable Not Found**: Ensure fluent is added before use
2. **Grounding Mismatch**: Check parameter order and object names
3. **Unsatisfiable Formula**: Use incremental solving to find conflicts
4. **Performance Degradation**: Monitor clause count and minimize regularly

## Future Enhancements
- **Type Hierarchy Support**: Handle subtype relationships
- **Quantified Formulas**: Support for universal/existential quantifiers
- **Incremental SAT Solving**: Reuse solver state across updates
- **Parallel Grounding**: Multi-threaded grounding for large domains