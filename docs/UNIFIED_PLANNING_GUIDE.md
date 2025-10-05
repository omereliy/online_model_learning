# Unified Planning Framework Guide

## Overview

This guide explains how the Unified Planning (UP) Framework is used throughout the online model learning codebase, focusing on its unique expression tree structure and how it differs from simpler set-based representations.

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Expression Tree Structure](#expression-tree-structure)
3. [Lifted vs Grounded vs Parameter-Bound](#lifted-vs-grounded-vs-parameter-bound)
4. [Preconditions and Effects](#preconditions-and-effects)
5. [String Conversion and Simplification](#string-conversion-and-simplification)
6. [Code Flow in Our Project](#code-flow-in-our-project)
7. [Common Patterns and Examples](#common-patterns-and-examples)
8. [Debugging Tips](#debugging-tips)

## Core Concepts

### Key UP Classes

```python
from unified_planning.model import Problem, Action, Fluent, Object
from unified_planning.model.fnode import FNode
from unified_planning.io import PDDLReader, PDDLWriter
```

- **Problem**: Container for domain + problem specification
- **Action**: Action schema with parameters, preconditions, effects
- **Fluent**: State variable (predicate) that can be true/false
- **Object**: Domain object (e.g., block 'a', 'b', 'c')
- **FNode**: Expression tree node representing logical formulas

### The Critical Difference: Expression Trees vs Sets

**What you might expect:**
```python
action.preconditions = {"clear(x)", "ontable(x)", "handempty"}  # ❌ NOT how UP works
```

**What UP actually gives you:**
```python
action.preconditions = [FNode(AND, [FNode(clear(x)), FNode(ontable(x)), FNode(handempty)])]  # ✅
```

## Expression Tree Structure

### FNode Types

UP uses FNode (Formula Node) objects to represent logical expressions:

```
FNode Types:
├── Logical Operators
│   ├── AND: Conjunction of conditions
│   ├── OR: Disjunction of conditions
│   └── NOT: Negation
├── Fluent Expressions
│   ├── With parameters: clear(x)
│   └── Without parameters: handempty
└── Other
    ├── Constants: TRUE, FALSE
    └── Comparisons: =, <, > (for numeric fluents)
```

### Example Expression Tree

For action `pick-up(?x)` with precondition `(and (clear ?x) (ontable ?x) (handempty))`:

```
FNode(type=AND)
├── FNode(type=FLUENT_EXP, fluent=clear, args=[x])
├── FNode(type=FLUENT_EXP, fluent=ontable, args=[x])
└── FNode(type=FLUENT_EXP, fluent=handempty, args=[])
```

### Checking FNode Types

```python
# Essential FNode methods for type checking
expr.is_and()          # Is this an AND expression?
expr.is_or()           # Is this an OR expression?
expr.is_not()          # Is this a NOT expression?
expr.is_fluent_exp()   # Is this a fluent/predicate?
expr.is_equals()       # Is this an equality?
expr.is_constant()     # Is this a constant value?

# Accessing components
expr.args              # Sub-expressions (for AND/OR/NOT)
expr.fluent()          # Get the fluent (for fluent expressions)
```

## Type-Safe Representations

The codebase provides type-safe classes for PDDL representations:

```python
from src.core.pddl_types import ParameterBinding, ParameterBoundLiteral, GroundedFluent

# Parameter binding
binding = ParameterBinding({'x': Object('a', block_type)})
binding.object_names()  # ['a']

# Parameter-bound literal (uses action's parameter names)
lit = ParameterBoundLiteral('clear', ['?x'])
lit.to_string()  # "clear(?x)"

# Grounded fluent
fluent = GroundedFluent('clear', ['a'])
fluent.to_string()  # "clear_a"
```

**Key point**: `ParameterBoundLiteral` preserves action's parameter names.

## Lifted vs Grounded vs Parameter-Bound

### 1. Lifted (Schema Level)

**Definition**: Action/fluent templates with parameters (variables)

```python
# Lifted action: pick-up(?x - block)
# Lifted fluent: clear(?x)

# In UP:
action = problem.action("pick-up")
action.parameters  # [Parameter('x', type='block')]

# Precondition is lifted - contains parameter 'x'
precond = action.preconditions[0]  # (clear(x) and ontable(x) and handempty)
```

**String representation**: `"clear(?x)"` or `"pick-up(?x)"`

### 2. Parameter-Bound (During Execution)

**Definition**: Parameters bound to specific objects during action instantiation

```python
# During grounding, we bind ?x to object 'a'
binding = {'x': Object('a', type='block')}

# The action becomes: pick-up(a)
# Preconditions become: clear(a), ontable(a), handempty
```

### 3. Fully Grounded

**Definition**: All parameters replaced with concrete objects

```python
# Grounded action: pick-up_a
# Grounded fluents: clear_a, ontable_a, handempty

# In our string format:
grounded_action = "pick-up_a"
grounded_fluents = {"clear_a", "ontable_a", "handempty"}
```

### Transformation Flow

```
Lifted → Parameter-Bound → Grounded
pick-up(?x) → pick-up(x=a) → pick-up_a
clear(?x) → clear(x=a) → clear_a
```

## Preconditions and Effects

### How UP Stores Preconditions

```python
# action.preconditions is a LIST of FNode expression trees
action.preconditions  # type: List[FNode]

# Usually contains one expression (but can have multiple)
# Example: [(clear(x) AND ontable(x) AND handempty)]
```

### Extracting Fluents from Preconditions

```python
def extract_precondition_fluents(action):
    """Extract individual fluents from UP precondition expressions."""
    fluents = []

    for precond_expr in action.preconditions:
        fluents.extend(extract_from_expression(precond_expr))

    return fluents

def extract_from_expression(expr):
    """Recursively extract fluents from an FNode expression tree."""
    fluents = []

    if expr.is_and():
        # AND: extract from all arguments
        for arg in expr.args:
            fluents.extend(extract_from_expression(arg))

    elif expr.is_or():
        # OR: need special handling (represents alternatives)
        for arg in expr.args:
            fluents.extend(extract_from_expression(arg))

    elif expr.is_not():
        # NOT: extract negated fluent
        inner = expr.args[0]
        if inner.is_fluent_exp():
            fluents.append(('NOT', inner))

    elif expr.is_fluent_exp():
        # Base case: actual fluent
        fluents.append(('POS', expr))

    return fluents
```

### How UP Stores Effects

```python
# Effects are also FNode expressions
for effect in action.effects:
    effect.fluent      # The fluent being affected
    effect.value       # New value (True/False for add/delete)
    effect.condition   # Optional condition for conditional effects

    # Check if add or delete effect
    if effect.value.is_true():
        # Add effect
    else:
        # Delete effect
```

## String Conversion and Simplification

### Converting FNode to String

Our codebase converts UP's expression trees to strings for:
1. Human-readable output
2. CNF formula construction
3. Comparison and storage

```python
def expression_to_string(expr, parameters=None):
    """Convert UP FNode expression to string representation."""

    if expr.is_and():
        parts = [expression_to_string(arg, parameters) for arg in expr.args]
        return f"({' AND '.join(parts)})"

    elif expr.is_or():
        parts = [expression_to_string(arg, parameters) for arg in expr.args]
        return f"({' OR '.join(parts)})"

    elif expr.is_not():
        inner = expression_to_string(expr.args[0], parameters)
        return f"(NOT {inner})"

    elif expr.is_fluent_exp():
        fluent = expr.fluent()
        if expr.args:
            # Has parameters
            param_strs = []
            for arg in expr.args:
                if parameters and str(arg) in parameters:
                    # Lifted representation
                    param_strs.append(f"?{arg}")
                else:
                    # Grounded or object reference
                    param_strs.append(str(arg))
            return f"{fluent.name}({','.join(param_strs)})"
        else:
            # No parameters (propositional)
            return str(fluent.name)

    return str(expr)  # Fallback
```

### String Conversion with ExpressionConverter

The `ExpressionConverter` class centralizes all FNode → string conversions:

```python
from src.core.expression_converter import ExpressionConverter
from src.core.pddl_types import ParameterBinding

# Convert to parameter-bound literal (uses action's parameter names)
param_bound = ExpressionConverter.to_parameter_bound_string(
    expr, action.parameters  # ← Action's parameters preserve names
)
# Returns: "clear(?x)" for pick-up(?x) action

# Convert to grounded fluent
binding = ParameterBinding({'x': Object('a', 'block')})
grounded = ExpressionConverter.to_grounded_string(expr, binding)
# Returns: "clear_a"

# Extract CNF clauses
clauses = ExpressionConverter.to_cnf_clauses(expr, action.parameters)
# Returns: [["clear(?x)"], ["ontable(?x)"]] for AND expression
```

**Key point**: Always pass `action.parameters` to preserve parameter names.

### Binding Operations (bindP and bindP⁻¹)

The `FluentBinder` class implements grounding and lifting operations from the Information Gain algorithm:

```python
from src.core.binding_operations import FluentBinder
from src.core.pddl_handler import PDDLHandler

# Initialize binder
binder = FluentBinder(pddl_handler)

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

**Key points**:
- Object order matters: `['a', 'b']` maps `?x→a`, `?y→b`
- Handles negation: `¬clear(?x)` → `¬clear_a`
- Propositional fluents unchanged: `handempty` → `handempty`

### Simplification Process

```python
# Lifted form (with variables)
"clear(?x)"

# After parameter binding
"clear(a)"

# Grounded form (our convention using underscores)
"clear_a"

# Complex expressions
"(clear(?x) AND ontable(?x) AND handempty)"
# Simplified to set: {"clear_a", "ontable_a", "handempty"}
```

## Code Flow in Our Project

### 1. Parsing Phase (PDDLHandler)

```python
# src/core/pddl_handler.py
handler = PDDLHandler()
problem = handler.parse_domain_and_problem(domain_file, problem_file)

# UP creates Problem object with:
# - Fluents (predicates)
# - Actions (with FNode preconditions/effects)
# - Objects (domain objects)
# - Initial state and goals
```

### 2. Extraction Phase

```python
# Extract lifted actions
for action in problem.actions:
    handler._lifted_actions[action.name] = action

    # Store lifted predicates
    for fluent in problem.fluents:
        if fluent.arity > 0:  # Has parameters
            handler._lifted_predicates[fluent.name] = (fluent.name, param_types)
```

### 3. Grounding Phase

```python
# Generate all possible groundings
for action in problem.actions:
    param_bindings = get_all_bindings(action.parameters, problem.objects)
    for binding in param_bindings:
        grounded_action = ground_action(action, binding)
        handler._grounded_actions.append((action, binding))
```

### 4. CNF Conversion Phase

```python
# Convert preconditions to CNF clauses
def extract_lifted_preconditions_cnf(action_name):
    action = handler._lifted_actions[action_name]
    cnf_clauses = []

    for precond in action.preconditions:
        # Traverse expression tree
        clauses = extract_clauses_from_expression(precond)
        cnf_clauses.extend(clauses)

    return cnf_clauses
```

## Common Patterns and Examples

### Pattern 1: Extracting Preconditions as a Set

```python
def get_precondition_set(action):
    """Convert UP preconditions to a simple set of strings."""
    precond_set = set()

    for precond_expr in action.preconditions:
        # Handle the common case: AND of fluents
        if precond_expr.is_and():
            for arg in precond_expr.args:
                if arg.is_fluent_exp():
                    fluent_str = fluent_to_string(arg)
                    precond_set.add(fluent_str)
        elif precond_expr.is_fluent_exp():
            # Single fluent
            precond_set.add(fluent_to_string(precond_expr))

    return precond_set

# Usage
action = problem.action("pick-up")
preconds = get_precondition_set(action)
# Result: {"clear(x)", "ontable(x)", "handempty"}
```

### Pattern 2: Grounding a Lifted Action

```python
def ground_action(action, objects):
    """Ground a lifted action with specific objects."""
    # Create binding
    binding = {}
    for param, obj in zip(action.parameters, objects):
        binding[param.name] = obj

    # Ground preconditions
    grounded_preconds = []
    for precond in action.preconditions:
        grounded = substitute_in_expression(precond, binding)
        grounded_preconds.append(grounded)

    return grounded_preconds
```

### Pattern 3: Checking Applicability

```python
def is_action_applicable(action, state, binding):
    """Check if action is applicable in state with given binding."""

    for precond_expr in action.preconditions:
        if not evaluate_expression(precond_expr, state, binding):
            return False

    return True

def evaluate_expression(expr, state, binding):
    """Evaluate expression in given state with binding."""

    if expr.is_and():
        return all(evaluate_expression(arg, state, binding)
                  for arg in expr.args)

    elif expr.is_or():
        return any(evaluate_expression(arg, state, binding)
                  for arg in expr.args)

    elif expr.is_not():
        return not evaluate_expression(expr.args[0], state, binding)

    elif expr.is_fluent_exp():
        # Ground the fluent with binding
        grounded_fluent = ground_fluent(expr, binding)
        return grounded_fluent in state

    return False
```

## Debugging Tips

### 1. Inspecting Expression Structure

```python
def debug_expression(expr, indent=0):
    """Print expression tree structure for debugging."""
    prefix = "  " * indent

    if expr.is_and():
        print(f"{prefix}AND:")
        for arg in expr.args:
            debug_expression(arg, indent + 1)
    elif expr.is_or():
        print(f"{prefix}OR:")
        for arg in expr.args:
            debug_expression(arg, indent + 1)
    elif expr.is_not():
        print(f"{prefix}NOT:")
        debug_expression(expr.args[0], indent + 1)
    elif expr.is_fluent_exp():
        print(f"{prefix}FLUENT: {expr.fluent().name}")
        if expr.args:
            print(f"{prefix}  params: {expr.args}")
```

### 2. Common Errors and Solutions

**Error**: `AttributeError: 'FNode' object has no attribute 'fluent'`
- **Cause**: Trying to access fluent() on non-fluent expression
- **Solution**: Check `expr.is_fluent_exp()` first

**Error**: `AssertionError` when calling `fluent()`
- **Cause**: Expression is not a fluent expression (might be AND/OR/NOT)
- **Solution**: Traverse tree properly, only call fluent() on leaf nodes

**Error**: Preconditions appearing as single complex string
- **Cause**: Not traversing the expression tree
- **Solution**: Use recursive extraction as shown above

### 3. Useful Debug Prints

```python
# See what UP parsed
print(f"Action: {action.name}")
print(f"Parameters: {action.parameters}")
print(f"Precondition type: {type(action.preconditions)}")
print(f"Precondition count: {len(action.preconditions)}")
for i, p in enumerate(action.preconditions):
    print(f"  Precond {i}: {p}")
    print(f"    Type: {type(p)}")
    print(f"    Is AND?: {p.is_and()}")
    if p.is_and():
        print(f"    Components: {p.args}")
```

## Key Takeaways

1. **UP uses expression trees, not sets** - Always traverse the tree structure
2. **Check expression type before accessing** - Use `is_and()`, `is_fluent_exp()`, etc.
3. **Lifted → Grounded is a multi-step process** - Parameters → Bindings → Strings
4. **Our string convention** - Use underscores for grounded (`clear_a`) vs parameters for lifted (`clear(?x)`)
5. **Preconditions are a list** - Usually one element, but can have multiple
6. **Effects need value checking** - Check if True (add) or False (delete)
7. **Recursive traversal is key** - Most operations require walking the expression tree

## Integration with CNF Manager

The expression tree structure directly impacts how we build CNF formulas:

```python
# Expression: (clear(x) AND ontable(x)) OR handempty
# Becomes CNF: (clear(x) OR handempty) AND (ontable(x) OR handempty)

def expression_to_cnf(expr):
    """Convert UP expression to CNF clauses."""
    if expr.is_and():
        # Each component becomes separate clause
        clauses = []
        for arg in expr.args:
            clauses.extend(expression_to_cnf(arg))
        return clauses
    elif expr.is_or():
        # Combine into single clause
        clause = []
        for arg in expr.args:
            # ... extract literals
        return [clause]
    # ... handle other cases
```

This is why the `extract_lifted_preconditions_cnf` method in PDDLHandler needs to carefully traverse the expression tree and convert it to the appropriate CNF representation for the SAT solver.