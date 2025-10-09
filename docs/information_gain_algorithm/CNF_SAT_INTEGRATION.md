# CNF/SAT Solver Integration for Information-Theoretic Learning

## Overview
This document describes the CNF formula representation and SAT solver integration for the information-theoretic action model learning algorithm.

## Core Concepts

### CNF Formula Representation
The algorithm represents uncertainty about action preconditions and effects using CNF (Conjunctive Normal Form) formulas over fluent variables.

```python
# Example: move(A,B) precondition uncertainty
# Variables: on_A_B (1), clear_B (2), handempty (3)
# CNF: (clear_B OR NOT on_A_B) AND (handempty)
# Clauses: [[2, -1], [3]]
```

### Variable Mapping
Each fluent in a state is mapped to a unique boolean variable:
- `on(a,b)` → variable 1
- `clear(c)` → variable 2
- `handempty()` → variable 3

## Key Components

### 1. CNF Manager (`src/core/cnf_manager.py`) ✅ IMPLEMENTED
Complete CNF formula management with PySAT integration.

**Key Functions:**
- `add_fluent(fluent_str)` - Map fluent to variable ID
- `add_clause(clause)` - Add clause with fluent strings
- `count_solutions()` - Count satisfying assignments
- `get_all_solutions()` - Get all models as fluent sets
- `minimize_qm()` - Quine-McCluskey minimization
- `get_entropy()` - Calculate formula entropy
- **Lifted Support**: `add_lifted_fluent()`, `instantiate_lifted_clause()`
- **Constraint Operations (Oct 5, 2025)**:
  - `create_with_state_constraints(state_constraints)` - Create copy with state constraints
  - `add_constraint_from_unsatisfied(unsatisfied_literals)` - Add failure constraint
  - `build_from_constraint_sets(constraint_sets)` - Build CNF from pre?(a)
  - `count_models_with_constraints(state_constraints)` - Model counting with constraints
  - `clear_formula()` - Clear clauses but preserve variables
  - `has_clauses()` - Check if formula has clauses
  - `add_unit_constraint(fluent, must_be_true)` - Add unit clause for fluent

### 2. PDDL Components (New Architecture - October 2025) ✅ REFACTORED
Clean layered architecture replacing monolithic PDDLHandler.

**Components:**
- **PDDLReader** (`src/core/pddl_io.py`) - Parse PDDL domain and problem files
- **LiftedDomainKnowledge** (`src/core/lifted_domain.py`) - Central domain representation with lifted actions/predicates
- **Grounding Utilities** (`src/core/grounding.py`) - Functional bindP/bindP⁻¹ operations for literal grounding/lifting
- **UPAdapter** (`src/core/up_adapter.py`) - Stateless conversions between UP and project types

**Key Functions:**
- `parse_pddl(domain_file, problem_file)` - Parse PDDL files (returns domain, initial_state)
- `domain.get_parameter_bound_literals(action_name)` - Get all literals for action schema
- `grounding.ground_literal_set(literals, objects)` - Ground literals (bindP⁻¹)
- `grounding.lift_fluent_set(fluents, objects, domain)` - Lift fluents (bindP)
- `domain.get_action(action_name)` - Access action schemas
- `domain.is_subtype(type1, type2)` - Type hierarchy operations

### 3. Future Components (To Be Implemented)
The following components will be created in `src/sat_integration/`:
- **Advanced CNF Builder**: Domain-specific formula construction
- **Incremental SAT Solving**: Reuse solver state
- **Parallel Model Counting**: Multi-threaded counting
- **Formula Learning**: Learn compact representations

**Key Functions:**
- `fluent_to_variable(fluent_name)` - Get variable ID for fluent
- `variable_to_fluent(variable_id)` - Get fluent name from variable
- `get_state_variables(state)` - Map entire state to variables

## Information Gain Calculation

### Entropy-Based Selection
The algorithm selects actions that maximize expected information gain:

```python
def calculate_information_gain(action, current_cnf, possible_outcomes):
    """Calculate expected information gain from executing action."""
    current_entropy = calculate_entropy(current_cnf)

    expected_entropy = 0
    for outcome, probability in possible_outcomes:
        updated_cnf = update_cnf_with_outcome(current_cnf, action, outcome)
        outcome_entropy = calculate_entropy(updated_cnf)
        expected_entropy += probability * outcome_entropy

    return current_entropy - expected_entropy

def calculate_entropy(cnf_formula):
    """Calculate entropy based on number of satisfying models."""
    num_models = sat_solver.count_models(cnf_formula)
    total_possible = 2 ** cnf_formula.num_variables
    if num_models == 0:
        return 0
    probability = num_models / total_possible
    return -probability * math.log2(probability)
```

## Architecture and Separation of Concerns

### Clean Separation (October 2025 Refactoring)
The implementation maintains strict separation between:

1. **Algorithm Logic** (`src/algorithms/information_gain.py`)
   - Information gain calculations
   - Action selection strategies
   - Model update rules
   - Observation processing

2. **CNF Management** (`src/core/cnf_manager.py`)
   - Formula construction and manipulation
   - SAT solving and model counting
   - Constraint handling
   - Variable mapping

3. **PDDL Components** (Layered Architecture)
   - **Domain Representation** (`src/core/lifted_domain.py`) - Parameter-bound literal generation, action schemas
   - **Grounding Operations** (`src/core/grounding.py`) - Fluent grounding/lifting (bindP/bindP⁻¹)
   - **PDDL I/O** (`src/core/pddl_io.py`) - Domain/problem parsing
   - **UP Integration** (`src/core/up_adapter.py`) - Format conversions

This separation ensures:
- Algorithm code focuses only on learning logic
- CNF operations are centralized and reusable
- PDDL handling is consistent across the framework
- Testing is modular and focused
- Clean data flow: UP → Adapter → Domain → Grounding → Algorithm

## Algorithm Workflow

### 1. Initialization
```python
# Initialize CNF formulas for each action
for action in actions:
    precond_cnf[action] = CNFFormula(state_variables)
    add_effect_cnf[action] = CNFFormula(state_variables)
    del_effect_cnf[action] = CNFFormula(state_variables)
```

### 2. Action Selection
```python
def select_action(self, state):
    """Select action with highest expected information gain."""
    best_action = None
    best_gain = -float('inf')

    for action in self.applicable_actions(state):
        gain = self.calculate_information_gain(action, state)
        if gain > best_gain:
            best_gain = gain
            best_action = action

    return best_action
```

### 3. Model Update
```python
def observe(self, state, action, next_state, success):
    """Update CNF formulas based on observation."""
    if success:
        # Update effect formulas
        self.update_effect_cnf(action, state, next_state)
        # Confirm preconditions were satisfied
        self.confirm_preconditions(action, state)
    else:
        # Add constraints to precondition formula
        self.update_precondition_cnf(action, state, failed=True)
```

## PySAT Integration Details

### Solver Configuration
```python
from pysat.solvers import Minisat22
from pysat.formula import CNF

# Initialize solver
solver = Minisat22()

# Add CNF formula
cnf = CNF()
cnf.append([1, -2, 3])  # (x1 OR NOT x2 OR x3)
cnf.append([-1, 2])     # (NOT x1 OR x2)

# Count models (approximate for large formulas)
num_models = count_models_approx(cnf)
```

### Model Counting Strategies
1. **Exact counting**: For small formulas (<20 variables)
2. **Approximate counting**: Using sampling for larger formulas
3. **Cached results**: Store counts for repeated subformulas

## Performance Optimizations

### 1. Formula Caching
```python
class CNFCache:
    def __init__(self):
        self.cache = {}

    def get_model_count(self, formula_hash):
        return self.cache.get(formula_hash)

    def store_model_count(self, formula_hash, count):
        self.cache[formula_hash] = count
```

### 2. Incremental Solving
- Reuse solver state between calls
- Add/remove clauses incrementally
- Use solver assumptions for temporary constraints

### 3. Formula Simplification
- Apply unit propagation before solving
- Remove subsumed clauses
- Merge equivalent variables

## Example Usage

```python
# Create CNF formula for move(a,b) preconditions
mapper = VariableMapper()
cnf_builder = CNFBuilder(mapper)

# Initial uncertainty: any combination possible
precond_cnf = CNFFormula()

# Observe failure: move(a,b) failed in state {on(a,table), on(b,c)}
failed_state = [mapper.fluent_to_variable('on_a_table'),
                mapper.fluent_to_variable('on_b_c')]
cnf_builder.add_negative_example(precond_cnf, 'move_a_b', failed_state)

# Count remaining possibilities
solver = SATSolver('minisat')
num_models = solver.count_models(precond_cnf)
print(f"Remaining precondition possibilities: {num_models}")
```

## Integration with Unified Planning

### State Conversion
```python
def up_state_to_variables(up_state, mapper):
    """Convert Unified Planning state to CNF variables."""
    variables = []
    for fluent in up_state.get_all_fluents():
        if up_state.get_value(fluent).is_true():
            var_id = mapper.fluent_to_variable(str(fluent))
            variables.append(var_id)
    return variables
```

### Action Grounding
```python
def ground_action_for_cnf(up_action, mapper):
    """Ground UP action for CNF variable mapping."""
    action_name = up_action.action.name
    params = [str(param) for param in up_action.actual_parameters]
    return f"{action_name}_{('_'.join(params))}"
```