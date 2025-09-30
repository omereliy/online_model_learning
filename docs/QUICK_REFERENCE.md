# Quick Reference - Code Patterns & Commands

## Commands

### Testing Commands
```bash
# Curated test suite (165 tests, 100% pass rate)
make test

# All tests including experimental (196 tests)
pytest tests/

# Quick tests (< 2 minutes)
make test-quick

# Specific module tests
make test-metrics
make test-integration
pytest tests/test_pddl_handler.py -v
pytest tests/test_cnf_manager.py -v
```

### Docker Commands
```bash
# Build all Docker images
make docker-build

# Run curated tests in Docker
make docker-test

# Quick tests without dependencies
make docker-test-quick

# Interactive development shell
make docker-shell

# Run experiments in container
make docker-experiment

# Jupyter notebook for analysis
make docker-notebook
```

### Development Commands
```bash
# Run CI pipeline locally
make ci-local

# Performance benchmarks
make benchmark
make benchmark-quick

# Code coverage
make coverage
make coverage-detailed

# Check implementation status
grep -r "TODO" src/ --include="*.py"
grep -r "NotImplementedError" src/ --include="*.py"
```

## Code Patterns

### State Format Conversions:
```python
# UP format: Problem/State objects
up_state.get_all_fluents()

# CNF variables: Integer IDs
mapper.fluent_to_variable('on_a_b') → 1

# OLAM format: Set of strings
{'on(a,b)', 'clear(c)', 'handempty()'}

# ModelLearner format: Dict
{"domain": {...}, "problem": {"init": {...}}}
```

### CNF Formula Manipulation with Lifted Support
```python
from src.core.cnf_manager import CNFManager
from src.core.pddl_handler import PDDLHandler

# CNF with lifted fluents
cnf = CNFManager()
cnf.add_lifted_fluent("on", ["?x", "?y"])
cnf.add_clause(["on(?x,?y)", "-clear(?y)"], lifted=True)

# Ground for specific objects
bindings = {"?x": "a", "?y": "b"}
grounded = cnf.instantiate_lifted_clause(["on(?x,?y)"], bindings)

# Count models for information gain
num_models = cnf.count_solutions()
entropy = cnf.get_entropy()
```

### Run Experiment with CNF Settings
```python
config = {
    'domain': 'blocksworld',
    'algorithms': ['information_gain'],
    'cnf_settings': {
        'solver': 'minisat',
        'minimize_formulas': True,
        'max_clauses': 1000
    }
}
```

## Import Patterns

### External Algorithm Imports
```python
import sys

# OLAM
sys.path.append('/home/omer/projects/OLAM')
from OLAM.Learner import Learner
from Util.PddlParser import PddlParser

# ModelLearner
sys.path.append('/home/omer/projects/ModelLearner/src')
from model_learner.ModelLearnerLifted import ModelLearnerLifted
```

### PySAT Import Pattern
```python
# Handle both pysat and python-sat packages
try:
    from pysat.solvers import Minisat22
    from pysat.formula import CNF
except ImportError:
    from pysat.solvers import Minisat22  # python-sat package
    from pysat.formula import CNF
```

### Unified Planning Import Pattern
```python
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import SequentialSimulator
from unified_planning.model import Problem, Action, FNode
```

## Debugging Patterns

### Common Issues & Solutions
| Issue | Solution |
|-------|----------|
| UP preconditions not sets | Use recursive FNode traversal |
| PySAT import errors | Try/except for both packages |
| Type hierarchy missing 'object' | Special case in `is_subtype_of()` |
| Test timeout/hanging | Check for Lock vs RLock |
| JSON serialization errors | Custom encoder for numpy types |
| State format mismatches | Verify conversion functions |

## Complete Working Examples

### CNF Formula with SAT Solving
```python
# Complete CNF workflow
from src.core.cnf_manager import CNFManager

cnf = CNFManager()

# Add fluents and build CNF: (clear_b OR NOT on_a_b)
cnf.add_clause(['clear_b', '-on_a_b'])

# Count models
count = cnf.count_solutions()
print(f"Number of satisfying assignments: {count}")

# Get all solutions
solutions = cnf.get_all_solutions()
for sol in solutions:
    print(f"Solution: {sol}")

# Calculate entropy
entropy = cnf.get_entropy()
print(f"Entropy: {entropy}")
```

### PDDL Parsing with UP
```python
from unified_planning.io import PDDLReader

reader = PDDLReader()
problem = reader.parse_problem(
    'benchmarks/olam-compatible/blocksworld/domain.pddl',
    'benchmarks/olam-compatible/blocksworld/p01.pddl'
)

# Access components
for action in problem.actions:
    print(f"Action: {action.name}")
    for param in action.parameters:
        print(f"  Param: {param.name} : {param.type}")
```

### UP Expression Tree Traversal
```python
from unified_planning.model import FNode

def traverse_expression(expr):
    """Recursively traverse UP expression tree."""
    if expr.is_and():
        return all(traverse_expression(arg) for arg in expr.args)
    elif expr.is_or():
        return any(traverse_expression(arg) for arg in expr.args)
    elif expr.is_not():
        return not traverse_expression(expr.arg(0))
    elif expr.is_fluent_exp():
        # Handle fluent
        return check_fluent_in_state(expr)
    else:
        return True
```

### OLAM State/Action Conversion
```python
# UP format to OLAM format
def up_state_to_olam(up_state):
    """Convert {'clear_a', 'on_a_b'} to ['(clear a)', '(on a b)']"""
    olam_state = []
    for fluent in up_state:
        parts = fluent.split('_')
        if len(parts) == 1:
            olam_state.append(f"({parts[0]})")
        elif len(parts) == 2:
            olam_state.append(f"({parts[0]} {parts[1]})")
        elif len(parts) == 3:
            olam_state.append(f"({parts[0]} {parts[1]} {parts[2]})")
    return olam_state

# Action conversion
def up_action_to_olam(action, objects):
    """Convert ('pick-up', ['a']) to 'pick-up(a)'"""
    if objects:
        return f"{action}({','.join(objects)})"
    else:
        return f"{action}()"
```

## External Dependencies
See [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) for complete external tool paths.