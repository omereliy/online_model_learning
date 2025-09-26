# Quick Reference Guide - Updated for CNF/SAT Integration

## Essential Context for AI Agents

### Project Purpose
Compare three online action model learning algorithms on PDDL domains:
1. **OLAM**: Learn from both success/failure (Python wrapper)
2. **Optimistic**: Start optimistic, refine on failures (Python wrapper)
3. **Information Gain**: Novel CNF-based approach using SAT solvers

### Tech Stack
- **Unified Planning Framework**: PDDL parsing, planning, simulation
- **PySAT**: CNF formula manipulation and SAT solving (minisat)
- **External repos**: OLAM and ModelLearner (Python implementations)

### Key Files to Check First
```bash
# Project structure and requirements
/docs/DEVELOPMENT_RULES.md          # Updated implementation rules
/docs/LIFTED_SUPPORT.md             # Lifted fluents and actions documentation
/docs/external_repos/               # Interface docs for OLAM/ModelLearner
/README.md                          # Updated project overview with lifted examples
```

### Critical Implementation Details

#### CNF/SAT Integration (Currently in `src/core/`)
- `cnf_manager.py` ✅ - Complete CNF formula management with PySAT
- `pddl_handler.py` ✅ - PDDL parsing with UP and CNF extraction
- Future: `src/sat_integration/` for advanced features

#### Information-Theoretic Algorithm Must:
- Represent uncertainty as CNF formulas over fluent variables
- Use SAT solver for model counting and information gain
- Select actions that maximize expected entropy reduction
- Minimize formulas for performance

#### Unified Planning Integration (`src/planning/`)
- Use UP for all PDDL parsing and planning
- Convert between UP objects and internal representations
- Handle UP Problem/State/Action objects

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

### Common Tasks

#### 1. CNF Formula Manipulation with Lifted Support
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

#### 2. Run Experiment with CNF Settings
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

#### 3. Debug CNF Issues
```python
# Common problems:
1. Variable mapping inconsistent → Check variable_mapper
2. CNF formulas too large → Enable formula minimization
3. SAT solver timeout → Use approximate model counting
4. Memory issues → Cache formula evaluations
```

### File Creation Priority for CNF Integration
1. `src/core/cnf_manager.py` - CNF formulas with lifted support ✓
2. `src/core/pddl_handler.py` - PDDL parsing with lifted actions ✓
3. `src/algorithms/information_gain.py` - Main algorithm
4. `src/planning/unified_planning_interface.py` - UP integration
5. Everything else

### External Dependencies Paths
```python
OLAM_PATH = "/home/omer/projects/OLAM"
MODELLEARNER_PATH = "/home/omer/projects/ModelLearner"

# New dependencies
from unified_planning.io import PDDLReader
from pysat.solvers import Minisat22
from pysat.formula import CNF
```

### Quick CNF/SAT Testing
```python
# Test PySAT integration
from pysat.solvers import Minisat22
from pysat.formula import CNF

cnf = CNF()
cnf.append([1, -2])  # (x1 OR NOT x2)
cnf.append([2])      # x2

solver = Minisat22(bootstrap_with=cnf)
is_sat = solver.solve()
print(f"Satisfiable: {is_sat}")

# Test UP integration
from unified_planning.io import PDDLReader
reader = PDDLReader()
problem = reader.parse_problem('benchmarks/blocksworld/domain.pddl',
                              'benchmarks/blocksworld/p01.pddl')
print(f"Problem loaded: {problem.name}")
```

### CNF-Specific Debugging Checklist
- [ ] PySAT installed? `pip install python-sat`
- [ ] UP installed? `pip install unified-planning`
- [ ] Variable mapping consistent? Print variable assignments
- [ ] CNF formulas valid? Test with small examples
- [ ] SAT solver working? Test basic satisfiability

### Information-Theoretic Algorithm Workflow
1. **Initialize**: Create empty CNF formulas for all actions
2. **Select Action**: Calculate information gain via model counting
3. **Execute**: Use UP simulator or external environment
4. **Observe**: Update CNF formulas based on success/failure
5. **Minimize**: Optimize formulas for performance

### When Stuck on CNF Integration
1. Read `/docs/CNF_SAT_INTEGRATION.md` for detailed examples
2. Test PySAT separately with simple formulas
3. Verify variable mapping with print statements
4. Use formula minimization for large CNF formulas
5. Check SAT solver logs for timeout issues

### Minimal CNF Working Example
```python
# Test complete CNF workflow with current implementation
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