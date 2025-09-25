# Implementation Tasks for Online Model Learning Framework

## Project Overview
Building an experiment framework to compare three online action model learning algorithms:
1. **OLAM** - Baseline (Python implementation)
2. **ModelLearner** - Optimistic Exploration (Python implementation)  
3. **Information-Theoretic** - Novel CNF-based approach with SAT solving

## Technology Stack
- **Unified Planning Framework** - PDDL parsing and planning integration
- **PySAT** - SAT solver (Minisat) for CNF formula manipulation
- **Python 3.9** - In conda environment "action-learning"
- **SLURM** - HPC job scheduling

## Implementation Phases

### Phase 1: Core Infrastructure with UP and PySAT

**Files to create:**
- `src/core/cnf_manager.py`
- `src/core/pddl_handler.py`

**Requirements:**
1. **CNF Manager** (`src/core/cnf_manager.py`)
   - Fluent to variable ID bidirectional mapping
   - CNF formula manipulation (add/remove clauses)
   - Solution enumeration using Minisat22
   - Solution counting for probability calculations
   - Formula minimization (Quine-McCluskey or Espresso)
   - String representation for debugging

2. **PDDL Handler** (`src/core/pddl_handler.py`)
   - Parse PDDL files using Unified Planning
   - Convert UP objects to internal representation
   - Handle grounded predicates
   - Support negative preconditions
   - Export learned models back to PDDL

### Phase 2: Information-Theoretic Algorithm with SAT

**File to read:**
- `docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md`(read)
- `docs/information_gain_algorithm/CNF_SAT_INTEGRATION.md`(read)
- 
**Files to update/create:**
- `src/algorithms/information_gain.py` (update)
- `src/algorithms/cnf_tracker.py` (create)

**Requirements:**
1. **Information Gain Algorithm Enhancement**
   - Replace set operations with CNF formulas
   - Implement precondition CNF tracking: `(p1 ∨ p2) ∧ (¬p3 ∨ p4)`
   - Calculate applicability probability via SAT counting
   - Expected information gain with CNF uncertainty

2. **CNF Tracker** (`src/algorithms/cnf_tracker.py`)
   - Track precondition uncertainty as CNF formula
   - Update CNF on successful execution (remove impossible preconditions)
   - Update CNF on failed execution (add disjunctive constraints)
   - Efficient solution counting for probability calculation
   - Formula minimization after updates

### Phase 3: External Algorithm Integration

**Files to create:**
- `src/algorithms/olam_adapter.py`
- `src/algorithms/optimistic_adapter.py`

**Requirements:**
1. **OLAM Adapter**
   - Import OLAM's main learning class
   - Convert UP problem representation to OLAM format
   - Implement BaseActionModelLearner interface
   - Handle OLAM's exploration strategy

2. **ModelLearner Adapter**
   - Import ModelLearner's optimistic exploration
   - Convert between UP and ModelLearner formats
   - Implement diverse planning integration
   - Maintain optimistic model estimates

### Phase 4: Planner Integration via UP

**Files to create:**
- `src/planners/up_planner.py`

**Requirements:**
1. **Unified Planning Planner Wrapper**
   - Use UP's Fast Downward integration
   - Implement diverse planning (multiple solutions)
   - Add timeout handling and error recovery
   - Support both optimal and satisficing planning
   - Parse and return plan objects

### Phase 5: Experiment Configuration

**Files to create:**
- `configs/domains/blocksworld.yaml`
- `configs/domains/gripper.yaml`
- `configs/domains/elevator.yaml`
- `configs/algorithms.yaml`
- `configs/experiment.yaml`

**Requirements:**
1. **Domain Configurations**
   - PDDL domain and problem files
   - Initial states and goals
   - Domain-specific parameters

2. **Algorithm Configurations**
   - SAT solver settings (timeout, max solutions)
   - CNF minimization frequency
   - Exploration parameters
   - Learning rates and thresholds

3. **Experiment Metrics**
   - Sample complexity
   - Model accuracy (precision/recall)
   - CNF formula size
   - Solution count evolution
   - Runtime performance
   - Minimization effectiveness

### Phase 6: HPC Deployment

**Files to create:**
- `slurm/run_single.sh`
- `slurm/run_batch.sh`
- `scripts/collect_results.py`

**Requirements:**
1. **SLURM Scripts**
   - Conda environment activation
   - Resource allocation (memory, CPUs, time)
   - Array jobs for multiple experiments
   - Checkpointing and restart capability

2. **Results Collection**
   - Aggregate results from multiple runs
   - Statistical analysis
   - Generate plots and tables
   - Export to CSV/JSON

## Key Dependencies

```bash
# Core dependencies
pip install unified-planning[fast-downward,tamer]
pip install python-sat        # PySAT for Minisat
pip install pyeda             # Optional: for Espresso minimization
pip install pyyaml            # Configuration files
pip install pandas matplotlib # Analysis and visualization

# Development dependencies
pip install pytest pytest-cov # Testing
pip install black flake8      # Code formatting
```

## CNF Manager Example Implementation

```python
from pysat.solvers import Minisat22
from pysat.formula import CNF
from typing import List, Dict, Set, Tuple

class CNFManager:
    """Manages CNF formulas for precondition/effect uncertainty."""
    
    def __init__(self):
        self.fluent_to_var: Dict[str, int] = {}
        self.var_to_fluent: Dict[int, str] = {}
        self.cnf = CNF()
        self.next_var = 1
    
    def add_fluent(self, fluent_str: str) -> int:
        """Map fluent string to variable ID."""
        if fluent_str not in self.fluent_to_var:
            self.fluent_to_var[fluent_str] = self.next_var
            self.var_to_fluent[self.next_var] = fluent_str
            self.next_var += 1
        return self.fluent_to_var[fluent_str]
    
    def add_clause(self, clause: List[str]):
        """Add clause with fluent strings (prefix '-' for negation)."""
        var_clause = []
        for lit in clause:
            if lit.startswith('-'):
                var_clause.append(-self.add_fluent(lit[1:]))
            else:
                var_clause.append(self.add_fluent(lit))
        self.cnf.append(var_clause)
    
    def count_solutions(self) -> int:
        """Count all satisfying assignments."""
        solver = Minisat22(bootstrap_with=self.cnf)
        count = 0
        while solver.solve():
            count += 1
            model = solver.get_model()
            # Block current solution
            solver.add_clause([-lit for lit in model])
        return count
    
    def get_all_solutions(self) -> List[Set[str]]:
        """Get all satisfying assignments as sets of true fluents."""
        solutions = []
        solver = Minisat22(bootstrap_with=self.cnf)
        while solver.solve():
            model = solver.get_model()
            solution = {
                self.var_to_fluent[abs(lit)] 
                for lit in model 
                if lit > 0 and abs(lit) in self.var_to_fluent
            }
            solutions.append(solution)
            solver.add_clause([-lit for lit in model])
        return solutions
    
    def minimize(self):
        """Minimize CNF formula using Quine-McCluskey."""
        # Implementation using pyeda or custom QM algorithm
        pass
    
    def to_string(self) -> str:
        """Human-readable CNF representation."""
        clauses = []
        for clause in self.cnf.clauses:
            literals = []
            for lit in clause:
                var = abs(lit)
                if var in self.var_to_fluent:
                    fluent = self.var_to_fluent[var]
                    literals.append(f"¬{fluent}" if lit < 0 else fluent)
            clauses.append(f"({' ∨ '.join(literals)})")
        return ' ∧ '.join(clauses)
```

## Testing Strategy

1. **Unit Tests** - Test each component in isolation
2. **Integration Tests** - Test algorithm adapters with UP
3. **System Tests** - Full experiments on toy problems
4. **Performance Tests** - Benchmark SAT solving and planning

## Success Criteria

- All three algorithms run on same domains
- CNF formulas correctly track uncertainty
- SAT solver provides accurate probability estimates
- Results reproducible across local and HPC environments
- Metrics clearly show algorithm differences

## Notes

- Start with simple domains (3-4 objects) for debugging
- Log CNF formulas at each step for verification
- Monitor SAT solver performance (may need limits on solution enumeration)
- Consider incremental SAT solving for efficiency
