# Implementation Tasks for Online Model Learning Framework

## Project Overview
Building an experiment framework to compare three online action model learning algorithms:
1. **OLAM** - Baseline (Python implementation)
2. **ModelLearner** - Optimistic Exploration (Python implementation)
3. **Information-Theoretic** - Novel CNF-based approach with SAT solving

## 🎯 Quick Start for New Session

**For next agent starting fresh:**
1. Read `docs/DEVELOPMENT_RULES.md` first (mandatory)
2. Check `CLAUDE.md` for navigation help
3. Start with Phase 2: Information-Theoretic Algorithm
4. Use existing CNFManager and PDDLHandler - they're complete!

## 📝 MANDATORY UPDATE RULE

**After completing ANY implementation task:**
1. **Update this file** (`docs/IMPLEMENTATION_TASKS.md`) to reflect:
   - What was implemented (move to ✅ Completed)
   - Any new insights or changes needed for remaining tasks
   - Updated file paths and dependencies
2. **Request user approval** with a clear prompt like:
   > "I've completed [specific task]. Should I update IMPLEMENTATION_TASKS.md to mark this as complete and adjust remaining tasks? Please review the proposed changes carefully."
3. **Wait for explicit approval** - Do NOT auto-approve or assume consent
4. **Commit the documentation update** separately from code changes

## Current Implementation Status

### ✅ Completed Components

#### Core Infrastructure
- **CNF Manager** (`src/core/cnf_manager.py`) - Full CNF formula management with:
  - Fluent to variable mapping (including lifted fluents)
  - Solution enumeration and counting
  - Formula minimization (QM and Espresso)
  - Entropy calculation for uncertainty
  - Lifted fluent support with instantiation

- **PDDL Handler** (`src/core/pddl_handler.py`) - Complete PDDL processing with:
  - Unified Planning Framework integration
  - Expression tree (FNode) traversal for preconditions/effects
  - Type hierarchy with 'object' as root
  - Lifted action and predicate extraction
  - CNF conversion from complex preconditions
  - Negative precondition support

#### Documentation
- **Development Rules** with mandatory review and Git safety guidelines
- **CLAUDE.md** for context guidance in new sessions
- **Unified Planning Guide** explaining expression trees vs sets
- **Lifted Support Guide** for parameterized actions/fluents
- **Information Gain Algorithm** with negative precondition examples

#### Testing
- Comprehensive tests for CNF Manager lifted fluent support
- Tests for PDDL Handler type hierarchy and lifted actions
- Tests validating expression tree traversal

## Technology Stack
- **Unified Planning Framework** - PDDL parsing and planning integration (expression trees, NOT simple sets!)
- **PySAT/python-sat** - SAT solver (Minisat) for CNF formula manipulation
- **Python 3.9+** - In conda environment
- **SLURM** - HPC job scheduling (future)

## Implementation Phases

### Phase 2: Information-Theoretic Algorithm ⏳ NEXT PRIORITY

**Essential Reading Before Implementation:**
- `docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md` - Complete algorithm specification with negative preconditions
- `docs/UNIFIED_PLANNING_GUIDE.md` - Critical for understanding UP's expression trees
- `docs/LIFTED_SUPPORT.md` - For handling parameterized actions

**Files to create:**
- `src/algorithms/base_learner.py` - Abstract base class
- `src/algorithms/information_gain.py` - Main algorithm implementation

**Implementation Requirements:**

1. **Base Learner Interface** (`src/algorithms/base_learner.py`)
   ```python
   class BaseActionModelLearner(ABC):
       def select_action(self, state) -> Tuple[str, List[str]]  # (action_name, objects)
       def observe(self, state, action, objects, success, next_state=None) -> None
       def get_learned_model(self) -> Dict  # Export learned model
       def has_converged(self) -> bool
   ```

2. **Information Gain Learner** (`src/algorithms/information_gain.py`)
   - Initialize with domain using PDDLHandler
   - Maintain 6 knowledge sets per action (pre, pre?, eff+, eff-, eff?+, eff?-)
   - Use CNFManager for precondition constraints (pre?)
   - Handle negative preconditions (¬fluent checks)
   - Calculate information gain metrics:
     - `preAppPotential`: Knowledge from successful preconditions
     - `preFailPotential`: Knowledge from failed preconditions
     - `effPotential`: Knowledge from observed effects
   - Action selection based on expected information gain
   - Update rules for success/failure observations

### Phase 3: External Algorithm Integration

**Prerequisites:**
- Ensure external repos are accessible:
  - `/home/omer/projects/OLAM/`
  - `/home/omer/projects/ModelLearner/`

**Files to create:**
- `src/algorithms/olam_adapter.py`
- `src/algorithms/optimistic_adapter.py`

**Implementation Requirements:**

1. **OLAM Adapter** (`src/algorithms/olam_adapter.py`)
   ```python
   import sys
   sys.path.append('/home/omer/projects/OLAM')
   from OLAM.Learner import Learner
   ```
   - Implement BaseActionModelLearner interface
   - Convert UP state format to OLAM's set-based format
   - Handle OLAM's specific methods:
     - `select_action()`: Use OLAM's exploration
     - `learn()`: Process successful actions
     - `learn_failed_action_precondition()`: Process failures
   - State format conversion (UP objects ↔ OLAM strings)

2. **ModelLearner Adapter** (`src/algorithms/optimistic_adapter.py`)
   ```python
   import sys
   sys.path.append('/home/omer/projects/ModelLearner/src')
   from model_learner.ModelLearnerLifted import ModelLearnerLifted
   ```
   - Implement BaseActionModelLearner interface
   - Handle ModelLearner's lifted_dict YAML requirement
   - Key method: `learning_step_all_actions_updated()`
   - Manage optimistic model updates

### Phase 4: Environment and Planning Integration

**Files to create:**
- `src/environments/pddl_environment.py` - Simulated PDDL environment
- `src/planning/unified_planner.py` - UP planner wrapper

**Implementation Requirements:**

1. **PDDL Environment** (`src/environments/pddl_environment.py`)
   - Use UP's SequentialSimulator for action execution
   - Track current state
   - Execute grounded actions and return success/failure
   - Handle action applicability checking
   - Provide state observations

2. **Unified Planner** (`src/planning/unified_planner.py`)
   - Wrap UP's OneshotPlanner
   - Support multiple planner backends (pyperplan, tamer, fast-downward)
   - Handle timeout and error cases
   - Convert between internal and UP plan formats
   - Support both optimal and satisficing modes

### Phase 5: Experiment Runner and Metrics

**Files to create:**
- `src/experiments/runner.py` - Main experiment orchestrator
- `src/experiments/metrics.py` - Performance metrics collector
- `configs/experiment.yaml` - Experiment configuration

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

## Key Implementation Notes

### Working with Unified Planning Framework

**CRITICAL**: UP uses expression trees (FNode), NOT simple sets!
- Preconditions are List[FNode] that need recursive traversal
- Use `expr.is_and()`, `expr.is_fluent_exp()` etc. to check types
- See `docs/UNIFIED_PLANNING_GUIDE.md` for details

### Handling Negative Preconditions

- Positive literal `clear(a)`: Check if `clear(a) ∈ state`
- Negative literal `¬clear(a)`: Check if `clear(a) ∉ state`
- CNF representation: `¬clear(a)` becomes negated variable `¬x_clear_a`
- See examples in `docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md`

### Import Paths

```python
# PySAT (handle both package names)
try:
    from pysat.solvers import Minisat22
    from pysat.formula import CNF
except ImportError:
    from pysat.solvers import Minisat22  # python-sat package
    from pysat.formula import CNF

# Unified Planning
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import SequentialSimulator

# External algorithms (add to sys.path)
import sys
sys.path.append('/home/omer/projects/OLAM')
sys.path.append('/home/omer/projects/ModelLearner/src')
```

## Testing Strategy

1. **Unit Tests** ✅ Completed for:
   - CNF Manager with lifted fluents
   - PDDL Handler with type hierarchy
   - Expression tree traversal

2. **Integration Tests** (Next Priority):
   - Information gain algorithm with CNF
   - Algorithm adapters with UP
   - Environment simulation

3. **System Tests** - Full experiments on domains:
   - Blocksworld (start here)
   - Gripper
   - Logistics

4. **Performance Tests**:
   - SAT solver scaling
   - Formula minimization effectiveness
   - Planning timeouts

## Success Criteria

- ✅ CNF formulas correctly represent uncertainty (with lifted support)
- ✅ Type hierarchy properly handled with 'object' as root
- ⏳ All three algorithms implement BaseActionModelLearner interface
- ⏳ Information gain correctly calculated from CNF model counting
- ⏳ Experiments run on standard PDDL domains
- ⏳ Results reproducible with seed control
- ⏳ Metrics show clear algorithm comparison

## Critical Reminders for Next Session

1. **ALWAYS review `docs/DEVELOPMENT_RULES.md` first**
2. **Check `CLAUDE.md` for quick context guidance**
3. **UP uses expression trees** - Never assume simple sets
4. **Negative preconditions** need special handling
5. **Test incrementally** - Start with blocksworld, 3-4 objects
6. **Use existing implementations** - CNFManager and PDDLHandler are complete

## Next Steps Priority Order

1. Create `BaseActionModelLearner` abstract class
2. Implement `InformationGainLearner` using existing CNFManager
3. Create simple PDDL environment for testing
4. Test on small blocksworld problems
5. Add OLAM and ModelLearner adapters
6. Run comparative experiments
