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

#### Phase 2: OLAM Adapter (Completed)
- **Base Action Model Learner** (`src/algorithms/base_learner.py`) - Abstract interface for all algorithms:
  - Unified interface methods: `select_action()`, `observe()`, `get_learned_model()`, `has_converged()`
  - Statistics tracking and reset functionality
  - Helper methods for state/action conversion

- **OLAM Adapter** (`src/algorithms/olam_adapter.py`) - Full integration with OLAM learner:
  - State format conversion (UP ↔ OLAM): `clear_a` ↔ `(clear a)`
  - Action format conversion: `('pick-up', ['a'])` ↔ `pick-up(a)`
  - Proper grounding of actions (18 actions for 3-block world)
  - Observation handling for success/failure learning
  - Model export functionality
  - Handles OLAM's directory structure requirements (PDDL/, Info/)

#### Testing
- Comprehensive tests for CNF Manager lifted fluent support
- Tests for PDDL Handler type hierarchy and lifted actions
- Tests validating expression tree traversal
- **OLAM Adapter Tests**:
  - Comprehensive test suite (`tests/test_olam_adapter.py`) with 9 test categories
  - Simple integration tests (`tests/test_olam_simple.py`) - 5 tests all passing
  - Validated state/action conversions and basic workflow

## Technology Stack
- **Unified Planning Framework** - PDDL parsing and planning integration (expression trees, NOT simple sets!)
- **PySAT/python-sat** - SAT solver (Minisat) for CNF formula manipulation
- **Python 3.9+** - In conda environment
- **SLURM** - HPC job scheduling (future)

## Implementation Phases

**🎯 Phase Ordering Rationale:**
Phases are ordered to enable complete testing of OLAM before implementing other algorithms:
1. Phase 2 (COMPLETED): OLAM adapter with BaseActionModelLearner interface
2. Phase 3: Experiment framework for running and measuring learning
3. Phase 4: Environment/Planning for actual PDDL execution
4. **Full OLAM validation and baseline establishment**
5. Phases 5-6: Other algorithms for comparison

This ensures we have a fully functional and tested baseline before adding complexity.

### ✅ Phase 2: OLAM Adapter Implementation (Test-Driven Development) - COMPLETED

**Context Documentation:**
- `docs/external_repos/OLAM_interface.md` - OLAM API and usage patterns
- `docs/external_repos/integration_guide.md` - Adapter pattern and state conversion
- `docs/UNIFIED_PLANNING_GUIDE.md` - UP state/action formats

**Test-Driven Development Approach:**

1. **FIRST: Write Comprehensive Test Suite** (`tests/test_olam_adapter.py`)

   **Test Categories Required:**

   a) **Basic Functionality Tests**
      - Test OLAM import and initialization
      - Test BaseActionModelLearner interface compliance
      - Test state format conversion (UP ↔ OLAM)
      - Test action format conversion

   b) **Action Selection Tests**
      - Test action selection returns valid actions
      - Test exploration vs exploitation strategies
      - Test handling of empty action sets
      - Test action applicability checking

   c) **Learning from Observations Tests**
      - Test successful action observation updates
      - Test failed action precondition learning
      - Test effect learning (add/delete effects)
      - Test model refinement over multiple observations

   d) **Edge Cases and Error Handling**
      - Test with empty states
      - Test with invalid actions
      - Test with contradictory observations
      - Test timeout and iteration limits
      - Test convergence detection

   e) **Plan Execution Tests**
      - Test plan generation with learned model
      - Test plan execution monitoring
      - Test replanning on failure
      - Test goal achievement validation

   f) **Integration Tests**
      - Test with real PDDL domains (blocksworld, gripper, rover\rovers(classical\discrete domains))
      - Test action sequence execution
      - Test model accuracy over time
      - Test compatibility with UP simulator

2. **THEN: Implement OLAM Adapter** (`src/algorithms/olam_adapter.py`)

   **Implementation Requirements:**
   ```python
   import sys
   sys.path.append('/home/omer/projects/OLAM')
   from OLAM.Learner import Learner
   ```

   - Implement BaseActionModelLearner interface
   - State conversion methods:
     - `_up_state_to_olam(up_state)` → Set of strings
     - `_olam_state_to_up(olam_state)` → UP State
   - Action conversion methods:
     - `_up_action_to_olam(action, objects)` → OLAM format
     - `_olam_action_to_up(action_idx)` → (name, objects)
   - Wrap OLAM methods:
     - `select_action()` → Use OLAM's exploration
     - `observe()` → Call appropriate OLAM learning method
     - `get_learned_model()` → Export OLAM's learned operators
     - `has_converged()` → Check OLAM's convergence flag

3. **Validation Requirements:**

   **All tests must pass before proceeding to experiments:**
   - Unit tests achieve 100% code coverage
   - Integration tests work with at least 3 PDDL domains
   - Edge cases handled gracefully without crashes
   - Performance benchmarks meet requirements (< 1s per action)
   - Memory usage stays within bounds
   - OLAM functionality correctly preserved

### Phase 3: Experiment Runner and Metrics Framework ⏳ NEXT PRIORITY

**Rationale:** Implementing the experiment framework first allows us to test and evaluate each algorithm as it's developed.

**Test-Driven Development Approach:**
1. Write test suite first (`tests/test_experiment_runner.py`, `tests/test_metrics.py`)
2. Test configuration loading and validation
3. Test metric collection and export functionality
4. Test integration with BaseActionModelLearner interface
5. Implement components to pass tests incrementally

**Files to create:**
- `tests/test_experiment_runner.py` - Test suite for runner (CREATE FIRST)
- `tests/test_metrics.py` - Test suite for metrics (CREATE FIRST)
- `src/experiments/runner.py` - Main experiment orchestrator
- `src/experiments/metrics.py` - Performance metrics collector
- `configs/experiment.yaml` - Experiment configuration

**Implementation Requirements:**

1. **Experiment Runner** (`src/experiments/runner.py`)
   - Load experiment configuration from YAML
   - Initialize selected algorithms (OLAM, ModelLearner, Information-Theoretic)
   - Setup PDDL environment and planner
   - Run learning episodes with configurable stopping criteria
   - Collect metrics at specified intervals
   - Save results in structured format

2. **Metrics Collector** (`src/experiments/metrics.py`)
   - **Mistake rate**: Track action failure rate over time (snapshot every x actions)
   - **Runtime performance**: Time to select/plan actions (should decrease as hypothesis space shrinks)
   - **Information gain metrics**: Reduction in uncertainty for preconditions/effects (from CNF entropy)
   - **Model accuracy**: Precision/recall for learned model (evaluate at end)
   - **Solution count evolution**: Track CNF formula solution space size
   - Export metrics to CSV/JSON for analysis

3. **Configuration Schema** (`configs/experiment.yaml`)
   ```yaml
   experiment:
     name: "blocksworld_comparison"
     algorithms: ["olam", "information_gain"]
     domain: "benchmarks/blocksworld/domain.pddl"
     problem: "benchmarks/blocksworld/p01.pddl"
     metrics_interval: 10  # Collect metrics every 10 actions
     max_iterations: 1000
   ```

### Phase 4: Environment and Planning Integration ⏳ REQUIRED FOR TESTING

**Rationale:** Need environment and planning components to run actual experiments and fully test OLAM adapter.

**Test-Driven Development Approach:**
1. Write test suite first (`tests/test_pddl_environment.py`)
2. Test environment initialization with PDDL files
3. Test action execution and state transitions
4. Test state observation format compatibility
5. Test reset functionality
6. Implement environment to pass tests incrementally

**Files to create:**
- `tests/test_pddl_environment.py` - Test suite for environment (CREATE FIRST)
- `src/environments/pddl_environment.py` - Simulated PDDL environment
- `src/planning/unified_planner.py` - UP planner wrapper (optional for initial testing)

**Implementation Requirements:**

1. **PDDL Environment** (`src/environments/pddl_environment.py`)
   - Use UP's SequentialSimulator for action execution
   - Track current state
   - Execute grounded actions and return success/failure
   - Handle action applicability checking
   - Provide state observations in BaseActionModelLearner-compatible format
   - Reset to initial state functionality

2. **Unified Planner** (`src/planning/unified_planner.py`) - Optional for Phase 4
   - Wrap UP's OneshotPlanner
   - Support multiple planner backends (pyperplan, tamer, fast-downward)
   - Handle timeout and error cases
   - Convert between internal and UP plan formats
   - Support both optimal and satisficing modes

**Integration with Experiment Runner:**
- Environment provides action execution for learning loop
- State observations compatible with OLAM adapter format
- Enables full end-to-end testing of OLAM learning

### Phase 5: Information-Theoretic Algorithm Implementation

**Context Documentation:**
- `docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md` - Complete algorithm specification with negative preconditions
- `docs/UNIFIED_PLANNING_GUIDE.md` - Critical for understanding UP's expression trees
- `docs/LIFTED_SUPPORT.md` - For handling parameterized actions

**Test-Driven Development Approach:**
1. Write comprehensive test suite first (`tests/test_information_gain.py`)
2. Test knowledge set management (pre, pre?, eff+, eff-, eff?+, eff?-)
3. Test CNF formula updates for preconditions
4. Test information gain calculations
5. Test action selection based on expected gain
6. Test learning from success/failure observations
7. Implement algorithm to pass tests incrementally

**Files to create:**
- `tests/test_information_gain.py` - Test suite for algorithm (CREATE FIRST)
- `src/algorithms/information_gain.py` - Main algorithm implementation

**Implementation Requirements:**

1. **Information Gain Learner** (`src/algorithms/information_gain.py`)
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
   - Integrate with experiment framework for comparison with OLAM

### Phase 6: ModelLearner Adapter Integration

**Prerequisites:**
- Ensure external repo is accessible: `/home/omer/projects/ModelLearner/`

**Test-Driven Development Approach:**
1. Write test suite first (`tests/test_optimistic_adapter.py`)
2. Test ModelLearner import and initialization
3. Test state/action format conversions
4. Test optimistic model updates
5. Test batch learning after plan execution
6. Implement adapter to pass tests (following OLAM adapter pattern)

**Files to create:**
- `tests/test_optimistic_adapter.py` - Test suite for adapter (CREATE FIRST)
- `src/algorithms/optimistic_adapter.py`

**Implementation Requirements:**

1. **ModelLearner Adapter** (`src/algorithms/optimistic_adapter.py`)
   ```python
   import sys
   sys.path.append('/home/omer/projects/ModelLearner/src')
   from model_learner.ModelLearnerLifted import ModelLearnerLifted
   ```
   - Implement BaseActionModelLearner interface
   - Handle ModelLearner's lifted_dict YAML requirement
   - Key method: `learning_step_all_actions_updated()`
   - Manage optimistic model updates
   - State/action format conversion similar to OLAM adapter
   - Integrate with experiment framework for three-way comparison

### Phase 7: HPC Deployment

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
- ✅ OLAM adapter implements BaseActionModelLearner interface
- ⏳ All three algorithms implement BaseActionModelLearner interface (1/3 complete)
- ⏳ Information gain correctly calculated from CNF model counting
- ⏳ Experiments run on standard PDDL domains
- ⏳ Results reproducible with seed control
- ⏳ Metrics show clear algorithm comparison

## Critical Reminders for Next Session

1. **ALWAYS review `docs/DEVELOPMENT_RULES.md` first**
2. **Check `CLAUDE.md` for quick context guidance**
3. **Follow Test-Driven Development** - Write tests before implementation
4. **UP uses expression trees** - Never assume simple sets
5. **Negative preconditions** need special handling
6. **Test incrementally** - Start with blocksworld, 3-4 objects
7. **Use existing implementations** - CNFManager and PDDLHandler are complete
8. **Complete OLAM testing** before implementing other algorithms

## Next Steps Priority Order

1. **Experiment Framework Implementation (Phase 3):**
   - Write tests first for runner and metrics components
   - Create experiment runner with YAML configuration support
   - Implement metrics collector with specified metrics
   - Initial testing capability with mocked environment

2. **Environment and Planning Integration (Phase 4):**
   - Write tests for PDDL environment functionality
   - Create PDDL environment using UP's SequentialSimulator
   - Enable full end-to-end testing with OLAM adapter
   - **MILESTONE: Fully test and validate OLAM learning on multiple domains**
   - Collect baseline metrics for OLAM performance

3. **Information-Theoretic Algorithm (Phase 5):**
   - Write comprehensive test suite first
   - Implement InformationGainLearner using existing CNFManager
   - Validate with small blocksworld problems
   - Compare performance with OLAM baseline

4. **ModelLearner Adapter (Phase 6):**
   - Apply same TDD approach as OLAM
   - Ensure compatibility with optimistic exploration
   - Complete three-way algorithm comparison
