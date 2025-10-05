# Implementation Tasks for Online Model Learning Framework

## Project Overview
Building experiment framework to compare three online action model learning algorithms:
1. **OLAM** - Baseline (Python implementation) ✅
2. **ModelLearner** - Optimistic Exploration (TODO)
3. **Information-Theoretic** - Novel CNF-based approach (TODO)

## Quick Start
**See [CLAUDE.md](../CLAUDE.md) for navigation and [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) for project rules.**

## Current Status (Updated: October 5, 2025 - 11:20 AM)

### 🔧 Recent CI/CD Fixes
- **NumPy Version**: Updated to support 2.x (was restricted to <2.0.0)
- **VAL Build**: Fixed CMake build process in CI workflow
- **ModelLearner**: Repository unavailable - commented out until fixed
- **Error Visibility**: Removed `continue-on-error` flags to expose real failures

### ✅ Completed Components

#### Core Infrastructure
- **CNF Manager** (`src/core/cnf_manager.py`) - Complete
  - **REFACTORED (Oct 5)**: Added constraint-based operations for algorithm separation
    - `create_with_state_constraints(state_constraints)` - CNF copy with state constraints
    - `add_constraint_from_unsatisfied(unsatisfied_literals)` - Failure constraint handling
    - `build_from_constraint_sets(constraint_sets)` - Build CNF from pre?(a)
    - `count_models_with_constraints(state_constraints)` - Model counting with constraints
    - `clear_formula()`, `has_clauses()`, `add_unit_constraint()` - Helper methods
- **PDDL Handler** (`src/core/pddl_handler.py`) - Complete with injective bindings
  - **NEW (Oct 5)**: Added parameter-bound literal manipulation methods
    - `get_parameter_bound_literals(action_name)` - Generate La for any action
    - `ground_literals(literals, objects)` - Convert lifted to grounded (bindP⁻¹)
    - `lift_fluents(fluents, objects)` - Convert grounded to lifted (bindP)
    - `extract_predicate_name(literal)` - Extract predicate from any literal format
  - **REFACTORING**: InformationGainLearner now delegates all PDDL operations to PDDLHandler
- **Domain Analyzer** (`src/core/domain_analyzer.py`) - Algorithm compatibility checking
- **PDDL Environment** (`src/environments/pddl_environment.py`) - Real PDDL execution

#### OLAM Integration
- **OLAM Adapter** (`src/algorithms/olam_adapter.py`) - Fully validated against paper
- **Base Learner** (`src/algorithms/base_learner.py`) - Abstract interface
- **Java bypass** with learned model filtering (no ground truth)
- **Domain compatibility** - Works with blocksworld/gripper (no negative preconditions)
- **Validation confirmed** - All 4 paper behaviors demonstrated

#### Experiment Framework
- **Runner** (`src/experiments/runner.py`) - YAML-based experiments
- **Metrics** (`src/experiments/metrics.py`) - Comprehensive tracking
- **Results export** - CSV/JSON formats

#### Information Gain Algorithm
- **Information Gain Learner** (`src/algorithms/information_gain.py`) - Complete with Phase 3
  - Applicability probability calculation using SAT model counting
  - Entropy and information gain calculations
  - Three selection strategies (greedy, epsilon-greedy, Boltzmann)
  - **REFACTORED (Oct 5)**: Separated CNF operations from algorithm logic
    - Now uses CNF Manager methods for all CNF manipulation
    - Cleaner separation between algorithm and formula management
  - **REFACTORED (Oct 5)**: Now contains only algorithm logic, delegates PDDL operations to PDDLHandler
    - Removed internal PDDL parsing methods (moved to PDDLHandler)
    - Uses `PDDLHandler.get_parameter_bound_literals()` for La generation
    - Uses `PDDLHandler.ground_literals()` for bindP⁻¹
    - Uses `PDDLHandler.lift_fluents()` for bindP
  - ~60 tests passing
- See [INFORMATION_GAIN_ALGORITHM.md](information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md)

### ⏳ TODO Components

#### ModelLearner Integration
- `src/algorithms/optimistic_adapter.py` - Adapter implementation
- See [ModelLearner_interface.md](external_repos/ModelLearner_interface.md)

## Recent Updates (October 5, 2025)

### Architecture Refactoring - Separation of Concerns
**Context**: The InformationGainLearner class previously contained both algorithm logic and PDDL manipulation code.

**Problem**: PDDL operations (lifting/grounding, literal parsing) were mixed with algorithm logic, violating single responsibility principle.

**Solution**: Refactored to delegate all PDDL operations to PDDLHandler:
- **PDDLHandler** now provides all PDDL manipulation methods
- **InformationGainLearner** focuses solely on the learning algorithm
- **Benefits**:
  - Clear separation between algorithm and PDDL handling
  - Reusable PDDL methods for other algorithms
  - Easier testing and maintenance
  - Better code organization

**Methods Moved to PDDLHandler**:
- `get_parameter_bound_literals()` - Generate La for actions
- `ground_literals()` (was `bindP_inverse`) - Lifted → Grounded
- `lift_fluents()` (was `bindP`) - Grounded → Lifted
- `extract_predicate_name()` - Parse predicate names
- Internal helpers for parameter indexing and naming

## Recent Fixes (September 28, 2025)

1. **OLAM Learning Validation**
   - Fixed Java bypass to use learned model only
   - Added injective binding support for OLAM
   - Validated hypothesis space reduction behavior

2. **Domain Compatibility**
   - Created domain analyzer for feature detection
   - Added OLAM-compatible domains (no negative preconditions)
   - Proper validation without success rate assumptions

## Next Implementation Tasks

### Phase 4: Information Gain Integration & Testing
1. Integrate `InformationGainLearner` with experiment runner
2. Add configuration for selection strategies (greedy/epsilon-greedy/Boltzmann)
3. Test on multiple domains (blocksworld, gripper, rover, depots)
4. Profile performance and optimize if needed

### Phase 5: Comparative Experiments
1. Create experiment configs for Information Gain vs OLAM
2. Run on multiple domains (blocksworld, gripper, rover, depots)
3. Analyze sample complexity and convergence
4. Generate comparison reports

### Phase 6: ModelLearner Adapter (BLOCKED - Low Priority)
⚠️ **Blocked**: Repository https://github.com/kcleung/ModelLearner.git not accessible
1. Find correct repository URL or alternative implementation
2. Create `OptimisticAdapter` class
3. Handle lifted_dict YAML requirements
4. Implement optimistic exploration strategy
5. Validate against ModelLearner paper

### Phase 7: Three-Way Comparison
1. Create experiment configs for all algorithms
2. Run on multiple domains (blocksworld, gripper, rover)
3. Analyze sample complexity and convergence
4. Generate comparison reports

## Testing Status

- **Unit tests**: 165/165 passing (`make test`)
- **Integration tests**: OLAM fully tested
- **Validation**: OLAM paper behaviors confirmed

## File Structure
See [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) for complete directory structure.

## External Dependencies
See [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md#external-tool-paths) for external tool paths and availability status.

## Documentation Index
See [CLAUDE.md](../CLAUDE.md) for complete documentation navigation.