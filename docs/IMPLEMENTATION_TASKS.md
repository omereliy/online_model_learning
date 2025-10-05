# Implementation Tasks for Online Model Learning Framework

## Project Overview
Building experiment framework to compare three online action model learning algorithms:
1. **OLAM** - Baseline (Python implementation) ✅
2. **Information Gain** - CNF/SAT-based information-theoretic approach ✅
3. **ModelLearner** - Optimistic Exploration (BLOCKED - repo unavailable)

## Quick Start
**See [CLAUDE.md](../CLAUDE.md) for navigation and [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) for project rules.**

## Current Status (Updated: October 5, 2025 - 10:30 PM)

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

### PDDLHandler Refactoring Phase 1 - Type-Safe Data Classes (October 5, 2025)
**Context**: PDDLHandler used primitive types (Dict[str, Object], str) throughout, reducing type safety and code clarity.

**Problem**:
- No type safety for parameter bindings
- String-based literals without semantic distinction
- No compile-time verification of types

**Solution**: Created type-safe data classes in `src/core/pddl_types.py`:
- **ParameterBinding**: Type-safe wrapper for Dict[str, Object]
  - Provides `get_object()`, `object_names()`, `to_dict()` methods
  - Clear semantic meaning for parameter→object mappings
- **ParameterBoundLiteral**: Literals using action's parameter names
  - CRITICAL: Uses action's specific parameter names (e.g., clear(?x) for pick-up(?x))
  - Supports `to_string()` and `from_string()` for conversions
  - Handles negation (¬) and propositional literals
- **GroundedFluent**: Fully instantiated fluents with objects
  - Converts to/from string format (e.g., "clear_a", "on_a_b")
  - Handles multi-parameter and propositional fluents

**Benefits**:
- Type safety with IDE support
- Self-documenting code
- Validation at creation time
- Foundation for further refactoring phases

**Tests**: 21 new tests in `tests/core/test_pddl_types.py`, all passing

### PDDLHandler Refactoring Phase 2 - Expression Conversion Logic (October 5, 2025)
**Context**: Expression conversion logic (FNode → string) was scattered throughout PDDLHandler, making it hard to test and reuse.

**Problem**:
- Three methods doing similar FNode→string conversions
- Duplicate traversal logic for AND/OR/NOT expressions
- No single source of truth for conversion rules

**Solution**: Created `ExpressionConverter` class in `src/core/expression_converter.py`:
- **`to_parameter_bound_string()`**: Converts FNode to parameter-bound literal
  - CRITICAL: Preserves action's specific parameter names (e.g., `clear(?x)`)
  - Fixed bug: Compare `str(arg) == param.name`, not `str(arg) == str(param)`
- **`to_grounded_string()`**: Converts FNode to grounded fluent with binding
  - Uses ParameterBinding for type-safe parameter→object mapping
  - Handles negation and multi-parameter fluents
- **`to_cnf_clauses()`**: Extracts CNF clauses from complex expressions
  - Recursively handles AND/OR/NOT operators
  - Returns List[List[str]] format for CNF formulas

**Refactoring**:
- PDDLHandler methods now delegate to ExpressionConverter:
  - `_expression_to_lifted_string()` → `ExpressionConverter.to_parameter_bound_string()`
  - `_ground_expression_to_string()` → `ExpressionConverter.to_grounded_string()`
  - `_extract_clauses_from_expression()` → `ExpressionConverter.to_cnf_clauses()`
- Old methods kept for backward compatibility (marked DEPRECATED)

**Benefits**:
- Centralized conversion logic - single source of truth
- Easier to test in isolation (9 new tests)
- Reusable across different components
- Reduced code duplication (~60 lines removed from PDDLHandler)

**Tests**: 9 new tests in `tests/core/test_expression_converter.py`, all passing

### PDDLHandler Refactoring Phase 3 - Grounding/Lifting Operations (October 5, 2025)
**Context**: Grounding (bindP⁻¹) and lifting (bindP) operations were embedded in PDDLHandler with ~150 lines of implementation code, making them hard to test and not clearly separated as algorithm-specific operations.

**Problem**:
- bindP/bindP⁻¹ operations scattered across multiple internal methods
- No clear separation between algorithm logic and PDDL manipulation
- Hard to test binding operations in isolation
- Used by Information Gain algorithm but buried in PDDLHandler

**Solution**: Created `FluentBinder` class in `src/core/binding_operations.py`:
- **`ground_literal()`**: Single literal grounding (e.g., `clear(?x)` + `['a']` → `clear_a`)
- **`lift_fluent()`**: Single fluent lifting (e.g., `clear_a` + `['a']` → `clear(?x)`)
- **`ground_literals()`**: Batch bindP⁻¹ operation for sets of literals
- **`lift_fluents()`**: Batch bindP operation for sets of fluents
- Handles negation, multi-parameter literals, and propositional fluents
- Uses PDDLHandler's static methods for parameter name generation

**Refactoring**:
- PDDLHandler now delegates to FluentBinder:
  - Added `_get_binder()` for lazy initialization (avoids circular import)
  - `ground_literals()` → `FluentBinder.ground_literals()`
  - `lift_fluents()` → `FluentBinder.lift_fluents()`
  - `_ground_lifted_literal_internal()` → marked DEPRECATED, delegates to FluentBinder
  - `_lift_grounded_fluent_internal()` → marked DEPRECATED, delegates to FluentBinder
- Old methods kept for backward compatibility during transition

**Benefits**:
- Clear algorithm semantics (bindP and bindP⁻¹)
- Easier to test in isolation (16 new tests)
- Reduced PDDLHandler complexity (~150 lines extracted)
- Better separation of concerns (algorithm operations vs PDDL parsing)
- Inverse operations verified: `bindP(bindP⁻¹(F, O), O) = F`

**Tests**: 16 new tests in `tests/core/test_binding_operations.py`, all passing
**Integration**: All existing tests pass, including Information Gain algorithm bindP tests

### PDDLHandler Refactoring Phase 1B - GroundedAction Type-Safe Class (October 5, 2025)
**Context**: Grounded actions were represented as `Tuple[Action, Dict[str, Object]]` throughout the codebase, lacking type safety and self-documentation.

**Problem**:
- Primitive tuple representation not self-documenting
- Requires tuple unpacking everywhere
- No type safety for grounded actions
- No built-in validation or convenience methods

**Solution**: Created `GroundedAction` dataclass in `src/core/pddl_types.py`:
- **GroundedAction**: Type-safe wrapper for grounded actions
  - Replaces `Tuple[Action, Dict[str, Object]]` representation
  - Provides `object_names()`, `to_string()` methods
  - Includes `from_components()` for backward compatibility
  - Includes `to_tuple()` for gradual migration
  - Follows same pattern as `GroundedFluent` for consistency

**Benefits**:
- Type safety with IDE support
- Self-documenting code
- Consistent API with GroundedFluent
- Foundation for future migration of PDDLHandler methods
- Easier to maintain and understand grounded action handling

**Tests**: 9 new tests in `tests/core/test_pddl_types.py`, all passing

### PDDLHandler Refactoring Phase 4 - Complex Method Refactoring (October 5, 2025)
**Context**: PDDLHandler contained complex methods with mixed responsibilities (~137 lines total), violating single responsibility principle and reducing testability.

**Problem**:
- `state_to_fluent_set()` handled both dict and state objects in one method (52 lines)
- `get_action_preconditions()` mixed lifted and grounded extraction with boolean flag (41 lines)
- `get_action_effects()` mixed lifted and grounded extraction (44 lines)
- Hard to test individual code paths in isolation

**Solution**: Applied dispatcher pattern to split complex methods:
- **state_to_fluent_set()** → dispatcher + `_fluents_from_dict()` + `_fluents_from_state_object()`
  - Each private method handles one state type (~18 lines each)
  - Dispatcher routes based on `isinstance(state, dict)`
- **get_action_preconditions()** → dispatcher + `_get_lifted_preconditions()` + `_get_grounded_preconditions()`
  - Lifted method extracts parameter-bound preconditions (~15 lines)
  - Grounded method extracts fully instantiated preconditions (~18 lines)
  - Dispatcher routes based on `lifted` flag
- **get_action_effects()** → dispatcher + `_get_lifted_effects()` + `_get_grounded_effects()`
  - Lifted method extracts parameter-bound effects (~15 lines)
  - Grounded method extracts fully instantiated effects (~18 lines)
  - Returns tuple of (add_effects, delete_effects)

**Benefits**:
- Single responsibility: each method handles one specific case
- Improved testability: can test each private method in isolation
- Reduced complexity: methods under 20 lines each
- Maintained backward compatibility: public API unchanged
- Code reduction: ~137 lines → ~117 lines (net -20 lines through consolidation)

**Tests**: 6 new tests in `tests/core/test_pddl_handler.py`, all passing
- `test_fluents_from_dict` / `test_fluents_from_state_object`
- `test_get_lifted_preconditions` / `test_get_grounded_preconditions`
- `test_get_lifted_effects` / `test_get_grounded_effects`

### PDDLHandler Refactoring Phase 5 - Migrate to Type-Safe Classes (October 5, 2025)
**Context**: Dependent files still used primitive tuple representations for grounded actions, missing the benefits of type-safe GroundedAction class.

**Problem**:
- `information_gain.py` used tuple unpacking: `for action, binding in grounded_actions:`
- `olam_adapter.py` accessed internal `_grounded_actions` list directly
- `pddl_environment.py` works with primitive types (no changes needed)
- Code not using available type-safe interfaces

**Solution**: Migrated dependent files to use type-safe GroundedAction:
- **PDDLHandler**: Added `get_all_grounded_actions_typed()` method
  - Returns `List[GroundedAction]` instead of `List[Tuple[Action, Dict[str, Object]]]`
  - Wraps internal `_grounded_actions` with type-safe classes
  - Clean API without exposing internal representation
- **information_gain.py**: Updated to use GroundedAction
  - Changed from: `for action, binding in grounded_actions:`
  - Changed to: `for grounded_action in grounded_actions:`
  - Uses `grounded_action.action.name` for action name
  - Uses `grounded_action.object_names()` for parameter extraction
  - Cleaner, more readable code
- **olam_adapter.py**: Updated to use GroundedAction
  - Changed from: `for action, binding in self.pddl_handler._grounded_actions:`
  - Changed to: `for grounded_action in self.pddl_handler.get_all_grounded_actions_typed():`
  - Uses `grounded_action.object_names()` instead of manual extraction
  - No longer accesses internal `_grounded_actions` attribute
- **pddl_environment.py**: Assessment complete - no changes needed
  - Works with action names and parameter lists directly
  - No parameter binding usage

**Benefits**:
- Type safety throughout the codebase
- Consistent API usage with GroundedAction
- Improved code readability and maintainability
- No direct access to internal PDDLHandler attributes
- Foundation complete for future refactoring phases

**Tests**: All 51 curated tests passing, including:
- Information Gain algorithm tests
- OLAM adapter integration tests
- Full pipeline integration tests

**Documentation Updates**:
- `UNIFIED_PLANNING_GUIDE.md`: Added `get_all_grounded_actions_typed()` usage example
- `QUICK_REFERENCE.md`: Added "Type-Safe Grounded Actions" pattern
- `pddl_handler_refactoring_plan.md`: Updated execution tracking to Phase 5 complete

### PDDLHandler Refactoring Phase 6 - Documentation Review and Updates (October 5, 2025)
**Context**: After completing Phases 1-5 of the PDDLHandler refactoring, comprehensive documentation updates needed to ensure all documentation reflects the refactored codebase.

**Problem**:
- Documentation partially updated during earlier phases
- Need comprehensive review to ensure consistency
- No centralized documentation of final refactoring metrics
- Missing references to new modules in some docs

**Solution**: Comprehensive documentation review and updates across all documentation files:
- **UNIFIED_PLANNING_GUIDE.md**: Verified complete with ExpressionConverter, FluentBinder, and GroundedAction examples
- **LIFTED_SUPPORT.md**: Added type-safe class section at top, FluentBinder operations, updated examples
- **QUICK_REFERENCE.md**: Added ExpressionConverter usage, FluentBinder (bindP/bindP⁻¹) usage, expanded type-safe patterns
- **IMPLEMENTATION_TASKS.md**: Added Phase 6 completion entry (this section)
- **pddl_handler_refactoring_plan.md**: Updated execution tracking to Phase 6 complete

**Benefits**:
- All documentation accurately reflects refactored codebase
- No outdated references to primitive types (only historical explanations)
- Comprehensive examples for all new modules
- Final refactoring metrics documented for future reference
- Documentation internally consistent

**Final Refactoring Metrics**:
- **New modules created**: 3
  - `src/core/pddl_types.py` (type-safe classes)
  - `src/core/expression_converter.py` (FNode conversion logic)
  - `src/core/binding_operations.py` (bindP/bindP⁻¹ operations)
- **New tests added**: 46 tests across 3 new test files
  - `tests/core/test_pddl_types.py`: 21 tests
  - `tests/core/test_expression_converter.py`: 9 tests
  - `tests/core/test_binding_operations.py`: 16 tests
- **Code reduction in PDDLHandler**: ~200 lines extracted to separate modules
- **Methods refactored**: 11 methods (3 expression converters, 2 binding operations, 3 state/precondition/effects dispatchers, 1 helper method, 2 grounded action methods)
- **Type safety improvements**: 4 primitive types replaced
  - `Dict[str, Object]` → `ParameterBinding`
  - Parameter-bound strings → `ParameterBoundLiteral`
  - Grounded fluent strings → `GroundedFluent`
  - `Tuple[Action, Dict[str, Object]]` → `GroundedAction`
- **Files updated in dependent code**: 2
  - `src/algorithms/information_gain.py`: Using GroundedAction
  - `src/algorithms/olam_adapter.py`: Using GroundedAction

**Tests**: All 51 curated tests passing throughout all phases

**Validation**: Phase 6 Checklist ✓
- All documentation reflects refactored codebase accurately
- No outdated references to primitive-type APIs
- All type-safe classes documented with examples
- ExpressionConverter usage documented
- FluentBinder (bindP/bindP⁻¹) usage documented
- All code examples verified
- All 51 curated tests passing
- Documentation internally consistent
- Refactoring plan execution tracking complete

### Experiment Readiness Assessment (October 5, 2025)
**Context**: Before conducting paper-ready experiments comparing OLAM and Information Gain, need to validate implementation readiness against research goals.

**Research Goals**:
1. Run comprehensive experiments with automated metrics collection
2. Gather informable statistics for academic paper
3. Ensure correct algorithm logic for honest results

**Assessment Process**:
- Analyzed current implementation against all three goals
- Identified components that are complete and production-ready
- Identified critical gaps preventing paper-ready experiments
- Documented implementation requirements for each gap
- Created 3-phase implementation plan

**Document Created**: `docs/validation/experiment_readiness_assessment.md`

**Key Findings**:

**✅ Complete Components**:
- ExperimentRunner fully automated (YAML config, stopping criteria, export)
- MetricsCollector comprehensive (action tracking, windowed mistake rates, per-action stats)
- OLAM validated against paper (see OLAM_VALIDATION_REPORT.md)
- Information Gain fully implemented (~60 tests passing)
- Real PDDL execution environment
- Configuration system for multiple domains

**❌ Critical Gaps Identified (5 gaps)**:

1. **No Statistical Significance Testing**
   - Problem: Cannot determine if algorithm differences are statistically significant
   - Missing: t-tests, effect sizes (Cohen's d), confidence intervals, p-values
   - Required: `src/experiments/statistical_analysis.py` with `StatisticalAnalyzer` class
   - Impact: Cannot make scientifically valid claims in paper

2. **No Automated Algorithm Comparison Pipeline**
   - Problem: Must manually run experiments and compare results
   - Missing: Batch runner for multiple trials, result aggregation, comparison reports
   - Required: `scripts/compare_algorithms.py` with `AlgorithmComparisonRunner`
   - Impact: Time-consuming manual process, prone to errors

3. **No Convergence Detection Validation**
   - Problem: `has_converged()` methods may be unreliable or unimplemented
   - Missing: Validated convergence criteria for both algorithms
   - Required: Implement and test convergence detection for OLAM and Information Gain
   - Impact: Experiments may run longer than necessary or stop prematurely

4. **No Ground Truth Model Comparison**
   - Problem: Cannot verify learned models match PDDL specifications
   - Missing: Precision/recall calculation, effect accuracy, model comparison metrics
   - Required: `src/core/model_validator.py` with `ModelValidator` class
   - Impact: Cannot validate algorithm correctness

5. **No Information Gain Validation Report**
   - Problem: Unlike OLAM (which has validation report), Information Gain lacks systematic validation
   - Missing: Evidence of hypothesis space reduction, information gain-based selection, model learning
   - Required: `scripts/validate_information_gain.py` and validation report document
   - Impact: Algorithm correctness unverified against theoretical description

**Overall Assessment**: ⚠️ **MOSTLY READY** with critical gaps

**Implementation Estimate**: 6-9 days (3 phases)
- Phase 1: Statistical foundation (2-3 days)
- Phase 2: Algorithm validation (2-3 days)
- Phase 3: Comparison pipeline (2-3 days)

**Current Readiness**: 30% (3/10 paper-ready criteria met)
**After Gap Implementation**: 100% (10/10 criteria met)

**Status**: Assessment complete, ready for implementation planning

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

### Priority: Paper-Ready Experiment Infrastructure

Based on the Experiment Readiness Assessment, the following gaps must be addressed before conducting publishable comparative experiments.

### Phase 1: Statistical Foundation (2-3 days) - HIGH PRIORITY

**Goal**: Enable statistically valid algorithm comparisons

#### Gap #2: Statistical Significance Testing
**Component**: `src/experiments/statistical_analysis.py`

**Tasks**:
1. Implement `StatisticalAnalyzer` class
   - Paired t-test (scipy.stats.ttest_rel)
   - Cohen's d effect size calculation
   - Confidence interval computation (95% CI)
   - Multiple comparison correction (Bonferroni)
2. Create `StatisticalResult` dataclass
   - Store means, stds, CIs for both algorithms
   - Store test statistic, p-value, effect size
   - Include interpretation string
3. Unit tests with known data
4. Integration tests with sample experiment results

**Deliverable**: Can compute statistical significance for algorithm comparisons

#### Gap #5: Ground Truth Model Comparison
**Component**: `src/core/model_validator.py`

**Tasks**:
1. Implement `ModelValidator` class
   - Parse ground truth from PDDL domain files
   - Extract action preconditions and effects
   - Store in comparable format
2. Implement `compare_preconditions()` method
   - Calculate precision: TP / (TP + FP)
   - Calculate recall: TP / (TP + FN)
   - Calculate F1-score
   - Identify false positives and false negatives
3. Implement `compare_effects()` method
   - Add effect accuracy
   - Delete effect accuracy
4. Unit tests with hand-crafted domains
5. Integration tests with blocksworld

**Deliverable**: Can verify learned models against ground truth

### Phase 2: Algorithm Validation (2-3 days) - HIGH PRIORITY

**Goal**: Verify algorithm correctness before comparison

#### Gap #3: Convergence Detection Validation
**Components**:
- `src/algorithms/olam_adapter.py` (update `has_converged()`)
- `src/algorithms/information_gain.py` (update `has_converged()`)

**Tasks**:
1. **OLAM Convergence**:
   - Implement hypothesis space stability check (no changes for N iterations)
   - Implement high success rate check (>95% in last M actions)
   - Unit tests for each criterion
2. **Information Gain Convergence**:
   - Implement model stability check (no pre(a) changes)
   - Implement low information gain check (max gain < ε)
   - Implement high success rate check
   - Unit tests for each criterion
3. Integration tests showing early stopping
4. Validation: Does algorithm converge before max_iterations?

**Deliverable**: Reliable convergence detection for both algorithms

#### Gap #4: Information Gain Validation Report
**Components**:
- `scripts/validate_information_gain.py` (new)
- `docs/validation/INFORMATION_GAIN_VALIDATION_REPORT.md` (new)

**Tasks**:
1. Create validation script with verbose logging
   - Log initial hypothesis space size
   - Log action selections with information gain values
   - Log model updates after observations
   - Log hypothesis space reduction over time
   - Log model entropy decrease
2. Run validation experiment on blocksworld
3. Generate validation report documenting:
   - Hypothesis space reduction (evidence)
   - Information gain-based selection (evidence)
   - Model entropy decrease (evidence)
   - Ground truth comparison (using ModelValidator from Phase 1)
4. Verify algorithm behaves as theoretically expected

**Deliverable**: Validation report confirming Information Gain correctness

### Phase 3: Comparison Pipeline (2-3 days) - MEDIUM PRIORITY

**Goal**: Automate algorithm comparison experiments

#### Gap #1: Algorithm Comparison Pipeline
**Component**: `scripts/compare_algorithms.py`

**Tasks**:
1. Implement `AlgorithmComparisonRunner` class
   - Run multiple trials (≥5) for each algorithm
   - Control RNG seeds for fair comparison
   - Aggregate results across trials
2. Implement `_run_single_trial()` method
   - Wrapper around ExperimentRunner
   - Consistent configuration across trials
3. Implement `_generate_comparison_report()` method
   - Call StatisticalAnalyzer (from Phase 1)
   - Generate human-readable report
   - Export to CSV and JSON
4. Create visualization tools
   - Box plots for sample complexity comparison
   - Learning curves with error bars (95% CI)
   - Convergence time comparison
   - Model accuracy comparison
5. Optional: LaTeX table generation for papers

**Deliverable**: One-command algorithm comparison pipeline

**Usage**:
```bash
python scripts/compare_algorithms.py \
  --domain blocksworld \
  --problem p01 \
  --algorithms olam information_gain \
  --trials 5 \
  --max-iterations 200
```

### Phase 4: ModelLearner Integration (BLOCKED - Low Priority)

⚠️ **Blocked**: Repository https://github.com/kcleung/ModelLearner.git not accessible

**Future tasks** (when unblocked):
1. Find correct repository URL or alternative implementation
2. Create `OptimisticAdapter` class
3. Handle lifted_dict YAML requirements
4. Implement optimistic exploration strategy
5. Validate against ModelLearner paper
6. Three-way comparison (OLAM vs Information Gain vs ModelLearner)

### Implementation Priority Order

**Week 1**: Phase 1 (Statistical foundation)
- Day 1-2: StatisticalAnalyzer implementation + tests
- Day 2-3: ModelValidator implementation + tests

**Week 2**: Phase 2 (Algorithm validation)
- Day 1: Convergence detection implementation + tests
- Day 2-3: Information Gain validation script + report

**Week 3**: Phase 3 (Comparison pipeline)
- Day 1-2: AlgorithmComparisonRunner implementation
- Day 3: Visualization tools + final testing

**Total estimated time**: 6-9 days for paper-ready infrastructure

## Testing Status

- **Unit tests**: 51/51 passing (`make test`)
- **Integration tests**: OLAM fully tested
- **Validation**: OLAM paper behaviors confirmed

## File Structure
See [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) for complete directory structure.

## External Dependencies
See [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md#external-tool-paths) for external tool paths and availability status.

## Documentation Index
See [CLAUDE.md](../CLAUDE.md) for complete documentation navigation.