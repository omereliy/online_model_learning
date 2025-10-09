# Implementation Tasks for Online Model Learning Framework

## Project Overview
Building experiment framework to compare three online action model learning algorithms:
1. **OLAM** - Baseline (Python implementation) ✅
2. **Information Gain** - CNF/SAT-based information-theoretic approach ✅
3. **ModelLearner** - Optimistic Exploration (BLOCKED - repo unavailable)

## Quick Start
**See [CLAUDE.md](../CLAUDE.md) for navigation and [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) for project rules.**

## Current Status (Updated: October 9, 2025)

### 🎯 Major Refactoring Complete - Clean Layered Architecture ✅
**Date**: October 8, 2025
**Status**: COMPLETE - All components migrated, all tests passing
**Objective**: Separate PDDLHandler/PDDLEnvironment concerns into focused, testable components

**New Architecture** (1916 lines):
1. **UPAdapter** (`src/core/up_adapter.py`, 316 lines) - Bidirectional UP ↔ Project types converter
2. **LiftedDomainKnowledge** (`src/core/lifted_domain.py`, 557 lines) - Intermediate domain representation
3. **PDDL I/O** (`src/core/pddl_io.py`, 212 lines) - Read/write wrappers using UPAdapter
4. **Grounding Utilities** (`src/core/grounding.py`, 486 lines) - Functional grounding operations
5. **ActiveEnvironment** (`src/environments/active_environment.py`, 345 lines) - Minimal execution interface

**Benefits**:
- ✅ Clean separation: UP layer → Domain layer → Grounding layer → Environment layer
- ✅ Functional grounding: stateless, pure functions (easier testing)
- ✅ Minimal interfaces: ActiveEnvironment only exposes execution methods
- ✅ Better for learning: LiftedDomainKnowledge supports partial knowledge
- ✅ All tests pass: 51/51 curated tests passing

**Migrated Components** (October 8, 2025):
- ✅ `src/algorithms/information_gain.py` - Uses PDDLReader + grounding utilities
- ✅ `src/algorithms/olam_adapter.py` - Uses PDDLReader + LiftedDomainKnowledge + grounding
- ✅ `src/core/model_validator.py` - Uses PDDLReader
- ✅ `src/experiments/runner.py` - Uses ActiveEnvironment
- ✅ `scripts/olam_paper_validation.py` - Uses ActiveEnvironment
- ✅ `scripts/olam_learning_trace.py` - Uses ActiveEnvironment
- ✅ `scripts/benchmark_performance.py` - Uses PDDLReader + grounding

**Configuration Updates**:
- ✅ Makefile - Removed test_pddl_handler.py from ci-local
- ✅ .github/workflows/ci.yml - Updated test paths
- ✅ docker-compose.yml - Updated test paths
- ✅ .claude/agents/test_guardian.py - Updated affected test mappings

**Old Files Ready for Deletion** (~2176 lines):
- `src/core/pddl_handler.py` (1137 lines) - Replaced by PDDLReader + LiftedDomainKnowledge
- `src/core/binding_operations.py` (139 lines) - Replaced by grounding module
- `src/environments/pddl_environment.py` (279 lines) - Replaced by ActiveEnvironment
- `tests/core/test_pddl_handler.py` (289 lines) - Old architecture tests
- `tests/core/test_binding_operations.py` (165 lines) - Old architecture tests
- `tests/environments/test_pddl_environment.py` (167 lines) - Old architecture tests

### ✅ Completed Components

#### Core Infrastructure (Refactored Architecture)
- **UPAdapter** (`src/core/up_adapter.py`) - Complete
  - Stateless conversions: UP ↔ fluent sets, actions, expressions
  - `up_state_to_fluent_set()`, `fluent_set_to_up_state()`
  - `get_all_grounded_fluents()`, `get_initial_state_as_fluent_set()`
- **LiftedDomainKnowledge** (`src/core/lifted_domain.py`) - Complete
  - Central domain representation with lifted actions/predicates
  - Supports partial knowledge for learning algorithms
  - Type hierarchy operations: `is_subtype()`, `get_type_ancestors()`
  - Parameter-bound literals: `get_parameter_bound_literals(action_name)`
- **PDDL I/O** (`src/core/pddl_io.py`) - Complete
  - `PDDLReader.parse_domain_and_problem()` → (domain, initial_state)
  - `PDDLWriter` (stub for future export)
  - Convenience function: `parse_pddl(domain_file, problem_file)`
- **Grounding Utilities** (`src/core/grounding.py`) - Complete
  - Functional/stateless grounding operations
  - `ground_action()`, `ground_all_actions(require_injective=bool)`
  - `ground_parameter_bound_literal()`, `lift_grounded_fluent()`
  - `parse_grounded_action_string()`, batch operations
- **ActiveEnvironment** (`src/environments/active_environment.py`) - Complete
  - Minimal execution-only interface (grounded actions only)
  - Methods: `get_state()`, `execute()`, `execute_plan()`, `reset()`, `is_goal_reached()`
  - Optional queries: `get_applicable_actions()`, `get_all_grounded_actions()`
- **CNF Manager** (`src/core/cnf_manager.py`) - Complete
  - Constraint-based operations for algorithm separation
  - SAT model counting with state constraints
- **Domain Analyzer** (`src/core/domain_analyzer.py`) - Algorithm compatibility checking

#### OLAM Integration
- **OLAM Adapter** (`src/algorithms/olam_adapter.py`) - Fully validated against paper
- **Base Learner** (`src/algorithms/base_learner.py`) - Abstract interface
- **Java Requirement** - Oracle JDK 17+ required for production use
  - Location: `/home/omer/projects/OLAM/Java/jdk-*/`
  - Bypass mode (`bypass_java=True`) only for testing
  - See [OLAM Experiment Review](reviews/experimenting_OLAM_review.md)
- **Domain compatibility** - Works with blocksworld/gripper (no negative preconditions)
- **Validation confirmed** - All 4 paper behaviors demonstrated

#### Experiment Framework
- **Runner** (`src/experiments/runner.py`) - YAML-based experiments
- **Metrics** (`src/experiments/metrics.py`) - Comprehensive tracking
- **Statistical Analysis** (`src/experiments/statistical_analysis.py`) - Complete (Oct 6)
  - Paired t-test for algorithm comparison
  - Cohen's d effect size calculation
  - 95% confidence intervals
  - Bonferroni correction for multiple comparisons
  - 9 tests passing
- **Model Validator** (`src/core/model_validator.py`) - Complete (Oct 6)
  - Ground truth parsing from PDDL domains
  - Precision/recall/F1-score for preconditions and effects
  - False positive/negative identification
  - 15 tests passing
- **Results export** - CSV/JSON formats

#### Information Gain Algorithm
- **Information Gain Learner** (`src/algorithms/information_gain.py`) - Complete with Phase 3
  - Applicability probability calculation using SAT model counting
  - Entropy and information gain calculations
  - Three selection strategies (greedy, epsilon-greedy, Boltzmann)
  - **REFACTORED (Oct 8)**: Updated to use new architecture
    - Now uses `PDDLReader` + `LiftedDomainKnowledge` instead of PDDLHandler
    - Uses `grounding` module for all bindP/bindP⁻¹ operations
    - Cleaner separation: algorithm logic vs. domain knowledge vs. grounding
  - ~60 tests passing
- See [INFORMATION_GAIN_ALGORITHM.md](information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md)

### ⏳ TODO Components

#### ModelLearner Integration
- `src/algorithms/optimistic_adapter.py` - Adapter implementation
- See [ModelLearner_interface.md](external_repos/ModelLearner_interface.md)

## Recent Updates (October 9, 2025)

### OLAM Configuration Parameters - Extended Support (October 9, 2025)
**Context**: OLAM review document identified missing OLAM Configuration.py parameters that weren't exposed through the adapter.

**Problem**:
- OLAM has 6 important Configuration.py parameters not accessible via adapter
- No way to fine-tune OLAM's advanced features from YAML configs
- Missing: `PLANNER_TIME_LIMIT`, `MAX_PRECS_LENGTH`, `NEG_EFF_ASSUMPTION`, `OUTPUT_CONSOLE`, `RANDOM_SEED`, `TIME_LIMIT_SECONDS`

**Solution**: Added configuration parameter passthrough to OLAMAdapter
- **OLAMAdapter parameters** (`src/algorithms/olam_adapter.py`):
  - `planner_time_limit`: OLAM planner subprocess timeout (seconds)
  - `max_precs_length`: Negative precondition search depth
  - `neg_eff_assumption`: STRIPS negative effects assumption
  - `output_console`: Console vs file logging
  - `random_seed`: Numpy random seed
  - `time_limit_seconds`: Total experiment timeout (seconds)
- **Configuration method** (`_configure_java_settings()`):
  - Applies parameters to OLAM's Configuration module if provided
  - None values use OLAM's defaults (no override)
  - Logs each parameter setting for debugging
- **YAML config** (`configs/experiment_blocksworld.yaml`):
  - Added all 6 parameters with inline documentation
  - Marked as optional (uses OLAM defaults if omitted)

**Tests**: 11 tests in `tests/algorithms/test_olam_configuration.py`
- Parameter passthrough validation
- Multiple parameter combinations
- Type checking
- None value handling (no override)
- All tests passing

**Benefits**:
- Full control over OLAM's advanced features
- Configuration documented in YAML files
- Simple passthrough (no validation added)
- Backward compatible (all parameters optional)
- ExperimentRunner automatically passes parameters via **kwargs

**Status**: COMPLETE - All 51 curated tests + 11 new tests passing

### Documentation Update - New Architecture (October 9, 2025)
**Context**: Updated all documentation to reflect the new layered architecture after major refactoring.

**Changes**:
1. **Archived**: `UNIFIED_PLANNING_GUIDE.md` → `archive/UNIFIED_PLANNING_GUIDE_OLD_ARCHITECTURE.md`
   - Extensive PDDLHandler references (deleted class)
   - FluentBinder references (deleted module)
   - Retained as historical reference

2. **Rewritten**: `LIFTED_SUPPORT.md` (433 lines)
   - Now focuses on new architecture: `LiftedDomainKnowledge`, `grounding.py`, `UPAdapter`
   - Removed all PDDLHandler/FluentBinder references
   - Added complete examples for new API
   - Migration guide from old to new architecture

3. **Updated**: `integration_guide.md`
   - Updated OLAM adapter example to use new architecture
   - Updated all challenge/solution examples
   - Marked ModelLearner as conceptual (repository unavailable)

4. **Updated**: `CLAUDE.md`
   - Removed UNIFIED_PLANNING_GUIDE reference
   - Updated navigation to point to LIFTED_SUPPORT for PDDL/domain work

**Benefits**:
- All documentation now accurate for new architecture
- Clear migration paths from old code
- No confusing references to deleted components

### Architecture Refactoring - Layered Architecture (October 8, 2025)
**Objective**: Replace monolithic PDDLHandler/PDDLEnvironment with clean, focused components.

**Problem**:
- PDDLHandler (1137 lines) had 9+ responsibilities: parsing, state conversion, action grounding, precondition extraction, CNF, type hierarchy, export
- PDDLEnvironment had duplicate fluent conversion logic
- Tight coupling between unrelated concerns

**Solution**: Created 5 new focused components with clean separation:
1. **UPAdapter** - Stateless UP ↔ Project type conversions
2. **LiftedDomainKnowledge** - Central domain representation (supports partial knowledge)
3. **PDDL I/O** - Thin wrappers for reading/writing PDDL
4. **Grounding Utilities** - Pure functions for grounding/lifting operations
5. **ActiveEnvironment** - Minimal execution-only interface

**Benefits**:
- Single Responsibility Principle - each component has one clear purpose
- Functional grounding - stateless, easier to test
- Minimal interfaces - only expose what's needed
- Better for learning algorithms - LiftedDomainKnowledge supports partial knowledge
- Zero test changes - all 51 curated tests still pass
- Clear data flow: UP → Adapter → Domain → Grounding → Environment

**Migration Path**:
- Old code using PDDLHandler/PDDLEnvironment still works (not removed yet)
- New code uses: `parse_pddl()` → `LiftedDomainKnowledge` + `grounding` utilities + `ActiveEnvironment`
- InformationGainLearner updated to new architecture
- ExperimentRunner updated to use ActiveEnvironment

### Legacy Architecture Refactoring (October 5-8, 2025) - COMPLETE ✅

**Historical Context**: This project underwent a major refactoring from PDDLHandler/PDDLEnvironment monolithic architecture to a clean layered architecture. The detailed history has been archived.

**Key Accomplishments**:
- Created type-safe data classes (ParameterBinding, ParameterBoundLiteral, GroundedFluent, GroundedAction)
- Extracted ExpressionConverter for FNode conversions
- Extracted FluentBinder for bindP/bindP⁻¹ operations
- Replaced PDDLHandler (1137 lines) with focused components: UPAdapter, LiftedDomainKnowledge, PDDL I/O
- Replaced PDDLEnvironment with ActiveEnvironment (minimal execution interface)
- Migrated all dependent code to new architecture
- All 51 curated tests passing throughout

**For Detailed History**: See archived document `docs/archive/pddl_handler_refactoring_history.md` (if needed for reference)

### Experiment Readiness Assessment (October 5, 2025)
**Context**: Assessment conducted to identify gaps before running paper-ready experiments comparing OLAM and Information Gain.

**Complete Infrastructure**:
- ExperimentRunner (YAML config, stopping criteria, export)
- MetricsCollector (comprehensive action tracking)
- OLAM validated (see OLAM_VALIDATION_REPORT.md)
- Information Gain implemented (~60 tests passing)
- Real PDDL execution environment

**Critical Gaps Identified**: 5 gaps documented in `docs/validation/experiment_readiness_assessment.md`
1. Statistical significance testing (t-tests, Cohen's d, CIs) → **RESOLVED (Phase 1, Oct 6)**
2. Automated comparison pipeline (batch runner, aggregation)
3. Convergence detection validation (reliable stopping criteria)
4. Ground truth model comparison (precision/recall) → **RESOLVED (Phase 1, Oct 6)**
5. Information Gain validation report (algorithm correctness verification)

**Status**: 2/5 gaps resolved (Phase 1 complete), 3 gaps remaining for paper-ready experiments

### Phase 1 Implementation - Statistical Foundation (October 6, 2025)
**Context**: Implemented critical infrastructure for statistically valid algorithm comparisons (Gaps #2 and #5 from Experiment Readiness Assessment).

**Implementation Summary**:

**Gap #2: Statistical Significance Testing - COMPLETE**
- **Component**: `src/experiments/statistical_analysis.py`
- **Classes**: `StatisticalAnalyzer`, `StatisticalResult` dataclass
- **Functionality**:
  - Paired t-test using scipy.stats.ttest_rel
  - Cohen's d effect size calculation with interpretation (small/medium/large)
  - 95% confidence interval computation using t-distribution
  - Bonferroni correction for multiple comparisons
  - Human-readable result interpretations
- **Tests**: 9 tests in `tests/experiments/test_statistical_analysis.py`
  - Known t-test examples with hand-calculated values
  - Cohen's d validation
  - Confidence interval calculation verification
  - Bonferroni correction validation
  - Integration test with realistic experiment data
  - Edge cases: equal performance, small sample sizes
- **Status**: All tests passing (9/9)

**Gap #5: Ground Truth Model Comparison - COMPLETE**
- **Component**: `src/core/model_validator.py`
- **Classes**: `ModelValidator`, `ModelComparisonResult` dataclass
- **Functionality**:
  - Ground truth parsing from PDDL domain files using PDDLHandler
  - Precondition precision/recall/F1-score calculation
  - Add effect precision/recall/F1-score calculation
  - Delete effect precision/recall/F1-score calculation
  - False positive/negative identification
  - Overall model accuracy metric (average of three F1 scores)
- **Tests**: 15 tests in `tests/core/test_model_validator.py`
  - Perfect match validation
  - Missing preconditions (false negatives)
  - Extra preconditions (false positives)
  - Missing/extra effects validation
  - Precision/recall/F1 calculation with known values
  - Edge cases: empty learned models, over-general models, propositional actions
  - Integration tests with real blocksworld PDDL domain
- **Status**: All tests passing (15/15)

**Integration Validation**:
- All 51 curated tests still passing (make test)
- No regressions in existing functionality
- Total new tests added: 24 (9 statistical + 15 model validator)
- Ready for use in Phase 2 (Algorithm Validation)

**Benefits Achieved**:
- Can compute statistical significance between OLAM and Information Gain results
- Can verify learned models against ground truth PDDL specifications
- Foundation established for algorithm validation (Gap #4, Phase 2)
- Foundation established for comparison pipeline (Gap #1, Phase 3)

### Experiment Configuration for Statistical Validity (October 9, 2025)
**Context**: Analysis of current configuration settings revealed aggressive convergence parameters and insufficient timeouts that may prevent collection of statistically valid experiment data.

**Critical Configuration Issues Identified**:

**1. Aggressive Convergence Parameters (Information Gain)**
- **Current (hardcoded)**: `MODEL_STABILITY_WINDOW=10`, `INFO_GAIN_EPSILON=0.01`, `SUCCESS_RATE_THRESHOLD=0.95`
- **Problem**:
  - Model stable for just 10 iterations triggers convergence (too short)
  - 95% success (19/20 actions) too easy to satisfy
  - Convergence logic: ANY 2 of 3 criteria (too aggressive)
- **Impact**: Premature convergence → insufficient data for statistical analysis
- **Recommended**: `MODEL_STABILITY_WINDOW=50`, `INFO_GAIN_EPSILON=0.001`, `SUCCESS_RATE_THRESHOLD=0.98`
- **Fix Required**: Make parameters configurable via YAML (currently hardcoded in `src/algorithms/information_gain.py:40-43`)

**2. Short Planner Timeouts**
- **Current**: 60 seconds per planning call
- **Problem**: Complex problems need longer planning → artificial failures contaminate learning signal
- **Impact**: False action failures affect model learning
- **Recommended**: 180-300 seconds for full experiments
- **Status**: ✅ Already configurable via `planner_time_limit` parameter (OLAM)

**3. Low Max Iterations**
- **Current**: 200 iterations
- **Problem**: Insufficient sample size for statistical power (need 500-1000+ for medium effect sizes)
- **Impact**: Weak statistical conclusions, high variance
- **Recommended**: 1000+ iterations for paper-ready experiments
- **Status**: ✅ Already configurable via `max_iterations` parameter

**4. Frequent Convergence Checks**
- **Current**: Every 20 iterations
- **Problem**: Increases chance of premature convergence
- **Recommended**: Every 100 iterations, or disable (set to 99999) for data collection
- **Status**: ✅ Already configurable via `convergence_check_interval`

**Configuration Guide Created**: `docs/EXPERIMENT_CONFIGURATION_GUIDE.md`
- Conservative configuration templates for full experiments
- Statistical power analysis and sample size requirements
- Rationale for each parameter recommendation
- Phase-based configuration strategy (exploratory → data collection → final validation)

**Status**: Analysis complete, configuration guide created, Information Gain parameters need code changes

## Next Implementation Tasks

### Priority: Paper-Ready Experiment Infrastructure

Based on the Experiment Readiness Assessment and configuration analysis, the following gaps must be addressed before conducting publishable comparative experiments.

### Missing Components Analysis for Full Experiments (October 9, 2025)

**Context**: Comprehensive assessment of what's needed to run full OLAM and Information Gain experiments for academic paper.

**Current Project Readiness: 60-70% Complete**

**✅ COMPLETE - Core Infrastructure**:
- **ExperimentRunner** (`src/experiments/runner.py`) - Fully automated with YAML configs
- **MetricsCollector** (`src/experiments/metrics.py`) - Comprehensive action tracking with windowing
- **StatisticalAnalyzer** (`src/experiments/statistical_analysis.py`) - Phase 1 complete (Oct 6)
  - Paired t-tests, Cohen's d, confidence intervals, Bonferroni correction
  - 9 tests passing
- **ModelValidator** (`src/core/model_validator.py`) - Phase 1 complete (Oct 6)
  - Ground truth comparison, precision/recall/F1-score
  - 15 tests passing
- **OLAM Integration** - Fully validated against paper with configurable parameters
- **Information Gain Implementation** - Algorithm complete with ~60 tests passing
- **PDDL Execution** - ActiveEnvironment with real action execution
- **Configuration System** - YAML-based with multiple domain support

**❌ MISSING - Critical Gaps (3 Remaining)**:

**Gap #1: Automated Comparison Pipeline** (NOT IMPLEMENTED)
- **File**: `scripts/compare_algorithms.py` - **DOES NOT EXIST**
- **Problem**: No batch execution of multiple trials with aggregated results
- **Required Functionality**:
  - `AlgorithmComparisonRunner` class for multi-trial experiments
  - Automated YAML config generation for trial batches
  - Result aggregation across trials (mean, std, CI)
  - Comparison report generation with statistical tests
  - Visualization: learning curves, box plots, convergence time
- **Impact**: Manual experiment execution, time-consuming, error-prone
- **Estimated Work**: 2-3 days

**Gap #2: Information Gain Convergence Configurability** (PARTIALLY IMPLEMENTED)
- **File**: `src/algorithms/information_gain.py:40-43` - **PARAMETERS HARDCODED**
- **Problem**: Convergence parameters are class constants, not configurable via YAML
- **Required Changes**:
  - Add convergence parameters to `__init__()` method
  - Accept from YAML configs via ExperimentRunner
  - Update convergence check logic to use instance variables
  - Add tests for configurable parameters
- **Current Hardcoded Values**: `MODEL_STABILITY_WINDOW=10`, `INFO_GAIN_EPSILON=0.01`, etc.
- **Impact**: Cannot use conservative settings for full experiments
- **Estimated Work**: 4-6 hours

**Gap #3: Information Gain Validation Report** (NOT CREATED)
- **File**: `scripts/validate_information_gain.py` - **DOES NOT EXIST**
- **Report**: `docs/validation/INFORMATION_GAIN_VALIDATION_REPORT.md` - **DOES NOT EXIST**
- **Problem**: Unlike OLAM (which has validation report), Information Gain lacks systematic validation
- **Required Validation**:
  - Hypothesis space reduction evidence (CNF formula size tracking)
  - Information gain calculation correctness
  - Action selection based on information gain values
  - Model learning convergence to ground truth
  - Comparison with theoretical behavior
- **Impact**: Algorithm correctness unverified against theory
- **Estimated Work**: 1-2 days

**Configuration Infrastructure Gaps**:
- ❌ Conservative configuration templates (blocksworld, gripper, etc.) with 1000+ iterations
- ❌ Multi-trial batch configuration generator
- ✅ Individual algorithm configs (exist but use exploratory settings)

**Visualization Gaps**:
- ❌ Learning curve plotting with confidence intervals
- ❌ Box plots for sample complexity comparison
- ❌ Convergence time visualization
- ✅ Basic metrics plotting (exists in MetricsCollector)

**Overall Assessment**:
- **OLAM Experiment Readiness**: 85% (only needs comparison pipeline)
- **Information Gain Experiment Readiness**: 70% (needs configurability + validation + pipeline)
- **Paper-Ready Analysis**: 60% (needs visualization + automated comparison)

**Estimated Work to 100%**: 3-4 days focused implementation

### Phase 1: Statistical Foundation - COMPLETE ✅

**Goal**: Enable statistically valid algorithm comparisons

**Status**: COMPLETE (October 6, 2025) - Both gaps implemented and tested

#### Gap #2: Statistical Significance Testing - COMPLETE ✅
**Component**: `src/experiments/statistical_analysis.py`

**Completed Tasks**:
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

#### Gap #5: Ground Truth Model Comparison - COMPLETE ✅
**Component**: `src/core/model_validator.py`

**Completed Tasks**:
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

**Goal**: Verify algorithm correctness and enable flexible configuration before comparison

#### Gap #2: Information Gain Convergence Configurability (NEW - CRITICAL)
**Component**: `src/algorithms/information_gain.py`
**Status**: ⚠️ BLOCKING ISSUE - Parameters hardcoded, prevents statistical validity

**Tasks**:
1. **Make convergence parameters configurable**:
   - Add to `__init__()`: `model_stability_window`, `info_gain_epsilon`, `success_rate_threshold`, `success_rate_window`
   - Set conservative defaults: 50, 0.001, 0.98, 50 (currently: 10, 0.01, 0.95, 20)
   - Update `has_converged()` to use instance variables instead of class constants
   - Update convergence logic: require ALL 3 criteria (currently ANY 2 of 3)
2. **Update configuration files**:
   - Add parameters to `configs/information_gain_blocksworld.yaml`
   - Create conservative template: `configs/full_experiment_information_gain.yaml`
3. **Add tests** (8-10 new tests):
   - Parameter passthrough validation
   - Conservative vs aggressive convergence behavior
   - YAML configuration parsing
   - Convergence logic correctness (ALL 3 criteria)
4. **Update ExperimentRunner**:
   - Verify parameters passed via `**kwargs`
   - Add validation for parameter ranges

**Deliverable**: Configurable convergence for statistical validity ⚠️ **REQUIRED FOR FULL EXPERIMENTS**

#### Gap #3: Convergence Detection Validation
**Components**:
- `src/algorithms/olam_adapter.py` (update `has_converged()`)
- `src/algorithms/information_gain.py` (validate updated logic)

**Tasks**:
1. **OLAM Convergence Validation**:
   - Test existing convergence with conservative settings (500+ iterations)
   - Verify hypothesis space stability tracking
   - Unit tests for convergence criteria
2. **Information Gain Convergence Validation** (after Gap #2 implementation):
   - Test model stability check with configurable window
   - Test information gain threshold with configurable epsilon
   - Test success rate with configurable threshold and window
   - Verify ALL 3 criteria required (not ANY 2)
   - Integration tests with conservative settings
3. Validation: Can algorithms run 1000+ iterations without premature convergence?

**Deliverable**: Reliable, tested convergence detection with conservative settings

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

**Goal**: Automate algorithm comparison experiments with visualization

#### Gap #1: Algorithm Comparison Pipeline
**Component**: `scripts/compare_algorithms.py` (NEW - DOES NOT EXIST)

**Tasks**:

**1. AlgorithmComparisonRunner Class Implementation**:
   - Constructor: Accept domain, problem, algorithms list, trials count, base config
   - `run_comparison()`: Main entry point for batch experiments
   - `_run_single_trial(algorithm, seed)`: Execute one experiment trial
     - Wrapper around ExperimentRunner
     - Load YAML config with trial-specific seed
     - Return MetricsCollector results
   - `_aggregate_results()`: Combine results across trials
     - Extract sample complexity (actions to convergence)
     - Extract final model accuracy (using ModelValidator)
     - Extract learning curves (windowed mistake rates)
     - Calculate mean, std, 95% CI for each metric
   - `_generate_comparison_report()`: Statistical analysis and report generation
     - Call StatisticalAnalyzer.compare_algorithms() for sample complexity
     - Call StatisticalAnalyzer.compare_algorithms() for final accuracy
     - Generate markdown report with tables
     - Export raw data to CSV and JSON
   - `_create_visualizations()`: Plot generation (see Gap #1B below)

**2. Configuration Management**:
   - Accept base YAML config file
   - Generate trial-specific configs with different seeds
   - Ensure consistent parameters across algorithms (except algorithm-specific)
   - Support conservative configuration templates

**3. Result Aggregation Format**:
   ```python
   {
     'algorithm_name': {
       'sample_complexity': {'mean': float, 'std': float, 'ci_95': (float, float), 'trials': List[int]},
       'final_accuracy': {'mean': float, 'std': float, 'ci_95': (float, float), 'trials': List[float]},
       'learning_curves': List[List[float]],  # Per-trial windowed mistake rates
       'convergence_iterations': List[int],   # Per-trial convergence points
       'runtime_seconds': {'mean': float, 'std': float, 'trials': List[float]}
     }
   }
   ```

**4. Command-Line Interface**:
   ```bash
   python scripts/compare_algorithms.py \
     --domain blocksworld \
     --problem p01 \
     --algorithms olam information_gain \
     --trials 10 \
     --max-iterations 1000 \
     --config configs/full_experiment_template.yaml \
     --output results/comparison_blocksworld_p01/
   ```

**5. Tests** (10-12 new tests):
   - Single trial execution
   - Multi-trial aggregation
   - Statistical analysis integration
   - Result export formats
   - Configuration override handling

**Deliverable**: Automated multi-trial comparison pipeline with statistical analysis

#### Gap #1B: Visualization Module
**Component**: `src/experiments/visualization.py` (NEW - DOES NOT EXIST)

**Tasks**:

**1. Learning Curve Plotting**:
   - `plot_learning_curves(results_dict, title, output_path)`:
     - X-axis: iteration number
     - Y-axis: windowed mistake rate or success rate
     - Plot mean curve with 95% confidence interval shading
     - Multiple algorithms on same plot (different colors)
     - Legend, axis labels, grid
   - Support matplotlib and seaborn backends
   - Publication-quality output (PDF, 300 DPI)

**2. Sample Complexity Comparison**:
   - `plot_sample_complexity_boxplot(results_dict, title, output_path)`:
     - Box plot showing distribution across trials
     - Algorithms on X-axis, sample complexity on Y-axis
     - Show mean, median, quartiles, outliers
     - Statistical significance markers (*, **, ***)
   - `plot_sample_complexity_bars(results_dict, title, output_path)`:
     - Bar chart with error bars (95% CI)
     - Cleaner alternative to box plots

**3. Convergence Analysis**:
   - `plot_convergence_times(results_dict, title, output_path)`:
     - Histogram of convergence iterations per algorithm
     - Side-by-side or overlapping distributions
   - `plot_convergence_comparison(results_dict, title, output_path)`:
     - CDF plot: P(convergence by iteration N)

**4. Model Accuracy Visualization**:
   - `plot_accuracy_comparison(results_dict, title, output_path)`:
     - Grouped bar chart: precondition F1, add effect F1, delete effect F1
     - Error bars for 95% CI
     - Overall accuracy as separate metric

**5. Utility Functions**:
   - `apply_publication_style()`: Set matplotlib rcParams for papers
   - `generate_comparison_dashboard(results_dict, output_dir)`: Create all plots
   - LaTeX table generation: `export_latex_table(results_dict, output_path)`

**6. Tests** (8-10 new tests):
   - Plot generation with sample data
   - Confidence interval rendering
   - Multi-algorithm plot handling
   - Export format validation

**Deliverable**: Comprehensive visualization suite for paper-ready figures

#### Configuration Templates
**Components**:
- `configs/full_experiment_olam.yaml` (NEW)
- `configs/full_experiment_information_gain.yaml` (NEW)

**Content**: Conservative settings for paper-ready experiments
- `max_iterations: 1000`
- `planner_time_limit: 180`
- `convergence_check_interval: 100` (or 99999 to disable)
- Information Gain: `model_stability_window: 50`, `info_gain_epsilon: 0.001`, `success_rate_threshold: 0.98`
- Reference to EXPERIMENT_CONFIGURATION_GUIDE.md for rationale

### Configuration Best Practices for Full Experiments

**Context**: Guidelines for setting up statistically valid experiments based on configuration analysis.

**Exploratory Phase** (quick validation, convergence enabled):
```yaml
max_iterations: 200
planner_time_limit: 60
convergence_check_interval: 20
# Use default aggressive convergence for quick feedback
```
**Use case**: Algorithm debugging, parameter tuning, quick sanity checks

**Data Collection Phase** (full runs, no early stopping):
```yaml
max_iterations: 1000
planner_time_limit: 180  # 3 minutes per planning call
convergence_check_interval: 99999  # Effectively disabled
algorithm_params:
  information_gain:
    model_stability_window: 50      # Conservative
    info_gain_epsilon: 0.001         # Conservative
    success_rate_threshold: 0.98     # Conservative
    success_rate_window: 50          # Conservative
```
**Use case**: Initial data collection for paper, algorithm comparison

**Publication-Ready Phase** (maximum rigor):
```yaml
max_iterations: 1000
planner_time_limit: 300  # 5 minutes per planning call
convergence_check_interval: 99999  # Disabled
# Run 10+ trials per algorithm with different seeds
# Conservative convergence parameters (as above)
```
**Use case**: Final experiments for paper submission

**Key Principles**:
1. **Disable convergence for data collection**: Set `convergence_check_interval: 99999`
2. **Longer is better**: Use 1000+ iterations for sufficient statistical power
3. **Generous timeouts**: 180-300 seconds prevents false failures
4. **Multiple trials**: 10+ trials per algorithm for robust statistics
5. **Fixed seeds**: Control randomness for reproducibility

**Sample Size Requirements** (for medium effect size, Cohen's d=0.5, α=0.05, power=0.80):
- **Minimum**: 5 trials per algorithm
- **Recommended**: 10 trials per algorithm
- **Ideal**: 20+ trials per algorithm

See `docs/EXPERIMENT_CONFIGURATION_GUIDE.md` for detailed rationale and examples.

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

**Phase 1**: Statistical foundation - ✅ **COMPLETE** (October 6, 2025)

**Phase 2**: Algorithm validation + configuration flexibility (2-3 days)
- Day 1: Make Information Gain convergence configurable (Gap #2) ⚠️ **BLOCKING**
- Day 2: Convergence detection validation (Gap #3)
- Day 3: Information Gain validation script + report (Gap #4)

**Phase 3**: Comparison pipeline + visualization (2-3 days)
- Day 1-2: AlgorithmComparisonRunner implementation (Gap #1)
- Day 2-3: Visualization module + configuration templates (Gap #1B)

**Total remaining time**: 4-6 days for paper-ready infrastructure (from 60% → 100%)

## Testing Status

- **Curated test suite**: 51/51 passing (`make test`)
- **Total test files**: 25 test modules
- **Total available tests**: 431 tests (via pytest)
- **Phase 1 new tests**: 24/24 passing
  - Statistical analysis: 9 tests
  - Model validator: 15 tests
- **Integration tests**: OLAM fully tested, ModelValidator integrated with PDDLHandler
- **Validation**: OLAM paper behaviors confirmed
- **Configuration flexibility**:
  - ✅ OLAM: 11 configuration tests passing (October 9) - planner timeouts, max iterations, convergence configurable
  - ❌ Information Gain: Convergence parameters hardcoded (Gap #2) - **NEEDS IMPLEMENTATION**
  - ✅ ExperimentRunner: Supports arbitrary algorithm parameters via `**kwargs`

## File Structure
See [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) for complete directory structure.

## External Dependencies
See [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md#external-tool-paths) for external tool paths and availability status.

## Documentation Index
See [CLAUDE.md](../CLAUDE.md) for complete documentation navigation.