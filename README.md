# Online Action Model Learning Experiment Framework

## Overview
A unified framework for testing and comparing online action model learning algorithms in PDDL domains, with advanced CNF/SAT solver integration for uncertainty representation.

## Algorithms Implemented
1. **OLAM** - Online Learning of Action Models (Lamanna et al., 2021) [Python]
2. **ModelLearner (Optimistic)** - Optimistic exploration with symbolic model estimates (Sreedharan et al., 2023) [Python]
3. **Information-Theoretic Selection** - Novel CNF-based approach using expected information gain with SAT solver integration

## Architecture

```

workspace/
├── online-model-learning-framework/  # Your project
├── OLAM/                             # External OLAM repo (GitHub)
├── ModelLearner/                     # External ModelLearner repo (GitHub)
└── unified-planning/                 # Unified Planning Framework


online-model-learning-framework/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pddl_handler.py        # PDDL parsing with lifted support
│   │   ├── cnf_manager.py         # CNF formulas with lifted fluents
│   │   ├── state.py               # State representation
│   │   ├── action.py              # Action representation
│   │   └── pddl_model.py          # PDDL model representation
│   │
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base_learner.py        # Abstract base class
│   │   ├── olam_adapter.py        # OLAM wrapper
│   │   ├── optimistic_adapter.py  # ModelLearner wrapper
│   │   └── information_gain.py    # CNF-based information-theoretic approach
│   │
│   ├── sat_integration/
│   │   ├── __init__.py
│   │   ├── cnf_builder.py         # Build CNF formulas from uncertainty
│   │   ├── sat_solver.py          # PySAT solver interface
│   │   ├── formula_minimizer.py   # Formula optimization
│   │   └── variable_mapper.py     # Fluent to CNF variable mapping
│   │
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── unified_planning_interface.py  # Unified Planning Framework wrapper
│   │   └── planner_utils.py       # Planning utilities
│   │
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── simulator.py           # PDDL simulator using UP
│   │   └── domains/               # Domain-specific files
│   │
│   └── experiments/
│       ├── __init__.py
│       ├── runner.py              # Experiment runner
│       ├── metrics.py             # Performance metrics
│       └── visualizer.py          # Results visualization
│
├── benchmarks/                     # PDDL domain files
│   ├── blocksworld/
│   ├── gripper/
│   └── ...
│
├── results/                        # Experiment results
├── tests/                          # Unit tests
└── configs/                        # Configuration files
```

## Key Design Principles

1. **Modular Architecture**: Each algorithm implements a common interface
2. **Unified Planning Integration**: Standardized PDDL parsing and planning
3. **CNF/SAT Solver Integration**: Advanced uncertainty representation using PySAT
4. **Metric Collection**: Unified metrics for comparison
5. **Reproducibility**: Seed control and configuration management

## Tech Stack

- **Unified Planning Framework**: PDDL parsing, planning, and simulation
- **PySAT**: CNF formula manipulation and SAT solving (minisat)
- **External Algorithms**: OLAM and ModelLearner Python implementations
- **Scientific Python**: NumPy, Pandas, Matplotlib for analysis

## Installation

```bash
# Install core dependencies
pip install unified-planning pysat numpy pandas matplotlib

# Install additional UP integrations (optional)
pip install unified-planning[pyperplan,tamer]

# Install requirements
pip install -r requirements.txt
```

## Usage

```python
from src.experiments.runner import ExperimentRunner
from src.algorithms.olam_adapter import OLAMAdapter
from src.algorithms.optimistic_adapter import OptimisticAdapter
from src.algorithms.information_gain import InformationGainLearner

# Configure experiment
config = {
    'domain': 'blocksworld',
    'problems': ['p01', 'p02', 'p03'],
    'algorithms': ['olam', 'optimistic', 'information_gain'],
    'metrics': ['sample_complexity', 'time_to_goal', 'model_accuracy', 'cnf_formula_size'],
    'seed': 42,
    'cnf_settings': {
        'solver': 'minisat',
        'minimize_formulas': True,
        'max_clauses': 1000
    }
}

# Run experiments
runner = ExperimentRunner(config)
results = runner.run()
runner.visualize_results(results)
```

## Information-Theoretic Algorithm Features

### CNF-Based Uncertainty Representation
- **Precondition Uncertainty**: CNF formulas over fluent variables
- **Effect Uncertainty**: Separate CNF formulas for add/delete effects
- **SAT Solver Integration**: Model counting for information gain calculation
- **Formula Minimization**: Compact representation of learned constraints

### Key Components

#### Lifted Fluent and Action Support
The framework now supports both lifted (parameterized) and grounded representations:

```python
from src.core.cnf_manager import CNFManager
from src.core.pddl_handler import PDDLHandler

# CNF Manager with lifted fluents
cnf = CNFManager()
cnf.add_lifted_fluent("on", ["?x", "?y"])  # Lifted predicate
cnf.add_lifted_fluent("clear", ["?x"])

# Add lifted clause: on(?x,?y) → ¬clear(?y)
cnf.add_clause(["on(?x,?y)", "-clear(?y)"], lifted=True)

# Ground the clause for specific objects
bindings = {"?x": "a", "?y": "b"}
grounded = cnf.instantiate_lifted_clause(["on(?x,?y)", "-clear(?y)"], bindings)
# Result: ['on_a_b', '-clear_b']

# PDDL Handler with lifted actions
handler = PDDLHandler()
handler.parse_domain_and_problem(domain_file, problem_file)

# Get lifted action structure
action = handler.get_lifted_action("pick-up")
lifted_preconds = handler.get_action_preconditions("pick-up", lifted=True)
# Returns: {'clear(?x)', 'ontable(?x)', 'handempty'}

# Extract CNF representation
cnf_clauses = handler.extract_lifted_preconditions_cnf("pick-up")
```

#### CNF-Based Uncertainty Representation
```python
# CNF formula for precondition uncertainty
precond_cnf = CNFManager()
precond_cnf.add_fluent('on_a_b')
precond_cnf.add_fluent('clear_c')
precond_cnf.add_clause(['on_a_b', '-clear_c'])  # (on_a_b OR NOT clear_c)

# SAT solver for model counting
num_models = precond_cnf.count_solutions()
entropy = precond_cnf.get_entropy()
```

## Current Status (September 27, 2025)

### ✅ Completed
- **Phase 1**: Core infrastructure (CNF Manager, PDDL Handler) with lifted support
- **Phase 2**: OLAM adapter implementation with complete test coverage
- **Phase 3**: Experiment runner and metrics framework with TDD approach
  - Comprehensive metrics tracking (cumulative mistakes, sliding windows)
  - Thread-safe implementation with deadlock prevention
  - Mock environment for testing
  - Planner path configuration (Fast Downward, VAL)
  - **100% test pass rate (165/165 tests passing)**
- **Multi-Domain Support**:
  - 5 domains ready: Blocksworld, Gripper, Logistics, Rover, Depots
  - Comprehensive test coverage for all domains
  - Extended experiment capability (1500+ actions demonstrated)
- **Performance & Quality Tools**:
  - Performance benchmarking suite
  - Code coverage reporting (53.8% module coverage)
  - Extended experiment scripts for long-running tests
- **CI/CD Infrastructure**:
  - Docker multi-stage builds for consistent environments
  - GitHub Actions CI pipeline
  - Automated testing across Python versions

### 🚧 In Progress
- **Phase 4**: PDDL environment and planning integration (needed for real experiments)

### 📋 TODO
- **Phase 5**: Information-Theoretic algorithm implementation
- **Phase 6**: ModelLearner (Optimistic) adapter
- **Phase 7**: Comparative experiments and analysis

## Testing

### Local Testing
Run the test suite:
```bash
# Quick tests (< 2 minutes)
make test-quick

# Full curated test suite (recommended)
make test  # Runs 165 tests, 100% pass rate

# All tests including experimental
pytest tests/  # Runs 196 tests, includes unstable experiments

# Specific module tests
make test-metrics
make test-integration
```

### Docker Testing
```bash
# Build Docker images
make docker-build

# Run curated tests in Docker (recommended)
make docker-test  # Same as 'make test' but in Docker

# Quick tests without dependencies
make docker-test-quick

# Interactive development
make docker-shell

# Note: Docker ensures consistent environment with all dependencies
# Use for CI/CD or when local dependencies are missing
```

### CI/CD
```bash
# Run local CI pipeline
make ci-local

# GitHub Actions runs automatically on push
```

**Test Status**: 165/165 tests passing (100% pass rate). See `docs/TEST_IMPLEMENTATION_REVIEW.md` for detailed analysis.
