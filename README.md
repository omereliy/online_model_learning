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
│   │   ├── pddl_model.py          # PDDL model representation
│   │   ├── state.py               # State representation
│   │   ├── action.py              # Action representation
│   │   └── cnf_formula.py         # CNF formula representation
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
```python
# CNF formula for precondition uncertainty
precond_cnf = CNFFormula(variables=['on_a_b', 'clear_c', 'handempty'])
precond_cnf.add_clause([1, -2])  # (on_a_b OR NOT clear_c)

# SAT solver for model counting
solver = SATSolver('minisat')
num_models = solver.count_models(precond_cnf)
information_gain = calculate_entropy_reduction(num_models)
```
