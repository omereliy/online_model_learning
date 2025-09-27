# Online Action Model Learning Experiment Framework

## Overview
A unified framework for testing and comparing online action model learning algorithms in PDDL domains, with advanced CNF/SAT solver integration for uncertainty representation.

## Algorithms Implemented
1. **OLAM** - Online Learning of Action Models (Lamanna et al., 2021) [Python]
2. **ModelLearner (Optimistic)** - Optimistic exploration with symbolic model estimates (Sreedharan et al., 2023) [Python]
3. **Information-Theoretic Selection** - Novel CNF-based approach using expected information gain with SAT solver integration

## Getting Started

For detailed architecture and design principles, see [DEVELOPMENT_RULES.md](docs/DEVELOPMENT_RULES.md)

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

## Code Examples

For detailed examples and code patterns, see [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)

## Project Status

For current implementation status and roadmap, see [IMPLEMENTATION_TASKS.md](docs/IMPLEMENTATION_TASKS.md)

## Testing

```bash
# Run curated test suite (165 tests, 100% pass rate)
make test

# Run with Docker for consistent environment
make docker-test
```

For complete testing options and Docker usage, see [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md#commands)
