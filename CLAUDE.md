# Claude Code Context Guide

## Project Overview
A framework for testing and comparing online action model learning algorithms in PDDL domains.

**Algorithms:**
1. **Information Gain** - CNF/SAT-based information-theoretic approach (primary)
2. **OLAM** - Online Learning of Action Models (external, post-processing analysis)

## Quick Reference

### Make Commands
```bash
make test              # Full test suite (51 tests)
make test-quick        # Critical tests only
make coverage          # Test coverage analysis
make benchmark         # Performance benchmarks
make docker-test       # Tests in Docker container
make clean             # Remove generated files
```

### Running Information Gain Experiments
```bash
# Using config file
python -m src.experiments.runner --config configs/paper/information_gain_blocksworld_1000.yaml

# Direct programmatic usage
from src.algorithms.information_gain import InformationGainLearner
from src.core.pddl_io import PDDLReader
```

### Analyzing OLAM Results (Post-Processing)
OLAM is run externally. Our code reads its output for analysis:
```bash
# Process OLAM checkpoint exports
python scripts/process_olam_results.py \
    --olam-results /path/to/olam_results/depots \
    --ground-truth benchmarks/olam-compatible/depots/domain.pddl \
    --output-dir results/olam_processed
```

See `docs/OLAM_INPUT_OUTPUT.md` for OLAM output format details.

## Project Structure

```
src/
├── algorithms/
│   ├── information_gain.py      # CNF-based learner (main algorithm)
│   ├── olam_external_runner.py  # Run OLAM as subprocess
│   └── base_learner.py          # Abstract interface
├── core/
│   ├── cnf_manager.py           # CNF formulas + SAT solving (PySAT)
│   ├── lifted_domain.py         # Domain representation
│   ├── grounding.py             # Action/state grounding
│   ├── pddl_io.py               # PDDL file reading
│   ├── model_metrics.py         # Precision/recall/F1
│   ├── model_validator.py       # Model correctness checking
│   ├── olam_trace_parser.py     # Parse OLAM execution logs
│   └── olam_knowledge_reconstructor.py  # Replay OLAM learning
├── experiments/
│   ├── runner.py                # Main experiment orchestration
│   ├── metrics.py               # Metric collection
│   └── olam_experiment.py       # OLAM-specific handling
└── environments/
    ├── active_environment.py    # Real environment execution
    └── mock_environment.py      # Testing environment
```

## Configuration

**Config files in `configs/`:**
- `configs/paper/*.yaml` - Paper experiment configs (1000 iterations)
- `configs/full_validation_experiment.yaml` - Full validation suite

Example config structure:
```yaml
domain: blocksworld
problems: [p01, p02, p03]
algorithm: information_gain
max_iterations: 1000
checkpoints: [1, 5, 10, 20, 50, 100, 200, 500, 1000]
```

## Benchmarks

23 PDDL domains in `benchmarks/olam-compatible/`:
- blocksworld, depots, driverlog, ferry, floortile, gold-miner, grid, gripper
- hanoi, matching-bw, miconic, n-puzzle, nomystery, parking, rover, satellite
- sokoban, spanner, tpp, transport, zenotravel, barman

Each domain has `domain.pddl` + multiple problem files (`p00.pddl` - `p09.pddl`).

## Testing

```bash
# Standard workflow
make test              # Before committing

# Specific test files
pytest tests/algorithms/test_information_gain.py -v
pytest tests/integration/ -v
```

Test structure:
- `tests/algorithms/` - Algorithm-specific tests
- `tests/core/` - Core module tests
- `tests/experiments/` - Experiment framework tests
- `tests/integration/` - End-to-end tests

## Key Documentation

| File | Purpose |
|------|---------|
| `docs/IMPLEMENTATION_TASKS.md` | Project status, completed work |
| `docs/OLAM_INPUT_OUTPUT.md` | OLAM file formats and workflow |
| `docs/LIFTED_SUPPORT.md` | Parameterized actions/fluents |
| `docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md` | Algorithm details |

## Development Guidelines

- Run `make test` before committing
- Prefer editing existing files over creating new ones
- Keep solutions simple - avoid over-engineering
- Use existing patterns in the codebase
