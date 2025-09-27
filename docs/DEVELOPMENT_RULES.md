# Development Rules and Context Management

## ⚠️ MANDATORY REVIEW
**This document MUST be reviewed at the start of every Claude Code session before any interaction.**

## Project Overview
**Goal**: Compare online action model learning algorithms (OLAM, Optimistic Exploration, CNF-based Information-Theoretic) on PDDL domains using Unified Planning Framework and PySAT integration.

## Tech Stack
- **Unified Planning Framework**: PDDL parsing, planning, and simulation
- **PySAT**: CNF formula manipulation and SAT solving (minisat)
- **External Algorithms**: OLAM and ModelLearner Python implementations
- **Scientific Python**: NumPy, Pandas, Matplotlib for analysis

## Directory Structure Rules

### Core Paths
```
/home/omer/projects/
├── online_model_learning/     # Main project (current working directory)
├── OLAM/                      # External OLAM implementation
├── ModelLearner/              # External Optimistic Exploration
├── fast-downward/             # External planner
└── Val/                       # PDDL validator
```

### Project Structure
```
online_model_learning/
├── src/
│   ├── algorithms/           # Algorithm implementations and adapters
│   ├── core/                # PDDL models, states, actions, CNF formulas
│   ├── sat_integration/     # CNF/SAT solver components
│   ├── planning/            # Unified Planning Framework integration
│   ├── environments/        # Simulators and domains
│   └── experiments/         # Runners and metrics
├── benchmarks/              # Domain/problem files
├── configs/                 # Experiment configurations
├── scripts/                 # SLURM and setup scripts
├── results/                 # Experiment outputs
└── docs/                    # Documentation
    ├── external_repos/      # Interface documentation for OLAM/ModelLearner
    └── CNF_SAT_INTEGRATION.md # CNF/SAT solver integration guide
```

## Key Implementation Rules

### 1. Algorithm Adapter Pattern
**Rule**: All algorithms must implement `BaseActionModelLearner` interface.
```python
class BaseActionModelLearner(ABC):
    def select_action(self, state) -> Action
    def observe(self, state, action, next_state, success) -> None
    def update_model(self) -> None
    def get_learned_model() -> PDDLModel
    def has_converged() -> bool
```

### 2. External Repository Integration
**OLAM Adapter** (`src/algorithms/olam_adapter.py`):
- Import: `from OLAM.Learner import Learner`
- Key methods: `select_action()`, `learn()`, `learn_failed_action_precondition()`
- Handle state format conversion

**ModelLearner Adapter** (`src/algorithms/optimistic_adapter.py`):
- Import: `from model_learner.ModelLearnerLifted import ModelLearnerLifted`
- Key method: `learning_step_all_actions_updated()`
- Requires lifted_dict YAML file

### 3. State/Action Representation
**Rule**: Use unified representations with converters.
```python
# Unified formats
State: Set of ground predicates
Action: name + parameters list
Model: Dict with preconditions, positive_effects, negative_effects
```

### 4. PDDL File Handling
**Rule**: All PDDL operations through Unified Planning Framework.
- Use UP for parsing, planning, and simulation
- Convert to unified internal format
- Handle both external algorithms and UP integration

### 5. CNF/SAT Integration Rules
**Rule**: CNF formulas in `src/sat_integration/` with PySAT backend.
- `cnf_builder.py`: Construct formulas from observations
- `sat_solver.py`: PySAT minisat interface
- `variable_mapper.py`: Fluent ↔ CNF variable mapping
- `formula_minimizer.py`: Optimize formulas for performance

### 6. Unified Planning Integration
**Rule**: Use UP as primary PDDL interface in `src/planning/`.
- `unified_planning_interface.py`: Main UP wrapper
- Support multiple planners through UP
- Handle UP Problem/State/Action objects

### 7. Experiment Configuration
**Rule**: YAML configs in `configs/`.
```yaml
domain: blocksworld
algorithms: [olam, optimistic, information_gain]
metrics: [sample_complexity, runtime, accuracy, cnf_formula_size]
seed: 42
cnf_settings:
  solver: minisat
  minimize_formulas: true
  max_clauses: 1000
```

## Context Optimization Rules

### 1. Documentation References
**Rule**: Key interfaces documented in `docs/external_repos/`.
- `OLAM_interface.md`: OLAM methods and usage
- `ModelLearner_interface.md`: ModelLearner methods
- `integration_guide.md`: Adapter implementation guide

### 2. Import Statements
```python
# Unified Planning imports
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import SequentialSimulator

# PySAT imports
from pysat.solvers import Minisat22
from pysat.formula import CNF

# OLAM imports
import sys
sys.path.append('/home/omer/projects/OLAM')
from OLAM.Learner import Learner
from Util.PddlParser import PddlParser

# ModelLearner imports
sys.path.append('/home/omer/projects/ModelLearner/src')
from model_learner.ModelLearnerLifted import ModelLearnerLifted
from model_learner.parser import Parser
```

### 3. Common Pitfalls to Avoid
- **Don't** modify external repositories (OLAM, ModelLearner)
- **Don't** assume planner output formats - always parse
- **Don't** mix state representations - always convert
- **Don't** hardcode paths - use config files

### 4. Testing Checklist
- [ ] Adapters return correct action format
- [ ] State conversions work both ways
- [ ] Planner wrapper handles timeouts
- [ ] Metrics collected at each step
- [ ] Seeds ensure reproducibility

## Test-Driven Development Requirements

### MANDATORY: Run Tests Before Marking Complete
**Rule**: No task or implementation can be marked as complete without passing all tests.

#### 1. TDD Workflow
1. **Write tests FIRST** - Tests define the expected behavior
2. **Implement to pass tests** - Code only what's needed
3. **Run tests frequently** - Use `make test-quick` during development
4. **Full test suite before complete** - Run `make test` or `python scripts/run_test_suite.py`
5. **Only mark "completed" if ALL tests pass** - No exceptions

#### 2. Available Test Commands
```bash
# Quick critical tests (< 2 minutes)
make test-quick

# Full test suite (all tests)
make test

# Specific module tests
make test-metrics
make test-integration

# Run with test runner script
python scripts/run_test_suite.py         # Full suite
python scripts/run_test_suite.py --quick # Quick tests only
python scripts/run_test_suite.py --list  # List available tests
```

#### 3. Common Test Issues and Solutions
- **Test timeout/hanging**: Likely a deadlock (e.g., threading.Lock instead of RLock)
- **Import errors**: Check sys.path and external repo availability
- **JSON serialization**: Use custom encoder for numpy types
- **State format mismatches**: Verify conversions between systems

#### 4. Test Categories
1. **Unit Tests** - Test individual methods/classes
2. **Integration Tests** - Test component interactions
3. **System Tests** - Test full pipeline execution
4. **Performance Tests** - Verify acceptable runtimes

#### 5. Pre-Commit Verification
```bash
# Before ANY commit, run:
make commit  # Runs full test suite and reports status

# Or manually:
make test && echo "Ready to commit" || echo "Fix tests first"
```

#### 6. Test Failure Protocol
If tests fail:
1. **STOP** - Do not proceed or mark complete
2. **DEBUG** - Use pytest -vv for verbose output
3. **FIX** - Address the root cause, not symptoms
4. **RETEST** - Verify fix with full test suite
5. **DOCUMENT** - Note any changes needed in other components

## Implementation Progress Tracking

### MANDATORY: Update IMPLEMENTATION_TASKS.md After Each Task
**Rule**: After completing ANY implementation:
1. Update `docs/IMPLEMENTATION_TASKS.md` with completed work
2. Ask user: "Should I update IMPLEMENTATION_TASKS.md to reflect the completed [task name]?"
3. Wait for explicit user approval (no auto-approve)
4. Only then update and commit the documentation

## Quick Start Commands

### Setup Environment
```bash
conda env create -f environment.yml
conda activate online_model_learning
```

### Run Single Experiment
```python
from src.experiments.runner import ExperimentRunner
runner = ExperimentRunner('configs/blocksworld.yaml')
results = runner.run()
```

### SLURM Submission
```bash
sbatch scripts/run_experiments.sh
```

## File Priorities for Implementation

1. **First**: `src/core/` (state, action, pddl_model)
2. **Second**: `src/algorithms/base_learner.py`
3. **Third**: Adapters (olam_adapter, optimistic_adapter)
4. **Fourth**: `src/planners/fd_wrapper.py`
5. **Fifth**: `src/experiments/runner.py`
6. **Last**: SLURM scripts and visualization

## Performance Considerations
- Cache planner results
- Batch state conversions
- Reuse PDDL parsers
- Profile adapter overhead

## Debugging Tips
- Check state format mismatches first
- Verify action name conventions
- Log planner inputs/outputs
- Compare learned models visually

## Required Dependencies
```
# Core framework
unified-planning[pyperplan,tamer]
python-sat[pblib,aiger]

# Scientific computing
numpy
pandas
matplotlib
scipy

# Utilities
pyyaml
tqdm
```

## External Tool Paths
- OLAM: `/home/omer/projects/OLAM/`
- ModelLearner: `/home/omer/projects/ModelLearner/`
- Val validator: `/home/omer/projects/Val/validate`
- PDDL domains: `/home/omer/projects/online_model_learning/benchmarks/`

## CNF/SAT Specific Rules
- Use minisat as default solver (fast and reliable)
- Cache CNF formula evaluations for performance
- Minimize formulas before SAT solving
- Map fluent names consistently to CNF variables
- Handle large formulas with approximate model counting

## GitHub MCP Safety Rules

### Git Operations Guidelines
**Rule**: All git operations must follow these safety protocols.

#### 1. Commit Safety
- **Never** commit sensitive data (passwords, API keys, tokens)
- **Never** commit large binary files or datasets
- **Always** review changes with `git status` and `git diff` before committing
- **Always** use descriptive commit messages
- **Never** credit claude or a person in a commit message
- **Never** force push to main/master branches

#### 2. Branch Management
- **Always** work on feature branches for new functionality
- **Never** directly modify main/master without review
- **Always** pull latest changes before creating new branches
- **Never** delete branches without confirming merged status

#### 3. File Modification Rules
- **Never** modify external repository files (OLAM, ModelLearner)
- **Always** respect .gitignore patterns
- **Never** commit temporary or cache files
- **Always** ensure tests pass before committing

#### 4. Pull Request Guidelines
- **Always** provide clear PR descriptions
- **Always** link related issues if applicable
- **Never** merge without reviewing changes
- **Always** resolve conflicts carefully

#### 5. Repository Boundaries
- **Never** push to repositories outside project scope
- **Always** verify repository URL before operations
- **Never** clone private repos without permission
- **Always** respect repository access levels

### MCP Server Usage
- **Always** verify MCP server connection status
- **Never** expose MCP credentials in code or commits
- **Always** use MCP operations within project boundaries
- **Never** use MCP to access unauthorized resources

## Testing Approach

### Test Suite Organization
The project maintains two test execution approaches:

#### 1. Curated Test Suite (`make test`)
- **165 tests** - Stable, reliable tests
- **100% pass rate** - All tests must pass
- **Use for**: CI/CD, validation, pre-commit checks
- **Excludes**: Experimental tests that may be unstable

#### 2. Full Test Suite (`pytest tests/`)
- **196 tests** - All tests including experimental
- **May include failures** - Some tests for future features
- **Use for**: Development, debugging, exploration
- **Includes**: All test files in tests/ directory

### When to Use Which
- **Before commits**: Always use `make test`
- **CI/CD pipeline**: Uses `make test` for stability
- **Local development**: Can use `pytest tests/` for broader coverage
- **Docker testing**: Both options available via make targets

## Docker Rationale and Usage

### Why Docker?
1. **Consistency**: Eliminates "works on my machine" problems
2. **Dependencies**: All required packages pre-installed
3. **CI/CD**: Same environment locally and in GitHub Actions
4. **Isolation**: No conflicts with system packages
5. **Reproducibility**: Exact same environment for all developers

### Docker Commands
```bash
# Build all Docker images
make docker-build

# Run curated tests in Docker (recommended)
make docker-test

# Quick tests without dependencies
make docker-test-quick

# Interactive development shell
make docker-shell

# Run experiments in container
make docker-experiment

# Jupyter notebook for analysis
make docker-notebook
```

### Docker Architecture
- **Multi-stage builds**: Optimized image sizes
- **Base image**: Ubuntu 22.04 with Python 3.9
- **Builder stage**: Compiles Fast Downward and VAL
- **Dev stage**: Full development environment
- **Test stage**: Minimal testing environment
- **Prod stage**: Production deployment ready

### When to Use Docker
- **Missing dependencies**: When planners/validators not installed
- **CI/CD testing**: Ensure tests pass in clean environment
- **Collaboration**: Share exact environment with team
- **Deployment**: Production-ready containers

## Markdown Documentation Rules

### File Purposes (Single Source of Truth)
Each markdown file has a specific purpose. Content should appear in exactly ONE location:

| File | Purpose | Content Owner |
|------|---------|---------------|
| **README.md** | Project overview | Installation, basic usage |
| **CLAUDE.md** | AI navigation index | Points to other docs only |
| **DEVELOPMENT_RULES.md** | Project conventions | Architecture, rules, guidelines |
| **QUICK_REFERENCE.md** | Code patterns & commands | Snippets, examples, all commands |
| **IMPLEMENTATION_TASKS.md** | Progress tracking | Current status, todos |
| **TEST_IMPLEMENTATION_REVIEW.md** | Test analysis | Test quality assessment |

### Editing Guidelines
1. **No Duplication**: Each fact appears in exactly ONE file
2. **Use References**: Link to other docs instead of copying content
3. **Update Single Source**: When info changes, update only the authoritative file
4. **Concise Writing**: Be direct, avoid verbose explanations
5. **No Credits**: Never add "Generated by", emojis, or attribution

### Content Ownership Examples
- **Architecture details** → DEVELOPMENT_RULES.md
- **Test commands** → QUICK_REFERENCE.md
- **Project status** → IMPLEMENTATION_TASKS.md
- **Installation steps** → README.md
- **Navigation help** → CLAUDE.md (links only)

### Before Editing Any Markdown
1. Check which file owns that content type
2. Verify information doesn't exist elsewhere
3. Update only the authoritative location
4. Add cross-references if needed
5. Remove any duplicate content found

### Documentation Maintenance
- Review for duplicates quarterly
- Update file purposes if roles change
- Keep CLAUDE.md as pure navigation
- Ensure README stays beginner-friendly
- Track all changes in git history