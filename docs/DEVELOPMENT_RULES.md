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
│   ├── configuration/       # Project configuration and paths (renamed from config)
│   ├── core/                # PDDL models, states, actions, CNF formulas
│   ├── sat_integration/     # CNF/SAT solver components (Phase 4a - TODO)
│   ├── planning/            # Unified Planning Framework integration (Phase 4b - TODO)
│   ├── environments/        # Simulators and domains
│   └── experiments/         # Runners and metrics
├── tests/                   # Test suite (mirrors src/ structure)
│   ├── core/                # Core functionality tests
│   ├── algorithms/          # Algorithm implementation tests
│   ├── environments/        # Environment tests
│   ├── experiments/         # Experiment framework tests
│   ├── integration/         # Cross-module integration tests
│   ├── utils/               # Test utilities and helpers
│   └── resources/           # Test resources (info, pddl, results)
├── benchmarks/              # Domain/problem files
├── configs/                 # Experiment configurations
├── scripts/                 # Experiment, testing, and analysis scripts
├── results/                 # Experiment outputs (organized)
│   ├── tests/              # Test result files (auto-cleaned)
│   ├── experiments/        # Real experiment results
│   └── detailed/           # Detailed experiment runs with logs
└── docs/                    # Documentation
    ├── external_repos/      # Interface documentation for OLAM/ModelLearner
    ├── information_gain_algorithm/ # Information-theoretic algorithm docs
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
- **Status**: Repository currently unavailable (https://github.com/kcleung/ModelLearner.git not found)
- **TODO**: Find correct repository URL or alternative implementation
- When available: Import from `model_learner.ModelLearnerLifted`
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

### 8. Results Organization
**Rule**: Experiment results are automatically organized by type.
- **Test results** (`results/tests/`): Automatically routed when experiment name contains "test"
  - Old test files auto-cleaned before new runs
  - Not tracked in git (files ignored, structure tracked)
- **Experiment results** (`results/experiments/`): Real experiment output files
  - Flat files: `{experiment_name}_{timestamp}_metrics.{csv,json}`
  - Summary files and learned models
- **Detailed runs** (`results/detailed/`): Full experiment directories with logs
  - Created by `scripts/run_experiments.py`
  - Format: `{name}_detailed_{timestamp}/`
  - Contains: logs, config snapshots, complete results

## Context Optimization Rules

### 1. Documentation References
**Rule**: Key interfaces documented in `docs/external_repos/`.
- `OLAM_interface.md`: OLAM methods and usage
- `ModelLearner_interface.md`: ModelLearner methods
- `integration_guide.md`: Adapter implementation guide

### 2. Import Statements
For import patterns and examples, see [QUICK_REFERENCE.md](QUICK_REFERENCE.md#import-patterns)

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

#### TDD Workflow
1. **Write tests FIRST** - Tests define the expected behavior
2. **Implement to pass tests** - Code only what's needed
3. **Run tests frequently** - During development
4. **Full test suite before complete** - Before marking any task complete
5. **Only mark "completed" if ALL tests pass** - No exceptions

#### Test Commands
For all available test commands, see [QUICK_REFERENCE.md](QUICK_REFERENCE.md#testing-commands)

#### Test Failure Protocol
If tests fail:
1. **STOP** - Do not proceed or mark complete
2. **DEBUG** - Use pytest -vv for verbose output
3. **FIX** - Address the root cause, not symptoms
4. **RETEST** - Verify fix with full test suite
5. **DOCUMENT** - Note any changes needed in other components

#### Common Issues
- **Test timeout/hanging**: Likely deadlock (threading.Lock vs RLock)
- **Import errors**: Check sys.path and external repo availability
- **JSON serialization**: Use custom encoder for numpy types
- **State format mismatches**: Verify conversions between systems

## Implementation Progress Tracking

### MANDATORY: Update IMPLEMENTATION_TASKS.md After Each Task
**Rule**: After completing ANY implementation:
1. Update `docs/IMPLEMENTATION_TASKS.md` with completed work
2. Ask user: "Should I update IMPLEMENTATION_TASKS.md to reflect the completed [task name]?"
3. Wait for explicit user approval (no auto-approve)
4. Only then update and commit the documentation

## Quick Start
For setup commands and code examples, see [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

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
unified-planning[fast-downward,tamer]>=1.2.0
python-sat>=1.8.dev22

# Scientific computing
numpy>=1.21.0,<3.0.0  # Updated to support numpy 2.x
pandas>=1.3.0
matplotlib>=3.5.0
scipy>=1.7.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
typing-extensions>=4.0.0

# Optional
pyeda>=0.28.0  # For formula minimization
```

## External Tool Paths
- OLAM: `/home/omer/projects/OLAM/` (Available)
- ModelLearner: `/home/omer/projects/ModelLearner/` (Repository unavailable - needs fix)
- Fast Downward: `/home/omer/projects/fast-downward/`
- Val validator: `/home/omer/projects/Val/`
- PDDL domains: `/home/omer/projects/online_model_learning/benchmarks/`

### OLAM Java Setup (REQUIRED)
OLAM requires Java JDK for action filtering via `compute_not_executable_actions.jar`.

**Installation**:
1. Download Oracle JDK 17+: https://www.oracle.com/java/technologies/downloads/
2. Extract to `/home/omer/projects/OLAM/Java/`:
   ```bash
   cd /home/omer/projects/OLAM/Java
   tar -xzvf jdk-17_linux-x64_bin.tar.gz
   ```
3. Verify: `ls /home/omer/projects/OLAM/Java/jdk-*/bin/java`

**Configuration**:
- OLAM sets `Configuration.JAVA_BIN_PATH` automatically from `Java/` directory
- OLAMAdapter defaults to using OLAM's bundled Java
- Bypass mode (`bypass_java=True`) only for testing without Java
- See [OLAM Experiment Review](reviews/experimenting_OLAM_review.md) for details

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
- **51 tests** - Stable, reliable tests
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
See [QUICK_REFERENCE.md](QUICK_REFERENCE.md#docker-commands) for all Docker commands.

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