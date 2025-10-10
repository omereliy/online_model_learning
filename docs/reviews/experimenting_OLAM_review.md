# OLAM Experiment Integration Review

## Overview
This document reviews the integration between native OLAM experiments and our adapter-based approach, identifying related files, code misalignments, and configuration differences.

## Related OLAM Files for Experiments

### Core Components
**Location**: `/home/omer/projects/OLAM/`

| File | Size | Purpose |
|------|------|---------|
| `main.py` | 789 lines | Entry point, experiment orchestration, domain iteration |
| `OLAM/Learner.py` | 228 KB | Core learning algorithm, hypothesis space management |
| `OLAM/Planner.py` | ~500 lines | Planning with learned models |
| `Configuration.py` | 71 lines | Global experiment parameters |

### Utility Modules
**Location**: `/home/omer/projects/OLAM/Util/`

| File | Purpose |
|------|---------|
| `PddlParser.py` | PDDL domain/problem parsing |
| `Simulator.py` | State tracking and simulation |
| `preprocessing.py` | Domain preprocessing for experiments |
| `Dataframe_generator.py` | Result aggregation to Excel |
| `LogReader.py` | Parse experiment logs for metrics |
| `metrics.py` | Statistical analysis of results |

### External Dependencies
**Required**:
- Java JDK 17+ (Oracle): `/home/omer/projects/OLAM/Java/jdk-XX/`
- FastDownward planner: `/home/omer/projects/OLAM/Planners/FD/`
- FastForward planner: `/home/omer/projects/OLAM/Planners/FF/ff`

**Purpose**:
- Java: Executes `compute_not_executable_actions.jar` for filtering (CORE REQUIREMENT)
- FastDownward: Primary planning engine
- FastForward: Action grounding via preprocessing

### Directory Structure
```
OLAM/
├── PDDL/                           # Working PDDL files
│   ├── domain.pddl                 # Current domain
│   ├── facts.pddl                  # Current problem/state
│   ├── domain_learned.pddl         # Learned model (uncertain)
│   ├── domain_learned_certain.pddl # Learned model (certain only)
│   └── domain_input.pddl           # Transfer learning input
├── Info/                           # Temporary action filtering files
├── Analysis/
│   ├── Benchmarks/                 # PDDL domains and problems
│   │   └── <domain>/
│   │       ├── domain.pddl         # Domain specification
│   │       └── N_pXX_<domain>_gen.pddl  # Problem instances
│   └── run_X/                      # Experiment run results
│       ├── Tests/<domain>/<instance>/  # Per-instance results
│       │   ├── <instance>_log      # Execution trace
│       │   ├── domain_learned.pddl
│       │   └── operator_*.json     # Learned knowledge
│       ├── Results_cert/           # Certain-only metrics
│       └── Results_uncert_neg/     # With uncertain effects
```

## Code Misalignments

### 1. PDDL Directory Management
**OLAM Native**:
- Expects permanent `PDDL/` directory in working directory
- Files persist across instances (transfer learning)
- `domain_input.pddl` carries learned knowledge forward

**Our Adapter** (`olam_adapter.py:160-190`):
```python
def _setup_olam_pddl_directory(self):
    pddl_dir = Path("PDDL")
    pddl_dir.mkdir(exist_ok=True)
    # Copies domain/problem each time
```
**Impact**: Creates temporary structure, no transfer learning support

### 2. Java Configuration
**OLAM Native** (`main.py:675-690`):
```python
java_jdk_dir = [d for d in os.listdir(Configuration.JAVA_DIR)]
Configuration.JAVA_BIN_PATH = os.path.join(
    os.getcwd(), Configuration.JAVA_DIR, java_jdk_dir[0], "bin", "java")
assert os.path.exists(Configuration.JAVA_BIN_PATH)
```
**Requirement**: Java JDK MUST be installed, no bypass

**Our Adapter** (`olam_adapter.py:234-262`):
```python
def _configure_java_settings(self):
    if self.bypass_java:
        Configuration.JAVA_BIN_PATH = ""  # Bypass mode
        return
```
**Impact**: Supports bypass mode (for testing), but Java is REQUIRED for production

### 3. Action Grounding
**OLAM Native** (`main.py:253-281`):
```python
def compute_all_action():
    """Compute via cartesian product of object types"""
    for op in all_op.keys():
        subclass_obj_types = [obj_types[el] for el in all_op[op]]
        all_obj_combinations = itertools.product(*op_obj_lists)
```
Also uses FastForward's preprocessing for advanced grounding.

**Our Adapter** (`olam_adapter.py:192-224`):
```python
def _extract_action_list(self):
    from src.core import grounding
    grounded_actions = grounding.ground_all_actions(
        self.domain, require_injective=False)
```
**Impact**: Reimplements grounding using our architecture (cleaner but different)

### 4. Result Analysis Integration
**OLAM Native**:
- `Dataframe_generator.py`: Aggregates results to Excel (per-domain, overall)
- `LogReader.py`: Parses `_log` files for precision/recall/F1
- Outputs: `<domain>_action_model_eval.xlsx`, `overall_summary.xlsx`

**Our Adapter**:
- Uses `MetricsCollector` for CSV/JSON output
- No Excel generation
- Different metric format

**Impact**: Results not compatible with OLAM's analysis tools

### 5. Experiment Directory Structure
**OLAM Native**:
```
Analysis/run_0/Tests/blocksworld/1_p00_blocksworld_gen/
    ├── 1_p00_blocksworld_gen_log
    ├── domain_learned.pddl
    ├── operator_certain_predicates.json
    └── ...
```

**Our Framework**:
```
results/experiments/blocksworld_detailed_20251009_153914_metrics.csv
results/experiments/blocksworld_detailed_20251009_153914_summary.json
```
**Impact**: Different organization, no per-instance directories

### 6. Multi-Instance Sequential Runs
**OLAM Native** (`main.py:741-773`):
```python
all_instances = sorted(os.listdir(instances_dir),
                      key=lambda x: int(x.split("_")[0]))
for instance_name in all_instances:
    # Solve instances in order, transfer learning between them
```
Naming: `1_p00_domain.pddl`, `2_p01_domain.pddl`, ...

**Our Framework**:
- Single problem per experiment configuration
- No automatic sequential runs
- No transfer learning between problems

**Impact**: Different experiment workflow

## Configuration Issues

### Missing OLAM Parameters in YAML

**OLAM Configuration.py Parameters**:
```python
TIME_LIMIT_SECONDS = 300        # Total experiment timeout
MAX_ITER = 100                  # Maximum iterations per instance
PLANNER_TIME_LIMIT = 60         # Planner subprocess timeout
MAX_PRECS_LENGTH = 8            # Negative precondition search depth
NEG_EFF_ASSUMPTION = False      # STRIPS negative effects assumption
OUTPUT_CONSOLE = False          # Console vs file logging
RANDOM_SEED = 0                 # Numpy random seed
```

**Our YAML** (`configs/experiment_blocksworld.yaml`):
```yaml
algorithm_params:
  olam:
    max_iterations: 200         # Maps to MAX_ITER
    eval_frequency: 10          # Adapter-specific

stopping_criteria:
  max_runtime_seconds: 600      # Maps to TIME_LIMIT_SECONDS
```

**Missing**:
- PLANNER_TIME_LIMIT
- MAX_PRECS_LENGTH
- NEG_EFF_ASSUMPTION
- OUTPUT_CONSOLE
- STRATEGIES configuration

**Impact**: Cannot fine-tune OLAM's advanced features

### Java Path Configuration

**OLAM Native**:
```python
Configuration.JAVA_BIN_PATH = "Java/jdk-17.../bin/java"
```
Fixed path relative to OLAM directory.

**Our Adapter**:
```python
bypass_java: bool = False       # Should be False for production
use_system_java: bool = False   # Try system 'java' command
```

**Recommendation**: Default to OLAM's Java, document setup clearly

### Result Format Differences

**OLAM Outputs**:
- `domain_learned.pddl`: Full model with uncertain effects
- `domain_learned_certain.pddl`: Certain preconditions only
- `operator_uncertain_precs.json`: Per-operator uncertain knowledge
- `operator_certain_positive_effects.json`: Learned effects

**Our Outputs**:
- `learned_model.json`: Unified JSON format
- No separate certain/uncertain models
- Different schema

**Impact**: Cannot directly use OLAM's result analysis tools

## Java Configuration (REQUIRED)

### Installation
Per OLAM EXPERIMENT_GUIDE.md:

1. Download Oracle JDK 17+: https://www.oracle.com/java/technologies/downloads/
2. Extract to `/home/omer/projects/OLAM/Java/`:
   ```bash
   cd /home/omer/projects/OLAM/Java
   tar -xzvf jdk-17_linux-x64_bin.tar.gz
   ```
3. Verify: `ls /home/omer/projects/OLAM/Java/jdk-*/bin/java`

### Why Java is Required
OLAM uses `compute_not_executable_actions.jar` for:
- Efficient action filtering based on learned preconditions
- Ground truth precondition checking (when available)
- Advanced constraint solving

**No Python bypass available in native OLAM**.

### Adapter Java Modes
Our adapter supports three modes:

1. **OLAM Bundled Java** (DEFAULT - RECOMMENDED):
   ```python
   OLAMAdapter(bypass_java=False, use_system_java=False)
   ```
   Uses Java from `/home/omer/projects/OLAM/Java/`

2. **System Java** (FALLBACK):
   ```python
   OLAMAdapter(bypass_java=False, use_system_java=True)
   ```
   Tries `java` command from PATH

3. **Bypass Mode** (TESTING ONLY):
   ```python
   OLAMAdapter(bypass_java=True)
   ```
   Python fallback - NOT for production experiments

**Verification**:
```bash
# Check OLAM Java installation
ls /home/omer/projects/OLAM/Java/*/bin/java

# Check system Java
java -version
```

## Recommendations

### 1. Maintain Adapter Approach
**Rationale**: Clean separation, reusable grounding utilities, unified architecture

**Keep**:
- Library-style integration (import from OLAM)
- Type-safe interfaces (LiftedDomainKnowledge, grounding module)
- Unified experiment framework (ExperimentRunner)

### 2. Add OLAM Configuration Passthrough
**Proposal**: Map OLAM parameters via `algorithm_params`:

```yaml
algorithm_params:
  olam:
    max_iterations: 200
    eval_frequency: 10
    planner_time_limit: 60        # NEW
    max_precs_length: 8           # NEW
    neg_eff_assumption: false     # NEW
    output_console: false         # NEW
```

**Implementation**: Pass through to Configuration.py in adapter initialization

### 3. Default to Java (Not Bypass)
**Change**: Make `bypass_java=False` the enforced default

**Documentation**: Prominently document Java setup requirements

**Testing**: Use bypass mode only for unit tests without Java dependency

### 4. Consider Result Format Converters
**Optional Enhancement**:
```python
class OLAMResultConverter:
    def to_excel_format(learned_model) -> pd.DataFrame
    def to_pddl_format(learned_model, certain_only=False) -> str
```

**Benefit**: Leverage OLAM's analysis tools

### 5. Document Workflow Differences
**Clarify**:
- Adapter: Single-problem experiments, research comparison focus
- Native OLAM: Multi-instance sequential runs, transfer learning
- Both valid for different use cases

## Current Adapter Strengths

1. **Clean Architecture**: Uses refactored grounding module (not OLAM's ad-hoc approach)
2. **Type Safety**: LiftedDomainKnowledge, GroundedAction classes
3. **Flexible**: Works with multiple algorithms in unified framework
4. **Testable**: Can run with/without external dependencies
5. **Modern**: YAML configs, structured logging, metric tracking

## Known Limitations

1. **No Transfer Learning**: Each experiment starts from scratch
2. **No Excel Analysis**: Results in CSV/JSON, not OLAM's Excel format
3. **Single Problem**: No automatic multi-instance runs
4. **Partial Config**: Some OLAM parameters not exposed
5. **Testing Bypass**: Java bypass mode may behave differently than native

## Integration Summary

| Aspect | Native OLAM | Our Adapter | Alignment |
|--------|-------------|-------------|-----------|
| Java Requirement | Required | Optional (bypass) | ⚠️ Should enforce |
| Action Grounding | FF + cartesian | grounding.py | ✓ Equivalent |
| PDDL Directory | Persistent | Temporary | ⚠️ Different |
| Result Format | Excel + JSON | CSV + JSON | ⚠️ Different |
| Multi-Instance | Sequential | Single | ⚠️ Different |
| Transfer Learning | Yes | No | ✗ Missing |
| Config Format | Configuration.py | YAML | ✓ Cleaner |
| Architecture | Monolithic | Layered | ✓ Better |

**Overall Assessment**: Adapter provides clean integration with trade-offs in advanced features. Suitable for comparative algorithm research, not for replicating native OLAM workflows.
