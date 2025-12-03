# OLAM Experiment Interface

This document describes the input/output interface for running OLAM (Online Learning of Action Models) experiments. this document is copied and pasted directly from olam repo.

## Table of Contents
- [Command Line Interface](#command-line-interface)
- [Input Requirements](#input-requirements)
- [Output Structure](#output-structure)
- [Configuration Parameters](#configuration-parameters)
- [Example Workflows](#example-workflows)

---

## Command Line Interface

### Basic Usage

```bash
# Single domain
python main.py -d <domain_name>

# Multiple domains (batch mode)
python main.py --domains <domain1> <domain2> <domain3>

# Custom output directory
python main.py -d depots --output-dir /path/to/results

# Full configuration
python main.py \
  --domains blocksworld depots gripper \
  --output-dir results/experiment_1 \
  --max-iter 500 \
  --planner-timeout 360 \
  --checkpoints 1 5 10 20 50 100
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d`, `--domain` | `str` | `None` | Single domain name to run |
| `--domains` | `str[]` | `None` | Multiple domain names for batch processing |
| `--output-dir` | `str` | `Analysis/run_0/` | Custom output directory for results |
| `--max-iter` | `int` | `100` | Maximum iterations per problem instance |
| `--planner-timeout` | `int` | `60` | Planner timeout in seconds per planning call |
| `--checkpoints` | `int[]` | `DEFAULT_CHECKPOINTS` | List of iteration numbers for checkpoint exports |

**Notes:**
- If both `-d` and `--domains` are specified, `--domains` takes precedence
- If neither is specified, all domains in `Analysis/Benchmarks/` are processed
- Domains `elevators` and `elevators-opt08-strips` are automatically excluded

### Default Checkpoint Schedule

If `--checkpoints` is not specified, the following default schedule is used:

```python
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,     # Every iteration for first 10
 12, 14, 16, 18, 20,                 # Every 2 iterations
 25, 30, 35, 40, 45, 50,             # Every 5 iterations
 60, 70, 80, 90, 100,                # Every 10 iterations
 120, 140, 160, 180, 200,            # Every 20 iterations
 250, 300, 350, 400, 450, 500]       # Every 50 iterations
```

---

## Input Requirements

### Directory Structure

```
Analysis/
└── Benchmarks/
    ├── <domain>.pddl                        # Domain definition file
    └── <domain>/                            # Problem instances directory
        ├── 1_p00_<domain>_gen.pddl         # Problem 1 (executed first)
        ├── 2_p01_<domain>_gen.pddl         # Problem 2 (executed second)
        ├── 3_p02_<domain>_gen.pddl         # Problem 3
        └── ...
```

### Domain Definition File

**Location:** `Analysis/Benchmarks/<domain>.pddl`

**Format:** Standard PDDL domain file containing:
- `(define (domain <domain-name>)`
- `:requirements` - PDDL features used
- `:predicates` - Domain predicates
- `:action` - Action definitions (with preconditions and effects)

**Example:**
```pddl
(define (domain gripper)
  (:requirements :strips)
  (:predicates (room ?r) (ball ?b) (gripper ?g) ...)
  (:action pick
    :parameters (?b ?r ?g)
    :precondition (and (ball ?b) (room ?r) ...)
    :effect (and (carry ?b ?g) (not (at-ball ?b ?r))))
  ...
)
```

### Problem Instance Files

**Location:** `Analysis/Benchmarks/<domain>/N_pXX_<domain>_gen.pddl`

**Naming Convention:**
- `N` - Execution order number (1, 2, 3, ...)
- `pXX` - Problem ID (p00, p01, p02, ...)
- Files are sorted and executed in order by `N`

**Format:** Standard PDDL problem file containing:
- `(define (problem <problem-name>)`
- `:domain` - Reference to domain name
- `:objects` - Problem objects
- `:init` - Initial state predicates
- `:goal` - Goal condition

**Example:**
```pddl
(define (problem gripper-1)
  (:domain gripper)
  (:objects room1 room2 ball1 ball2 left right)
  (:init (room room1) (room room2) (ball ball1) ...)
  (:goal (and (at-ball ball1 room2) (at-ball ball2 room2)))
)
```

### Learning Mode

**Independent Learning (Default):**
- Each problem learns from scratch
- No knowledge transfer between problems
- `PDDL/domain_input.pddl` is deleted before each problem

**Sequential Learning (Disabled):**
- Currently commented out in code (main.py:698)
- Would pass learned model from previous problem to next

---

## Output Structure

### Top-Level Output Directory

```
<output-dir>/                               # e.g., Analysis/run_0/ or custom path
├── Tests/                                   # Per-domain results
│   └── <domain>/                           # e.g., blocksworld/
│       ├── <problem1>/                     # e.g., 1_p00_blocksworld_gen/
│       ├── <problem2>/
│       └── ...
├── Results_cert/                           # Evaluation results (certain model)
│   └── <domain>_action_model_eval.xlsx
├── Results_uncert_neg/                     # Evaluation results (uncertain model)
│   └── <domain>_action_model_eval.xlsx
└── trace_complete.json                     # Aggregated trace (all domains)
```

### Per-Problem Output

**Location:** `<output-dir>/Tests/<domain>/<problem>/`

```
<problem>/                                   # e.g., 1_p00_depots_gen/
├── <problem>_log                           # Detailed execution log
├── domain_learned.pddl                     # Final learned domain (with uncertain)
├── domain_learned_certain.pddl             # Final learned domain (certain only)
├── trace.json                              # JSON lines trace of all iterations
├── operator_certain_predicates.json        # Final certain preconditions per operator
├── operator_uncertain_precs.json           # Final uncertain preconditions per operator
├── operator_certain_positive_effects.json  # Final certain positive effects
├── operator_certain_negative_effects.json  # Final certain negative effects
├── operator_uncertain_positive_effects.json # Final uncertain positive effects
├── operator_uncertain_negative_effects.json # Final uncertain negative effects
├── operator_useless_possible_precs.json    # Final useless preconditions
├── operator_useless_negated_precs.json     # Final useless negated preconditions
├── checkpoints/                            # Model snapshots at checkpoint iterations
│   ├── iter_1/
│   │   ├── domain_learned.pddl
│   │   ├── domain_learned_certain.pddl
│   │   ├── operator_certain_predicates.json
│   │   ├── operator_uncertain_precs.json
│   │   ├── operator_certain_positive_effects.json
│   │   ├── operator_certain_negative_effects.json
│   │   ├── operator_uncertain_positive_effects.json
│   │   ├── operator_uncertain_negative_effects.json
│   │   ├── operator_useless_possible_precs.json
│   │   └── operator_useless_negated_precs.json
│   ├── iter_5/
│   │   └── ... (same structure)
│   ├── iter_10/
│   └── ...
└── final/                                  # Copy of final model + all JSONs
    ├── domain_learned.pddl
    ├── domain_learned_certain.pddl
    └── ... (all 8 JSON files)
```

### Output File Descriptions

#### Execution Log (`<problem>_log`)
- **Format:** Plain text
- **Content:** Timestamped progress messages, iteration metrics, action execution results
- **Example:**
  ```
  [2025-01-19 10:30:15] Starting learning for problem: 1_p00_depots_gen
  [2025-01-19 10:30:16] Iteration 1: Action drive(truck1,depot1,distributor1) - Success
  ...
  ```

#### Trace File (`trace.json`)
- **Format:** JSON Lines (one JSON object per line)
- **Content:** Complete trace of every action execution
- **Schema:**
  ```json
  {
    "domain": "depots",
    "problem": "1_p00_depots_gen",
    "iter": 1,
    "action": "drive(truck1,depot1,distributor1)",
    "success": true,
    "strategy": "FD",
    "timestamp": "2025-01-19T10:30:16.123456"
  }
  ```
- **Fields:**
  - `domain`: Domain name
  - `problem`: Problem instance name
  - `iter`: Iteration number (1-indexed)
  - `action`: Ground action executed
  - `success`: Boolean indicating if action succeeded
  - `strategy`: Strategy used ("FD" for planner, "Random" for random selection, or function name)
  - `timestamp`: ISO 8601 timestamp

#### Aggregated Trace (`trace_complete.json`)
- **Format:** JSON array
- **Content:** Concatenation of all trace.json files across all domains/problems
- **Use:** Cross-domain analysis, aggregated statistics

#### Learned Domain Files
- **`domain_learned.pddl`**: Learned domain with all effects (certain + uncertain)
- **`domain_learned_certain.pddl`**: Learned domain with only certain preconditions/effects

#### Model State JSON Files

Each JSON file contains a dictionary mapping operator names to lists of predicates:

**`operator_certain_predicates.json`**
```json
{
  "drive": ["(truck ?param_1)", "(at ?param_1 ?param_2)", "(location ?param_2)", "(location ?param_3)"],
  "load": ["(hoist ?param_1)", "(crate ?param_2)", ...]
}
```

**`operator_uncertain_precs.json`**
```json
{
  "drive": ["(available ?param_1)"],
  "load": []
}
```

**`operator_certain_positive_effects.json`**
```json
{
  "drive": ["(at ?param_1 ?param_3)"],
  "load": ["(in ?param_2 ?param_1)"]
}
```

**`operator_certain_negative_effects.json`**
```json
{
  "drive": ["(not (at ?param_1 ?param_2))"],
  "load": ["(not (available ?param_1))"]
}
```

**`operator_uncertain_positive_effects.json`**
- Positive effects that were observed but not confirmed certain

**`operator_uncertain_negative_effects.json`**
- Negative effects that were observed but not confirmed certain

**`operator_useless_possible_precs.json`**
- Preconditions tested and found to be unnecessary

**`operator_useless_negated_precs.json`**
- Negated preconditions tested and found to be unnecessary

#### Evaluation Files (`Results_cert/`, `Results_uncert_neg/`)

**Format:** Excel (.xlsx) with multiple sheets

**Sheets:**
- `Summary`: High-level metrics per instance
- `Total`: Aggregated metrics across all instances
- Per-operator sheets: Detailed metrics for each operator

**Metrics:**
- **Recall**: Percentage of actual preconditions/effects that were learned
- **Precision**: Percentage of learned preconditions/effects that are correct
- **Computation Time**: Time taken in seconds
- **Iterations**: Number of iterations executed

**Differences:**
- `Results_cert/`: Evaluation using only certain preconditions/effects
- `Results_uncert_neg/`: Evaluation including uncertain negative effects

---

## Configuration Parameters

### Configuration.py Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OUTPUT_CONSOLE` | `True` | Print output to console (True) or redirect to log file (False) |
| `NEG_EFF_ASSUMPTION` | `False` | STRIPS assumption: remove uncertain negative effects not in preconditions |
| `TIME_LIMIT_SECONDS` | `3600000` | Maximum time per problem instance (seconds) |
| `MAX_ITER` | `500` | Maximum iterations per problem instance |
| `PLANNER_TIME_LIMIT` | `60` | Timeout for single planner call (seconds) |
| `MAX_PRECS_LENGTH` | `8` | Maximum precondition complexity to test |
| `RANDOM_SEED` | `42` | NumPy random seed for reproducible action selection |
| `DEFAULT_CHECKPOINTS` | `[1,2,3,...]` | Default checkpoint iteration schedule |

### Override Priority

1. **CLI arguments** (highest priority)
2. **Configuration.py** values
3. **Hard-coded defaults** (lowest priority)

Example:
```bash
# This overrides Configuration.MAX_ITER
python main.py -d depots --max-iter 200
```

---

## Example Workflows

### 1. Quick Test on Single Domain

```bash
# Edit Configuration.py first:
# MAX_ITER = 20
# TIME_LIMIT_SECONDS = 300
# PLANNER_TIME_LIMIT = 30

python main.py -d gripper
```

**Expected Output:**
```
Analysis/run_0/
├── Tests/gripper/
│   ├── 1_p00_gripper_gen/
│   │   ├── 1_p00_gripper_gen_log
│   │   ├── trace.json
│   │   ├── domain_learned.pddl
│   │   └── checkpoints/iter_1/, iter_5/, ...
│   └── ...
└── Results_cert/gripper_action_model_eval.xlsx
```

### 2. Batch Experiment with Multiple Domains

```bash
python main.py \
  --domains blocksworld depots gripper logistics \
  --output-dir experiments/batch_run_1 \
  --max-iter 500 \
  --planner-timeout 360 \
  --checkpoints 1 5 10 20 50 100 200 500
```

**Expected Output:**
```
experiments/batch_run_1/
├── Tests/
│   ├── blocksworld/
│   ├── depots/
│   ├── gripper/
│   └── logistics/
├── Results_cert/
│   ├── blocksworld_action_model_eval.xlsx
│   ├── depots_action_model_eval.xlsx
│   ├── gripper_action_model_eval.xlsx
│   └── logistics_action_model_eval.xlsx
├── Results_uncert_neg/
│   └── ... (same structure)
└── trace_complete.json
```

### 3. Custom Checkpoint Schedule

```bash
# Dense checkpoints early, very sparse later
python main.py -d depots \
  --max-iter 1000 \
  --checkpoints 1 2 3 4 5 10 20 50 100 500 1000
```

### 4. High-Performance Long Run

```bash
python main.py \
  --domains blocksworld depots gripper logistics satellite \
  --output-dir experiments/long_run_3600s \
  --max-iter 1000 \
  --planner-timeout 600 \
  --checkpoints 1 5 10 25 50 100 200 500 1000
```

**Recommended Settings:**
- `MAX_ITER`: 1000 for complex domains
- `PLANNER_TIME_LIMIT`: 600 (10 minutes) for hard problems
- `TIME_LIMIT_SECONDS`: 3600 (1 hour) per instance

---

## Reading Results

### Analyzing Trace Data

```python
import json

# Read trace for single problem
with open('Analysis/run_0/Tests/depots/1_p00_depots_gen/trace.json') as f:
    trace = [json.loads(line) for line in f]

print(f"Total iterations: {len(trace)}")
print(f"Successful actions: {sum(1 for t in trace if t['success'])}")
print(f"Failed actions: {sum(1 for t in trace if not t['success'])}")

# Read aggregated trace
with open('Analysis/run_0/trace_complete.json') as f:
    all_traces = json.load(f)

print(f"Total actions across all domains: {len(all_traces)}")
```

### Analyzing Learned Models

```python
import json

# Read learned model state at iteration 50
checkpoint_dir = 'Analysis/run_0/Tests/depots/1_p00_depots_gen/checkpoints/iter_50'

with open(f'{checkpoint_dir}/operator_certain_predicates.json') as f:
    certain_precs = json.load(f)

print("Learned preconditions for 'drive' operator:")
print(certain_precs['drive'])
```

### Comparing Checkpoints

```bash
# Compare learned models at different iterations
diff checkpoints/iter_10/domain_learned.pddl \
     checkpoints/iter_50/domain_learned.pddl
```

---

## Troubleshooting

### Common Issues

**Problem:** `No such file or directory: 'Analysis/Benchmarks/<domain>.pddl'`
- **Solution:** Ensure domain definition file exists in `Analysis/Benchmarks/`

**Problem:** `No problem instances found for domain <domain>`
- **Solution:** Check that problem files exist in `Analysis/Benchmarks/<domain>/` and follow naming convention `N_pXX_<domain>_gen.pddl`

**Problem:** Missing checkpoints in output
- **Solution:** Verify checkpoint iterations are less than `MAX_ITER` and that learning reached those iterations

**Problem:** Empty trace.json
- **Solution:** Check that actions were actually executed (not just planning without execution)

---

## Version Information

- **OLAM Version:** Research implementation (IJCAI 2021)
- **PDDL Support:** STRIPS, ADL (via adl2strips)
- **Planners:** FastDownward, FastForward
- **Python Version:** 3.7+
- **Dependencies:** numpy, pandas, openpyxl, xlsxwriter
