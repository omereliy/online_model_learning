# Information Gain Post-Processing Guide

## Overview

Information Gain is a CNF-based action model learning algorithm that runs within our codebase. This guide explains how to reconstruct model checkpoints from experiment results and compute precision/recall metrics for analysis.

## Results Location

Information Gain experiments are stored in: `results/paper/consolidated_experiments/information_gain/`

Each domain has its own directory structure with problem subdirectories.

## Results Directory Structure

```
results/paper/consolidated_experiments/information_gain/
├── <domain_name>/                    # e.g., blocksworld, depots
│   ├── <problem_name>/               # e.g., p00, p01
│   │   ├── observations/
│   │   │   ├── obs_001.json          # Individual observations
│   │   │   ├── obs_002.json
│   │   │   └── ...
│   │   ├── models/
│   │   │   ├── model_iter_001.json   # Checkpoint at iteration 1
│   │   │   ├── model_iter_002.json   # Checkpoint at iteration 2
│   │   │   └── ...
│   │   ├── experiment_config.json    # Experiment parameters
│   │   └── experiment_log.txt        # Execution log
│   ├── p01/
│   └── ... (all problems for this domain)
```

## Checkpoint Format

Each checkpoint file (`model_iter_NNN.json`) contains:

```json
{
  "iteration": 1,
  "algorithm": "information_gain",
  "metadata": {
    "domain": "blocksworld",
    "problem": "p00",
    "export_timestamp": "2025-01-20T12:34:56.789012"
  },
  "actions": {
    "pick-up": {
      "parameters": ["?x"],
      "possible_preconditions": ["clear(?0)", "ontable(?0)", "handempty"],
      "certain_preconditions": ["ontable(?0)"],
      "uncertain_preconditions": ["clear(?0)", "handempty"],
      "confirmed_add_effects": ["holding(?0)"],
      "confirmed_del_effects": ["ontable(?0)", "handempty"],
      "possible_add_effects": [],
      "possible_del_effects": [],
      "constraint_sets": [["clear(?0)", "handempty"]]
    }
  }
}
```

### Key Fields

**Model Components**:
- `possible_preconditions` - All preconditions that MAY be required (certain + uncertain)
- `certain_preconditions` - Preconditions confirmed as required
- `uncertain_preconditions` - Preconditions not yet confirmed (possible - certain)
- `confirmed_add_effects` - Add effects confirmed by all observations
- `confirmed_del_effects` - Delete effects confirmed by all observations
- `possible_add_effects` - Add effects observed but not certain
- `possible_del_effects` - Delete effects observed but not certain
- `constraint_sets` - CNF clauses representing precondition constraints

**Metadata**:
- `iteration` - Learning iteration number (1-indexed)
- `algorithm` - Always "information_gain"
- `metadata` - Domain/problem context and export timestamp

## Key Concepts

### Safe/Sound Model
**Definition**: Conservative model that avoids false negatives

**Construction**:
- **Preconditions**: ALL possible preconditions (possible_preconditions)
- **Add effects**: ONLY confirmed add effects (confirmed_add_effects)
- **Delete effects**: ONLY confirmed delete effects (confirmed_del_effects)

**Expected Metrics**:
- Precondition recall → 100% (includes all possible preconditions)
- Effect precision → High (only confirmed effects)

**Use Case**: Planning with a safe model guarantees no missing preconditions

### Complete Model
**Definition**: Optimistic model that avoids false positives

**Construction**:
- **Preconditions**: ONLY certain preconditions (certain_preconditions)
- **Add effects**: confirmed + possible add effects
- **Delete effects**: confirmed + possible delete effects

**Expected Metrics**:
- Precondition precision → High (only confirmed preconditions)
- Effect recall → High (includes all possible effects)

**Use Case**: Planning with complete model may be overly restrictive but is precise

## Generating Checkpoints

### Reconstruction Script

**`scripts/reconstruct_early_models.py`** - Regenerate checkpoints from observations

**Usage**:
```bash
# Reconstruct checkpoints for specific domain/problem
python scripts/reconstruct_early_models.py \
    --domain blocksworld \
    --problem p00 \
    --experiments-dir results/paper/consolidated_experiments/information_gain

# Reconstruct all problems in a domain
python scripts/reconstruct_early_models.py \
    --domain blocksworld \
    --experiments-dir results/paper/consolidated_experiments/information_gain

# Reconstruct everything
python scripts/reconstruct_early_models.py \
    --experiments-dir results/paper/consolidated_experiments/information_gain
```

**What it does**:
1. Loads experiment configuration (domain PDDL, problem PDDL, checkpoint frequency)
2. Initializes Information Gain learner with domain
3. Replays observations sequentially
4. Exports model snapshot at each checkpoint iteration
5. Saves to `models/model_iter_NNN.json` with proper metadata

**Important**: Uses `export_model_snapshot()` method which produces the metadata-rich format required by post-processing.

## Computing Metrics

### Main Analysis Script

**`scripts/analyze_information_gain_metrics.py`** - Complete metrics computation pipeline

**Usage**:
```bash
# Analyze all domains
python scripts/analyze_information_gain_metrics.py \
    --consolidated-dir results/paper/consolidated_experiments \
    --benchmarks-dir benchmarks/olam-compatible \
    --output-dir results/information_gain_metrics

# Analyze specific domain
python scripts/analyze_information_gain_metrics.py \
    --domain blocksworld \
    --consolidated-dir results/paper/consolidated_experiments \
    --benchmarks-dir benchmarks/olam-compatible \
    --output-dir results/information_gain_metrics

# Analyze single problem
python scripts/analyze_information_gain_metrics.py \
    --domain blocksworld \
    --problem p00 \
    --consolidated-dir results/paper/consolidated_experiments \
    --benchmarks-dir benchmarks/olam-compatible \
    --output-dir results/information_gain_metrics
```

**What it does**:
1. Finds all checkpoint files (`model_iter_*.json`) in problem directories
2. For each checkpoint:
   - Reconstructs safe and complete models
   - Compares against ground truth PDDL domain
   - Computes precision/recall/F1 for preconditions, add effects, delete effects
3. Saves per-problem metrics to JSON
4. Aggregates metrics across problems to domain level
5. Exports flat CSV for analysis

## Output Files

### Per-Problem Metrics
**Location**: `results/information_gain_metrics/{domain}/{problem}/metrics_per_iteration.json`

**Format**:
```json
{
  "1": {
    "safe_model": {
      "precision": 0.260,
      "recall": 0.500,
      "f1": 0.301,
      "precondition_precision": 0.279,
      "precondition_recall": 1.000,
      "effect_precision": 1.000,
      "effect_recall": 0.222,
      "detailed_per_action": {
        "pick-up": {
          "precondition_precision": 0.333,
          "precondition_recall": 1.000,
          "add_effect_precision": 1.000,
          "add_effect_recall": 1.000,
          "delete_effect_precision": 1.000,
          "delete_effect_recall": 0.667
        }
      }
    },
    "complete_model": {
      "precision": 0.188,
      "recall": 0.450,
      "f1": 0.241,
      "precondition_precision": 0.400,
      "precondition_recall": 0.667,
      "effect_precision": 0.667,
      "effect_recall": 0.333,
      "detailed_per_action": {...}
    },
    "execution_stats": {
      "pick-up": {
        "executions": 3,
        "successes": 2,
        "failures": 1
      },
      "put-down": {
        "executions": 2,
        "successes": 1,
        "failures": 1
      },
      "stack": {
        "executions": 1,
        "successes": 0,
        "failures": 1
      }
    }
  },
  "2": {...},
  "3": {...}
}
```

**Execution Statistics** (NEW):
The `execution_stats` field tracks action execution counts cumulative up to each iteration:
- `executions`: Total number of times this action was attempted (successes + failures)
- `successes`: Number of successful executions (action preconditions satisfied, expected effects observed)
- `failures`: Number of failed executions (preconditions not satisfied or unexpected effects)

**Notes**:
- Execution stats are cumulative from iteration 1 to N
- Actions not yet executed won't appear in execution_stats
- Success/failure counts help understand learning difficulty per action
- High failure rates indicate complex or hard-to-learn preconditions

### Domain-Level Metrics
**Location**: `results/information_gain_metrics/{domain}/domain_metrics.json`

**Format**:
```json
{
  "1": {
    "safe": {
      "overall": {
        "precision": 0.099,
        "recall": 0.319,
        "f1": 0.133,
        "problem_count": 10
      },
      "preconditions": {
        "precision": 0.173,
        "recall": 0.833,
        "problem_count": 10
      },
      "effects": {
        "precision": 0.225,
        "recall": 0.050,
        "problem_count": 10
      }
    },
    "complete": {...}
  },
  "2": {...}
}
```

**Notes**:
- Metrics are **averaged** across all problems in the domain
- `problem_count` indicates how many problems contributed to each iteration
- Not all problems reach the same max iteration

### Flat CSV Export
**Location**: `results/information_gain_metrics/checkpoint_metrics.csv`

**Format**:
```csv
algorithm,domain,problem,iteration,model_type,component,precision,recall,f1
information_gain,blocksworld,p00,1,safe,preconditions,0.279,1.000,
information_gain,blocksworld,p00,1,safe,add_effects,1.000,1.000,
information_gain,blocksworld,p00,1,safe,delete_effects,1.000,0.667,
information_gain,blocksworld,p00,1,safe,overall,0.260,0.500,0.301
information_gain,blocksworld,p00,1,complete,preconditions,0.400,0.667,
```

**Use Cases**:
- Import into pandas for analysis
- Generate learning curves
- Compare safe vs complete models
- Compare Information Gain vs OLAM

## Core Processing Infrastructure

### Model Reconstruction
**`src/core/model_reconstructor.py`** - Converts checkpoints to ReconstructedModel

**Key Methods**:
```python
# Reconstruct safe model (conservative)
safe_model = ModelReconstructor.reconstruct_information_gain_safe(checkpoint_data)

# Reconstruct complete model (optimistic)
complete_model = ModelReconstructor.reconstruct_information_gain_complete(checkpoint_data)
```

**Returns**: `ReconstructedModel` with:
- `model_type` - "safe" or "complete"
- `algorithm` - "information_gain"
- `iteration` - Checkpoint iteration number
- `actions` - Dict mapping action names to ReconstructedAction objects

### Metrics Computation
**`src/core/model_metrics.py`** - Computes precision/recall/F1

**Key Method**:
```python
# Initialize with ground truth
metrics_calculator = ModelMetrics(ground_truth_domain, ground_truth_problem)

# Compute metrics for a model
metrics = metrics_calculator.compute_metrics(safe_model)
```

**Returns**: Flat dictionary with:
- `precision`, `recall`, `f1` - Overall averages
- `precondition_precision`, `precondition_recall` - Precondition-only metrics
- `effect_precision`, `effect_recall` - Effect-only metrics (add + delete combined)
- `detailed_per_action` - Per-action breakdown with TP/FP/FN counts

### Ground Truth Validation
**`src/core/model_validator.py`** - Parses ground truth and compares models

**Key Method**:
```python
# Initialize with ground truth PDDL
validator = ModelValidator(domain_file, problem_file)

# Compare learned action against ground truth
result = validator.compare_action(
    action_name="pick-up",
    learned_preconditions={"clear(?0)", "ontable(?0)", "handempty"},
    learned_add_effects={"holding(?0)"},
    learned_delete_effects={"ontable(?0)", "handempty"}
)
```

**Returns**: `ModelComparisonResult` with per-component metrics and error sets

## Parameter Normalization

### Critical Normalization Rules

**1. Positional Parameter Normalization**
Converts different parameter naming schemes to positional format:
- Ground truth: `(lifting ?x ?y)` → `(lifting ?0 ?1)`
- OLAM format: `(lifting ?param_1 ?param_2)` → `(lifting ?0 ?1)`
- Information Gain: `(lifting ?0 ?1)` → `(lifting ?0 ?1)` (already normalized)

**2. Zero-Arity Predicate Normalization**
Handles predicates with no parameters:
- Ground truth: `handempty()` (from ExpressionConverter)
- Information Gain: `handempty` (from LiftedDomainKnowledge)
- **Solution**: Strip trailing `()` during comparison

**Implementation**: `src/core/model_validator.py:18-61`
```python
def normalize_predicate_parameters(predicate: str) -> str:
    """Normalize for comparison."""
    # Strip trailing () for 0-arity predicates
    if predicate.endswith('()'):
        predicate = predicate[:-2]

    # Normalize parameters to positional
    params = re.findall(r'\?[\w_]+', predicate)
    param_map = {p: f"?{i}" for i, p in enumerate(params)}

    # Replace (longest first to avoid partial replacements)
    for old, new in sorted(param_map.items(), key=lambda x: len(x[0]), reverse=True):
        predicate = predicate.replace(old, new)

    return predicate
```

### Impact of Normalization

**Before Fix** (blocksworld p00, iteration 1):
- Precondition recall: 83.3% (missing `handempty` match)
- Effect precision: 75.0% (false positive on `handempty()`)

**After Fix**:
- Precondition recall: **100.0%** (+16.7%)
- Effect precision: **100.0%** (+25.0%)

## Metrics Calculation Details

### Effect Metrics Computation
Effects combine add and delete components:
```python
# Sum TP/FP/FN across add and delete
total_tp = add_tp + delete_tp
total_fp = add_fp + delete_fp
total_fn = add_fn + delete_fn

# Compute combined effect metrics
effect_precision = total_tp / (total_tp + total_fp)
effect_recall = total_tp / (total_tp + total_fn)
```

### Precondition Metrics Computation
Preconditions are analyzed separately per action, then averaged:
```python
# Per action
prec_precision = prec_tp / (prec_tp + prec_fp)
prec_recall = prec_tp / (prec_tp + prec_fn)

# Averaged across all actions
avg_prec_precision = sum(prec_precisions) / action_count
avg_prec_recall = sum(prec_recalls) / action_count
```

### Overall Metrics
Overall metrics average precondition and effect components:
```python
overall_precision = (prec_precision + effect_precision) / 2
overall_recall = (prec_recall + effect_recall) / 2
overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
```

## Common Issues

### Empty Actions at Early Checkpoints
**Problem**: Some actions show 0% metrics at early iterations

**Cause**: Actions not yet observed, so no preconditions/effects learned

**Solution**: Normal behavior - focus analysis on later iterations after all actions observed

### Safe Model Recall Not 100%
**Problem**: Safe model precondition recall < 100% at some checkpoints

**Cause**:
1. Unobserved actions have empty possible_preconditions → 0% recall
2. Averaging includes these zeros

**Solution**: Check final checkpoint after all actions observed, or filter to only observed actions

### Complete Model Shows Lower Recall
**Problem**: Complete model has lower effect recall than expected

**Cause**: Complete model only includes confirmed effects (not possible effects)

**Solution**: By design - complete model prioritizes precision over recall

### Checkpoint Files Missing
**Problem**: No checkpoint files found in `models/` directory

**Cause**: Checkpoints not yet generated or wrong format used

**Solution**: Run `scripts/reconstruct_early_models.py` to generate proper checkpoints

### Metrics All Zeros
**Problem**: All metrics show 0.0 across all iterations

**Cause**: Format mismatch or normalization issue

**Solution**:
1. Verify checkpoint format has `iteration`, `algorithm`, `actions` fields
2. Check normalization fix is applied to `model_validator.py`
3. Verify ground truth PDDL files are correct

## Related Files

**Core Processing**:
- `src/algorithms/information_gain.py` - Main learning algorithm with `export_model_snapshot()`
- `src/core/model_reconstructor.py` - Checkpoint → ReconstructedModel conversion
- `src/core/model_metrics.py` - Precision/recall computation
- `src/core/model_validator.py` - Ground truth comparison with normalization

**Scripts**:
- `scripts/reconstruct_early_models.py` - Regenerate checkpoints from observations
- `scripts/analyze_information_gain_metrics.py` - Full metrics analysis pipeline

**Support**:
- `src/core/lifted_domain.py` - PDDL domain representation (parameter-bound literals)
- `src/core/expression_converter.py` - UP expression → string conversion

## Comparison with OLAM

| Aspect | Information Gain | OLAM |
|--------|------------------|------|
| **Execution** | Runs in our codebase | External (user-run) |
| **Checkpoint Format** | Single JSON with metadata | 8-10 separate JSON files |
| **Reconstruction** | Replay observations | Parse JSON exports |
| **Safe Model** | All possible precs + certain effects | Certain + uncertain precs + certain effects |
| **Complete Model** | Certain precs + all effects | Certain precs + certain + uncertain effects |
| **Normalization** | Already positional (?0, ?1) | Needs parameter normalization |
| **Output Structure** | Per-iteration JSON + domain aggregation + CSV | Same pattern |

Both use the **same metrics computation infrastructure** (ModelMetrics, ModelValidator) for consistent comparison.
