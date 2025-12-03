# OLAM Post-Processing Guide

## Overview

OLAM (Online Learning of Action Models) is run externally by the user. This guide explains how to process OLAM's output results for analysis and metrics computation.

## OLAM Results Location

OLAM results are stored in: `/path/to/project/olam_results/`

Each domain has its own directory structure.

## Results Directory Structure

```
olam_results/
├── <domain_name>/                    # e.g., depots, blocksworld
│   ├── trace_complete.json           # Concatenated trace of all problems
│   ├── 1_p00_<domain>_gen/           # Problem directory
│   │   ├── trace.json                # JSON Lines format (one JSON per line)
│   │   ├── checkpoints/
│   │   │   ├── iter_1/               # Checkpoint at iteration 1
│   │   │   │   ├── operator_certain_predicates.json
│   │   │   │   ├── operator_uncertain_precs.json
│   │   │   │   ├── operator_certain_positive_effects.json
│   │   │   │   ├── operator_certain_negative_effects.json
│   │   │   │   ├── operator_uncertain_positive_effects.json
│   │   │   │   ├── operator_uncertain_negative_effects.json
│   │   │   │   ├── operator_useless_negated_precs.json
│   │   │   │   ├── operator_useless_possible_precs.json
│   │   │   ├── iter_2/
│   │   │   ├── iter_5/
│   │   │   └── ... (checkpoints may end earlier than max iterations)
│   │   ├── final/                    # Final checkpoint (same structure)
│   │   └── domain_learned.pddl       # Final learned domain
│   ├── 2_p01_<domain>_gen/
│   └── ... (all problems for this domain)
```

### Key Files in Each Checkpoint

**8 JSON Export Files** (per checkpoint):
1. `operator_certain_predicates.json` - Preconditions that appear in ALL successful executions
2. `operator_uncertain_precs.json` - Preconditions that appear in SOME executions
3. `operator_certain_positive_effects.json` - Add effects confirmed by all observations
4. `operator_certain_negative_effects.json` - Delete effects confirmed by all observations
5. `operator_uncertain_positive_effects.json` - Possible add effects
6. `operator_uncertain_negative_effects.json` - Possible delete effects
7. `operator_useless_negated_precs.json` - Preconditions ruled out as unnecessary
8. `operator_useless_possible_precs.json` - Possible preconditions ruled out

### Notes
- Checkpoint numbering is dynamic: `iter_1, iter_2, ..., iter_N`
- Not all problems reach the same max iteration (may terminate early)
- `trace.json` is JSON Lines format: one JSON object per line
- Each line: `{"domain": "...", "iter": 1, "action": "...", "success": true, ...}`

## Key Concepts

### Safe/Sound Model
**Definition**: Conservative model that avoids false negatives

**Construction**:
- **Preconditions**: certain + uncertain (all possible preconditions)
- **Add effects**: ONLY certain (confirmed positive effects)
- **Delete effects**: ONLY certain (confirmed negative effects)

**Expected Metrics**:
- Precondition recall → 100% (no false negatives)
- Effect precision → High (only confirmed effects)

### Complete Model
**Definition**: Optimistic model that avoids false positives

**Construction**:
- **Preconditions**: ONLY certain (minimal preconditions)
- **Add effects**: certain + uncertain (all possible positive effects)
- **Delete effects**: certain + uncertain (all possible negative effects)

**Contradiction Handling**: If same literal appears in both add and delete, it goes in add effects.

**Expected Metrics**:
- Precondition precision → High (only confirmed preconditions)
- Effect recall → High (includes all possible effects)

## Processing OLAM Results

### Main Script

**`scripts/process_olam_results.py`** - Complete post-processing pipeline

**Usage**:
```bash
python scripts/process_olam_results.py \
    --olam-results /path/to/olam_results/depots \
    --ground-truth benchmarks/olam-compatible/depots/domain.pddl \
    --output-dir results/olam_processed
```

**What it does**:
1. Auto-detects all problem directories (1_p00_*, 2_p01_*, ...)
2. Auto-detects checkpoints per problem (iter_1, iter_2, ...)
3. Loads JSON exports from each checkpoint
4. Reconstructs safe and complete models
5. Computes metrics (precision/recall) vs ground truth
6. Aggregates metrics across problems to domain level

### Output Files

**Per-Problem Files** (in `results/olam_processed/`):
- `p00_safe_metrics.json` - Safe model metrics for problem p00
- `p00_complete_metrics.json` - Complete model metrics for problem p00
- `p00_detailed_metrics.json` - Detailed per-action TP/FP/FN breakdown (for debugging)

**Domain-Level Files**:
- `domain_safe_metrics.json` - Aggregated safe metrics across all problems
- `domain_complete_metrics.json` - Aggregated complete metrics across all problems
- `domain_metrics.json` - Combined safe and complete metrics

### Metrics Format

**Main Metrics** (safe/complete files):
```json
{
  "checkpoints": {
    "1": {
      "precondition_precision": 0.0,
      "precondition_recall": 0.0,
      "effect_precision": 0.0,
      "effect_recall": 0.0,
      "overall_precision": 0.0,
      "overall_recall": 0.0
    },
    "50": {
      "precondition_precision": 0.36,
      "precondition_recall": 0.36,
      "effect_precision": 0.16,
      "effect_recall": 0.16,
      "overall_precision": 0.23,
      "overall_recall": 0.23
    }
  }
}
```

**Detailed Metrics** (detailed files):
```json
{
  "checkpoints": {
    "50": {
      "safe": {
        "lift": {
          "add_tp": 2, "add_fp": 1, "add_fn": 0,
          "add_precision": 0.67, "add_recall": 1.0,
          "del_tp": 1, "del_fp": 0, "del_fn": 1,
          "del_precision": 1.0, "del_recall": 0.5,
          "prec_precision": 0.6, "prec_recall": 1.0
        }
      }
    }
  }
}
```

## Important Notes

### Effect Metrics Calculation
- Effects are computed by **summing TP/FP/FN** across add and delete effects
- Formula: `effect_precision = total_tp / (total_tp + total_fp)`
- Where: `total_tp = add_tp + delete_tp`

### Precondition vs Effect Metrics
- **Precondition metrics**: Averaged per action, then averaged across actions
- **Effect metrics**: Summed TP/FP/FN across all actions, then computed once
- These are kept **separate** - not combined into "overall domain precision"

### Overall Precision/Recall
- Kept for informational purposes (averages precondition and effect metrics per action)
- Not used for primary analysis - use separated precondition/effect metrics instead

## Related Files

**Core Processing**:
- `src/core/olam_trace_parser.py` - Parse OLAM's JSON exports
- `src/core/olam_knowledge_reconstructor.py` - Reconstruct knowledge from exports
- `src/core/model_reconstructor.py` - Build safe and complete models
- `src/core/model_metrics.py` - Compute precision/recall metrics

**Validation**:
- `src/core/model_validator.py` - Compare learned models vs ground truth

## Common Issues

### Empty Metrics at Early Checkpoints
**Problem**: All metrics show 0% at early iterations (< 20)

**Cause**: Actions haven't been executed yet, so no preconditions/effects learned

**Solution**: Normal behavior - focus analysis on later checkpoints (50+)

### Safe Model Recall < 100%
**Problem**: Safe model precondition recall is not 100% at intermediate checkpoints

**Cause**: Unobserved actions have empty preconditions → 0% recall drags average down

**Solution**: Check final checkpoint - all observed actions should have 100% recall

### Contradicting Effects
**Problem**: Same literal in both certain_add_effects and certain_del_effects

**Cause**: OLAM may export contradictions in some cases

**Solution**: Complete model puts contradiction in add_effects (fixed choice)
