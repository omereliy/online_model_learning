# Algorithm Comparison Semantics: OLAM vs Information Gain

This document describes the semantic differences between OLAM and Information Gain algorithms
and how to ensure fair comparison of their learned models.

## Overview

Both algorithms learn action models from observations, but they use fundamentally different
learning mechanisms that produce different model representations. Understanding these differences
is crucial for fair benchmarking and comparison.

## Key Semantic Differences

### 1. Safe Model Preconditions

The "safe" model represents a conservative estimate with maximal preconditions (fewer false negatives).

| Aspect | Information Gain | OLAM |
|--------|------------------|------|
| **Source** | `possible_preconditions` (all non-ruled-out) | ALL type-compatible - `useless_precs` |
| **Initialization** | La (all parameter-bound literals) | Empty (no observations) |
| **Unexecuted Action** | Full La | Empty (BUG - should be La) |

**Important:** For fair comparison of unexecuted actions, OLAM safe model reconstruction
requires a `domain_file` parameter to generate all possible preconditions.

### 2. Complete Model Preconditions

The "complete" model represents an optimistic estimate with minimal preconditions (fewer false positives).

| Aspect | Information Gain | OLAM |
|--------|------------------|------|
| **Source** | `certain_preconditions` (singleton constraints) | `certain_preconditions` (intersection of successes) |
| **Learning Mechanism** | Failure-based (constraint from failed states) | Success-based (intersection of successful states) |
| **Semantic Meaning** | "Must be true (proven by failures)" | "Always was true (observed in successes)" |

**These are fundamentally different semantics!** InfoGain learns what MUST be true from failures,
while OLAM learns what IS ALWAYS true from successes. Same terminology, different meanings.

### 3. Effect Contradiction Handling

When the same fluent appears in both add and delete effects:

| Algorithm | Strategy |
|-----------|----------|
| Information Gain | Remove from BOTH sets (conservative) |
| OLAM | Keep in add, remove from delete (prefer add) |

This produces different complete models from the same observations.

## Expected Metrics for Edge Cases

### Unexecuted Action (0 observations)

| Model Type | Component | Precision | Recall | Notes |
|------------|-----------|-----------|--------|-------|
| **Safe** | Preconditions | 0.0 | 1.0 | All possible precs included (many FP, no FN) |
| **Safe** | Effects | 1.0 | 0.0 | Empty (no claims, all missing) |
| **Complete** | Preconditions | 1.0 | 0.0 | Empty (no claims, all missing) |
| **Complete** | Effects | 1.0 | 0.0 | Empty (no claims, all missing) |

### Only Failed Action (failures, no successes)

| Algorithm | Safe Preconditions | Complete Preconditions | Effects |
|-----------|-------------------|------------------------|---------|
| Information Gain | La (all possible) | Singletons from constraints | Empty (no successes) |
| OLAM | Empty* | Empty | Empty |

*OLAM cannot learn from failures alone - needs at least one success for comparison.

### Only Succeeded Action (successes, no failures)

| Algorithm | Safe Preconditions | Complete Preconditions | Effects |
|-----------|-------------------|------------------------|---------|
| Information Gain | Refined by intersection | Empty (no singleton constraints) | Learned from state changes |
| OLAM | Certain + Uncertain | Intersection of successful states | Learned from state changes |

## Normalization for Fair Comparison

Use the `ModelNormalizer` class to ensure fair comparison:

```python
from src.core.model_normalizer import ModelNormalizer

# Generate all possible literals for an action
all_possible = ModelNormalizer.generate_all_possible_literals(domain_file, action_name)

# Normalize safe model preconditions
normalized_model = ModelNormalizer.normalize_safe_preconditions(
    model, algorithm="olam", all_possible_literals=all_possible, strategy="infogain_style"
)

# Normalize contradiction handling
add_eff, del_eff = ModelNormalizer.normalize_contradiction_handling(
    add_effects, del_effects, strategy="remove_both"
)

# Compare models side-by-side
comparison = ModelNormalizer.compare_models_side_by_side(
    infogain_model, olam_model, normalize=True, all_possible_literals=all_possible
)
```

## Filtering Unexecuted Actions from Metrics

To exclude unexecuted actions from aggregate metrics:

```python
from src.core.model_metrics import ModelMetrics

# Extract observation counts from snapshot
observation_counts = {
    action_name: action_data.get("observations", 0)
    for action_name, action_data in snapshot["actions"].items()
}

# Compute metrics excluding unexecuted actions
metrics = model_metrics.compute_metrics(
    model,
    observation_counts=observation_counts,
    min_observations=1  # Exclude actions with 0 observations
)

# Check which actions were excluded
print(f"Excluded: {metrics['excluded_actions']}")
print(f"Actions evaluated: {metrics['actions_evaluated']}")
```

## Intentional vs Bug Differences

### Intentional Differences (Do Not "Fix")

1. **Certain precondition semantics**: InfoGain uses failure analysis, OLAM uses success intersection
2. **Learning from failures alone**: OLAM requires successes, InfoGain can learn from failures
3. **Different model philosophies**: Constraint-based (InfoGain) vs Observation-based (OLAM)

### Bugs to Fix

1. **OLAM Safe model for unexecuted actions**: Should use all possible predicates (La),
   currently returns empty set without domain_file. Fixed by providing domain_file parameter.

### Configurable Behaviors

1. **Contradiction handling**: Use `ModelNormalizer.normalize_contradiction_handling()` with
   desired strategy
2. **Precondition universe**: Use `ModelNormalizer.normalize_safe_preconditions()` with
   desired strategy
3. **Action filtering**: Use `ModelMetrics.compute_metrics()` with `min_observations` parameter

## Recommended Comparison Workflow

1. **Always provide domain_file** when reconstructing OLAM safe models
2. **Use normalization** when comparing across algorithms
3. **Filter unexecuted actions** from aggregate metrics using `min_observations=1`
4. **Document which normalization** strategy was used in your results
5. **Report per-action metrics** alongside aggregates to identify edge cases

## Files Reference

- `src/core/model_reconstructor.py`: Safe/complete model construction
- `src/core/model_normalizer.py`: Normalization utilities
- `src/core/model_metrics.py`: Metrics computation with filtering
- `src/core/model_validator.py`: Precision/recall calculation
- `tests/validation/test_algorithm_comparison_fairness.py`: Validation tests
