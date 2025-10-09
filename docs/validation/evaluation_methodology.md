# Online Model Learning Evaluation Methodology

## Overview

Two-phase evaluation approach inspired by SAM Learning discrete experiments:

1. **Online Phase**: Metrics about the learning process itself
2. **Offline Phase**: Semantic evaluation of intermediate domain snapshots

---

## Phase 1: Online Learning Metrics

**Purpose**: Track learning progress and efficiency during execution

**Collected During**: Active learning episodes

### Metrics

| Metric | Description | Collection |
|--------|-------------|------------|
| Episode success rate | % of episodes reaching goal | Per episode |
| Steps per episode | Efficiency of learned policy | Per episode |
| Exploration coverage | # unique states visited | Cumulative |
| Action selection time | Query time per decision | Per action |
| Model updates | # of precondition/effect updates | Per observation |
| Query type distribution | Observation vs. membership queries | Cumulative |

### Implementation

```python
class OnlineLearningMetrics:
    """Track metrics during online learning."""

    def __init__(self):
        self.episodes = []
        self.total_observations = 0
        self.total_queries = 0
        self.states_visited = set()

    def record_episode(self, episode_data):
        self.episodes.append({
            'episode_num': len(self.episodes),
            'success': episode_data.goal_reached,
            'steps': episode_data.step_count,
            'observations': len(episode_data.trajectory),
            'queries': episode_data.query_count
        })
        self.total_observations += len(episode_data.trajectory)
        self.states_visited.update(episode_data.states)

    def export_online_stats(self, output_path):
        """Export CSV with per-episode metrics."""
        df = pd.DataFrame(self.episodes)
        df['cumulative_observations'] = df['observations'].cumsum()
        df['success_rate_rolling'] = df['success'].rolling(10).mean()
        df.to_csv(output_path, index=False)
```

**Output**: `online_metrics_fold_{fold}.csv`

---

## Phase 2: Offline Domain Evaluation

**Purpose**: Measure quality of intermediate learned domains

**Execution**: After learning completes, evaluate domain snapshots against ground truth

### 2.1 Domain Snapshots

**Strategy**: Save intermediate domains at observation count milestones

```python
# During online learning
checkpoints = [10, 20, 50, 100, 200, 500, 1000]

if total_observations in checkpoints:
    intermediate_domain = learner.get_learned_model()
    export_domain(intermediate_domain, f"domain_obs_{total_observations}.pddl")
```

**Stored**: `results_directory/intermediate_domains/fold_{fold}/domain_obs_{count}.pddl`

### 2.2 Semantic Performance Evaluation

**Adapted from**: `sam_learning/statistics/encoded_performance_calculator.py`

**Method**: Compare learned domain behavior against ground truth on test observations

#### Precondition Evaluation

Test action applicability:

```
For each test observation:
    For each state-action pair:
        applicable_ground_truth = can_apply(action, state, ground_truth_domain)
        applicable_learned = can_apply(action, state, learned_domain)

        TP: both applicable
        FP: learned applicable, ground truth not (UNSAFE)
        FN: learned not applicable, ground truth is (INCOMPLETE)
```

**Metrics**:
- **Precondition Precision** = TP / (TP + FP)
- **Precondition Recall** = TP / (TP + FN)

#### Effect Evaluation

Compare state transitions:

```
For each test observation:
    For each (s, a, s') triplet:
        ground_truth_effects = predicates(s') - predicates(s)

        if action not in learned_domain:
            learned_effects = {}
        else:
            s'_learned = apply(a, s, learned_domain)
            learned_effects = predicates(s'_learned) - predicates(s)

        TP: effects present in both
        FP: effects in learned but not ground truth
        FN: effects in ground truth but not learned
```

**Metrics**:
- **Effect Precision** = TP / (TP + FP)
- **Effect Recall** = TP / (TP + FN)

---

## Soundness vs. Completeness Tradeoff

### Definitions

- **Sound Domain**: Only allows safe actions (no false positives)
- **Complete Domain**: Allows all legal actions (no false negatives)

### Precision/Recall Interpretation

| Domain Type | Precondition Precision | Precondition Recall | Effect Precision | Effect Recall |
|-------------|------------------------|---------------------|------------------|---------------|
| Sound | **High** (few FP) | Low (many FN) | **High** (few FP) | Low (many FN) |
| Complete | Low (many FP) | **High** (few FN) | Low (many FP) | **High** (few FN) |
| Ideal | 1.0 | 1.0 | 1.0 | 1.0 |

### Key Insights

1. **Sound domains are restrictive**:
   - Learned preconditions are stricter than necessary
   - Actions rejected when they should be applicable → High precision, low recall
   - Safer but less useful

2. **Complete domains are permissive**:
   - Learned preconditions allow actions that shouldn't apply
   - Actions accepted when they're actually invalid → Low precision, high recall
   - More useful but potentially unsafe

3. **OLAM targets soundness** (via CNF approach):
   - Conservative precondition learning
   - Expect: Precision ≈ 1.0, Recall increases with observations

4. **Information Gain balances both**:
   - Active query selection to minimize both FP and FN
   - Expect: Precision and recall both increase together

---

## Implementation Plan

### Phase 1: Offline-Only Evaluation

**Start simple**: Evaluate final learned domains only

```python
# After online learning completes
final_domain = learner.get_learned_model()
export_domain(final_domain, "final_domain.pddl")

# Run semantic evaluation
performance_calc = SemanticPerformanceCalculator(
    model_domain=ground_truth_domain,
    observations=test_observations,
    working_directory_path=results_dir,
    learning_algorithm=algorithm_type
)

performance_calc.calculate_performance(
    learned_domain_path="final_domain.pddl",
    num_used_observations=total_observations
)
performance_calc.export_semantic_performance(fold_num=0)
```

**Output**: Single row CSV with precision/recall for final model

### Phase 2: Add Checkpoints

**Once Phase 1 works**: Add intermediate domain evaluation

```python
# Save domains during learning
for obs_count in [10, 20, 50, 100, 200]:
    if total_observations >= obs_count and obs_count not in evaluated_checkpoints:
        intermediate_domain = learner.get_learned_model()
        domain_path = save_domain(intermediate_domain, obs_count)
        domain_snapshots.append((obs_count, domain_path))
        evaluated_checkpoints.add(obs_count)

# After learning, evaluate all snapshots
for obs_count, domain_path in domain_snapshots:
    performance_calc.calculate_performance(domain_path, obs_count)

performance_calc.export_semantic_performance(fold_num=0)
```

**Output**: Multi-row CSV showing convergence over time

---

## Adapter-Specific Considerations

### OLAM (Sound Domain Learner)

**Encoding**: OLAM learns lifted schemas, need ground action mapping

```python
def olam_encode(ground_action):
    """Map ground action to OLAM schema instances."""
    for schema in olam_adapter.get_action_schemas():
        if matches_signature(ground_action, schema):
            return [instantiate_schema(schema, ground_action.parameters)]
    return []  # Action not yet learned

# Set in performance calculator
performance_calc.encode = olam_encode
performance_calc.decode = lambda x: x
```

**Expected Metrics**:
- Precondition Precision: ~1.0 (sound by design)
- Precondition Recall: Starts low, increases with observations
- Effect Precision: ~1.0
- Effect Recall: Increases as more effects observed

### Information Gain (Balanced Learner)

**Encoding**: Similar CNF-based approach

```python
def infogain_encode(ground_action):
    """Map ground action to InfoGain learned schemas."""
    # Similar to OLAM
    schemas = infogain_adapter.get_action_schemas()
    return [instantiate_schema(schema, ground_action.parameters)
            for schema in schemas if matches(ground_action, schema)]
```

**Expected Metrics**:
- Both precision and recall should increase together
- Active querying targets high-uncertainty regions
- May converge faster than OLAM

---

## File Organization

```
results_directory/
├── online_metrics/
│   ├── fold_0_online_metrics.csv          # Phase 1: Online learning stats
│   ├── fold_1_online_metrics.csv
│   └── ...
├── intermediate_domains/
│   ├── fold_0/
│   │   ├── domain_obs_10.pddl             # Saved snapshots
│   │   ├── domain_obs_20.pddl
│   │   └── ...
│   └── ...
└── semantic_performance/
    ├── olam_fold_0_semantic_performance.csv   # Phase 2: Offline evaluation
    ├── infogain_fold_0_semantic_performance.csv
    └── ...
```

---

## CSV Output Formats

### Online Metrics CSV

```csv
episode_num,success,steps,observations,queries,cumulative_observations,success_rate_rolling
0,True,15,15,0,15,1.0
1,False,50,50,2,65,0.5
...
```

### Semantic Performance CSV (from SAM Learning)

```csv
action_name,learning_algorithm,num_trajectories,precondition_precision,precondition_recall,effects_precision,effects_recall
pickup,olam,10,1.0,0.3,1.0,0.4
stack,olam,10,1.0,0.2,1.0,0.35
pickup,olam,50,1.0,0.7,1.0,0.75
...
```

---

## Summary

**Key Simplifications**:

1. ✅ **Separate online/offline**: Don't mix real-time metrics with domain evaluation
2. ✅ **Checkpoint-based**: Save domains periodically, evaluate later
3. ✅ **Reuse SAM evaluation**: Adapt `SemanticPerformanceCalculator` with minimal changes
4. ✅ **Clear soundness interpretation**: High precision = sound, high recall = complete

**Implementation Order**:

1. Add `get_learned_model()` to adapters (if not present)
2. Implement domain export at checkpoints
3. Create simple `SemanticPerformanceCalculator` wrapper
4. Start with final-model-only evaluation
5. Add incremental checkpoints once working

**Next Steps**: Implement Phase 1 (offline-only) for single checkpoint evaluation
