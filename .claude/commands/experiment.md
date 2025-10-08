---
description: Run and analyze experiment
args:
  - name: config
    description: Experiment config file or description (e.g., "configs/blocksworld_olam.yaml" or "compare OLAM vs InfoGain on blocksworld")
    required: true
---

# Run Experiment

**Context**: Load @.claude/commands/_project.md

**Experiment**: $ARGUMENT

## Execution Steps

1. **Setup**
   - Parse config file or create config from description
   - Validate algorithms available
   - Check domain/problem files exist

2. **Run**
   - Execute experiment with ExperimentRunner
   - Collect metrics (sample complexity, runtime, accuracy)
   - Monitor for convergence

3. **Analyze**
   - Compare algorithm performance
   - Statistical significance (if multiple runs)
   - Key insights from results

4. **Report**
   - Save results to results/ directory
   - Generate comparison plots if needed
   - Brief summary of findings

## Output Format

```
EXPERIMENT: OLAM vs InfoGain on blocksworld
Config: configs/comparison.yaml
Trials: 5 per algorithm

Running...
[Progress indicator]

RESULTS:
---------
OLAM:
  Sample complexity: 145.2 ± 12.3 actions
  Convergence time: 8.3s
  Model accuracy: 92%

Information Gain:
  Sample complexity: 98.6 ± 8.7 actions
  Convergence time: 12.1s
  Model accuracy: 95%

Statistical test: InfoGain significantly better (p=0.023)

Saved: results/blocksworld_comparison_2024_10_07.json
```

Focus on key metrics and conclusions.