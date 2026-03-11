---
description: Rules for experiments and scripts
paths:
  - "scripts/**"
---

## Experiments Rules

- The `scripts/` directory contains experiment and analysis scripts

## Running experiments

### Via AMLGym (preferred)

Requires: `pip install -e .` in this project, and AMLGym installed.

```bash
# Single domain
python3 scripts/run_amlgym_experiment.py --domain blocksworld

# Multiple domains
python3 scripts/run_amlgym_experiment.py --domain blocksworld gripper hanoi

# All 20 AMLGym benchmark domains
python3 scripts/run_amlgym_experiment.py --all-domains

# With evaluation metrics (syntactic precision/recall, predictive power)
python3 scripts/run_amlgym_experiment.py --domain blocksworld --evaluate

# Custom settings
python3 scripts/run_amlgym_experiment.py --domain blocksworld --max-steps 300 --model-mode complete --seed 123 --verbose
```

Options:
- `--max-steps` (default 500): learning iterations
- `--model-mode` (`safe`|`complete`): safe = all possible preconditions + confirmed effects; complete = certain preconditions + all possible effects
- `--no-object-subset`: disable object subset selection (use all objects)
- `--spare-objects N` (default 2): extra objects per type for subset selection
- `--seed` (default 42): random seed
- `--output-dir` (default `results/amlgym`): where to save learned models
- `--evaluate`: run AMLGym evaluation metrics after learning

### Via local runner (legacy)

```bash
python3 scripts/run_full_experiments.py --mode quick
python3 scripts/run_full_experiments.py --domains blocksworld hanoi --iterations 200
```

## Current experiments remain as verification

Scripts in `scripts/` are for analysis and verification only. The core algorithm should not depend on experiment infrastructure.
