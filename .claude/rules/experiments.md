---
description: Rules for experiments and scripts
paths:
  - "scripts/**"
  - "configs/**"
---

## Experiments Rules

- The `scripts/` directory contains analysis and experiment scripts
- The `configs/` directory contains YAML experiment configurations

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
- `--seed` (default 42): random seed
- `--output-dir` (default `results/amlgym`): where to save learned models
- `--evaluate`: run AMLGym evaluation metrics after learning

### Via internal runner (legacy)

```bash
python3 -m src.experiments.runner --config configs/<file>.yaml
```

## Current experiments remain as verification

Scripts in `scripts/` are for analysis and verification only. The core algorithm should not depend on experiment infrastructure.
