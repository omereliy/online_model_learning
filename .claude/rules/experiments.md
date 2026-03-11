---
description: Rules for experiments and scripts
paths:
  - "scripts/**"
  - "configs/**"
---

## Experiments Rules

- The `scripts/` directory contains analysis and experiment scripts
- The `configs/` directory contains YAML experiment configurations
- Experiments run via: `python3 -m src.experiments.runner --config configs/<file>.yaml`

## Migration target

Once aml-gym integration is complete, production usage will be:
```python
from online_model_learning import SATKnowledgeBase, InformationGainLearner
from amlgym.algorithms import OnlineAlgorithmAdapter
```

## Current experiments remain as verification

Scripts in `scripts/` are for analysis and verification only. The core algorithm should not depend on experiment infrastructure.
