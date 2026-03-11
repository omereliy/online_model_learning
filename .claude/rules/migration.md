---
description: Rules for aml-gym migration direction
paths:
  - "information_gain_aml/**"
---

## AML-Gym Migration Direction

This project is being simplified to work as an aml-gym importable algorithm, similar to the KG-AML project.

### Target architecture (3 core classes):
1. **SATKnowledgeBase** — CNF precondition management (from `cnf_manager.py`)
2. **InformationGainLearner** — Effect tracking + gain computation (from `information_gain.py`)
3. **Agent** — Orchestration, action selection (from `runner.py` + environment logic)

### What PDDLGym/AMLGym will replace:
- `pddl_io.py` + `up_adapter.py` + `expression_converter.py` → PDDLGym parses PDDL
- `lifted_domain.py` + `pddl_types.py` → PDDLGym `env.domain.operators` / `env.domain.predicates`
- `grounding.py` → PDDLGym binding utilities (forward grounding)
- `active_environment.py` + `mock_environment.py` → AMLGym environment
- `runner.py` + `metrics.py` → AMLGym `OnlineAlgorithmAdapter` interface

### Migration rules:
1. **Do not add new Unified Planning imports** in core algorithm code. Move toward framework-agnostic representations.
2. **Prefer string-based representations** (like `"on(?x,?y)"` format) over UP FNode objects in algorithm logic.
3. **Keep `information_gain.py` and `cnf_manager.py` framework-agnostic** where possible. These are the core that survives migration.
4. **When fixing bugs in BRIDGE modules**, keep fixes minimal. Do not refactor code that will be replaced.
