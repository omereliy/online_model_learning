# Online Model Learning

## Project Status
In simplification. Migrating toward aml-gym-importable architecture.

## Environment
- Python 3.10+, macOS
- Package manager: `uv`
- SAT solver: PySAT

## Architecture
Current: multi-module with Unified Planning. Target: 3-class design importable by aml-gym.
- `SATKnowledgeBase` — Precondition CNF management (from `cnf_manager.py`)
- `InformationGainLearner` — Effect tracking and gain computation (from `information_gain.py`)
- `Agent` — Orchestration and action selection (from `runner.py` + environment)

## Code Quality
- `uv run pytest tests/ -v` after changes
- `uv run mypy src/` after changes
- All tests must pass before committing

## Workflow
1. Check existing code before writing new code (especially `grounding.py`, `cnf_manager.py`)
2. See `.claude/rules/simplification.md` for simplification principles
3. See `.claude/rules/migration.md` for migration direction

## Key Files
| File | Purpose |
|------|---------|
| `src/algorithms/information_gain.py` | CNF-based learner (core algorithm) |
| `src/core/cnf_manager.py` | CNF formulas + SAT solving |
| `src/core/grounding.py` | Action/state grounding utilities |
| `src/core/lifted_domain.py` | Domain representation |
| `src/experiments/runner.py` | Experiment orchestration |

## Reference Docs
- `docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md` — Algorithm formulas
- `.claude/rules/` — Simplification, migration, testing, experiment rules

## Integration
- Target: importable from AMLGym via `OnlineAlgorithmAdapter`
- Currently: standalone with Unified Planning environment
- Never write custom PDDL parsers — use library APIs
