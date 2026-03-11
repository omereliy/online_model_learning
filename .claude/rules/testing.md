---
description: Test file mapping and testing rules
paths:
  - "tests/**/*.py"
---

## Testing Rules

- Run `uv run pytest tests/ -v` to verify all changes
- All tests must pass before committing

## Test directories:

| Directory | Coverage |
|-----------|----------|
| `tests/algorithms/` | Algorithm-specific tests (information gain, parallel gain) |
| `tests/core/` | Core module tests (cnf_manager, grounding, expression converter, etc.) |
| `tests/environments/` | Environment tests |
| `tests/experiments/` | Runner and metrics tests |
| `tests/information_gain/` | Information gain specific scenarios |
| `tests/integration/` | End-to-end learning loops, multi-domain |
| `tests/validation/` | Algorithm comparison fairness |
| `tests/utils/` | Test utilities |
