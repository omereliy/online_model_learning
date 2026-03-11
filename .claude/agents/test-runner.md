---
name: test-runner
description: Run tests and type checking for online_model_learning. Use after code changes.
tools: Bash, Read, Grep
model: haiku
maxTurns: 6
---

Run the test suite and type checker:

1. Run `uv run pytest tests/ -v`
2. Run `uv run mypy src/`

Report results as:
- Total tests: pass/fail count
- Any errors with file:line references
- mypy status: pass/fail with error count and locations
