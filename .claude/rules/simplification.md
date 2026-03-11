---
description: Global simplification principle applied to all code changes
paths:
  - "**"
---

## Simplification Principle

Before writing or modifying code, verify:

1. **Existing code check**: Search the codebase for existing utilities, patterns, or functions. Especially check `grounding.py` for binding utilities and `cnf_manager.py` for SAT operations. Do not duplicate.
2. **Minimal change**: Implement the smallest change that solves the problem. No "just in case" code.
3. **Proportional complexity**: The solution complexity should match the problem complexity.
4. **One consumer rule**: Do not create abstractions (base classes, utility functions) with only one consumer. Inline until 2+ consumers exist.
5. **File count check**: If your change creates new source files in `src/`, reconsider. The goal is to reduce file count, not increase it.
6. **Algorithm fidelity**: Changes to SAT encoding, information gain computation, or effect tracking must reference and comply with `docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md`.
7. **No growth in BRIDGE modules**: Do not add significant new functionality to modules classified as BRIDGE (will be replaced by PDDLGym/AMLGym). Fix bugs only.
