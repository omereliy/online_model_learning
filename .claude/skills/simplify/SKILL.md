---
name: simplify
description: Review current plan or code changes for unnecessary complexity and algorithm correctness. Flags over-engineering, premature abstractions, SAT encoding issues, and migration anti-patterns.
context: fork
agent: simplifier
argument-hint: [description of what to review]
---

Review the current work for unnecessary complexity and algorithm correctness.

$ARGUMENTS

If no specific target is given, review the most recent changes (check git diff or the current plan).

Key reference files for correctness review:
- docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md — Mathematical formulas
- docs/information_gain_algorithm/cnf/ — CNF analysis
