---
name: simplify
description: Review current plan or code changes for unnecessary complexity and algorithm correctness. Flags over-engineering, premature abstractions, SAT encoding issues, and migration anti-patterns.
context: fork
agent: simplifier
argument-hint: [description of what to review]
paths: information_gain_aml/**
---

Review the current work for unnecessary complexity and algorithm correctness.

> **Layering with bundled `/code-review`:** As of Claude Code v2.1.147, bundled `/code-review` (formerly `/simplify`) handles general correctness at the chosen effort level (e.g. `/code-review high`). This skill adds project-specific concerns the bundled reviewer cannot know: SAT mutual-exclusion invariant, eff_pos/eff_neg disjointness, CNF cache invalidation, conformance to `docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md`, and the CORE/BRIDGE/REMOVABLE migration direction toward aml-gym. Run both — `/code-review high` first or in parallel.

$ARGUMENTS

If no specific target is given, review the most recent changes (check git diff or the current plan).

Key reference files for correctness review:
- docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md — Mathematical formulas
- docs/information_gain_algorithm/cnf/ — CNF analysis
