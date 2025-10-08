---
type: context
description: Shared project context for all commands
---

# Project Context (Auto-loaded by all commands)

## Core Documentation
Always load these files for context:
- @CLAUDE.md - Navigation index and command overview
- @docs/DEVELOPMENT_RULES.md - Conventions, architecture, rules
- @docs/IMPLEMENTATION_TASKS.md - Current progress and status
- @docs/QUICK_REFERENCE.md - Code patterns and snippets

## Project Details
- **Goal**: Compare online action model learning algorithms
- **Tech Stack**: Python, Unified Planning Framework, PySAT, pytest
- **Methodology**: Test-Driven Development (tests written FIRST)
- **Location**: /home/omer/projects/online_model_learning/

## Key Paths
- Source: src/ (algorithms/, core/, environments/, experiments/)
- Tests: tests/ (mirrors src/ structure)
- Benchmarks: benchmarks/ (PDDL domains)
- Results: results/ (experiment outputs)

## External Dependencies
- OLAM: /home/omer/projects/OLAM/
- ModelLearner: /home/omer/projects/ModelLearner/ (currently unavailable)
- Fast Downward: /home/omer/projects/fast-downward/