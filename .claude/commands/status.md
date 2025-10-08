---
description: Quick project status check
---

# Project Status

**Context**: Load @.claude/commands/_project.md

## Status Checks

1. **Current Progress**
   - Check IMPLEMENTATION_TASKS.md for current phase
   - List recently completed components
   - Identify next priority tasks

2. **Test Suite**
   - Run make test for curated suite
   - Report pass/fail count
   - Note any failing tests

3. **Git Status**
   - Show modified files
   - Recent commits (last 3)
   - Current branch

4. **Next Actions**
   - Priority tasks from IMPLEMENTATION_TASKS.md
   - Any blocking issues
   - Suggested next command

## Output Format

```
PROJECT STATUS
==============
Current Phase: Phase 2 - Algorithm Validation
Completed: Statistical analysis, Model validator

Tests: 75/75 passing ✓

Git: master branch
- Modified: src/core/state_converter.py (new)
- Last commit: "Add state converter extraction"

Next Priority:
1. Implement convergence detection for OLAM
2. Create Information Gain validation report

Suggested: /continue "convergence detection"
```

Keep it brief and actionable.