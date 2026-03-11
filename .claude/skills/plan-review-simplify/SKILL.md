---
name: plan-review-simplify
description: Create an execution plan with built-in review for correctness and simplification. Use for multi-file changes, new features, or refactoring.
disable-model-invocation: true
argument-hint: [task description]
---

## Planning Workflow with Review and Simplification

For the task described in $ARGUMENTS:

### Phase 1: Explore
1. Read all relevant existing code using Grep and Glob
2. Identify existing patterns that can be reused
3. Note existing utilities in `grounding.py` (binding, lifting, grounding) and `cnf_manager.py` (SAT ops)
4. **Run `test-runner` agent** to verify baseline test/type-check state before planning changes

### Phase 2: Plan
Design the implementation approach covering:
- **Objective**: One sentence describing the goal
- **Analysis**: Current state, what needs to change, existing code to reuse
- **Algorithm alignment**: Which sections of INFORMATION_GAIN_ALGORITHM.md are affected?
- **Module classification**: Are affected files CORE, BRIDGE, or REMOVABLE?
- **Files to modify**: Table of file | action | description
- **Execution steps**: Numbered checklist
- **Validation strategy**: `uv run pytest tests/ -v`, `uv run mypy src/`

### Phase 3: Review
Before presenting the plan, review it for simplification and correctness:

**Simplification:**
- Can any proposed new file be merged into an existing file?
- Can any proposed new class be a method on an existing class?
- Are there existing utilities that eliminate proposed helpers?
- Would a senior engineer say "this is more code than necessary"?

**Migration alignment:**
- Does the change add new UP dependencies to CORE modules? (avoid)
- Does the change grow BRIDGE modules significantly? (avoid)
- Does the change move toward or away from the 3-class target architecture?

**Correctness:**
- Does the change maintain mutual exclusion invariant in SAT encoding?
- Does the change maintain eff_pos / eff_neg disjointness?
- Does the change match INFORMATION_GAIN_ALGORITHM.md formulas?
- Are CNF caches invalidated when clauses change?

If concerns found: revise the plan. Note what changed and why.

### Phase 4: Present for Approval
Present plan to user, noting open decisions. Wait for approval.

Do NOT proceed until approved.

### Phase 5: Execute
Execute steps in order. After completion:
1. Run `uv run pytest tests/ -v`
2. Run `uv run mypy src/`
3. Summarize changes, key decisions, validation results

## Fast Mode
If user says "fast mode", "just do it", or "skip planning" — execute immediately without the planning workflow.

## Simple Tasks (No Planning Required)
Skip planning for:
- Single-file edits under 50 lines
- Answering questions or explaining code
- Running tests without modification
- Git operations
