---
description: Execute a pre-defined plan with TDD methodology
args:
  - name: plan
    description: The implementation plan to execute
    required: true
---

# Plan Execution Protocol

## Phase 0: Plan Validation
Review the provided plan:
- Verify all steps are clear and actionable
- Identify dependencies between steps
- Flag any ambiguous or missing details

**If plan is unclear**: Request clarification before proceeding.

## Phase 1: Context Loading
Load and verify project context:
1. Read @CLAUDE.md - project overview and conventions
2. Read @docs/DEVELOPMENT_RULES.md - coding standards
3. Read @docs/QUICK_REFERENCE.md - common patterns
4. Survey @src/ structure (directories only)

**If any required file missing**: Report and ask how to proceed.

## Phase 2: Plan Analysis
**Plan to Execute:** $ARGUMENT

Analyze plan to determine:
- Implementation order and dependencies
- Files that will be modified or created
- Tests required for validation
- Potential risks or edge cases

## Phase 3: Execution Strategy
Present execution approach:

**Files to modify:**
- `path/to/file.py` - Why: [reason]
- `path/to/test.py` - Why: [validation]

**Execution order:**
1. [Step 1 with brief description]
2. [Step 2 with brief description]
3. [Step 3 with brief description]

**Estimated scope:** [small/medium/large]

**Proceed?** Reply 'yes' or request changes.

## Phase 4: TDD Implementation
Execute plan following TDD methodology:

For each step in the plan:
1. **Tests First**: Write/update test cases with expected inputs/outputs
2. **Run Tests**: Verify they fail appropriately
3. **Implement**: Write minimal code to pass tests
4. **Validate**: Confirm tests pass
5. **Refactor**: Improve code quality while keeping tests green

Follow DEVELOPMENT_RULES.md for:
- Error handling patterns
- Type annotations
- Naming conventions
- Security checks

## Phase 5: Progress Tracking
After each major step:
- [ ] Report completion status
- [ ] Note any deviations from plan
- [ ] Identify blockers or issues
- [ ] Confirm readiness for next step

## Phase 6: Final Validation & Report
**Pre-delivery checks:**
- [ ] All plan steps completed
- [ ] All tests pass
- [ ] No syntax/import errors
- [ ] Security review complete

**Summary:**
- **Plan steps completed:** [X of Y]
- **Files modified:** [list]
- **Changes made:** [concise description]
- **Tests added/updated:** [count and coverage]
- **Deviations from plan:** [if any, with justification]
- **Issues encountered:** [if any, with solutions]

**Status:** Ready for review / Needs attention: [details]
