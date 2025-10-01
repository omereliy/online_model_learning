---
description: TDD task executor with comprehensive context loading
args: 
  - name: task_description
    description: The task to perform
    required: true
---

# Task Execution Protocol

## Phase 1: Context Validation
Load and verify project context:
1. Read @CLAUDE.md - project overview and conventions
2. Read @docs/DEVELOPMENT_RULES.md - coding standards
3. Read @docs/QUICK_REFERENCE.md - common patterns
4. Survey @src/ structure (directories only)

**If any required file missing**: Report and ask how to proceed.

## Phase 2: Task Analysis
**Task:** $ARGUMENT

Analyze task to determine:
- Primary objective and success criteria
- Required vs optional functionality
- Potential edge cases or failure modes


## Phase 3: Scope Definition & Confirmation
Present analysis:

**Files to modify:**
- `path/to/file.py` - Why: [reason]
- `path/to/test.py` - Why: [validation]

**Approach:**
[1-2 sentence implementation strategy]

**Proceed?** Reply 'yes' or request changes.

## Phase 4: TDD Implementation
Execute in this order:
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

## Phase 5: Validation & Report
**Pre-delivery checks:**
- [ ] All tests pass
- [ ] No syntax/import errors
- [ ] Security review complete

**Summary:**
- **Files modified:** [list]
- **Changes made:** [concise description]
- **Tests added/updated:** [count and coverage]
- **Issues resolved:** [if any, with solutions]

**Status:** Ready for review / Needs attention: [details]