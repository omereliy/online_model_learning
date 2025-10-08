---
description: Complete TDD workflow for a task
args:
  - name: task
    description: Task from IMPLEMENTATION_TASKS.md or direct description
    required: true
---

# TDD Implementation

**Context**: Load @.claude/commands/_project.md

**Task**: $ARGUMENT

## Execution

1. **Understand the task**
   - Check IMPLEMENTATION_TASKS.md if task references a phase/component
   - Identify required functionality and acceptance criteria

2. **Plan test-first approach**
   - Break down into testable components
   - Define test cases covering normal and edge cases

3. **Write tests FIRST**
   - Create test file if needed
   - Write failing tests that define expected behavior
   - Run tests to confirm they fail correctly

4. **Implement minimal code**
   - Write only enough code to pass tests
   - No premature optimization or extra features
   - Follow project conventions from DEVELOPMENT_RULES.md

5. **Validate**
   - Run make test (curated suite must pass)
   - Run specific tests for new functionality
   - Ensure no regressions

6. **Update documentation**
   - Update IMPLEMENTATION_TASKS.md with completion status
   - Add any discovered tasks or notes

## Output
Brief confirmation of completion with:
- Tests written and passing
- Files modified
- Any important notes or next steps

Keep output concise - no verbose step-by-step narration.