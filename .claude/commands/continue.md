---
description: Continue work from previous session
args:
  - name: context
    description: Brief reminder of what you were working on
    optional: true
---

# Continue Previous Work

**Context**: Load @.claude/commands/_project.md

**Previous work hint**: $ARGUMENT

## Resume Process

1. **Check Recent Activity**
   - Read IMPLEMENTATION_TASKS.md for last updates
   - Check git log for recent commits
   - Review modified but uncommitted files

2. **Identify Stopping Point**
   - Find incomplete tasks marked "in progress"
   - Check for failing tests that need fixing
   - Look for TODO comments added recently

3. **Resume Work**
   - Continue from identified stopping point
   - Complete any half-finished implementations
   - Run tests to verify current state

## Output Format

```
RESUMING: [Task description]
Last activity: [What was done]
Current state: [Tests passing/failing, files modified]

Continuing with: [Specific next action]
```

Then proceed directly with the work.
No need for verbose context repetition.

## Example Usage

```
/continue "refactoring state converter"
```

Or just:
```
/continue
```
(Will auto-detect from git and IMPLEMENTATION_TASKS.md)