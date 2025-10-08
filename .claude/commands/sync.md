---
description: Synchronize all project documentation
---

# Documentation Sync

**Context**: Load @.claude/commands/_project.md

## Synchronization Tasks

1. **Update IMPLEMENTATION_TASKS.md**
   - Mark newly completed tasks
   - Add any discovered tasks
   - Update component status section
   - Refresh timestamps and test counts

2. **Verify Documentation Consistency**
   - Test counts match actual (run make test)
   - Line counts accurate where stated
   - File paths still valid
   - Cross-references between docs work

3. **Update Key Metrics**
   - Component completion status
   - Test suite results (X/Y passing)
   - Recent refactoring changes
   - Current phase in progress

4. **Enforce Documentation Rules** (from DEVELOPMENT_RULES.md)
   - No duplicate content between files
   - Each fact in exactly ONE location:
     - README.md: Installation and basic usage
     - CLAUDE.md: Navigation only (links to other docs)
     - DEVELOPMENT_RULES.md: Conventions and architecture
     - QUICK_REFERENCE.md: Code snippets and commands
     - IMPLEMENTATION_TASKS.md: Progress and status
   - Remove any duplicates found

## Output Format

Brief summary:
```
Updated:
- IMPLEMENTATION_TASKS.md: Added Phase 2 completion, updated test count (75→78)
- QUICK_REFERENCE.md: Added new StateConverter usage pattern
- Removed duplicate content: [description]

All documentation synced and consistent.
```

Keep output minimal - just what changed.