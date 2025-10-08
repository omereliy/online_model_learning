---
description: Quick refactoring analysis
args:
  - name: files
    description: Files to analyze (supports wildcards, e.g., "src/core/*.py")
    required: true
---

# Refactoring Analysis

**Context**: Load @.claude/commands/_project.md

**Analyze**: $ARGUMENT

## Focus Areas

1. **Code Duplication**
   - Identical blocks >6 lines
   - Similar logic patterns
   - Repeated string literals

2. **Complexity Issues**
   - Methods >50 lines
   - Cyclomatic complexity >10
   - Deep nesting (>4 levels)
   - Classes >500 lines

3. **Naming Problems**
   - Unclear variable names (x, tmp, data)
   - Inconsistent naming conventions
   - Misleading names

4. **Error Handling**
   - Missing try/except where needed
   - Bare except clauses
   - Silent failures (empty except blocks)
   - Missing input validation

## Output Format

List only actionable issues with:
- Location (file:line)
- Issue type and severity
- Specific recommendation

Example:
```
src/core/pddl_handler.py:450 - Long method (87 lines)
→ Extract state conversion logic to StateConverter class

src/algorithms/olam_adapter.py:573,609 - Duplicate code
→ Both methods share state conversion logic, consolidate
```

Skip minor issues. Focus on high-impact improvements.
No verbose templates or theoretical discussions.