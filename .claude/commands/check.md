---
description: Validate implementation correctness
args:
  - name: component
    description: Algorithm, class, or component to validate
    required: true
---

# Correctness Validation

**Context**: Load @.claude/commands/_project.md

**Validate**: $ARGUMENT

## Validation Checks

1. **Core Logic**
   - Algorithm implements theoretical description correctly
   - Data structures used appropriately
   - No logical errors in flow

2. **Edge Cases**
   - Empty inputs handled
   - Boundary conditions checked
   - Error states managed

3. **Test Coverage**
   - Unit tests exist and pass
   - Edge cases covered by tests
   - Integration with rest of system tested

## Output Format

Concise validation report:
```
✓ Core logic correct
✓ Edge cases handled
✗ Missing test for convergence threshold
  → Add test_convergence_with_epsilon()

Overall: NEEDS WORK - 1 issue found
```

Focus on correctness, not style or optimization.
Report only actual problems, not theoretical concerns.