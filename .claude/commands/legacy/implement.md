---
description: Execute task implementation with strict TDD methodology
args:
  - name: task_reference
    description: Task description or reference (must match /plan if plan exists)
    required: true
---

# Task Implementation Command

## Purpose
Execute task implementation following Test-Driven Development methodology. ALWAYS writes tests BEFORE implementation code.

## Pre-Execution Check

### Check for Existing Plan
- [ ] Search conversation history for `/plan "[task_reference]"`
- [ ] If plan exists: Load plan details
- [ ] If no plan exists: Ask user if they want to create one first

**Decision Point**:
- **Plan exists** → Proceed with implementation
- **No plan** → Offer to run `/plan "[task_reference]"` first

**Task Reference**: $ARGUMENT

## Execution Steps

### Step 1: Context Loading
Load project context and requirements:
- [ ] Read @CLAUDE.md for project overview
- [ ] Read @docs/DEVELOPMENT_RULES.md for coding standards
- [ ] Read @docs/QUICK_REFERENCE.md for existing patterns
- [ ] Read @docs/IMPLEMENTATION_TASKS.md for task status
- [ ] Load implementation plan (if available)

**Expected Outcome**: Full context for implementation

### Step 2: Task Status Check
Update IMPLEMENTATION_TASKS.md:
- [ ] Locate task in IMPLEMENTATION_TASKS.md
- [ ] Mark task status as "in_progress"
- [ ] Add timestamp and notes
- [ ] Commit status update

**Expected Outcome**: Task marked as in-progress

### Step 3: Environment Validation
Verify development environment:
- [ ] Check conda environment is active (`action-learning`)
- [ ] Verify all dependencies installed
- [ ] Run baseline tests: `make test`
- [ ] Confirm all tests pass before starting

**Expected Outcome**: Clean baseline confirmed

### Step 4: TDD Cycle - Tests FIRST

#### 4a. Write Test Cases
**MANDATORY: Tests written BEFORE any implementation code**

- [ ] Create test file: `tests/[path]/test_[name].py`
- [ ] Import necessary testing utilities
- [ ] Write test class/functions following project conventions
- [ ] Implement each test case from plan:
  - Arrange: Set up test data and conditions
  - Act: Execute the functionality (that doesn't exist yet)
  - Assert: Verify expected outcomes

**Test Template**:
```python
import pytest
from src.[module] import [Component]  # Will fail - not implemented yet

def test_[component]_[scenario]():
    """Test [description]."""
    # Arrange
    [setup test data]

    # Act
    result = [call functionality]

    # Assert
    assert [expected outcome]
    assert [additional checks]

def test_[component]_[edge_case]():
    """Test [edge case description]."""
    # Arrange
    [setup edge case]

    # Act & Assert
    with pytest.raises([ExpectedException]):
        [call that should fail]
```

**Expected Outcome**: Complete test suite written

#### 4b. Validate Test Correctness
**CRITICAL: Ensure tests are testing the right thing**

- [ ] Review test logic for correctness
- [ ] Check test assertions are meaningful
- [ ] Verify edge cases covered
- [ ] Ensure tests will fail for right reasons
- [ ] Ask user to confirm if expected behavior is unclear

**Validation Questions**:
- Are the test assertions correct?
- Do the tests cover all requirements?
- Are edge cases properly handled?
- Is the expected behavior clear?

**Expected Outcome**: Validated test suite

#### 4c. Run Tests (Should FAIL)
- [ ] Run pytest on new test file
- [ ] Confirm tests fail with ImportError or NotImplementedError
- [ ] Verify failures are expected (not test bugs)

**Expected Outcome**: Tests fail appropriately (RED phase)

### Step 5: TDD Cycle - Implementation

#### 5a. Minimal Implementation
Write MINIMAL code to pass tests:

- [ ] Create source file: `src/[path]/[name].py`
- [ ] Add module docstring
- [ ] Implement class/functions to pass tests
- [ ] Follow DEVELOPMENT_RULES.md conventions:
  - Type annotations for all parameters and returns
  - Docstrings for all public methods
  - Error handling for edge cases
  - Security checks if applicable

**Implementation Template**:
```python
"""Module description.

Implements [functionality] for [purpose].
"""

from typing import [types]

class [ComponentName]:
    """Class description.

    Attributes:
        [attribute]: [description]
    """

    def __init__(self, [params]):
        """Initialize [component].

        Args:
            [param]: [description]
        """
        [minimal implementation]

    def [method](self, [params]) -> [ReturnType]:
        """Method description.

        Args:
            [param]: [description]

        Returns:
            [description]

        Raises:
            [Exception]: [when]
        """
        [minimal implementation to pass tests]
```

**Expected Outcome**: Code written to pass tests

#### 5b. Run Tests (Should PASS)
- [ ] Run pytest on test file
- [ ] Confirm all tests pass
- [ ] Check test coverage: `pytest --cov=src.[module] tests/[path]/test_[name].py`
- [ ] Verify coverage ≥ 90%

**Expected Outcome**: All tests pass (GREEN phase)

#### 5c. Integration Testing
- [ ] Run full test suite: `make test`
- [ ] Ensure no regressions in other tests
- [ ] Fix any integration issues
- [ ] Re-run until all tests pass

**Expected Outcome**: No regressions, all tests pass

### Step 6: TDD Cycle - Refactor

#### 6a. Code Quality Improvements
Improve code while keeping tests green:

- [ ] Extract duplicate code to functions
- [ ] Improve variable/function naming
- [ ] Optimize performance if needed
- [ ] Add defensive programming (assertions, validation)
- [ ] Enhance error messages
- [ ] Run tests after each change

**Expected Outcome**: Cleaner code, all tests still pass (REFACTOR phase)

#### 6b. Documentation
Complete all documentation:

- [ ] Verify docstrings are comprehensive
- [ ] Add inline comments for complex logic
- [ ] Update type hints if changed
- [ ] Check all public APIs documented

**Expected Outcome**: Well-documented code

### Step 7: Final Validation

#### 7a. Test Suite Validation
- [ ] Run curated tests: `make test` (must pass 100%)
- [ ] Run full tests: `pytest tests/` (check for new failures)
- [ ] Run with verbose: `pytest -vv` for detailed output
- [ ] Verify coverage: `make coverage`

**Expected Outcome**: All required tests pass

#### 7b. Code Quality Checks
- [ ] No syntax errors
- [ ] No import errors
- [ ] Type hints complete
- [ ] Docstrings complete
- [ ] No security vulnerabilities (if applicable)
- [ ] Follows DEVELOPMENT_RULES.md conventions

**Expected Outcome**: Code quality verified

#### 7c. Pre-Commit Hook Test
- [ ] Stage changes: `git add [files]`
- [ ] Test pre-commit hook: `python .claude/agents/test_guardian.py`
- [ ] Verify hook passes with new changes
- [ ] Ensure `action-learning` conda env active

**Expected Outcome**: Pre-commit validation passes

### Step 8: Task Completion

#### 8a. Update IMPLEMENTATION_TASKS.md
- [ ] Mark task as "completed" in IMPLEMENTATION_TASKS.md
- [ ] Add completion date and summary
- [ ] Note files created/modified
- [ ] Document test results
- [ ] Add any follow-up tasks discovered

**Update Template**:
```markdown
### [Task Name] - COMPLETE ✅
**Context**: [Brief description]

**Implementation**: ([Date])
- **Files Created**:
  - `src/[path]/[file].py` - [purpose]
  - `tests/[path]/test_[file].py` - [tests]
- **Tests**: [X] tests passing
- **Coverage**: [Y]%
- **Key Features**:
  - [Feature 1]
  - [Feature 2]

**Follow-up Tasks** (if any):
- [Task 1]
- [Task 2]
```

**Expected Outcome**: Task marked complete with full details

#### 8b. Documentation Updates
Update related documentation:

- [ ] README.md (if public API changed)
- [ ] QUICK_REFERENCE.md (if new patterns added)
- [ ] DEVELOPMENT_RULES.md (if new rules/conventions)
- [ ] Architecture docs (if structure changed)

**Expected Outcome**: All docs synchronized

#### 8c. Git Commit
Create commit with implemented changes:

- [ ] Review changes: `git status` and `git diff`
- [ ] Stage all changes: `git add [files]`
- [ ] Create commit with descriptive message
- [ ] Verify pre-commit hook passes
- [ ] Push if appropriate

**Commit Message Template**:
```
[action] [component]: [brief description]

- Implemented [feature/functionality]
- Added [X] tests with [Y]% coverage
- Updated [documentation]
```

**Expected Outcome**: Changes committed to git

### Step 9: Implementation Report

#### 9a. Generate Summary
Present comprehensive implementation report:

```markdown
# Implementation Report: [Task Name]

## ✅ Status: COMPLETE

## Implementation Summary
[Brief description of what was implemented]

## Files Created/Modified
- `src/[path]/[file].py` - [purpose, X lines]
- `tests/[path]/test_[file].py` - [Y tests, Z lines]
- `docs/IMPLEMENTATION_TASKS.md` - Updated task status

## Test Results
- **Curated Tests** (`make test`): ✓ [X]/[X] passing (100%)
- **Full Tests** (`pytest tests/`): ✓ [Y] passing, ⚠️ [Z] failures (if any)
- **New Tests**: [count] tests added
- **Coverage**: [percentage]%

## Key Features Implemented
1. [Feature 1] - [description]
2. [Feature 2] - [description]
3. [Feature 3] - [description]

## TDD Compliance
- ✓ Tests written BEFORE implementation
- ✓ All tests passing
- ✓ Code coverage ≥ 90%
- ✓ No regressions
- ✓ Pre-commit hook passes

## Follow-up Tasks
[If discovered during implementation]
- [ ] [Task 1]
- [ ] [Task 2]

## Next Steps
[Recommendations for what to do next]
```

**Expected Outcome**: Complete implementation report

#### 9b. Validation Checklist
Final confirmation:

- [ ] Task fully implemented per requirements
- [ ] All tests pass (make test = 100%)
- [ ] Code quality meets standards
- [ ] Documentation complete
- [ ] IMPLEMENTATION_TASKS.md updated
- [ ] Git commit created
- [ ] No blockers or issues remaining

**Expected Outcome**: Task confirmed complete

## Error Handling Protocol

### If Tests Don't Pass Expected Way
1. **STOP** - Do not proceed
2. **ANALYZE** - Review test logic
3. **VALIDATE** - Confirm expected behavior with user
4. **FIX** - Correct tests or assumptions
5. **RESTART** - Re-run TDD cycle

### If Implementation Fails
1. **STOP** - Do not continue
2. **DEBUG** - Use pytest -vv for details
3. **IDENTIFY** - Root cause analysis
4. **FIX** - Address root cause
5. **RETEST** - Verify fix

### If Integration Breaks
1. **ROLLBACK** - Revert to last working state
2. **ANALYZE** - Identify breaking change
3. **REFACTOR** - Fix integration issue
4. **VALIDATE** - Ensure all tests pass

## Validation Criteria
- [ ] Tests written BEFORE implementation (mandatory)
- [ ] All curated tests pass (make test = 100%)
- [ ] Code follows DEVELOPMENT_RULES.md
- [ ] Documentation complete
- [ ] IMPLEMENTATION_TASKS.md updated
- [ ] Git commit created
- [ ] No security issues

## Output Format
Progress updates at each phase:
1. Test creation status
2. Test validation results
3. Implementation progress
4. Test pass/fail status
5. Refactoring improvements
6. Final validation results
7. Complete implementation report

## Integration Points
- **/plan**: Uses plan as implementation guide
- **IMPLEMENTATION_TASKS.md**: Updates task status
- **DEVELOPMENT_RULES.md**: Enforces coding standards
- **make test**: Validates implementation
- **Git pre-commit hook**: Final validation

## Critical Rules
1. **NEVER write implementation code before tests**
2. **ALWAYS validate test correctness before implementing**
3. **ALWAYS run make test before marking complete**
4. **ALWAYS update IMPLEMENTATION_TASKS.md**
5. **ALWAYS ask if expected behavior is unclear**
