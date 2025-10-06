---
description: Plan task implementation with TDD methodology
args:
  - name: task_reference
    description: Task description or reference from IMPLEMENTATION_TASKS.md (e.g., "Phase 2", "convergence detection", or direct description)
    required: true
---

# Task Planning Command

## Purpose
Plan implementation approach for a task using Test-Driven Development methodology. Creates detailed implementation outline WITHOUT writing code.

## Execution Steps

### Step 1: Task Identification
Parse and understand the task:
- [ ] Read @docs/IMPLEMENTATION_TASKS.md
- [ ] If task_reference matches a phase/task ID, extract full description
- [ ] If task_reference is a description, use it directly
- [ ] Identify task scope and boundaries
- [ ] Extract any existing requirements or constraints

**Task Reference**: $ARGUMENT

**Expected Outcome**: Clear understanding of task requirements

### Step 2: Context Gathering
Load relevant project context:
- [ ] Read @CLAUDE.md for project overview
- [ ] Read @docs/DEVELOPMENT_RULES.md for coding standards
- [ ] Read @docs/QUICK_REFERENCE.md for existing patterns
- [ ] Scan relevant source files (read-only)
- [ ] Review related test files

**Expected Outcome**: Comprehensive context for planning

### Step 3: Dependency Analysis
Identify dependencies and prerequisites:
- [ ] List required external libraries/tools
- [ ] Identify dependent modules/classes
- [ ] Check for prerequisite tasks in IMPLEMENTATION_TASKS.md
- [ ] Verify all dependencies are available
- [ ] Flag any blockers or missing prerequisites

**Expected Outcome**: Dependency map with blockers identified

### Step 4: Component Breakdown
Decompose task into testable components:
- [ ] Identify main functionality to implement
- [ ] Break down into smallest logical units
- [ ] Define interfaces and contracts
- [ ] Identify edge cases and failure modes
- [ ] Determine integration points

**Example Breakdown**:
```
Task: Implement convergence detection for Information Gain

Components:
1. Model Stability Checker
   - Input: Current model, previous model
   - Output: Boolean (models are identical)
   - Edge cases: First iteration, empty models

2. Information Gain Threshold Checker
   - Input: Latest information gain value, threshold ε
   - Output: Boolean (gain below threshold)
   - Edge cases: No actions available, all gains equal

3. Success Rate Monitor
   - Input: Recent action history, window size
   - Output: Float (success rate 0-1)
   - Edge cases: Empty history, window larger than history
```

**Expected Outcome**: Component list with clear responsibilities

### Step 5: Test Specification
Define test cases for each component (TDD):
- [ ] **Unit Tests**: Individual component behavior
- [ ] **Integration Tests**: Component interactions
- [ ] **Edge Cases**: Boundary conditions
- [ ] **Failure Tests**: Error handling

**Test Template**:
```
Component: [Name]
Test File: tests/[path]/test_[name].py

Test Cases:
1. test_[component]_[scenario]
   - Given: [initial state]
   - When: [action]
   - Then: [expected outcome]
   - Assertions: [specific checks]

2. test_[component]_[edge_case]
   - Given: [edge condition]
   - When: [action]
   - Then: [expected behavior]
```

**Expected Outcome**: Complete test specification

### Step 6: Acceptance Criteria Definition
Define success criteria:
- [ ] Functional requirements (what must work)
- [ ] Performance requirements (speed, memory)
- [ ] Quality requirements (test coverage, documentation)
- [ ] Integration requirements (compatibility with existing code)

**Acceptance Criteria Template**:
```
✓ All unit tests pass
✓ Integration tests pass
✓ Edge cases handled gracefully
✓ Test coverage ≥ 90%
✓ Docstrings for all public methods
✓ Type annotations complete
✓ No breaking changes to existing API
✓ Performance meets benchmarks (if applicable)
✓ Security review passed (if applicable)
```

**Expected Outcome**: Clear definition of "done"

### Step 7: Risk Assessment
Identify potential risks and mitigation:
- [ ] Technical risks (complexity, unknowns)
- [ ] Integration risks (breaking changes)
- [ ] Performance risks (bottlenecks)
- [ ] Security risks (vulnerabilities)
- [ ] Timeline risks (dependencies on others)

**Risk Template**:
```
Risk: [Description]
Likelihood: High/Medium/Low
Impact: High/Medium/Low
Mitigation: [Strategy]
```

**Expected Outcome**: Risk register with mitigations

### Step 8: Implementation Outline
Create step-by-step implementation plan:
- [ ] Order components by dependency
- [ ] Define implementation sequence
- [ ] Specify TDD workflow for each component
- [ ] Identify refactoring opportunities
- [ ] Estimate effort for each step

**Implementation Plan Template**:
```
Phase 1: [Name] (Estimated: X hours)
1. Write tests for [component A]
   - Test file: tests/[path]/test_[name].py
   - Test cases: [list]
2. Implement [component A]
   - Source file: src/[path]/[name].py
   - Key methods: [list]
3. Validate and refactor
   - Run: make test
   - Check: coverage, type hints, docstrings

Phase 2: [Name] (Estimated: Y hours)
[...]

Phase 3: Integration (Estimated: Z hours)
[...]
```

**Expected Outcome**: Actionable implementation roadmap

### Step 9: Documentation Requirements
Plan documentation updates:
- [ ] Code documentation (docstrings, type hints)
- [ ] API documentation (if adding public interfaces)
- [ ] Architecture documentation (if changing structure)
- [ ] IMPLEMENTATION_TASKS.md updates
- [ ] README/CLAUDE.md updates (if needed)

**Expected Outcome**: Documentation checklist

### Step 10: Plan Summary
Present complete plan for user approval:

```markdown
# Implementation Plan: [Task Name]

## Task Description
[Clear description of what will be implemented]

## Components to Implement
1. [Component 1] - [purpose]
2. [Component 2] - [purpose]
[...]

## Test Strategy
- Unit tests: [count] tests
- Integration tests: [count] tests
- Files to create:
  - tests/[path]/test_[name].py

## Implementation Files
- src/[path]/[file1].py - [purpose]
- src/[path]/[file2].py - [purpose]

## Dependencies
- ✓ Available: [list]
- ⚠️ Blockers: [list if any]

## Risks
- [Risk 1]: [mitigation]
- [Risk 2]: [mitigation]

## Acceptance Criteria
- [ ] [Criterion 1]
- [ ] [Criterion 2]
[...]

## Implementation Phases
1. [Phase 1]: [description] (~X hours)
2. [Phase 2]: [description] (~Y hours)
3. [Phase 3]: [description] (~Z hours)

**Total Estimate**: [X+Y+Z] hours

## Next Steps
1. Review and approve this plan
2. Run `/implement "[task reference]"` to execute
3. Monitor progress in IMPLEMENTATION_TASKS.md
```

**Expected Outcome**: Complete implementation plan ready for approval

## Validation Criteria
- [ ] All components clearly defined
- [ ] Test strategy comprehensive
- [ ] Dependencies verified
- [ ] Risks identified with mitigations
- [ ] Acceptance criteria measurable
- [ ] Implementation sequence logical
- [ ] Effort estimates reasonable
- [ ] No code written (planning only)

## Output Format
Present plan as structured markdown document with:
1. Task overview
2. Component breakdown
3. Test specifications
4. Implementation phases
5. Risk assessment
6. Acceptance criteria

## Integration Points
- **IMPLEMENTATION_TASKS.md**: Source of task definitions
- **/implement**: Executes this plan with TDD
- **/validate-theory**: Called if algorithm validation needed
- **/inspect-refactor**: Called if code quality analysis needed

## User Interaction
After presenting plan:
- **Ask**: "Shall I proceed with `/implement` to execute this plan?"
- **Wait**: For user approval or requested changes
- **Iterate**: Refine plan based on feedback if needed
