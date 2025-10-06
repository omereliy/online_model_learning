# TDD Workflow Guide - Slash Commands

## Overview

This guide demonstrates how to use slash commands for Test-Driven Development workflows. The commands work together to enforce TDD methodology, maintain documentation, and ensure code quality.

## Available Commands

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `/docs-sync` | Synchronize all documentation | After major changes, before releases |
| `/plan` | Plan task implementation | Before starting any new feature/fix |
| `/implement` | Execute implementation with TDD | After planning, to write code |
| `/validate-theory` | Validate algorithm correctness | Before implementing complex algorithms |
| `/inspect-refactor` | Analyze code quality | Regular code reviews, before refactoring |
| `/engineer-prompt` | Engineer optimized prompt from free text | Before starting work in new/other sessions |

## Command Configuration Guide

### Recommended Settings Per Command

| Command | Model | Mode | Extended Thinking | Notes |
|---------|-------|------|-------------------|-------|
| `/docs-sync` | **Sonnet 4.5** | Plan | ❌ No | Mechanical sync - fast execution preferred |
| `/plan` | **Opus 4** | Regular | ⚠️ Complex only | Use extended thinking for large/ambiguous tasks |
| `/implement` | **Sonnet 4.5** | Regular | ❌ No | TDD needs fast, confident execution |
| `/validate-theory` | **Opus 4** | Regular | ✅ Yes | Deep reasoning for correctness proofs |
| `/inspect-refactor` | **Opus 4** | Regular | ✅ Yes | Architectural insights need deep analysis |
| `/engineer-prompt` | **Opus 4** | Regular | ✅ Yes | Meta-reasoning about prompt quality |

### Model Selection Rationale

**Sonnet 4.5** (Fast, structured execution):
- `/docs-sync`: Pattern matching, link validation, consistency checks
- `/implement`: Clear TDD workflow with defined steps

**Opus 4** (Deep reasoning):
- `/plan`: Architectural thinking, dependency analysis, test strategy
- `/validate-theory`: Theoretical correctness, complexity analysis, edge cases
- `/inspect-refactor`: Code smells, security issues, design patterns
- `/engineer-prompt`: Meta-cognitive prompt analysis and optimization

### Extended Thinking Guidelines

**✅ Always Use Extended Thinking**:
- `/validate-theory` - Complex correctness proofs, subtle edge cases
- `/inspect-refactor` - Non-obvious architectural issues, security analysis
- `/engineer-prompt` - Understanding implicit intent, resolving ambiguities

**⚠️ Use Conditionally**:
- `/plan` - Only for complex/ambiguous tasks with many dependencies
  - Simple tasks from IMPLEMENTATION_TASKS.md: No extended thinking
  - Novel features with unclear requirements: Yes extended thinking

**❌ Never Use Extended Thinking**:
- `/implement` - May create false uncertainty in TDD process, needs confidence
- `/docs-sync` - Mechanical task with no ambiguity, overthinking hurts

### Mode Selection

**Plan Mode** (shows plan before execution):
- `/docs-sync` - Preview documentation changes before applying
- `/implement` - Preview implementation approach before writing code

**Regular Mode** (analysis/generation only):
- `/plan` - Creates plan output, makes no changes
- `/validate-theory` - Analysis only, no modifications
- `/inspect-refactor` - Identifies issues, doesn't fix them
- `/engineer-prompt` - Generates prompt, makes no changes

### Key Principle

**Match reasoning depth to task complexity:**
- Mechanical tasks → Fast execution (Sonnet, no extended thinking)
- Structured workflows → Confident execution (Sonnet, no extended thinking)
- Deep analysis → Maximum reasoning (Opus, extended thinking)
- Planning → Context-dependent (Opus, extended thinking if complex)

## Core Workflows

### Workflow 1: New Feature Implementation

**Scenario**: Implement convergence detection for Information Gain algorithm

```bash
# Step 1: Plan the implementation
/plan "Implement convergence detection for Information Gain"

# Claude analyzes IMPLEMENTATION_TASKS.md, breaks down into components,
# defines test strategy, and presents plan

# Step 2: Review plan, then execute with TDD
/implement "Implement convergence detection for Information Gain"

# Claude writes tests FIRST, validates them, implements code to pass tests,
# runs validation, updates IMPLEMENTATION_TASKS.md

# Step 3: Update documentation
/docs-sync

# Claude ensures all docs are consistent with new feature
```

**Expected Flow**:
1. `/plan` reads IMPLEMENTATION_TASKS.md, creates detailed plan
2. User reviews and approves plan
3. `/implement` executes with strict TDD: tests → validate → implement → verify
4. IMPLEMENTATION_TASKS.md auto-updated
5. `/docs-sync` ensures documentation consistency

---

### Workflow 2: Algorithm Implementation with Validation

**Scenario**: Implement new CNF-based action selection algorithm from paper

```bash
# Step 1: Validate theoretical correctness
/validate-theory "docs/papers/new_algorithm.md"

# Claude analyzes algorithm, checks correctness, complexity,
# edge cases, and generates test suggestions

# Step 2: If validation passes, create implementation plan
/plan "Implement CNF action selection from validated algorithm"

# Claude uses validation results (edge cases, test suggestions)
# to create robust implementation plan

# Step 3: Execute implementation
/implement "Implement CNF action selection from validated algorithm"

# Claude implements with validation insights,
# includes edge case tests from step 1

# Step 4: Sync documentation
/docs-sync
```

**Expected Flow**:
1. `/validate-theory` ensures algorithm is theoretically sound
2. Identifies edge cases and complexity issues BEFORE coding
3. `/plan` incorporates validation insights into plan
4. `/implement` uses edge cases for comprehensive testing
5. Reduced implementation bugs and rework

---

### Workflow 3: Code Quality Improvement

**Scenario**: Refactor messy CNF manager code

```bash
# Step 1: Analyze code quality
/inspect-refactor "src/core/cnf_manager.py"

# Claude analyzes: code smells, duplication, security,
# performance, and generates prioritized refactoring plan

# Step 2: For critical issues, create fix plan
/plan "Fix security issue: SQL injection in CNF query builder"

# Claude creates plan to address specific issue with TDD

# Step 3: Implement fix with tests
/implement "Fix security issue: SQL injection in CNF query builder"

# Step 4: For architecture improvements, validate approach
/validate-theory "Strategy pattern for CNF builder operations"

# Step 5: Plan architecture refactoring
/plan "Refactor CNF builder to use strategy pattern"

# Step 6: Execute refactoring with comprehensive tests
/implement "Refactor CNF builder to use strategy pattern"

# Step 7: Verify no regressions and sync docs
/docs-sync
```

**Expected Flow**:
1. `/inspect-refactor` identifies issues with priority
2. Critical issues fixed immediately with `/plan` + `/implement`
3. Architecture improvements validated with `/validate-theory`
4. Refactoring planned and executed with TDD
5. All changes validated and documented

---

### Workflow 4: Task from IMPLEMENTATION_TASKS.md

**Scenario**: Work on "Phase 2: Algorithm Validation" from task list

```bash
# Step 1: Plan from task reference
/plan "Phase 2"

# Claude reads IMPLEMENTATION_TASKS.md, finds Phase 2,
# extracts requirements, creates detailed plan

# Step 2: Execute implementation
/implement "Phase 2"

# Claude implements with TDD, updates task status in
# IMPLEMENTATION_TASKS.md automatically

# Step 3: Update docs
/docs-sync
```

**Expected Flow**:
1. `/plan` parses task from IMPLEMENTATION_TASKS.md by phase/name
2. Extracts requirements and constraints from task description
3. `/implement` marks task as in-progress, then completed
4. IMPLEMENTATION_TASKS.md stays synchronized

---

### Workflow 5: Bug Fix with Root Cause Analysis

**Scenario**: Fix failing test in information gain algorithm

```bash
# Step 1: Analyze code for issues
/inspect-refactor "src/algorithms/information_gain.py"

# Identifies: missing error handling in probability calculation

# Step 2: Plan the fix
/plan "Add error handling for zero probability in information gain"

# Step 3: Implement with tests
/implement "Add error handling for zero probability in information gain"

# Claude writes test for edge case FIRST,
# then implements fix to handle it
```

**Expected Flow**:
1. `/inspect-refactor` identifies root cause (missing error handling)
2. `/plan` creates focused fix plan with edge case coverage
3. `/implement` writes edge case test first, then fix
4. Bug fixed with test to prevent regression

---

### Workflow 6: Regular Maintenance

**Scenario**: Monthly code quality review

```bash
# Step 1: Check documentation consistency
/docs-sync

# Step 2: Inspect core modules for issues
/inspect-refactor "src/core/*.py"

# Step 3: Review and prioritize findings
# (Claude presents prioritized refactoring plan)

# Step 4: Plan top priority refactoring
/plan "Refactor PDDLHandler to reduce complexity"

# Step 5: Execute refactoring
/implement "Refactor PDDLHandler to reduce complexity"

# Step 6: Final documentation sync
/docs-sync
```

**Expected Flow**:
1. Regular quality checks with `/docs-sync` and `/inspect-refactor`
2. Systematic refactoring of high-priority issues
3. Documentation kept current
4. Code quality maintained over time

---

### Workflow 7: Cross-Session Prompt Engineering

**Scenario**: Need to start complex work in a new session or hand off to another developer

```bash
# Step 1: Engineer the prompt from rough idea
/engineer-prompt "optimize the SAT solver performance"

# Claude asks: New or continued session?
# User responds: 1 (new session)

# Claude gathers context:
# - Role needed
# - Background/constraints
# - Output format
# - Success criteria

# Claude produces engineered prompt ready to copy-paste

# Step 2: Copy prompt to new/other session
# (User copies the ---BEGIN PROMPT--- section)

# Step 3: In new session, paste engineered prompt
# Claude has full context and can work effectively
```

**Expected Flow**:
1. `/engineer-prompt` detects if target is new/continued session
2. Gathers appropriate context based on session type
3. Transforms vague idea → specific, actionable prompt
4. Outputs copy-paste ready prompt with all context
5. User uses in new session with full effectiveness

**Session Type Benefits**:
- **New session**: Includes role, background, constraints, success criteria
- **Continued session**: References prior work, maintains continuity, avoids redundancy

---

## Command Composition Patterns

### Pattern 1: Validate → Plan → Implement
**Use**: Complex algorithms from papers

```
/validate-theory → /plan → /implement → /docs-sync
```

**Benefits**:
- Theoretical validation before coding
- Edge cases identified early
- Comprehensive test coverage
- Reduced implementation bugs

---

### Pattern 2: Inspect → Plan → Implement
**Use**: Refactoring existing code

```
/inspect-refactor → /plan → /implement → /docs-sync
```

**Benefits**:
- Data-driven refactoring decisions
- Prioritized improvements
- Quality metrics tracked
- No regressions

---

### Pattern 3: Plan → Implement (Rapid)
**Use**: Simple, well-understood tasks

```
/plan → /implement → /docs-sync
```

**Benefits**:
- Fast iteration for simple tasks
- Still maintains TDD discipline
- Documentation synchronized

---

### Pattern 4: Inspect → Validate → Plan → Implement (Full)
**Use**: Major architectural changes

```
/inspect-refactor → /validate-theory → /plan → /implement → /docs-sync
```

**Benefits**:
- Complete analysis before changes
- Theoretical and practical validation
- Systematic execution
- Full documentation

---

### Pattern 5: Engineer → Execute (Cross-Session)
**Use**: Starting work in new session or handing off to others

```
/engineer-prompt → [copy to new session] → /plan → /implement
```

**Benefits**:
- Proper context transfer to new sessions
- Clear, actionable prompts for others
- No context loss across sessions
- Optimized for session type (new vs continued)

---

## Best Practices

### 1. Always Plan Before Implementing
**Don't**: Jump straight to `/implement`
**Do**: Run `/plan` first to think through approach

```bash
# Bad
/implement "Add caching"

# Good
/plan "Add caching to expensive CNF operations"
/implement "Add caching to expensive CNF operations"
```

### 2. Validate Complex Algorithms
**Don't**: Implement algorithms without validation
**Do**: Use `/validate-theory` for complexity analysis

```bash
# Bad
/plan "Implement new SAT solver algorithm"

# Good
/validate-theory "docs/algorithms/new_sat_solver.md"
/plan "Implement new SAT solver algorithm"  # Uses validation insights
/implement "Implement new SAT solver algorithm"
```

### 3. Regular Code Quality Checks
**Don't**: Wait until code is unmaintainable
**Do**: Regular `/inspect-refactor` during development

```bash
# Weekly code review
/inspect-refactor "src/algorithms/*.py"
# Address high-priority issues immediately
```

### 4. Keep Documentation Synchronized
**Don't**: Let docs get stale
**Do**: Run `/docs-sync` after major changes

```bash
# After completing feature
/implement "Feature X"
/docs-sync  # Immediately sync docs
```

### 5. Reference Tasks by Name or Phase
**Don't**: Re-describe tasks from IMPLEMENTATION_TASKS.md
**Do**: Reference by phase/name for consistency

```bash
# Bad (duplicates task description)
/plan "Implement convergence detection with model stability check and info gain threshold"

# Good (references task)
/plan "Phase 2: Convergence Detection"
```

### 6. Engineer Prompts for New Sessions
**Don't**: Start new sessions with vague prompts
**Do**: Use `/engineer-prompt` to create optimized prompts

```bash
# Bad (in new session - no context)
"fix the performance issue"

# Good (engineer first)
/engineer-prompt "fix the performance issue"
# Claude gathers context, creates optimized prompt
# Copy to new session with full context
```

---

## Integration with Git Workflow

### Pre-Commit
```bash
# 1. Implement feature with TDD
/plan "Feature X"
/implement "Feature X"

# 2. Inspect code quality
/inspect-refactor "src/path/to/new_code.py"

# 3. Fix any issues found
/plan "Address issues from inspection"
/implement "Address issues from inspection"

# 4. Sync docs
/docs-sync

# 5. Commit (pre-commit hook runs make test automatically)
git add .
git commit -m "Add Feature X with tests"
```

### Pre-Release
```bash
# 1. Full documentation sync
/docs-sync

# 2. Quality check all code
/inspect-refactor "src/**/*.py"

# 3. Address critical issues
# (Use /plan and /implement for each)

# 4. Final documentation update
/docs-sync

# 5. Create release
git tag v1.0.0
```

---

## Troubleshooting

### Issue: Plan doesn't find task in IMPLEMENTATION_TASKS.md

**Solution**: Check task reference format
```bash
# Try different formats
/plan "Phase 2"  # By phase number
/plan "Convergence Detection"  # By task name
/plan "Implement convergence detection for Information Gain"  # Full description
```

### Issue: Implementation breaks existing tests

**Cause**: Integration issues not caught in planning

**Solution**:
```bash
# 1. Check what broke
make test -v

# 2. Inspect affected code
/inspect-refactor "src/path/to/affected_code.py"

# 3. Plan fix with integration tests
/plan "Fix integration issue in [component]"

# 4. Implement with comprehensive integration tests
/implement "Fix integration issue in [component]"
```

### Issue: Algorithm validation fails

**Cause**: Theoretical issues in algorithm

**Solution**: Address validation issues before implementation
```bash
/validate-theory "algorithm description"
# Review issues identified
# Modify algorithm or add constraints
# Re-validate before proceeding
```

### Issue: Documentation out of sync

**Cause**: Changes made without running /docs-sync

**Solution**: Regular synchronization
```bash
# After any implementation
/implement "Feature"
/docs-sync  # Always sync after changes
```

---

## Examples from This Project

### Example 1: PDDLHandler Refactoring (Real)

```bash
# Step 1: Identify refactoring needs
/inspect-refactor "src/core/pddl_handler.py"
# Found: God class (500+ LOC), missing type safety, mixed responsibilities

# Step 2: Plan Phase 1 - Type safety
/plan "Phase 1: Create type-safe data classes for PDDL operations"

# Step 3: Implement with tests
/implement "Phase 1: Create type-safe data classes for PDDL operations"
# Created: ParameterBinding, ParameterBoundLiteral, GroundedFluent classes
# Tests: 21 new tests in test_pddl_types.py

# Step 4: Update docs
/docs-sync

# Step 5: Plan Phase 2 - Extract converters
/plan "Phase 2: Extract expression conversion logic to separate class"

# Step 6: Implement Phase 2
/implement "Phase 2: Extract expression conversion logic to separate class"
# Created: ExpressionConverter class
# Tests: 9 new tests

# Repeated for Phases 3-6 until complete refactoring
```

### Example 2: Statistical Analysis Implementation (Real)

```bash
# Step 1: Plan from task list
/plan "Gap #2: Statistical Significance Testing"

# Step 2: Implement with TDD
/implement "Gap #2: Statistical Significance Testing"
# Wrote 9 tests FIRST
# Implemented StatisticalAnalyzer class
# All tests passing

# Step 3: Sync docs
/docs-sync
# Updated IMPLEMENTATION_TASKS.md with completion status
```

### Example 3: Information Gain Validation (Future)

```bash
# Step 1: Inspect current implementation
/inspect-refactor "src/algorithms/information_gain.py"

# Step 2: Validate theoretical correctness
/validate-theory "docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md"

# Step 3: Plan validation script
/plan "Gap #4: Information Gain Validation Report"

# Step 4: Implement validation
/implement "Gap #4: Information Gain Validation Report"

# Step 5: Update docs
/docs-sync
```

---

## Summary

**Key Principles**:
1. **Always plan first** - Think before coding
2. **Validate complexity** - Use `/validate-theory` for algorithms
3. **Maintain quality** - Regular `/inspect-refactor` checks
4. **Keep docs current** - `/docs-sync` after changes
5. **Follow TDD strictly** - Tests before implementation

**Command Flow**:
```
[Validate Theory] → [Plan] → [Implement] → [Docs Sync]
        ↑                                        ↓
    [Inspect Refactor] ← ← ← ← ← ← ← ← ← ← ← ← ←
```

**Remember**:
- Tests ALWAYS written before implementation
- IMPLEMENTATION_TASKS.md automatically updated
- Pre-commit hook enforces test success
- Documentation stays synchronized
