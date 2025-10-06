---
description: Identify refactoring opportunities and code quality issues
args:
  - name: target_files
    description: File(s) to inspect (supports wildcards, e.g., "src/core/*.py" or specific file)
    required: true
---

# Refactoring Inspection Command

## Purpose
Systematically analyze code quality, identify anti-patterns, and generate prioritized refactoring suggestions.

## Execution Steps

### Step 1: Target Resolution
Resolve files to inspect:
- [ ] Parse target argument: $ARGUMENT
- [ ] If wildcard pattern: Expand to file list
- [ ] If directory: Include all Python files
- [ ] If single file: Use directly
- [ ] Validate all files exist

**Target Files**: [List resolved files]

**Expected Outcome**: List of files to inspect

### Step 2: Code Quality Metrics Collection

#### 2a. File-Level Metrics
For each file, collect:
- [ ] **Lines of Code (LOC)**: Total, code, comments, blank
- [ ] **Cyclomatic Complexity**: Per function and file average
- [ ] **Maintainability Index**: 0-100 scale
- [ ] **Dependencies**: Import count and depth
- [ ] **Class Count**: Number of classes defined
- [ ] **Function Count**: Number of functions defined

#### 2b. Function/Method Metrics
For each function/method:
- [ ] **Length**: Line count
- [ ] **Complexity**: Cyclomatic complexity
- [ ] **Parameter Count**: Number of parameters
- [ ] **Return Points**: Number of return statements
- [ ] **Nesting Depth**: Maximum nesting level
- [ ] **Documentation**: Docstring presence and quality

#### 2c. Class Metrics
For each class:
- [ ] **Size**: Line count
- [ ] **Method Count**: Number of methods
- [ ] **Attribute Count**: Number of attributes
- [ ] **Inheritance Depth**: Levels of inheritance
- [ ] **Coupling**: Dependencies on other classes

**Metrics Template**:
```markdown
## Metrics Summary

### File: [file_path]
- LOC: [total] (code: [X], comments: [Y], blank: [Z])
- Complexity: [average] (max: [max_complexity])
- Maintainability Index: [0-100]
- Classes: [count], Functions: [count]

### Functions ([count] total)
| Function | LOC | Complexity | Params | Issues |
|----------|-----|------------|--------|--------|
| func1    | 45  | 8          | 5      | Too long, complex |
| func2    | 12  | 2          | 2      | OK |

### Classes ([count] total)
| Class | LOC | Methods | Complexity | Issues |
|-------|-----|---------|------------|--------|
| Class1| 120 | 8       | 25         | Too large |
```

**Expected Outcome**: Comprehensive metrics report

### Step 3: Code Smell Detection

#### 3a. Structural Smells
Identify structural issues:
- [ ] **God Class**: Classes with too many responsibilities (>200 LOC or >10 methods)
- [ ] **Long Method**: Methods exceeding 50 lines
- [ ] **Long Parameter List**: >5 parameters
- [ ] **Feature Envy**: Methods using more data from other classes
- [ ] **Data Clumps**: Same group of parameters appearing together

#### 3b. Duplication Detection
Find duplicate code:
- [ ] **Exact Duplicates**: Identical code blocks (≥6 lines)
- [ ] **Similar Code**: Nearly identical with minor differences
- [ ] **Copy-Paste Patterns**: Same logic in multiple places
- [ ] **Repeated Literals**: Magic numbers/strings used multiple times

#### 3c. Naming Issues
Check naming quality:
- [ ] **Unclear Names**: Variables like x, tmp, data
- [ ] **Inconsistent Naming**: Mixed camelCase/snake_case
- [ ] **Misleading Names**: Names that don't match behavior
- [ ] **Too Short**: Single letter names (except loop counters)
- [ ] **Too Long**: Names exceeding 40 characters

#### 3d. Logic Smells
Identify logic issues:
- [ ] **Deep Nesting**: More than 4 levels
- [ ] **Complex Conditions**: >3 boolean operators
- [ ] **Duplicated Conditionals**: Same if/else logic repeated
- [ ] **Switch Statements**: Long switch/if-elif chains
- [ ] **Dead Code**: Unreachable code or unused variables

**Code Smell Template**:
```markdown
## Code Smells Detected

### Structural Issues ([count])
⚠️ **God Class**: `ClassName` in [file]:[line]
- LOC: [count], Methods: [count]
- Issue: Too many responsibilities
- Recommendation: Split into [X] smaller classes

⚠️ **Long Method**: `method_name` in [file]:[line]
- LOC: [count], Complexity: [score]
- Issue: Method does too much
- Recommendation: Extract [X] helper methods

### Duplication ([count] instances)
⚠️ **Exact Duplicate**: [lines] in [file1]:[line1] and [file2]:[line2]
- Duplicated: [code snippet]
- Recommendation: Extract to function `[suggested_name]`

### Naming Issues ([count])
⚠️ **Unclear Name**: Variable `x` in [file]:[line]
- Context: [usage]
- Recommendation: Rename to `[suggested_name]`

### Logic Issues ([count])
⚠️ **Deep Nesting**: [file]:[line]
- Nesting depth: [count]
- Recommendation: Use early returns or extract methods
```

**Expected Outcome**: Categorized code smell report

### Step 4: Error Handling Analysis

#### 4a. Exception Handling
Check error handling patterns:
- [ ] **Bare Except**: except: without exception type
- [ ] **Too Broad**: except Exception: catches everything
- [ ] **Silent Failures**: Empty except blocks
- [ ] **Missing Error Handling**: Functions that should handle errors but don't
- [ ] **Error Swallowing**: Catching but not logging/re-raising

#### 4b. Input Validation
Check defensive programming:
- [ ] **Missing Validation**: No checks on inputs
- [ ] **Type Checking**: Missing type validation
- [ ] **Boundary Checks**: No validation of ranges/limits
- [ ] **None Checks**: No null/None handling

#### 4c. Resource Management
Check resource handling:
- [ ] **Missing Context Managers**: Files/connections not using `with`
- [ ] **Unclosed Resources**: Files/connections not explicitly closed
- [ ] **Memory Leaks**: Circular references or uncleaned resources

**Error Handling Template**:
```markdown
## Error Handling Issues

### Exception Handling ([count])
⚠️ **Bare Except**: [file]:[line]
```python
try:
    [code]
except:  # Too broad!
    pass
```
- Recommendation: Specify exception type and handle appropriately

### Input Validation ([count])
⚠️ **Missing Validation**: `function_name` in [file]:[line]
- Parameter: `[param_name]` not validated
- Risk: [what can go wrong]
- Recommendation: Add validation: `if not [condition]: raise ValueError(...)`

### Resource Management ([count])
⚠️ **Missing Context Manager**: [file]:[line]
```python
f = open(filename)  # Not using 'with'
data = f.read()
```
- Recommendation: Use `with open(filename) as f:`
```

**Expected Outcome**: Error handling assessment

### Step 5: Security Analysis

#### 5a. Security Vulnerabilities
Check for common vulnerabilities:
- [ ] **SQL Injection**: String concatenation in queries
- [ ] **Command Injection**: Unsafe shell command construction
- [ ] **Path Traversal**: Unvalidated file paths
- [ ] **Hardcoded Secrets**: API keys, passwords in code
- [ ] **Weak Cryptography**: Deprecated crypto algorithms

#### 5b. Data Protection
Check data handling:
- [ ] **Sensitive Data Logging**: Passwords/tokens in logs
- [ ] **Insecure Storage**: Plaintext sensitive data
- [ ] **Missing Input Sanitization**: User input not cleaned
- [ ] **Information Disclosure**: Detailed error messages to users

**Security Template**:
```markdown
## Security Issues

### Critical ([count])
🔴 **Hardcoded Secret**: [file]:[line]
```python
API_KEY = "sk-1234567890abcdef"  # Hardcoded!
```
- Risk: Credential exposure
- Recommendation: Use environment variables or secrets manager

### High ([count])
🟠 **Command Injection**: [file]:[line]
```python
os.system(f"rm {user_input}")  # Unsafe!
```
- Risk: Arbitrary command execution
- Recommendation: Use subprocess with shell=False and validate input
```

**Expected Outcome**: Security vulnerability report

### Step 6: Performance Analysis

#### 6a. Performance Anti-Patterns
Identify performance issues:
- [ ] **Premature Optimization**: Over-engineered for no benefit
- [ ] **Inefficient Algorithms**: O(n²) when O(n log n) possible
- [ ] **Repeated Computation**: Same calculation in loops
- [ ] **Inefficient Data Structures**: Wrong structure for use case
- [ ] **Memory Waste**: Unnecessary copies or large allocations

#### 6b. Optimization Opportunities
Find optimization potential:
- [ ] **Caching Opportunities**: Repeated expensive calls
- [ ] **Lazy Evaluation**: Computing values never used
- [ ] **Batch Operations**: Multiple individual ops vs batch
- [ ] **Generator Opportunities**: Lists that could be generators

**Performance Template**:
```markdown
## Performance Issues

### Critical ([count])
⚠️ **Inefficient Algorithm**: [file]:[line]
- Current: O(n²) nested loops
- Recommendation: Use hash map for O(n) lookup

### Optimization Opportunities ([count])
💡 **Caching**: `expensive_function` in [file]:[line]
- Called [X] times with same arguments
- Recommendation: Add @lru_cache decorator
```

**Expected Outcome**: Performance analysis report

### Step 7: Architecture & Design Analysis

#### 7a. SOLID Principles
Check adherence to SOLID:
- [ ] **Single Responsibility**: Classes doing one thing
- [ ] **Open/Closed**: Open for extension, closed for modification
- [ ] **Liskov Substitution**: Subtypes substitutable for base types
- [ ] **Interface Segregation**: No unused interface methods
- [ ] **Dependency Inversion**: Depend on abstractions

#### 7b. Design Patterns
Identify pattern opportunities:
- [ ] **Strategy Pattern**: If-else chains that could use strategy
- [ ] **Factory Pattern**: Complex object creation
- [ ] **Observer Pattern**: Manual notification systems
- [ ] **Decorator Pattern**: Nested wrapper functions
- [ ] **Singleton Pattern**: Global state management

**Architecture Template**:
```markdown
## Architecture & Design

### SOLID Violations ([count])
⚠️ **Single Responsibility**: `ClassName` in [file]
- Responsibilities: [list]
- Recommendation: Split into [X] classes

### Design Pattern Opportunities ([count])
💡 **Strategy Pattern**: [file]:[line]
- Current: if-elif chain for [different algorithms]
- Recommendation: Use strategy pattern for better extensibility
```

**Expected Outcome**: Architecture assessment

### Step 8: Documentation Quality

#### 8a. Docstring Coverage
Check documentation:
- [ ] **Missing Docstrings**: Functions/classes without docs
- [ ] **Incomplete Docstrings**: Missing parameters/returns
- [ ] **Outdated Docstrings**: Don't match implementation
- [ ] **Type Hints**: Missing type annotations

#### 8b. Comment Quality
Analyze comments:
- [ ] **No Comments**: Complex code without explanation
- [ ] **Outdated Comments**: Comments contradict code
- [ ] **Obvious Comments**: Comments stating the obvious
- [ ] **Commented Code**: Dead code left in comments

**Documentation Template**:
```markdown
## Documentation Issues

### Missing Docstrings ([count])
⚠️ `function_name` in [file]:[line]
- Public function without docstring
- Recommendation: Add comprehensive docstring

### Missing Type Hints ([count])
⚠️ `function_name` in [file]:[line]
- Parameters without type annotations
- Recommendation: Add types: `def func(param: Type) -> ReturnType:`
```

**Expected Outcome**: Documentation quality report

### Step 9: Prioritization & Recommendations

#### 9a. Issue Prioritization
Rank issues by impact:
- [ ] **Critical** (Fix immediately):
  - Security vulnerabilities
  - Major bugs/correctness issues
  - Performance bottlenecks
- [ ] **High** (Fix soon):
  - Code smells affecting maintainability
  - Missing error handling
  - Architecture violations
- [ ] **Medium** (Plan to fix):
  - Minor duplication
  - Naming improvements
  - Documentation gaps
- [ ] **Low** (Nice to have):
  - Style improvements
  - Minor optimizations

#### 9b. Refactoring Recommendations
Generate actionable recommendations:
- [ ] Specific changes to make
- [ ] Order of refactoring (dependencies)
- [ ] Estimated effort (hours/days)
- [ ] Risk assessment for each change
- [ ] Testing strategy for validation

**Prioritization Template**:
```markdown
## Prioritized Refactoring Plan

### Critical Issues ([count]) - Fix Immediately
1. **Security**: [Issue] in [file]:[line]
   - Impact: [Severity]
   - Effort: [hours]
   - Action: [Specific fix]

### High Priority ([count]) - Fix This Sprint
2. **Code Smell**: [Issue] in [file]:[line]
   - Impact: [Maintainability cost]
   - Effort: [hours]
   - Action: [Specific refactoring]

### Medium Priority ([count]) - Plan for Next Sprint
3. **Documentation**: [Issue] in [file]:[line]
   - Impact: [Developer productivity]
   - Effort: [hours]
   - Action: [Specific improvement]

### Low Priority ([count]) - Backlog
4. **Style**: [Issue] in [file]:[line]
   - Impact: [Minor]
   - Effort: [hours]
   - Action: [Optional improvement]
```

**Expected Outcome**: Prioritized action plan

### Step 10: Refactoring Integration

#### 10a. Generate Refactoring Tasks
For each high-priority issue:
- [ ] Create task description for `/plan`
- [ ] Define acceptance criteria
- [ ] List affected files
- [ ] Estimate complexity

#### 10b. Suggest Workflow
Provide integration with other commands:
```markdown
## Suggested Workflow

### For Critical Issues (Security/Bugs)
1. Run `/plan "Fix [security issue] in [file]"`
2. Run `/implement "Fix [security issue]"` with TDD
3. Verify with security tests

### For Architecture Refactoring
1. Run `/validate-theory "[new design approach]"` (if algorithmic)
2. Run `/plan "[refactoring description]"`
3. Run `/implement` with extensive tests
4. Run `/docs-sync` to update documentation

### For Code Quality Improvements
1. Run `/plan "Refactor [component] to address [smell]"`
2. Run `/implement` with refactoring tests
3. Verify no regressions: `make test`
```

**Expected Outcome**: Integration with TDD workflow

### Step 11: Comprehensive Inspection Report

Generate final report:

```markdown
# Code Quality Inspection Report

## Executive Summary
- **Files Inspected**: [count]
- **Total Issues**: [count]
  - Critical: [count]
  - High: [count]
  - Medium: [count]
  - Low: [count]
- **Quality Score**: [0-100]

## Metrics Summary
[Table of metrics per file]

## Issues by Category

### Code Smells ([count])
- Structural: [count]
- Duplication: [count]
- Naming: [count]
- Logic: [count]

### Error Handling ([count])
- Exception handling: [count]
- Input validation: [count]
- Resource management: [count]

### Security ([count])
- Critical: [count]
- High: [count]

### Performance ([count])
- Inefficient algorithms: [count]
- Optimization opportunities: [count]

### Architecture ([count])
- SOLID violations: [count]
- Design pattern opportunities: [count]

### Documentation ([count])
- Missing docstrings: [count]
- Missing type hints: [count]

## Detailed Findings
[Full details for each issue, organized by priority]

## Refactoring Plan
[Prioritized list with effort estimates]

## Next Steps
1. [Immediate action 1]
2. [Immediate action 2]
3. [Plan for sprint 1]
4. [Plan for sprint 2]

## Command Integration
To fix these issues:
```bash
# For critical issues
/plan "Fix [critical issue 1]"
/implement "Fix [critical issue 1]"

# For refactoring
/plan "Refactor [component] to address [smells]"
/implement "Refactor [component]"

# Update documentation
/docs-sync
```
```

**Expected Outcome**: Complete inspection report

## Validation Criteria
- [ ] All files analyzed
- [ ] Metrics collected and reported
- [ ] Code smells identified and categorized
- [ ] Security vulnerabilities flagged
- [ ] Performance issues noted
- [ ] Recommendations prioritized
- [ ] Integration with workflow provided
- [ ] Actionable next steps clear

## Output Format
Present structured report with:
1. Executive summary (quality score and counts)
2. Metrics dashboard
3. Issues by category and priority
4. Detailed findings with code examples
5. Prioritized refactoring plan
6. Integration commands for fixes

## Integration Points
- **/plan**: Generate refactoring plans from findings
- **/implement**: Execute refactoring with TDD
- **/docs-sync**: Update docs after refactoring
- **IMPLEMENTATION_TASKS.md**: Track refactoring tasks

## Decision Point
After inspection, ask user:
- "Found [count] issues. Shall I generate `/plan` for critical issues or review findings first?"
