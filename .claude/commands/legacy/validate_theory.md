---
description: Validate algorithm theoretical correctness before implementation
args:
  - name: algorithm_description
    description: Algorithm description (academic paper reference, markdown summary, or file path)
    required: true
extended_thinking_required: true
---

⚠️ **STOP: Extended Thinking Required**

Before proceeding, verify extended thinking is enabled.

**If NOT enabled**: Stop and tell user:
"This command requires extended thinking for correctness proofs, complexity analysis, and edge case discovery. Please enable it or confirm to proceed anyway (not recommended)."

**If enabled or user confirms**: Continue with Step 1.

---

# Algorithm Theoretical Validation Command

## Purpose
Validate theoretical and algorithmic correctness BEFORE implementation to identify issues, edge cases, and complexity considerations.

## Execution Steps

### Step 1: Algorithm Acquisition
Load algorithm description:
- [ ] If argument is file path: Read file content
- [ ] If argument is paper reference: Extract key details
- [ ] If argument is description: Use directly
- [ ] Parse algorithm into structured format

**Algorithm Source**: $ARGUMENT

**Expected Outcome**: Algorithm specification loaded

### Step 2: Algorithm Decomposition
Break down algorithm into components:
- [ ] Identify core algorithmic steps
- [ ] Extract mathematical formulas
- [ ] List data structures used
- [ ] Identify control flow (loops, recursion, conditionals)
- [ ] Note key variables and their domains

**Decomposition Template**:
```
Algorithm: [Name]

Input: [Parameters and types]
Output: [Return value and type]

Steps:
1. [Step 1] - [Operation]
   - Data structures: [list]
   - Formulas: [list]
2. [Step 2] - [Operation]
   ...

Key Variables:
- var1: [type, domain, purpose]
- var2: [type, domain, purpose]

Control Flow:
- [Description of loops, recursion, branches]
```

**Expected Outcome**: Structured algorithm specification

### Step 3: Theoretical Correctness Analysis
Validate algorithm logic and soundness:

#### 3a. Correctness Verification
- [ ] **Termination**: Does algorithm terminate for all valid inputs?
- [ ] **Partial Correctness**: If it terminates, is output correct?
- [ ] **Total Correctness**: Termination + Partial Correctness
- [ ] **Invariants**: What properties hold throughout execution?
- [ ] **Preconditions**: What must be true before execution?
- [ ] **Postconditions**: What is guaranteed after execution?

#### 3b. Mathematical Validation
- [ ] Verify all formulas are mathematically sound
- [ ] Check for division by zero risks
- [ ] Validate probability calculations (sum to 1, range [0,1])
- [ ] Verify set operations are well-defined
- [ ] Confirm numerical stability (no overflow/underflow)

#### 3c. Logic Validation
- [ ] Check for infinite loops or recursion
- [ ] Verify all branches are reachable
- [ ] Confirm no contradictory conditions
- [ ] Validate state transitions are well-defined

**Validation Report Template**:
```markdown
## Correctness Analysis

### Termination
- **Status**: ✓ Guaranteed / ⚠️ Conditional / ✗ Not guaranteed
- **Reasoning**: [Explanation]
- **Conditions**: [If conditional]

### Partial Correctness
- **Status**: ✓ Verified / ⚠️ Assumptions needed / ✗ Issues found
- **Invariants**: [List]
- **Preconditions**: [List]
- **Postconditions**: [List]

### Mathematical Soundness
- ✓ [Formula 1] - Verified
- ⚠️ [Formula 2] - Needs constraint: [condition]
- ✗ [Formula 3] - Issue: [problem]
```

**Expected Outcome**: Correctness validation report

### Step 4: Complexity Analysis

#### 4a. Time Complexity
- [ ] Analyze worst-case time complexity
- [ ] Calculate best-case time complexity
- [ ] Determine average-case complexity (if applicable)
- [ ] Identify complexity of each step
- [ ] Calculate overall Big-O notation

**Time Complexity Template**:
```
Step 1: [Description]
- Operations: [count or formula]
- Complexity: O([expression])

Step 2: [Description]
- Operations: [count or formula]
- Complexity: O([expression])

...

Overall Time Complexity:
- Best case: O([expression]) - When: [condition]
- Average case: O([expression]) - Assumptions: [list]
- Worst case: O([expression]) - When: [condition]
```

#### 4b. Space Complexity
- [ ] Identify all data structures and their sizes
- [ ] Calculate memory for variables
- [ ] Account for recursion stack (if applicable)
- [ ] Determine auxiliary space used
- [ ] Calculate overall space complexity

**Space Complexity Template**:
```
Data Structures:
- [Structure 1]: Size = [expression] → Space: O([expression])
- [Structure 2]: Size = [expression] → Space: O([expression])

Recursion Stack: Depth = [expression] → Space: O([expression])

Overall Space Complexity: O([expression])
```

**Expected Outcome**: Complete complexity analysis

### Step 5: Edge Case Identification
Identify boundary conditions and special cases:

#### 5a. Input Edge Cases
- [ ] Empty inputs (empty sets, lists, strings)
- [ ] Minimum values (0, 1, -∞)
- [ ] Maximum values (∞, MAX_INT)
- [ ] Single element inputs
- [ ] All identical elements
- [ ] Sorted vs unsorted inputs

#### 5b. Algorithmic Edge Cases
- [ ] First iteration vs subsequent iterations
- [ ] Convergence conditions
- [ ] Ties in selection/comparison
- [ ] Numerical precision limits
- [ ] State space exhaustion

#### 5c. Domain-Specific Edge Cases
- [ ] No valid actions available
- [ ] All actions have equal priority
- [ ] Cyclic dependencies
- [ ] Unreachable states
- [ ] Contradictory constraints

**Edge Case Template**:
```markdown
## Edge Cases

### Critical (Must Handle)
1. [Edge Case 1]
   - Condition: [When it occurs]
   - Impact: [What breaks]
   - Handling: [How to handle]

### Important (Should Handle)
2. [Edge Case 2]
   - Condition: [When it occurs]
   - Impact: [Degraded behavior]
   - Handling: [Recommended approach]

### Optional (Nice to Have)
3. [Edge Case 3]
   - Condition: [Rare scenario]
   - Impact: [Minor issue]
   - Handling: [Possible approach]
```

**Expected Outcome**: Comprehensive edge case catalog

### Step 6: Implementation Pitfall Detection
Identify potential implementation challenges:

#### 6a. Numerical Issues
- [ ] Floating point precision errors
- [ ] Integer overflow/underflow
- [ ] Division by zero
- [ ] Logarithm of zero or negative
- [ ] Square root of negative
- [ ] NaN or Inf propagation

#### 6b. Performance Pitfalls
- [ ] Unnecessary recomputation
- [ ] Inefficient data structures
- [ ] Cache-unfriendly access patterns
- [ ] Excessive memory allocation
- [ ] Redundant operations

#### 6c. Logic Pitfalls
- [ ] Off-by-one errors
- [ ] Uninitialized variables
- [ ] Race conditions (if parallel)
- [ ] State inconsistency
- [ ] Memory leaks

#### 6d. API Design Issues
- [ ] Unclear parameter semantics
- [ ] Missing error handling
- [ ] Mutable vs immutable expectations
- [ ] Side effects not documented
- [ ] Unclear ownership (who frees memory)

**Pitfall Report Template**:
```markdown
## Implementation Pitfalls

### High Priority
⚠️ [Pitfall 1]: [Description]
- **Risk**: [What can go wrong]
- **Prevention**: [How to avoid]
- **Detection**: [How to test for it]

### Medium Priority
⚠️ [Pitfall 2]: [Description]
- **Risk**: [What can go wrong]
- **Prevention**: [How to avoid]
- **Detection**: [How to test for it]
```

**Expected Outcome**: Pitfall prevention guide

### Step 7: Test Case Generation
Suggest test cases based on analysis:

#### 7a. Boundary Tests
Generate tests for edge cases:
- [ ] Minimum input size
- [ ] Maximum input size
- [ ] Boundary values (0, 1, -1, MAX, MIN)
- [ ] Just above/below boundaries

#### 7b. Correctness Tests
Generate tests for core functionality:
- [ ] Normal operation (typical inputs)
- [ ] Known outputs (hand-calculated examples)
- [ ] Invariant verification tests
- [ ] Postcondition validation tests

#### 7c. Performance Tests
Generate tests for complexity validation:
- [ ] Scaling tests (increasing input sizes)
- [ ] Worst-case scenario tests
- [ ] Timeout/resource limit tests

#### 7d. Robustness Tests
Generate tests for error handling:
- [ ] Invalid inputs
- [ ] Unexpected states
- [ ] Error conditions
- [ ] Resource exhaustion

**Test Case Template**:
```python
# Boundary Test: [Description]
def test_algorithm_[boundary_case]():
    """Test [algorithm] with [edge case]."""
    input_data = [boundary condition]
    expected = [calculated result]

    result = algorithm(input_data)

    assert result == expected
    # Additional invariant checks

# Correctness Test: [Description]
def test_algorithm_[correctness_case]():
    """Verify [algorithm] correctness for [scenario]."""
    # ... test implementation

# Performance Test: [Description]
def test_algorithm_[performance_case]():
    """Verify [algorithm] complexity for [size]."""
    # ... test implementation
```

**Expected Outcome**: Complete test suite suggestions

### Step 8: Computer Science Validation
Validate against known CS principles:

#### 8a. Algorithm Classification
- [ ] Algorithm type (greedy, dynamic programming, divide & conquer, etc.)
- [ ] Problem class (P, NP, NP-complete, etc.)
- [ ] Optimization type (exact, approximate, heuristic)
- [ ] Deterministic vs randomized

#### 8b. Theoretical Bounds
- [ ] Compare to known lower bounds
- [ ] Check if matches upper bounds
- [ ] Identify if optimal or approximation
- [ ] Note approximation ratio (if applicable)

#### 8c. Related Algorithms
- [ ] Similar existing algorithms
- [ ] Known optimizations
- [ ] Alternative approaches
- [ ] Trade-offs comparison

**CS Validation Template**:
```markdown
## Computer Science Validation

### Algorithm Classification
- **Type**: [Greedy/DP/etc.]
- **Problem Class**: [P/NP/etc.]
- **Optimization**: [Exact/Approximate/Heuristic]

### Theoretical Bounds
- **Lower Bound**: Ω([expression])
- **This Algorithm**: O([expression])
- **Upper Bound**: O([expression])
- **Optimality**: [Optimal / [ratio]-approximation]

### Related Work
- **Similar Algorithms**: [List with citations]
- **Key Differences**: [Comparison]
- **Known Issues**: [From literature]
```

**Expected Outcome**: CS principle validation

### Step 9: Validation Summary Report
Generate comprehensive validation report:

```markdown
# Algorithm Validation Report: [Algorithm Name]

## Executive Summary
- ✓ **Theoretically Correct**: [Yes/No/Conditional]
- ✓ **Time Complexity**: O([expression])
- ✓ **Space Complexity**: O([expression])
- ⚠️ **Critical Issues**: [count] found
- ✓ **Edge Cases**: [count] identified
- ✓ **Implementation Risks**: [count] flagged

## Algorithm Overview
[Brief description]

**Input**: [Specification]
**Output**: [Specification]

## Correctness Analysis
### Termination: [Status]
[Details]

### Partial Correctness: [Status]
- **Invariants**: [List]
- **Preconditions**: [List]
- **Postconditions**: [List]

### Mathematical Soundness: [Status]
[Details]

## Complexity Analysis
### Time Complexity
- Best: O([expression])
- Average: O([expression])
- Worst: O([expression])

### Space Complexity
- O([expression])

### Justification
[Detailed breakdown]

## Edge Cases ([count] identified)
1. [Edge case 1] - [Handling strategy]
2. [Edge case 2] - [Handling strategy]
...

## Implementation Pitfalls ([count] identified)
### Critical
⚠️ [Pitfall 1]: [Description and prevention]

### Important
⚠️ [Pitfall 2]: [Description and prevention]

## Suggested Test Cases
### Boundary Tests ([count])
- [Test 1]
- [Test 2]

### Correctness Tests ([count])
- [Test 1]
- [Test 2]

### Performance Tests ([count])
- [Test 1]
- [Test 2]

## Recommendations
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

## Ready for Implementation?
- [x] Theoretical correctness verified
- [x] Complexity acceptable
- [x] Edge cases identified
- [x] Pitfalls documented
- [x] Test cases designed

**Status**: ✓ Ready / ⚠️ Issues to address / ✗ Not recommended
**Next Steps**: [What to do next]
```

**Expected Outcome**: Complete validation report

## Validation Criteria
- [ ] Algorithm correctness analyzed
- [ ] Complexity bounds calculated
- [ ] Edge cases comprehensively identified
- [ ] Implementation pitfalls documented
- [ ] Test cases suggested
- [ ] CS principles validated
- [ ] Recommendations provided
- [ ] Clear ready/not-ready decision

## Output Format
Present structured validation report with:
1. Executive summary (quick decision)
2. Detailed correctness analysis
3. Complexity analysis with justification
4. Edge case catalog
5. Pitfall prevention guide
6. Test case suggestions
7. Final recommendation

## Integration Points
- **/plan**: Use validation results in planning
- **/implement**: Reference pitfalls during implementation
- **Test files**: Use suggested test cases
- **Documentation**: Include complexity analysis

## Decision Point
After validation, ask user:
- "This algorithm is [ready/not ready] for implementation. Proceed with `/plan` or address issues first?"
