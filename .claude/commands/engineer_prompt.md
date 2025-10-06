---
description: Engineer an optimized prompt from free-text user input
args:
  - name: user_prompt
    description: Raw prompt idea in free text
    required: true
extended_thinking_required: true
---

⚠️ **STOP: Extended Thinking Required**

Before proceeding, verify extended thinking is enabled.

**If NOT enabled**: Stop and tell user:
"This command requires extended thinking for understanding implicit intent, resolving ambiguities, and meta-analysis of prompt quality. Please enable it or confirm to proceed anyway (not recommended)."

**If enabled or user confirms**: Continue with Step 1.

---

# Prompt Engineering Command

## Purpose
Transform free-text user input into a well-engineered prompt optimized for clarity, specificity, and effectiveness. Accounts for target session type (new vs continued).

## Execution Steps

### Step 1: Session Type Detection
Determine target session context:

**Ask user:**
```
Is this prompt for:
1. **New session** (no prior context)
2. **Continued session** (has existing context)

Reply with: 1 or 2
```

**Decision Point**: Session type determines context gathering

---

### Step 2A: New Session - Context Gathering

**If new session**, collect essential context:

#### 2A.1: Role/Persona Definition
- [ ] Ask: "What role should Claude assume? (e.g., Python expert, architect, code reviewer)"
- [ ] Ask: "Any specific expertise needed? (e.g., PDDL, SAT solvers, TDD)"

#### 2A.2: Background Context
- [ ] Ask: "What background should Claude know?"
  - Project type/domain
  - Technology stack
  - Constraints or requirements
  - Relevant files or documentation

#### 2A.3: Output Format
- [ ] Ask: "What format should the response take?"
  - Code with explanations
  - Step-by-step plan
  - Analysis report
  - Implementation guide

#### 2A.4: Success Criteria
- [ ] Ask: "How will you know the response is successful?"
  - Specific deliverables
  - Quality criteria
  - Edge cases to handle

**Collected Context Template**:
```
Role: [role and expertise]
Background:
- [context point 1]
- [context point 2]
Output Format: [desired format]
Success Criteria:
- [criterion 1]
- [criterion 2]
```

---

### Step 2B: Continued Session - Context Reference

**If continued session**, identify what to reference:

#### 2B.1: Previous Work Reference
- [ ] Ask: "What prior context should be referenced?"
  - Previous tasks completed
  - Existing codebase elements
  - Decisions already made
  - Patterns established

#### 2B.2: Incremental Nature
- [ ] Ask: "Is this:"
  - Building on previous work (extend/modify)
  - New task using same context (parallel work)
  - Follow-up question (clarification)

#### 2B.3: Continuity Assumptions
- [ ] Ask: "What can Claude assume is already understood?"
  - Project conventions
  - Architecture decisions
  - Previously discussed approaches

**Collected Context Template**:
```
References:
- [prior work/context 1]
- [prior work/context 2]
Relationship: [building on / parallel / follow-up]
Assumed Knowledge:
- [assumption 1]
- [assumption 2]
```

---

### Step 3: Prompt Analysis
Analyze the user's raw prompt:

#### 3.1: Intent Identification
- [ ] Extract core objective
- [ ] Identify implicit requirements
- [ ] Detect ambiguities
- [ ] Note missing specifications

#### 3.2: Complexity Assessment
- [ ] Scope: Small/Medium/Large
- [ ] Clarity: Clear/Vague/Ambiguous
- [ ] Completeness: Sufficient/Missing details

#### 3.3: Improvement Opportunities
Identify where to enhance:
- [ ] **Specificity**: Vague → Precise
- [ ] **Structure**: Unorganized → Organized
- [ ] **Context**: Missing → Complete
- [ ] **Constraints**: Implicit → Explicit
- [ ] **Examples**: None → Concrete examples

**Analysis Template**:
```markdown
## Raw Prompt Analysis

**User Input**: "$ARGUMENT"

**Core Objective**: [What user actually wants]

**Implicit Requirements**:
- [requirement 1]
- [requirement 2]

**Ambiguities Detected**:
- [ambiguity 1] - Need: [clarification]
- [ambiguity 2] - Need: [clarification]

**Missing Specifications**:
- [missing 1]
- [missing 2]

**Complexity**: [Small/Medium/Large]
```

---

### Step 4: Prompt Engineering

#### 4.1: Structure Selection
Choose optimal structure based on task:

**Task Types & Structures**:

1. **Implementation Task**:
```
[Role/Context]
[Specific task]
[Requirements/Constraints]
[Success criteria]
[Output format]
```

2. **Analysis Task**:
```
[Context]
[What to analyze]
[Analysis dimensions]
[Depth/scope]
[Output format]
```

3. **Planning Task**:
```
[Background]
[Goal]
[Constraints]
[Required deliverables]
[Planning horizon]
```

4. **Problem-Solving Task**:
```
[Problem statement]
[Current situation]
[Desired outcome]
[Constraints]
[Approach preferences]
```

#### 4.2: Clarity Enhancement
Transform vague → specific:

**Techniques**:
- Add concrete examples
- Specify edge cases
- Define success criteria
- Include constraints explicitly
- Reference specific files/patterns

**Example Transformation**:
```
Before: "Fix the algorithm"
After: "Debug the CNF-based action selection in src/algorithms/information_gain.py:
- Issue: Returns wrong action when all information gains are equal
- Expected: Should use epsilon-greedy fallback
- Test: Fails test_equal_information_gains in tests/algorithms/test_information_gain.py"
```

#### 4.3: Context Integration
Weave in collected context:

**For New Sessions**:
```
You are a [role] with expertise in [domain].

Context:
[Background points]

Task:
[Engineered task description]

Requirements:
[Explicit requirements]

Constraints:
[Explicit constraints]

Success Criteria:
[Measurable criteria]

Output Format:
[Desired format with example]
```

**For Continued Sessions**:
```
Building on [previous work/context]:

[Reference to established context]

Next Task:
[Engineered task description]

Continuation Details:
[How this extends prior work]

Maintain:
[What to preserve from previous work]

Success Criteria:
[Criteria for this increment]
```

#### 4.4: Add Effectiveness Boosters

**Include as appropriate**:

- **Examples**: "For example: [concrete case]"
- **Counter-examples**: "Not like: [what to avoid]"
- **Edge cases**: "Consider when: [boundary condition]"
- **Output sample**: "Format like: [template]"
- **Verification**: "Verify by: [test/check]"
- **Constraints**: "Must: [hard requirement]", "Should: [preference]", "Could: [optional]"

---

### Step 5: Quality Validation

Check engineered prompt against criteria:

#### 5.1: Clarity Check
- [ ] No ambiguous terms
- [ ] Specific, measurable requirements
- [ ] Clear success criteria
- [ ] Explicit constraints

#### 5.2: Completeness Check
- [ ] All necessary context provided
- [ ] Role/expertise specified (if new session)
- [ ] Output format defined
- [ ] Edge cases mentioned

#### 5.3: Effectiveness Check
- [ ] Single, clear objective
- [ ] Actionable (Claude knows what to do)
- [ ] Measurable (success is verifiable)
- [ ] Scoped (not too broad/narrow)

**Validation Report**:
```
✓ Clarity: [score/10] - [notes]
✓ Completeness: [score/10] - [notes]
✓ Effectiveness: [score/10] - [notes]

Overall Quality: [score/10]
```

---

### Step 6: Final Prompt Generation

#### 6.1: Format for Copy-Paste
Present in copyable format:

**Output Structure**:
```markdown
# Engineered Prompt (Ready to Copy)

---BEGIN PROMPT---

[Engineered prompt with all context, structure, and enhancements]

---END PROMPT---

## Engineering Notes

**Original**: "$ARGUMENT"

**Enhancements Made**:
1. [Enhancement 1]
2. [Enhancement 2]
3. [Enhancement 3]

**Session Type**: [New/Continued]
**Context Added**: [Summary of context]
**Structure Used**: [Structure type]

**Quality Score**: [X/10]

**Usage Instructions**:
- Copy everything between ---BEGIN PROMPT--- and ---END PROMPT---
- Paste into [new/continued] session
- [Any additional setup needed]

**Expected Outcome**:
[What user should expect from Claude's response]

**If Response Unsatisfactory**:
[Suggested refinements or follow-up prompts]
```

#### 6.1b: Slash Command Recommendation
Analyze the engineered prompt and suggest appropriate slash commands:

**Determine if task matches existing slash commands**:
- [ ] Check if task is **planning** → Suggest `/plan "[task]"`
- [ ] Check if task is **implementation** → Suggest `/implement "[task]"`
- [ ] Check if task is **algorithm validation** → Suggest `/validate-theory "[algorithm]"`
- [ ] Check if task is **code quality analysis** → Suggest `/inspect-refactor "[files]"`
- [ ] Check if task is **documentation sync** → Suggest `/docs-sync`

**Add to output if slash command applicable**:
```markdown
## ⚡ Recommended Slash Command

Instead of copying the prompt manually, you can use:

```bash
/[command-name] "[argument]"
```

This slash command will:
- [Benefit 1 of using command]
- [Benefit 2 of using command]
- [Benefit 3 of using command]

**If you prefer manual prompt**: Use the ---BEGIN PROMPT--- version above.
```

**If no slash command matches**:
- Don't add this section
- Proceed with standard prompt output

**Expected Outcome**: User knows if a slash command can replace the manual prompt

#### 6.2: Alternative Versions (Optional)
If ambiguities remain, provide variants:

```markdown
## Alternative Interpretations

### Variant A: [Interpretation 1]
---BEGIN PROMPT---
[Prompt for interpretation 1]
---END PROMPT---

### Variant B: [Interpretation 2]
---BEGIN PROMPT---
[Prompt for interpretation 2]
---END PROMPT---

**Choose based on**: [Decision criteria]
```

---

### Step 7: Meta-Learning Suggestions

Provide tips for future prompts:

```markdown
## Tips for Better Prompts

Based on your input, consider:

1. **Be More Specific About**: [area that was vague]
   - Instead of: [vague phrase]
   - Try: [specific phrase]

2. **Always Include**: [commonly missing element]
   - Example: [how to include it]

3. **Structure Recommendation**: [suggested structure for similar tasks]

4. **Context Checklist**: For similar prompts, always specify:
   - [ ] [context item 1]
   - [ ] [context item 2]
   - [ ] [context item 3]
```

---

## Validation Criteria
- [ ] Session type identified (new vs continued)
- [ ] Necessary context gathered
- [ ] Ambiguities resolved
- [ ] Prompt is specific and actionable
- [ ] Success criteria clear
- [ ] Output format defined
- [ ] Constraints explicit
- [ ] Ready to copy-paste
- [ ] Quality score ≥ 8/10

## Output Format

**Primary Output**: Copyable engineered prompt in markdown code block

**Supporting Output**:
- Engineering notes and improvements made
- Quality assessment
- Usage instructions
- Tips for future prompts

## Integration Points
- **Used before starting work** in new or continued sessions
- **Can reference** DEVELOPMENT_RULES.md for project-specific context
- **Can chain with** other commands: engineer prompt → use in /plan or /implement

## Example Transformation

**User Input (Raw)**:
```
"make the algorithm faster"
```

**Engineered Output (New Session)**:
```markdown
---BEGIN PROMPT---

You are a performance optimization expert with deep knowledge of Python and algorithm complexity.

Context:
- Working on CNF-based information gain algorithm in `src/algorithms/information_gain.py`
- Current implementation: O(n²) for action selection due to nested model counting
- Bottleneck identified: SAT solver calls in tight loop (lines 145-160)
- Target: Process 50 actions in <100ms (currently 800ms)

Task:
Optimize the action selection performance in InformationGainLearner.select_action() method:

1. Analyze current complexity bottleneck
2. Propose optimization strategy (caching, batching, or algorithmic improvement)
3. Implement optimization maintaining correctness
4. Verify performance improvement with benchmarks

Requirements:
- Must maintain exact same action selection logic (no approximations)
- All existing tests must pass
- Performance improvement ≥ 5x
- Code must remain readable with clear comments

Constraints:
- Cannot change CNFManager API
- Must work with PySAT minisat solver
- Memory usage <100MB additional

Success Criteria:
- 50 actions processed in <100ms (currently 800ms)
- All 60 tests in test_information_gain.py pass
- Benchmark shows ≥5x speedup
- Code review approves readability

Output Format:
1. Analysis of bottleneck (with profiling data)
2. Optimization strategy explanation
3. Optimized code implementation
4. Benchmark results comparison
5. Testing verification

---END PROMPT---

## ⚡ Recommended Slash Command

Instead of copying the prompt manually, you can use:

```bash
/plan "Optimize action selection performance in InformationGainLearner"
```

This slash command will:
- Follow TDD methodology (write tests first)
- Break down optimization into testable components
- Create implementation roadmap with acceptance criteria
- Integrate with IMPLEMENTATION_TASKS.md for tracking

**If you prefer manual prompt**: Use the ---BEGIN PROMPT--- version above.
```

**Engineered Output (Continued Session)**:
```markdown
---BEGIN PROMPT---

Following our work on the Information Gain algorithm in `src/algorithms/information_gain.py`:

We identified the O(n²) bottleneck in select_action() during action selection. Now optimize this method:

Build on:
- Existing CNFManager integration
- 60 passing tests in test_information_gain.py
- Project coding standards from DEVELOPMENT_RULES.md

Task:
Optimize select_action() performance from 800ms → <100ms for 50 actions while maintaining:
- Exact same selection logic
- All test compatibility
- Code readability

Approach:
1. Profile current bottleneck (SAT solver calls)
2. Implement optimization (caching/batching)
3. Verify with benchmarks
4. Ensure tests pass

Success: ≥5x speedup, all tests pass, code approved

---END PROMPT---
```

## Usage Notes

**When to Use**:
- Before starting complex work in new session
- When user has vague idea needing structure
- To ensure new session has proper context
- When continuing work needs clear framing

**Session Type Impact**:
- **New session**: Requires full context, role definition, background
- **Continued session**: Can reference prior work, be more concise, assume knowledge

**The session type distinction is VALUABLE because**:
- Prevents context loss in new sessions
- Avoids redundancy in continued sessions
- Ensures proper framing for each case
- Results in more effective prompts
