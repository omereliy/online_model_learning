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
Transform free-text user input into a well-engineered prompt optimized for clarity, specificity, and effectiveness. Uses project documentation (CLAUDE.md, DEVELOPMENT_RULES.md, IMPLEMENTATION_TASKS.md) as automatic context.

## Execution Steps

### Step 1: Project Context Loading
Load standard project context (always available):

- [ ] Read @CLAUDE.md for project overview and navigation
- [ ] Read @docs/DEVELOPMENT_RULES.md for conventions and rules
- [ ] Read @docs/IMPLEMENTATION_TASKS.md for current status
- [ ] Note technology stack: Unified Planning, PySAT, Python, TDD methodology
- [ ] Note key paths: `src/`, `tests/`, `docs/`, `benchmarks/`

**Automatic Context Available**:
```
Project: Online Model Learning Framework
Tech Stack: Python, Unified Planning, PySAT, pytest
Methodology: Test-Driven Development (tests first)
Key Documentation: CLAUDE.md, DEVELOPMENT_RULES.md, IMPLEMENTATION_TASKS.md
Conventions: Type hints, docstrings, make test before commits
```

**Expected Outcome**: Full project context loaded

---

### Step 2: User Intent Analysis
Gather additional context from user if needed:

#### 2.1: Clarification Questions (if needed)
Ask only if the raw prompt is ambiguous:

- [ ] **Output format**: If unclear, ask: "What format should the output take? (code/plan/analysis/report)"
- [ ] **Success criteria**: If unclear, ask: "How will you know the task is complete?"
- [ ] **Scope**: If unclear, ask: "Should this be a quick fix or comprehensive solution?"

**Skip questions if**:
- Raw prompt is clear and specific
- Task type is obvious from context
- Standard project workflow applies

**Expected Outcome**: Any ambiguities resolved

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
Weave in project context automatically:

**Standard Prompt Structure**:
```
Project Context:
- Working on Online Model Learning Framework
- Tech Stack: Python, Unified Planning, PySAT, pytest
- Methodology: Test-Driven Development (tests written FIRST)
- Key docs: CLAUDE.md, DEVELOPMENT_RULES.md, IMPLEMENTATION_TASKS.md
- Conventions: Type hints, docstrings, make test before commits

Task:
[Engineered task description]

Requirements:
[Explicit requirements from analysis]

Constraints:
[Explicit constraints]

Success Criteria:
[Measurable criteria]

Output Format:
[Desired format with example]

See @CLAUDE.md and @docs/DEVELOPMENT_RULES.md for full project context.
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
- [ ] Project documentation referenced
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

**Context Added**: [Summary of context from project docs]
**Structure Used**: [Structure type]

**Quality Score**: [X/10]

**Usage Instructions**:
- Copy everything between ---BEGIN PROMPT--- and ---END PROMPT---
- Paste into new session
- Project context automatically loaded from CLAUDE.md and DEVELOPMENT_RULES.md

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
- [ ] Project context loaded from documentation
- [ ] Necessary clarifications gathered (if needed)
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
- Slash command suggestions (if applicable)
- Tips for future prompts

## Integration Points
- **Automatically loads** CLAUDE.md, DEVELOPMENT_RULES.md, IMPLEMENTATION_TASKS.md
- **Can chain with** other commands: engineer prompt → use in /plan or /implement
- **Works for any session** - all sessions use same project documentation

## Example Transformation

**User Input (Raw)**:
```
"make the algorithm faster"
```

**Engineered Output**:
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

## Usage Notes

**When to Use**:
- Before starting work in new session
- When user has vague idea needing structure
- To transform rough ideas into actionable prompts
- When complex tasks need structured framing

**Project Context**:
- All sessions share same project documentation
- CLAUDE.md, DEVELOPMENT_RULES.md, IMPLEMENTATION_TASKS.md always referenced
- No need to distinguish new vs continued sessions
- Context automatically consistent across all sessions

**Benefits**:
- Transforms vague requests into specific prompts
- Ensures TDD methodology referenced
- Suggests slash commands when applicable
- Maintains project conventions automatically
