---
description: Transform free text into optimized prompt
args:
  - name: raw_prompt
    description: Your rough idea or request in natural language
    required: true
---

# Prompt Engineering

**Context**: Load @.claude/commands/_project.md

**Raw Input**: $ARGUMENT

## Analysis Process

1. **Identify Intent**
   - Extract core objective
   - Detect missing specifications
   - Note ambiguities

2. **Add Context**
   - Include relevant project background
   - Reference appropriate documentation
   - Specify technical constraints

3. **Structure for Clarity**
   - Clear task statement
   - Explicit requirements
   - Success criteria
   - Output format

4. **Suggest Commands**
   - Check if existing slash command fits
   - Recommend: /tdd, /refactor, /check, etc.

## Output Format

```markdown
## Engineered Prompt

---BEGIN PROMPT---

Context: Online Model Learning Framework
- Tech: Python, Unified Planning, PySAT
- Method: Test-Driven Development
- Docs: See @CLAUDE.md, @docs/DEVELOPMENT_RULES.md

Task: [Clear, specific task description]

Requirements:
- [Explicit requirement 1]
- [Explicit requirement 2]

Success Criteria:
- [Measurable outcome 1]
- [Measurable outcome 2]

Output: [Expected format/structure]

---END PROMPT---

## Alternative: Use Slash Command

Instead of the prompt above, consider:
/[command] "[argument]"

This would achieve the same goal more efficiently.
```

## Example

**Input**: "make the algorithm faster"

**Output**:
```
---BEGIN PROMPT---
Context: Online Model Learning Framework - Information Gain algorithm

Task: Optimize action selection performance in src/algorithms/information_gain.py

Requirements:
- Profile current bottlenecks (likely SAT solver calls)
- Maintain exact algorithm behavior
- Target: 50 actions in <100ms (currently ~800ms)

Success Criteria:
- 5x+ performance improvement
- All tests still passing
- Code remains readable

Output: Profiling results, optimization approach, implementation

---END PROMPT---

Alternative: /tdd "optimize Information Gain action selection performance"
```

important!:
- Keep transformations practical and concise.
- do not execute the instruction in the output prompt without the user approval.