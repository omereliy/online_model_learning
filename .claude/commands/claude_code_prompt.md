# Claude Code Task: Rewrite Information Gain Algorithm

## Overview

You are tasked with completely rewriting the Information Gain-based action model learning algorithm. The current implementation has critical issues with CNF handling, model counting, and parallel execution that make it produce incorrect results.

**Your goal**: Create a new, clean, modular implementation that is correct, testable, and maintainable.

---

## Step 0: Read Documentation First (MANDATORY)

Before writing ANY code, you MUST read these documents thoroughly:

```bash
# Read the issues analysis
cat docs/information_gain_algorithm/cnf/cnf_issues_analysis.md

# Read the fix strategy  
cat docs/information_gain_algorithm/cnf/cnf_fix_strategy.md
```

These documents contain:
- 13 critical issues with the current implementation
- Code references and line numbers for each issue
- Quantitative impact analysis (model count inflation of 18,000x+)
- Recommended architecture with complete code examples
- Testing strategy and migration guide

**Do NOT proceed until you have read and understood these documents.**

---

## Step 1: Analyze Current Implementation (Reference Only)

Read the current implementation to understand the domain, but **DO NOT copy-paste code**:

```
src/algorithms/information_gain.py    # Main learner (~900 lines - too long!)
src/algorithms/parallel_gain.py       # Parallel computation
src/core/cnf_manager.py               # CNF handling (has critical bugs)
src/core/grounding.py                 # Grounding utilities
src/core/lifted_domain.py             # Domain representation
```

Note the problems:
- `information_gain.py` is ~900 lines - should be <300 lines
- CNF handling mixed with learning logic
- Parallel code duplicates sequential logic
- No separation of concerns

---

## Step 2: Architecture Design Using Sub-Agents

Before implementing, use sub-agents to design each component independently. This ensures clean interfaces and avoids the copy-paste problems in the current code.

### Sub-Agent Task 1: CNF Module Architecture

**Dispatch this task to a sub-agent:**

```
TASK: Design the CNF management module architecture

CONTEXT:
- Read docs/information_gain_algorithm/cnf/cnf_issues_analysis.md sections CRIT-01 through CRIT-04
- Read docs/information_gain_algorithm/cnf/cnf_fix_strategy.md section "Phase 1: Core Architecture Fixes"

REQUIREMENTS:
1. Literal Pair Model:
   - Each fluent f creates TWO variables: l (primary) and neg_l (auxiliary)
   - Mutex constraint [-l, -neg_l] added automatically at creation
   - Only 3 valid states per pair: (F,F), (T,F), (F,T)

2. Clear Negation Semantics:
   - NO confusion between literal negation (¬) and CNF negation (-)
   - Document the meaning of every operation clearly

3. Projection Support:
   - Track primary vs auxiliary variables
   - Provide projection set for model counting

DELIVERABLES:
1. Class diagram showing CNFManager structure
2. Method signatures with docstrings explaining semantics
3. Invariants that must always hold
4. Example usage code

DO NOT write implementation - only design the interface.
```

### Sub-Agent Task 2: Model Counting Architecture

**Dispatch this task to a sub-agent:**

```
TASK: Design the model counting module architecture

CONTEXT:
- Read docs/information_gain_algorithm/cnf/cnf_issues_analysis.md section CRIT-03
- Read docs/information_gain_algorithm/cnf/cnf_fix_strategy.md section "Phase 2: Model Counting Fixes"

REQUIREMENTS:
1. Projected Counting:
   - Count over PRIMARY variables only
   - Project out AUXILIARY variables
   - Handle fixed variables correctly

2. Counting Methods:
   - Exact counting for small formulas
   - Approximate counting (ApproxMC) for large formulas
   - Adaptive selection based on problem size

3. Assumptions Support:
   - Use solver's native assumption mechanism
   - NO modification of CNF for temporary constraints
   - Thread-safe for parallel execution

DELIVERABLES:
1. Interface for model counting operations
2. Strategy pattern for exact vs approximate
3. Caching strategy that doesn't cause staleness
4. Example showing projected vs non-projected counting

DO NOT write implementation - only design the interface.
```

### Sub-Agent Task 3: Information Gain Calculator Architecture

**Dispatch this task to a sub-agent:**

```
TASK: Design the information gain calculation module

CONTEXT:
- Read src/algorithms/information_gain.py methods:
  - _calculate_applicability_probability
  - _calculate_potential_gain_success  
  - _calculate_potential_gain_failure
  - _calculate_expected_information_gain

REQUIREMENTS:
1. Separation of Concerns:
   - Gain calculation should NOT know about CNF internals
   - Use abstract "hypothesis counter" interface
   - Pure functions where possible

2. Testability:
   - Each gain component independently testable
   - Mock-friendly interfaces
   - Clear input/output contracts

3. Parallel-Friendly:
   - Stateless computation functions
   - All state passed as parameters
   - No shared mutable state

DELIVERABLES:
1. Interface for gain calculation
2. Data classes for computation context
3. Strategy for parallel vs sequential execution
4. Formula documentation matching the algorithm specification

DO NOT write implementation - only design the interface.
```

### Sub-Agent Task 4: Learner Orchestration Architecture

**Dispatch this task to a sub-agent:**

```
TASK: Design the main learner orchestrator

CONTEXT:
- Read src/algorithms/information_gain.py class InformationGainLearner
- Identify what should be in the orchestrator vs delegated to components

REQUIREMENTS:
1. Thin Orchestrator:
   - Main learner should be <200 lines
   - Delegates to specialized components
   - Only handles high-level flow

2. Clean State Management:
   - Clear ownership of mutable state
   - Explicit state transitions
   - Easy to serialize/deserialize for checkpointing

3. Extensibility:
   - Easy to add new selection strategies
   - Easy to swap counting backends
   - Easy to add new update rules

DELIVERABLES:
1. Class diagram showing component relationships
2. Sequence diagram for one learning iteration
3. State ownership documentation
4. Configuration/dependency injection approach

DO NOT write implementation - only design the interface.
```

---

## Step 3: Review Sub-Agent Designs

After receiving sub-agent designs, review them for:

1. **Interface Consistency**: Do the modules fit together cleanly?
2. **No Duplication**: Is logic in exactly one place?
3. **Testability**: Can each component be unit tested in isolation?
4. **Parallel Safety**: Can gain calculation run in parallel without issues?

Create a unified architecture document combining the sub-agent outputs.

---

## Step 4: Implementation

Create the new module at the project root:

```
information_gain/
├── __init__.py              # Public API exports
├── cnf/
│   ├── __init__.py
│   ├── literal_pair.py      # LiteralPair dataclass
│   ├── manager.py           # CNFManager with proper literal pairs
│   └── counting.py          # Projected model counting
├── model/
│   ├── __init__.py
│   ├── action_model.py      # Per-action learned state
│   └── update.py            # Success/failure update logic
├── gain/
│   ├── __init__.py
│   ├── calculator.py        # Information gain formulas
│   └── parallel.py          # Parallel computation (if needed)
├── learner.py               # Main orchestrator (<200 lines!)
└── types.py                 # Shared type definitions
```

### Implementation Guidelines

1. **Start with types.py**: Define all data classes and type aliases first

2. **Implement bottom-up**: 
   - `cnf/literal_pair.py` → `cnf/manager.py` → `cnf/counting.py`
   - `model/action_model.py` → `model/update.py`
   - `gain/calculator.py`
   - Finally `learner.py`

3. **Test as you go**: Write tests for each module before moving to the next

4. **Keep files short**: 
   - No file should exceed 300 lines
   - If it does, split it

5. **Document invariants**: Every class should document its invariants in the docstring

---

## Step 5: Critical Requirements Checklist

Your implementation MUST satisfy these requirements from the issues analysis:

### CNF Handling (CRIT-01, CRIT-02, CRIT-03)
- [ ] Literal pairs: each fluent creates TWO variables (l, neg_l)
- [ ] Mutex constraints added automatically at pair creation
- [ ] Only 3 valid states per pair, not 4
- [ ] Projection set tracked (primary vars only)
- [ ] Model counting projects onto primary variables

### Update Logic (CRIT-04, CRIT-05)
- [ ] NO `refine_clauses_by_intersection` - use exclusion clauses instead
- [ ] Success update adds exclusion clauses for unsatisfied literals
- [ ] Failure update adds coverage clauses for unsatisfied literals
- [ ] Clear separation of constraint sets vs CNF clauses

### Parallel Execution (CRIT-06)
- [ ] Workers do NOT trust parent's solution cache
- [ ] Use solver's native assumptions, not CNF modification
- [ ] Context serialization is complete and correct

### Negation Handling (CRIT-07)
- [ ] Clear documentation of ¬ (literal) vs - (CNF) negation
- [ ] No double-negation bugs
- [ ] Validation of literal format

---

## Step 6: Validation

After implementation, verify:

1. **Invariant Checks**:
```python
def validate_cnf(mgr):
    # Every pair has mutex
    # Primary/auxiliary types correct
    # No conflicting unit clauses
    # Formula is satisfiable
```

2. **Model Count Sanity**:
```python
# For n fluent pairs with no constraints:
# Projected count should be 2^n (primary vars)
# NOT 4^n (all vars) or 3^n (states per pair)
```

3. **Parallel = Sequential**:
```python
# Same inputs should give same outputs
# regardless of parallel vs sequential execution
```

---

## Key Principles

1. **Correctness over Performance**: Get it right first, then optimize
2. **Explicit over Implicit**: No magic, clear data flow
3. **Small over Large**: Many small focused modules, not one giant file
4. **Tested over Untested**: If it's not tested, it's broken

---

## Example: What Success Looks Like

### Before (Current - BAD)
```python
# information_gain.py - 900+ lines, mixed concerns
class InformationGainLearner:
    def _update_success(self, ...):
        # 80 lines mixing:
        # - State conversion
        # - Literal satisfaction checking
        # - Constraint set updates (WRONG)
        # - CNF updates (some right, some wrong)
        # - Effect tracking
        # - Logging
```

### After (New - GOOD)
```python
# learner.py - <200 lines, orchestration only
class InformationGainLearner:
    def _update_success(self, action: str, binding: ObjectBinding, 
                        observation: SuccessObservation) -> None:
        """Update model after successful action."""
        model = self.action_models[action]
        
        # Delegate to focused update function
        updates = compute_success_updates(
            model=model,
            observation=observation,
            grounding=self.grounding
        )
        
        # Apply updates
        model.apply_updates(updates)
        
        # Invalidate caches
        self._invalidate_caches(action)
```

```python
# model/update.py - focused update logic
def compute_success_updates(model: ActionModel, 
                            observation: SuccessObservation,
                            grounding: GroundingContext) -> ModelUpdates:
    """Compute updates for successful action (pure function)."""
    satisfied = get_satisfied_literals(model.pre, observation.state, grounding)
    unsatisfied = model.pre - satisfied
    
    return ModelUpdates(
        pre_refinement=satisfied,
        exclusions=unsatisfied,  # These are NOT preconditions
        effect_updates=compute_effect_updates(observation)
    )
```

---

## Final Notes

- The current implementation is **fundamentally broken** for model counting
- Do NOT try to "fix" the current code - rewrite from scratch
- Use the architecture designs from sub-agents as your guide
- Ask clarifying questions if anything is unclear
- Commit frequently with meaningful messages

Good luck! This is a significant refactoring but will result in correct, maintainable code.
