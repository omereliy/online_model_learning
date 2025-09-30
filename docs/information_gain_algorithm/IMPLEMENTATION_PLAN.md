# Information Gain Algorithm Implementation Plan

## 🎯 Document Purpose
A phase-based implementation guide for the CNF-based Information-Theoretic action model learning algorithm. Each phase includes specific tasks, model recommendations, initialization prompts, and success criteria.

## 📍 Document Maintenance Rule
**CRITICAL**: This document maintains ONLY current phase information. When a phase is completed:
1. Archive completed phase details to `COMPLETED_PHASES.md` in same directory
2. Update initialization prompt with phase completion summary
3. Remove outdated implementation details
4. Keep only relevant context for next phase

## 📊 Overall Progress

| Phase | Status | Duration |
|-------|--------|----------|
| Phase 1: Core Data Structures | ✅ Complete | 1 session |
| Phase 2: CNF & Update Rules | 🔵 Current | 3-4 sessions |
| Phase 3: Information Gain | ⏳ Pending | 3-4 sessions |
| Phase 4: Integration & Testing | ⏳ Pending | 2-3 sessions |
| Phase 5: Performance | ⏳ Pending | 2 sessions |
| Phase 6: Comparative Analysis | ⏳ Pending | 1-2 sessions |

---

## 🔵 Current Phase: Phase 2 - CNF Formula Construction & Update Rules

**Model**: Opus 4.1 (complex logic, mathematical correctness)
**Duration**: 3-4 sessions
**Previous Phase**: Phase 1 complete - Core data structures and binding functions implemented

### Phase 1 Completion Summary

**What was completed**:
- ✅ `InformationGainLearner` class with all state variables
- ✅ Binding functions (`bindP`, `bindP_inverse`) working correctly
- ✅ La generation for all actions
- ✅ Observation recording infrastructure
- ✅ 24 passing tests with independent La validation

**Key files**:
- `src/algorithms/information_gain.py` - Main implementation (442 lines)
- `tests/test_information_gain.py` - Test suite (395 lines, 25 tests)

**What's placeholder**:
- `select_action()` - Returns first action (Phase 3 will add info gain logic)
- `observe()` - Only records observations (Phase 2 will add update rules)

### Phase 2 Tasks

#### 1. Integrate with CNFManager
- [ ] Import and initialize `CNFManager` from `src/core/cnf_manager.py`
- [ ] Create CNF formula for each action's precondition constraints (`pre?`)
- [ ] Handle PDDL ↔ internal negation conversion (¬ vs `(not ...)`)

#### 2. Implement Success Update Rules
When action succeeds from state s → s':

```python
# Update preconditions (narrow down)
pre(a) = pre(a) ∩ bindP(s, O)

# Update confirmed effects
eff+(a) = eff+(a) ∪ bindP(s' \ s, O)  # Added fluents
eff-(a) = eff-(a) ∪ bindP(s \ s', O)  # Deleted fluents

# Update possible effects
eff?+(a) = eff?+(a) ∩ bindP(s ∩ s', O)  # Keep unchanged
eff?-(a) = eff?-(a) \ bindP(s ∪ s', O)  # Remove if was/became true

# Update constraint sets
pre?(a) = {B ∩ bindP(s, O) | B ∈ pre?(a)}
```

- [ ] Implement `_update_success()` method with all rules
- [ ] Test with realistic state transitions from depots domain

#### 3. Implement Failure Update Rules
When action fails from state s:

```python
# Add constraint: at least one unsatisfied literal is a precondition
pre?(a) = pre?(a) ∪ {pre(a) \ bindP(s, O)}
```

- [ ] Implement `_update_failure()` method
- [ ] Build CNF formula from constraint sets
- [ ] Test constraint accumulation over multiple failures

#### 4. CNF Construction from Constraints
Build CNF formula representing precondition uncertainty:

```python
cnf_pre?(a) = ⋀(⋁xl) for B ∈ pre?(a), l ∈ B
# Each constraint set B becomes a clause (disjunction)
```

- [ ] Implement `_build_cnf_formula(action_name)` method
- [ ] Convert internal ¬ notation to CNFManager format
- [ ] Handle both positive and negative literals in clauses

#### 5. State Format Conversion
- [ ] Implement `_state_to_internal(state)` - converts UP/set format to internal
- [ ] Implement `_internal_to_pddl(literals)` - converts internal ¬ to PDDL `(not ...)`
- [ ] Test conversions preserve semantics

### What to Avoid

- **Don't** implement model counting yet (Phase 3)
- **Don't** implement applicability probability calculation (Phase 3)
- **Don't** add performance optimizations (Phase 5)
- **Don't** modify CNFManager internals (use as-is)

### Success Criteria

- [ ] CNF formulas correctly updated on observations
- [ ] Negative preconditions properly handled in CNF
- [ ] All update rules match algorithm specification
- [ ] Test `test_eff_maybe_sets_become_disjoint_after_success` passes (currently skipped)
- [ ] New tests validate update rules correctness
- [ ] All existing tests still pass

### Initialization Prompt for Phase 2

```
I need to implement Phase 2 of the Information Gain Algorithm.

Phase 1 Status: ✅ COMPLETE
- Core data structures implemented
- Binding functions (bindP, bindP_inverse) working
- La generation correct
- 24 tests passing

Phase 2 Goal: Implement CNF formulas and observation update rules

Read:
- @docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md (sections: Update Rules, CNF Construction)
- @docs/information_gain_algorithm/COMPLETED_PHASES.md (Phase 1 summary)
- @src/core/cnf_manager.py (CNF integration)
- @src/algorithms/information_gain.py (current implementation)

Tasks:
1. Integrate CNFManager for formula management
2. Implement success update rules (preconditions + effects)
3. Implement failure update rules (constraint addition)
4. Build CNF formulas from constraint sets
5. Handle PDDL ↔ internal negation conversion

Focus: Mathematical correctness of update rules, not performance.
```

### Test Requirements for Phase 2

1. **Update Rule Tests**:
   - Test precondition narrowing on success
   - Test effect learning (add/delete)
   - Test eff_maybe set separation
   - Test constraint accumulation on failure

2. **CNF Construction Tests**:
   - Test CNF building from constraints
   - Test negative literal handling
   - Test formula validity

3. **Integration Tests**:
   - Enable skipped test: `test_eff_maybe_sets_become_disjoint_after_success`
   - Test realistic observation sequences
   - Verify invariants maintained

### Key Algorithm Sections to Reference

**From INFORMATION_GAIN_ALGORITHM.md**:
- Lines 107-129: Success update rules
- Lines 131-139: Failure update rules
- Lines 141-201: CNF formula construction
- Lines 46-78: Negative precondition handling

---

## 🚫 Common Pitfalls to Avoid

### Algorithm Understanding
1. **State Format Confusion**: Always verify UP vs internal format conversions
2. **Negative Literal Handling**: ¬p means p ∉ state, not a negated fluent
3. **Binding Semantics**: bindP and bindP_inverse have specific mathematical definitions

### Implementation Details
4. **CNF Variable Mapping**: Maintain consistent fluent→variable mapping
5. **Lifted vs Grounded**: Use proper binding functions for conversions
6. **Set Operations**: Python set difference operator: `\` not `/`

### Testing Practices
7. **Test Independence**: Each test must be isolated, no shared state
8. **Domain Compatibility**: Test with both simple (drive) and complex (lift) actions
9. **Convergence Validation**: Verify algorithm converges correctly

---

## 📝 Progress Tracking Protocol

After each session:
1. Update `COMPLETED_PHASES.md` if phase complete
2. Run `make test` to ensure no regressions
3. Update initialization prompt for next session
4. Document any blockers or design decisions

---

## 🔍 Phase 2 Validation Checkpoints

### During Implementation
- [ ] Update rules match algorithm specification line-by-line
- [ ] CNF formulas correctly represent uncertainty
- [ ] Negative preconditions handled in CNF clauses
- [ ] State conversions preserve semantics

### After Implementation
- [ ] All Phase 1 tests still pass (24 tests)
- [ ] New update rule tests pass
- [ ] Skipped test now passes
- [ ] `make test` shows no regressions
- [ ] Manual inspection: eff_maybe_add and eff_maybe_del separate after observations

---

## 📌 Quick Reference Links

### Core Documentation
- **Algorithm Details**: `docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md`
- **CNF Integration**: `docs/information_gain_algorithm/CNF_SAT_INTEGRATION.md`
- **Development Rules**: `docs/DEVELOPMENT_RULES.md`
- **Completed Phases**: `docs/information_gain_algorithm/COMPLETED_PHASES.md`

### Implementation References
- **Current Implementation**: `src/algorithms/information_gain.py`
- **CNF Manager**: `src/core/cnf_manager.py`
- **PDDL Handler**: `src/core/pddl_handler.py`
- **Base Interface**: `src/algorithms/base_learner.py`

### Testing Resources
- **Test Suite**: `tests/test_information_gain.py`
- **Test Patterns**: `docs/QUICK_REFERENCE.md#code-patterns`
- **CNF Tests**: `tests/test_cnf_manager.py`

---

## 💡 Model Selection Rationale

### Opus 4.1 (Phase 2)
- **Why**: Complex mathematical logic requiring deep understanding
- **Strengths**: Theoretical reasoning, formula manipulation, correctness proofs
- **Focus**: Mathematical correctness over code efficiency
- **Use for**: Update rules implementation, CNF construction

### Sonnet 4.5 (Phases 1, 4-6)
- **Why**: Straightforward implementation and testing
- **Strengths**: Clean code, test writing, performance optimization
- **Focus**: Code quality and engineering best practices

---

## 🔄 Next Phases Preview

### Phase 3: Information Gain Calculation & Action Selection
- Implement applicability probability using SAT model counting
- Create potential gain functions
- Implement expected information gain calculation
- Add greedy and probabilistic selection strategies

### Phase 4: Integration & Testing
- Comprehensive test suite
- Test with negative precondition domains (rover)
- Integrate with experiment runner
- Validate convergence guarantees

### Phase 5: Performance Optimization
- Formula caching
- Incremental CNF updates
- Approximate counting for large formulas

### Phase 6: Comparative Analysis
- Run experiments comparing OLAM, ModelLearner, Information Gain
- Generate performance reports
- Document findings
