# Information Gain Algorithm Implementation Plan

## 🎯 Document Purpose
A phase-based implementation guide for the CNF-based Information-Theoretic action model learning algorithm. Each phase includes specific tasks, model recommendations, initialization prompts, and success criteria.

## 📍 Document Maintenance Protocol
**CRITICAL**: This document maintains ONLY current phase information. When a phase is completed:
1. Archive completed phase details to `COMPLETED_PHASES.md` in same directory
2. Update initialization prompt with phase completion summary
3. Remove outdated implementation details
4. Keep only relevant context for next phase

## 📊 Overall Progress

| Phase | Status | Duration |
|-------|--------|----------|
| Phase 1: Core Data Structures | ✅ Complete | 1 session |
| Phase 2: CNF & Update Rules | ✅ Complete | 1 session |
| Phase 3: Information Gain | 🔵 Current | 3-4 sessions |
| Phase 4: Integration & Testing | ⏳ Pending | 2-3 sessions |
| Phase 5: Performance | ⏳ Pending | 2 sessions |
| Phase 6: Comparative Analysis | ⏳ Pending | 1-2 sessions |

---

## 🔵 Current Phase: Phase 3 - Information Gain Calculation & Action Selection

**Model**: Sonnet 4.5 (straightforward implementation)
**Duration**: 3-4 sessions
**Previous Phases**: Phase 1 & 2 complete - Core structures, CNF formulas, and update rules implemented

### Phase 2 Completion Summary

**What was completed**:
- ✅ CNF Manager integration for formula management
- ✅ Success update rules (preconditions + effects)
- ✅ Failure update rules (constraint addition)
- ✅ CNF formula construction from constraint sets
- ✅ State format conversion (PDDL ↔ internal)
- ✅ 38 passing tests (100% pass rate)

**Key achievements**:
- All update rules match algorithm specification precisely
- Negative preconditions properly handled in CNF
- eff_maybe sets correctly become disjoint after observations
- Clean separation between internal and external formats

**What's still placeholder**:
- `select_action()` - Returns first action (Phase 3 will add info gain logic)

### Phase 3 Tasks

#### 1. Implement Applicability Probability Calculation
Calculate probability that action is applicable in current state:

```python
# P(a applicable in s) = |models of cnf_pre?(a) where bindP(s,O) is true| / |all models|
# Use SAT model counting on CNF formula
```

- [ ] Implement `_calculate_applicability_probability(action, state)` method
- [ ] Use CNFManager's model counting capabilities
- [ ] Handle edge cases (empty formulas, contradictions)
- [ ] Test with various precondition certainty levels

#### 2. Implement Potential Information Gain Functions
Calculate expected reduction in uncertainty:

```python
# Info gain from successful execution
gain_success = entropy(current_model) - entropy(model_after_success)

# Info gain from failed execution
gain_failure = entropy(current_model) - entropy(model_after_failure)

# Expected information gain
E[gain] = P(success) * gain_success + P(failure) * gain_failure
```

- [ ] Implement `_calculate_entropy(action)` method
- [ ] Implement `_calculate_potential_gain_success(action, state)` method
- [ ] Implement `_calculate_potential_gain_failure(action, state)` method
- [ ] Implement `_calculate_expected_information_gain(action, state)` method

#### 3. Implement Action Selection Strategies
Replace placeholder with actual selection logic:

```python
# Greedy: Select action with maximum expected information gain
# Epsilon-greedy: Explore with probability ε, exploit otherwise
# Boltzmann: Probabilistic selection based on gain values
```

- [ ] Implement greedy selection in `select_action()`
- [ ] Add epsilon-greedy variant (configurable)
- [ ] Add Boltzmann/softmax selection (optional)
- [ ] Test selection strategies with different scenarios

#### 4. Handle Edge Cases
- [ ] Actions with no uncertainty (known preconditions)
- [ ] States where no actions are possibly applicable
- [ ] Tie-breaking when multiple actions have same gain
- [ ] Computational limits for large CNF formulas

#### 5. Integration and Validation
- [ ] Ensure convergence detection works with info gain
- [ ] Validate that high-gain actions actually reduce uncertainty
- [ ] Test on domains with negative preconditions (rover)
- [ ] Compare with random action selection baseline

### What to Avoid

- **Don't** implement caching optimizations yet (Phase 5)
- **Don't** modify core update rules from Phase 2
- **Don't** add approximate counting yet (Phase 5)
- **Don't** implement full experiment runner integration (Phase 4)

### Success Criteria

- [ ] Information gain correctly calculated from CNF formulas
- [ ] Action selection chooses high-information actions
- [ ] Algorithm converges faster than random selection
- [ ] All Phase 1-2 tests still pass
- [ ] New tests validate info gain calculations
- [ ] Handles domains with negative preconditions

### Initialization Prompt for Phase 3

```
I need to implement Phase 3 of the Information Gain Algorithm.

Phase 1 & 2 Status: ✅ COMPLETE
- Core data structures implemented
- Binding functions working correctly
- CNF formulas and update rules implemented
- 38 tests passing (100% pass rate)

Phase 3 Goal: Implement information gain calculation and action selection

Read:
- @docs/information_gain_algorithm/INFORMATION_GAIN_ALGORITHM.md (sections: Information Gain, Action Selection)
- @docs/information_gain_algorithm/COMPLETED_PHASES.md (Phase 1-2 summaries)
- @docs/information_gain_algorithm/IMPLEMENTATION_PLAN.md (phase 3 - completion)
- @src/algorithms/information_gain.py (current implementation)
- @tests/test_information_gain.py (existing tests)

Tasks:
1. Implement applicability probability using SAT model counting
2. Calculate entropy and potential information gain
3. Implement expected information gain calculation
4. Replace placeholder select_action() with greedy selection
5. Add alternative selection strategies (epsilon-greedy, Boltzmann)

Focus: Correct information gain calculation that drives learning efficiency.
```

### Test Requirements for Phase 3

1. **Applicability Probability Tests**:
   - Test probability calculation with known preconditions
   - Test with partial knowledge (constraints)
   - Test edge cases (empty CNF, contradictions)
   - Test with negative preconditions

2. **Information Gain Tests**:
   - Test entropy calculation
   - Test potential gain calculations
   - Test expected information gain
   - Verify gain is positive for uncertain actions

3. **Action Selection Tests**:
   - Test greedy selection chooses max gain action
   - Test epsilon-greedy exploration
   - Test handling of ties
   - Test convergence improvement over random

### Key Algorithm Sections to Reference

**From INFORMATION_GAIN_ALGORITHM.md**:
- Lines 203-240: Information gain calculation
- Lines 242-280: Action selection strategies
- Lines 282-320: Applicability probability
- Lines 322-360: Entropy and model counting

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
1. Run `make test` to ensure no regressions
2. execute the file maintenance protocol at the start of this file
3. Document any blockers or design decisions

---

## 🔍 Phase 3 Validation Checkpoints

### During Implementation
- [ ] Applicability probability correctly uses model counting
- [ ] Entropy calculation matches information theory principles
- [ ] Expected information gain combines success/failure gains
- [ ] Action selection prioritizes high-information actions

### After Implementation
- [ ] All Phase 1-2 tests still pass (38 tests)
- [ ] New information gain tests pass
- [ ] Algorithm converges faster than random baseline
- [ ] `make test` shows no regressions
- [ ] Manual inspection: Selected actions reduce model uncertainty

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
