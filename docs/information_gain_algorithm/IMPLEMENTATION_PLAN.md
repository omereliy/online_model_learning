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
| Phase 3: Information Gain | ✅ Complete | 1 session |
| Phase 4: Integration & Testing | 🔵 Current | 2-3 sessions |
| Phase 5: Performance | ⏳ Pending | 2 sessions |
| Phase 6: Comparative Analysis | ⏳ Pending | 1-2 sessions |

---

## 🔵 Current Phase: Phase 4 - Integration & Testing

**Model**: Sonnet 4.5 (integration and testing focus)
**Duration**: 2-3 sessions
**Previous Phases**: Phase 1-3 complete - Full information gain algorithm implemented

### Phase 3 Completion Summary

**What was completed**:
- ✅ Applicability probability calculation using SAT model counting
- ✅ Entropy calculation for action model uncertainty
- ✅ Potential information gain from success/failure
- ✅ Expected information gain calculation
- ✅ Three action selection strategies (greedy, epsilon-greedy, Boltzmann)
- ✅ Replaced placeholder select_action() with full implementation
- ✅ ~60 total tests across all phases

**Key achievements**:
- Information gain correctly drives action selection
- Handles edge cases (empty state, no constraints, zero gains)
- Numerical stability in probability calculations
- Main test suite still passes (51 tests)

**Ready for Phase 4**:
- Full algorithm implementation complete
- Ready for integration with experiment runner
- Can be compared against OLAM and other baselines

### Phase 4 Tasks

#### 1. Integration with Experiment Runner
- [ ] Ensure InformationGainLearner works with experiment runner
- [ ] Add configuration options for selection strategies
- [ ] Integrate with metrics collection
- [ ] Test end-to-end pipeline

#### 2. Comprehensive Testing
- [ ] Test on multiple domains (blocksworld, gripper, rover, depots)
- [ ] Validate convergence behavior
- [ ] Compare against random baseline
- [ ] Stress test with large state spaces

#### 3. Performance Profiling
- [ ] Identify bottlenecks in SAT counting
- [ ] Measure time per action selection
- [ ] Profile memory usage
- [ ] Document performance characteristics

#### 4. Documentation
- [ ] Update API documentation
- [ ] Add usage examples
- [ ] Document configuration options
- [ ] Create comparison with OLAM

### Success Criteria

- [ ] Algorithm runs in experiment framework
- [ ] Converges on all test domains
- [ ] Outperforms random selection
- [ ] Performance acceptable for experiments
- [ ] Documentation complete

### Initialization Prompt for Phase 4

```
I need to implement Phase 4 of the Information Gain Algorithm.

Phase 1-3 Status: ✅ COMPLETE
- Full information gain algorithm implemented
- Applicability probability, entropy, and gain calculations working
- Three selection strategies implemented
- ~60 tests passing

Phase 4 Goal: Integration, testing, and validation

Read:
- @docs/information_gain_algorithm/COMPLETED_PHASES.md (Phase 1-3 summaries)
- @src/algorithms/information_gain.py (complete implementation)
- @src/experiments/runner.py (experiment framework)
- @tests/ (existing test structure)

Tasks:
1. Integrate with experiment runner
2. Test on multiple domains
3. Profile performance
4. Complete documentation
```

---

## Previous Phase Archives

For detailed information about completed phases, see [COMPLETED_PHASES.md](COMPLETED_PHASES.md).

- **Phase 1**: Core Data Structures & State Management
- **Phase 2**: CNF Formula Management & Update Rules
- **Phase 3**: Information Gain Calculation & Action Selection

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
- **Test Suite**: `tests/algorithms/test_information_gain.py`
- **Test Patterns**: `docs/QUICK_REFERENCE.md  #code-patterns`
- **CNF Tests**: `tests/core/test_cnf_manager.py`

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
