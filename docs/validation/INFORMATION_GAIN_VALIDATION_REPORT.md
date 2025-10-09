# Information Gain Algorithm Validation Report

## Executive Summary

**INFORMATION GAIN IMPLEMENTATION SHOWS PARTIAL VALIDATION** ⚠

Validation Score: **1/4 behaviors confirmed**

Only information gain-based action selection is working correctly. Critical issues discovered:
1. ✗ Hypothesis space reduction - NO reduction observed
2. ✗ Model entropy decrease - NO decrease observed
3. ✓ Information gain-based selection - Working correctly (100% greedy)
4. ✗ Ground truth convergence - Very low accuracy (F1=0.07)

**Status**: Implementation requires debugging before use in experiments.

## Experiment Details

- **Domain**: Blocksworld
- **Problem**: p01.pddl (3 blocks, 24 grounded actions)
- **Iterations**: 100 learning iterations
- **Convergence**: Aggressive parameters (window=10, epsilon=0.01, threshold=0.95)
- **Environment**: Real PDDL execution
- **Date**: October 9, 2025

## Evidence of Behavior

### 1. Hypothesis Space Reduction ✗ FAILED

**Expected**: CNF satisfying assignments should decrease as observations constrain the model

**Actual Result**:
```
Initial hypothesis space: 2,099,200 satisfying assignments
Final hypothesis space: 2,099,200 satisfying assignments
Reduction: 0 assignments (0.0%)

Hypothesis Space Over Time (CNF Satisfying Assignments):
  Iter   0: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2,099,200 (100.0% remaining)
  Iter  10: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2,099,200 (100.0% remaining)
  Iter  20: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2,099,200 (100.0% remaining)
  ...
  Iter 100: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2,099,200 (100.0% remaining)
```

**Analysis**: The hypothesis space never decreased. Observations are not constraining the CNF formulas.

**Diagnostic Data**:
- CNF formulas built: 0 clauses for all actions throughout execution
- All actions started and ended with no CNF constraints
- Observations collected but not translated to CNF clauses

### 2. Model Entropy Decrease ✗ FAILED

**Expected**: Entropy should decrease as uncertainty is resolved through learning

**Actual Result**:
```
Initial entropy: 0.00
Final entropy: 0.00
Reduction: 0.00 (0.0%)

Note: Entropy ≈ log(hypothesis_space), so both metrics track together
```

**Analysis**: Entropy remained at 0.00 throughout, consistent with no hypothesis space reduction. This suggests the entropy calculation may be incorrect or the hypothesis space representation is flawed.

### 3. Information Gain-Based Selection ✓ CONFIRMED

**Expected**: Algorithm should select actions with positive information gain using greedy strategy

**Actual Result**:
```
Actions selected with positive IG: 100/100
Greedy selection percentage: 100.0%

First 10 Action Selections (with information gain values):
  Iter   1: pick-up(a)   IG=1.5000 (24 candidates)
  Iter   2: pick-up(a)   IG=1.5000 (24 candidates)
  Iter   3: pick-up(a)   IG=1.5000 (24 candidates)
  ...
  Iter  10: pick-up(a)   IG=1.5000 (24 candidates)
```

**Analysis**: Action selection mechanism is working correctly - all selections had positive information gain. However, the algorithm got stuck selecting the same action repeatedly because the information gain never changed (suggesting the model never updated).

**Issue**: While greedy selection works, the repeated selection of the same failing action indicates the algorithm is not learning from observations.

### 4. Ground Truth Model Convergence ✗ FAILED

**Expected**: Learned model should converge toward ground truth with high F1 scores (≥0.8)

**Actual Result**:
```
Model Accuracy by Action Schema:
  pick-up:
    Preconditions: P=0.20, R=0.67, F1=0.31
    Add Effects: P=0.00, R=0.00, F1=0.00
    Del Effects: P=0.00, R=0.00, F1=0.00
    Overall F1: 0.10

  put-down:
    Preconditions: P=0.10, R=1.00, F1=0.18
    Add Effects: P=0.00, R=0.00, F1=0.00
    Del Effects: P=0.00, R=0.00, F1=0.00
    Overall F1: 0.06

  stack:
    Preconditions: P=0.10, R=1.00, F1=0.18
    Add Effects: P=0.00, R=0.00, F1=0.00
    Del Effects: P=0.00, R=0.00, F1=0.00
    Overall F1: 0.06

  unstack:
    Preconditions: P=0.10, R=0.67, F1=0.17
    Add Effects: P=0.00, R=0.00, F1=0.00
    Del Effects: P=0.00, R=0.00, F1=0.00
    Overall F1: 0.06

Aggregate Model Accuracy:
  Average Precision: 0.12
  Average Recall: 0.83
  Average F1-Score: 0.07
```

**Analysis**:
- Very low F1 scores (0.07 average) indicate minimal learning
- No effects were learned at all (F1=0.00 for all add/delete effects)
- High recall but low precision suggests overly permissive learned models
- The algorithm maintained overly broad hypotheses instead of refining them

## Learning Trace Analysis

**Action Selection Pattern**:
```
All 100 iterations selected: pick-up(a)
All 100 attempts resulted in: FAILURE
Information gain remained constant at: 1.500
```

**Key Observations**:
1. Algorithm repeatedly selected the same failing action
2. Information gain value never changed, suggesting model never updated
3. 0% success rate (0/100 actions succeeded)
4. No observable learning occurred

**Expected Behavior**:
- After observing `pick-up(a)` fail, the algorithm should update its model
- Information gain for `pick-up(a)` should decrease as uncertainty resolves
- Other actions should become more attractive
- Success rate should increase as the model improves

**Actual Behavior**:
- Model never updated despite 100 failure observations
- Same action selected repeatedly with identical information gain
- Algorithm appears stuck in a non-learning state

## Root Cause Analysis

### Primary Issues

1. **CNF Formula Construction**: CNF formulas are not being built from observations
   - All actions show "0 clauses, 0 unique variables" throughout execution
   - The `_build_cnf_formula()` method may not be properly encoding observations

2. **Observation Integration**: The `observe()` method may not be updating internal data structures
   - Despite 100 failure observations for `pick-up(a)`, no constraints were added
   - Pre/post conditions may not be properly tracked

3. **Effect Learning**: No effects were learned at all
   - All add/delete effects have F1=0.00
   - Effect learning mechanism appears non-functional

4. **Entropy Calculation**: Entropy remained at 0.00 throughout
   - May indicate incorrect entropy calculation
   - Or hypothesis space may not be properly initialized

### Secondary Issues

1. **Action Filtering**: No executable/non-executable distinction observed
   - All actions remained candidates throughout
   - Filtering mechanism may not be using CNF satisfiability

2. **Convergence Detection**: Algorithm ran to max iterations
   - Convergence criteria never triggered despite no learning occurring
   - May need better early stopping conditions

## Comparison with Expected Behavior

| Expected Behavior | Status | Evidence |
|-------------------|--------|----------|
| CNF satisfying assignments decrease | ✗ Failed | Remained at 2,099,200 |
| Model entropy decreases | ✗ Failed | Remained at 0.00 |
| Information gain-based selection | ✓ Confirmed | 100% greedy selections |
| Varied action selection | ✗ Failed | Same action selected 100 times |
| Learning from failures | ✗ Failed | No model updates observed |
| Effect learning | ✗ Failed | All F1=0.00 for effects |
| Ground truth convergence | ✗ Failed | F1=0.07 (very low) |
| Hypothesis space reduction | ✗ Failed | No reduction occurred |

## Critical Implementation Issues

### Must Fix Before Paper Experiments

1. **CNF Observation Encoding** (src/algorithms/information_gain.py)
   - Debug why observations don't create CNF clauses
   - Verify `_add_failure_observation()` and `_add_success_observation()` methods
   - Check CNF variable mapping and clause construction

2. **Hypothesis Space Tracking** (src/algorithms/information_gain.py)
   - Verify hypothesis space calculation uses CNF satisfiability
   - Ensure `_calculate_entropy()` correctly computes from CNF formulas
   - Check that `pre`, `add`, and `delete` sets are properly maintained

3. **Effect Learning** (src/algorithms/information_gain.py)
   - Implement or fix effect learning mechanism
   - Verify add/delete effects are extracted from successful observations
   - Ensure effects are incorporated into CNF formulas

4. **Model Export** (src/algorithms/information_gain.py)
   - Verify `get_learned_model()` correctly extracts learned constraints
   - Check that CNF formulas are properly decoded to preconditions/effects
   - Ensure format matches ModelValidator expectations

### Testing Recommendations

1. **Unit Tests Needed**:
   - Test CNF formula construction from single observation
   - Test hypothesis space counting with simple CNF formulas
   - Test entropy calculation with known distributions
   - Test effect extraction from successful observations

2. **Integration Tests Needed**:
   - Test full learning cycle with 1-2 actions
   - Verify observations create CNF clauses
   - Verify hypothesis space decreases
   - Verify model converges to ground truth

3. **Debug Instrumentation**:
   - Add detailed logging in `observe()` method
   - Log CNF clause additions
   - Log hypothesis space size after each update
   - Log information gain calculations in detail

## Recommendation

**DO NOT USE** this implementation for paper experiments until critical issues are fixed.

### Next Steps

1. **Immediate** (Required for basic functionality):
   - Debug CNF formula construction from observations
   - Fix hypothesis space tracking
   - Implement effect learning mechanism
   - Add comprehensive unit tests

2. **Short-term** (Required for experiments):
   - Validate on simple 2-action domain
   - Verify hypothesis space reduction occurs
   - Achieve F1 > 0.8 on blocksworld
   - Compare with OLAM baseline

3. **Long-term** (Nice to have):
   - Optimize CNF solving performance
   - Add early stopping for non-learning cases
   - Implement better action selection strategies
   - Add detailed learning diagnostics

## Key Code Locations

- Algorithm implementation: `src/algorithms/information_gain.py`
- CNF manager: `src/core/cnf_manager.py`
- Validation script: `scripts/validate_information_gain.py`
- Test suite: `tests/algorithms/test_information_gain_convergence.py`

## Data Files

- Validation results: `validation_logs/information_gain_validation_20251009_185254.json`
- Full execution log: Available in script output

---

*Report generated: October 9, 2025*
*Validation reveals critical implementation issues requiring immediate attention*
*Algorithm is NOT ready for paper experiments*
