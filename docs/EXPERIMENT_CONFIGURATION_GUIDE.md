# Experiment Configuration Guide for Statistical Validity

## Overview

This guide explains configuration choices for running statistically valid experiments comparing OLAM and Information Gain algorithms. Conservative settings ensure large sample sizes and avoid premature convergence.

## Critical Configuration Parameters

### 1. Max Iterations

**Purpose**: Upper bound on experiment length
**Current defaults**: 200 iterations
**Recommended for full experiments**: 500-1000 iterations

**Rationale**:
- Larger samples improve statistical power
- Some domains require many iterations to learn complete models
- Better to err on side of too much data than too little
- Can always truncate data during analysis

**Configuration**:
```yaml
stopping_criteria:
  max_iterations: 1000  # Conservative: ensure sufficient data
```

### 2. Planner Timeouts (OLAM)

**Purpose**: How long planner can run before timeout
**Current default**: 60 seconds
**Recommended for full experiments**: 120-300 seconds

**Rationale**:
- Complex problems may need longer planning time
- Premature timeouts counted as action failures
- Artificial failures contaminate learning signal
- Better to wait than get false failure data

**Configuration**:
```yaml
algorithm_params:
  olam:
    planner_time_limit: 180  # 3 minutes per planning call
    time_limit_seconds: 7200  # 2 hours total experiment limit
```

### 3. Convergence Detection Settings

#### OLAM Convergence

**Current**: Uses OLAM's internal `model_convergence` flag
**Issue**: May converge prematurely based on OLAM's criteria

**Recommendation**: Disable or make less aggressive
- Set very high `convergence_check_interval` (e.g., 500)
- Or disable convergence checking for data collection runs

**Configuration**:
```yaml
stopping_criteria:
  max_iterations: 1000
  convergence_check_interval: 500  # Check very rarely
  # Alternative: set to 99999 to effectively disable
```

#### Information Gain Convergence

**Current settings** (hardcoded in src/algorithms/information_gain.py:40-43):
```python
MODEL_STABILITY_WINDOW = 10      # ⚠️ TOO SHORT!
INFO_GAIN_EPSILON = 0.01         # ⚠️ MAY BE TOO LOW!
SUCCESS_RATE_THRESHOLD = 0.95    # ⚠️ TOO AGGRESSIVE!
SUCCESS_RATE_WINDOW = 20
```

**Issues**:
1. **MODEL_STABILITY_WINDOW = 10**: Model unchanged for just 10 iterations triggers convergence
   - Early in learning, model may stabilize temporarily then change again
   - 10 iterations is insufficient to confirm true convergence

2. **SUCCESS_RATE_THRESHOLD = 0.95**: 95% success in 20 actions (19/20 successes)
   - Just 1-2 failures in 20 actions triggers convergence
   - Too sensitive - normal learning fluctuations can trigger this

3. **Convergence logic**: Requires ANY 2 of 3 criteria
   - Too aggressive - easy to accidentally satisfy 2 criteria
   - Should require ALL 3 for true convergence

**Recommended values for full experiments**:
```python
MODEL_STABILITY_WINDOW = 50      # 50 iterations without model changes
INFO_GAIN_EPSILON = 0.001        # Lower threshold (more conservative)
SUCCESS_RATE_THRESHOLD = 0.98    # 98% success rate (nearly perfect)
SUCCESS_RATE_WINDOW = 50         # Larger window for stability
```

**Better: Make configurable via YAML** (see Implementation section below)

### 4. Convergence Check Interval

**Purpose**: How often to check convergence
**Current**: 20 iterations
**Recommended for full experiments**: 100 iterations

**Rationale**:
- Convergence checks have computational cost
- Checking every 100 iterations is sufficient
- Reduces overhead while still catching convergence

**Configuration**:
```yaml
stopping_criteria:
  convergence_check_interval: 100
```

## Recommended Full Experiment Configuration

### Template: `configs/full_experiment_template.yaml`

```yaml
experiment:
  name: "full_experiment_{domain}_{algorithm}"
  algorithm: "olam"  # or "information_gain"
  seed: 42

domain_problem:
  domain: "benchmarks/{domain}/domain.pddl"
  problem: "benchmarks/{domain}/p01.pddl"

# Algorithm-specific parameters
algorithm_params:
  olam:
    max_iterations: 1000
    eval_frequency: 10
    planner_time_limit: 180      # 3 minutes per planning attempt
    max_precs_length: 8
    neg_eff_assumption: false
    output_console: false
    random_seed: 42
    time_limit_seconds: 7200     # 2 hours total (safety limit)

  information_gain:
    selection_strategy: "greedy"
    max_iterations: 1000
    # Convergence parameters (conservative settings)
    model_stability_window: 50   # Require 50 stable iterations
    info_gain_epsilon: 0.001     # Very low threshold
    success_rate_threshold: 0.98 # 98% success required
    success_rate_window: 50      # Over 50 actions

# Metrics collection
metrics:
  interval: 1  # Record every action
  window_size: 50

# Stopping criteria - CONSERVATIVE
stopping_criteria:
  max_iterations: 1000           # Primary stopping criterion
  max_runtime_seconds: 7200      # 2 hours safety limit
  convergence_check_interval: 100  # Check infrequently

# Output configuration
output:
  directory: "results/"
  formats: ["csv", "json"]
  save_learned_model: true
```

## Implementation: Making Information Gain Parameters Configurable

Currently, Information Gain convergence parameters are hardcoded class constants. For flexible experimentation, these should be configurable via YAML.

### Proposed Changes to `src/algorithms/information_gain.py`

```python
def __init__(self,
             domain_file: str,
             problem_file: str,
             max_iterations: int = DEFAULT_MAX_ITERATIONS,
             # NEW: Convergence parameters with conservative defaults
             model_stability_window: int = 50,      # Increased from 10
             info_gain_epsilon: float = 0.001,      # Decreased from 0.01
             success_rate_threshold: float = 0.98,  # Increased from 0.95
             success_rate_window: int = 50,         # Increased from 20
             **kwargs):
    """
    Initialize Information Gain learner.

    Args:
        domain_file: Path to PDDL domain file
        problem_file: Path to PDDL problem file
        max_iterations: Maximum learning iterations
        model_stability_window: Iterations without model change for convergence
        info_gain_epsilon: Threshold for low information gain
        success_rate_threshold: Success rate threshold for convergence
        success_rate_window: Window size for success rate calculation
        **kwargs: Additional parameters
    """
    # ... existing validation ...

    # Store convergence parameters (override class constants)
    self.model_stability_window = model_stability_window
    self.info_gain_epsilon = info_gain_epsilon
    self.success_rate_threshold = success_rate_threshold
    self.success_rate_window = success_rate_window

    # ... rest of __init__ ...
```

Then update convergence check methods to use instance variables instead of class constants.

## Experimental Design Recommendations

### For Statistical Validity

1. **Run multiple trials** (5-10 per algorithm)
   - Use different random seeds
   - Collect multiple samples for statistical tests

2. **Disable convergence for data collection**
   - Set `convergence_check_interval: 99999` to effectively disable
   - Let experiments run to max_iterations
   - Ensures equal sample sizes for comparison

3. **Use generous timeouts**
   - Planner timeouts: 120-300 seconds
   - Total experiment timeout: 2-4 hours
   - Better to wait than get contaminated data

4. **Large iteration counts**
   - Minimum 500 iterations per trial
   - Preferably 1000+ for complex domains
   - Can subsample during analysis if needed

### Sample Size Calculation

For detecting medium effect size (Cohen's d = 0.5) with 80% power at α=0.05:
- **Minimum**: 5 trials per algorithm (N=10 total)
- **Recommended**: 10 trials per algorithm (N=20 total)
- **Ideal**: 20+ trials per algorithm (N=40+ total)

### Configuration Strategy

**Phase 1: Exploratory** (quick, convergence enabled)
```yaml
max_iterations: 200
convergence_check_interval: 20
# Use default convergence settings
```

**Phase 2: Data Collection** (full runs, no early stopping)
```yaml
max_iterations: 1000
convergence_check_interval: 99999  # Effectively disabled
planner_time_limit: 180
```

**Phase 3: Final Validation** (publication-ready)
```yaml
max_iterations: 1000
convergence_check_interval: 99999
planner_time_limit: 300
# Run 10+ trials per algorithm
```

## Rationale Summary

### Why Conservative Settings Matter

1. **Statistical Power**
   - Larger samples → more reliable statistics
   - Smaller p-values → stronger conclusions
   - Tighter confidence intervals → more precise estimates

2. **Avoiding False Convergence**
   - Models may temporarily stabilize then change
   - Success rate fluctuates during learning
   - Premature stopping loses valuable data

3. **Fair Comparison**
   - Both algorithms should run for same iterations
   - Equal opportunity to learn complete models
   - No bias from different convergence rates

4. **Robustness**
   - Accounts for domain variability
   - Handles occasional planner timeouts
   - Works across easy and hard problems

### When to Use Convergence Detection

**Use convergence** (aggressive settings):
- Interactive demos
- Quick validation tests
- Resource-constrained environments
- Real-time applications

**Disable convergence** (conservative settings):
- Academic paper experiments
- Algorithm comparison studies
- Statistical analysis
- Publication-quality results

## Quick Reference

### Conservative Settings Checklist

- [ ] `max_iterations: 1000` (or higher)
- [ ] `planner_time_limit: 180` (3+ minutes)
- [ ] `convergence_check_interval: 100` (or 99999 to disable)
- [ ] `model_stability_window: 50` (InfoGain)
- [ ] `info_gain_epsilon: 0.001` (InfoGain)
- [ ] `success_rate_threshold: 0.98` (InfoGain)
- [ ] Multiple trials per algorithm (5-10+)
- [ ] Different random seeds per trial
- [ ] Total runtime budget: 2-4 hours per trial

## References

- Statistical power analysis: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Sample size guidelines: Button et al. (2013). "Power failure: why small sample size undermines the reliability of neuroscience"
- Experimental design: Montgomery, D. C. (2017). Design and Analysis of Experiments
