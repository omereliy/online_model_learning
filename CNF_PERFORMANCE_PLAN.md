# CNF SAT Performance Enhancement - Implementation Plan

## Working Directory
- **Worktree**: `/tmp/cnf_performance_optimization`
- **Branch**: `feature/cnf-performance-optimization`
- **Base**: `master` (fec154b)

## Phase 1: Cache Base CNF Model Counts (3-4x speedup)

### Problem
Same base CNF counted 3-4 times per action during same iteration:
- `_calculate_applicability_probability()` line 208
- `_calculate_potential_gain_failure()` line 366
- `_calculate_entropy()` line 276 (via get_entropy)

### Solution
Add `_base_cnf_count_cache` dictionary, invalidate after CNF rebuild

### Implementation Steps
1. ✅ Add cache dict to `__init__()` line 129
2. ⏳ Implement `_get_base_model_count()` method
3. ⏳ Update `_calculate_applicability_probability()` line 208
4. ⏳ Update `_calculate_potential_gain_failure()` line 366
5. ⏳ Update `_calculate_entropy()` - modify CNFManager.get_entropy()
6. ⏳ Add cache invalidation in `update_model()` line 773

## Phase 2: Solver Assumptions Instead of Deep Copies (2-3x speedup)

### Problem
- `cnf.copy()` creates expensive deep copies (line 374, 540, 653)
- Each copy duplicates all clauses, variable mappings, metadata

### Solution
Use PySAT's `solve(assumptions=[...])` feature - no copy needed!

### Implementation Steps
1. ⏳ Add `count_models_with_assumptions()` to CNFManager
2. ⏳ Add `state_constraints_to_assumptions()` helper to CNFManager
3. ⏳ Update `_calculate_applicability_probability()` line 236
4. ⏳ Update `_calculate_potential_gain_failure()` line 374-378

## Expected Performance Gains
- **Phase 1**: 3-4x speedup (eliminates redundant base counts)
- **Phase 2**: 2-3x speedup (eliminates deep copy overhead)
- **Combined**: 6-12x overall speedup

## Testing Checkpoints
1. After Phase 1: Run `make test` from worktree
2. After Phase 2: Run `make test` from worktree
3. Performance benchmark: blocksworld p01 before/after

## Key Files
- `/tmp/cnf_performance_optimization/src/algorithms/information_gain.py`
- `/tmp/cnf_performance_optimization/src/core/cnf_manager.py`
