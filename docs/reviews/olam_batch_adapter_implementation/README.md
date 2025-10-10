# OLAM Batch Adapter - Implementation Files

This directory contains the implementation files for the simplified OLAM batch adapter proof-of-concept.

## Files

### Core Implementation

**`batch_learner.py`** (112 lines)
- Base class for batch-oriented algorithms
- Defines the `BatchAlgorithmAdapter` interface
- Raises `NotImplementedError` for per-iteration methods (`select_action`, `observe`)
- Defines `run_experiment()` abstract method

**`olam_batch_adapter.py`** (400 lines)
- Simplified OLAM adapter implementing `BatchAlgorithmAdapter`
- Runs OLAM's `learn()` method autonomously
- Parses learned model from OLAM's internal state
- Key simplifications vs original 1089-line adapter:
  - No monkey-patching of OLAM internals
  - Minimal state synchronization
  - Disables eval_log methods (file dependencies)

### Test Configurations

**`olam_depots_batch_test.yaml`**
- Test configuration for depots domain
- Result: ✅ SUCCESS (11.26s, 51 iterations, 918 actions)

**`olam_blocksworld_batch_test.yaml`**
- Test configuration for blocksworld domain
- Result: ⚠️ PARTIAL (hit OLAM preprocessing error at iteration ~3)

## Integration

To integrate these files into the main codebase:

1. Copy `batch_learner.py` to `src/algorithms/`
2. Copy `olam_batch_adapter.py` to `src/algorithms/`
3. Update `src/experiments/runner.py` with batch algorithm detection (see worktree)
4. Copy config files to `configs/`
5. Register "olam_batch" algorithm in algorithm factory

## Notes

- Full implementation in git worktree: `/tmp/olam_batch_refactor/`
- Branch: `feature/olam-batch-adapter`
- See parent directory's `olam_batch_adapter_review.md` for complete analysis

## Status

**Proof of concept complete.** Ready for decision on whether to:
- Keep both adapters (Option A)
- Go fully external with subprocess approach (Option B)
- Fix preprocessing in batch adapter (Option C)
