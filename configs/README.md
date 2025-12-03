# Configuration Files

## ⚠️ Important: OLAM Refactor (Nov 2025)

All OLAM-specific config files have been removed after the major refactor from real-time integration to post-processing approach.

### What Changed
- **Removed**: All `olam_*.yaml` files (used old OLAMAdapter)
- **New Approach**: OLAM now runs via command line with `scripts/analyze_olam_results.py`
- **Example**: See `olam_post_processing_example.yaml` for new approach

## Current Config Files

### Information Gain Configs
- `information_gain_*_300.yaml` - 300 iteration experiments
- `information_gain_*_500iter.yaml` - 500 iteration experiments
- `paper/information_gain_*_1000.yaml` - 1000 iteration experiments for paper

### General Experiment Configs
- `full_validation_experiment.yaml` - Full validation suite
- `paper/challenging_domains.yaml` - Complex domains with more iterations
- `paper/full_benchmark_suite.yaml` - Complete benchmark evaluation
- `paper/quick_test.yaml` - Quick test configuration

## Running Experiments

### Information Gain (Still uses configs)
```bash
python scripts/run_experiment.py --config configs/information_gain_blocksworld_300.yaml
```

### OLAM (New post-processing approach)
```bash
# Direct command line (recommended)
python scripts/analyze_olam_results.py \
    --domain benchmarks/olam-compatible/blocksworld/domain.pddl \
    --problem benchmarks/olam-compatible/blocksworld/p01.pddl \
    --checkpoints 5 10 20 50 100 200 300 \
    --max-iterations 300
```

## Note for Future Development

If you need to run OLAM experiments:
1. DO NOT look for `olam_*.yaml` configs - they've been removed
2. DO NOT try to use `OLAMAdapter` - it's been deleted
3. DO use `scripts/analyze_olam_results.py` for OLAM experiments
4. See `docs/external_repos/OLAM_interface_new.md` for full documentation