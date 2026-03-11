# Configuration Files

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

```bash
python scripts/run_experiment.py --config configs/information_gain_blocksworld_300.yaml
```
