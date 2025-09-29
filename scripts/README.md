# Scripts Directory

## Essential Scripts

### Experiment Running
- **`run_experiments.py`** - Main experiment runner using YAML configs
- **`run_test_suite.py`** - Runs the complete test suite

### OLAM Validation
- **`olam_paper_validation.py`** - Comprehensive OLAM paper validation against Lamanna et al.
- **`olam_learning_trace.py`** - Shows detailed OLAM learning behavior trace

### Analysis & Reporting
- **`analyze_results.py`** - Analyzes experiment results from JSON/CSV outputs
- **`benchmark_performance.py`** - Performance benchmarking for algorithms

### Testing & Coverage
- **`run_coverage.py`** - Runs test coverage analysis
- **`simple_coverage_report.py`** - Generates simplified coverage reports

## Removed Scripts (Redundant/Obsolete)

The following scripts were removed as redundant or obsolete:
- `validate_olam_rover.py` - Used incompatible Rover domain
- `validate_olam_hypothesis_space.py` - Superseded by olam_paper_validation.py
- `validate_olam_simple.py` - Superseded by olam_learning_trace.py
- `test_mock_experiment.py` - Used obsolete mock environment
- `run_long_mock_experiment.py` - Used obsolete mock environment
- `debug_experiment.py` - Development/debugging only
- `run_extended_experiment.py` - Early testing, superseded

## Usage Examples

### Run OLAM Validation
```bash
python scripts/olam_paper_validation.py
```

### Run Full Experiment
```bash
python scripts/run_experiments.py configs/olam_blocksworld.yaml
```

### Analyze Results
```bash
python scripts/analyze_results.py results/
```

### Run Tests with Coverage
```bash
python scripts/run_coverage.py
```