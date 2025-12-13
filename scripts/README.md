# Scripts Directory

## Quick Start: Run Information Gain Experiments

```bash
# Activate virtual environment
source venv/bin/activate

# Run quick test (5 domains, 100 iterations)
python scripts/run_full_experiments.py --mode quick

# Run standard benchmarks (8 domains, 500 iterations)
python scripts/run_full_experiments.py --mode standard

# Run custom domains
python scripts/run_full_experiments.py --domains blocksworld depots --iterations 400
```

---

## Core Scripts

### 1. Run Information Gain Experiments
Run the Information Gain algorithm on PDDL benchmark domains.

```bash
# Predefined modes
python scripts/run_full_experiments.py --mode quick      # 5 domains, 100 iterations
python scripts/run_full_experiments.py --mode standard   # 8 domains, 500 iterations
python scripts/run_full_experiments.py --mode full       # 22 domains, 500 iterations

# Custom configuration
python scripts/run_full_experiments.py --domains blocksworld hanoi --iterations 200

# Resume from failure
python scripts/run_full_experiments.py --mode standard --resume-from "rover/p01"

# Dry run (show what would run)
python scripts/run_full_experiments.py --mode standard --dry-run
```

**Options:**
- `--mode {quick,standard,full}` - Predefined experiment configuration
- `--domains` - Custom list of domains to run
- `--problems` - Problems to run (default: p00-p09)
- `--all-problems` - Auto-detect all problems in each domain
- `--iterations` - Number of iterations per experiment (default: 400)
- `--output-dir` - Output directory (default: results/paper/comparison_TIMESTAMP)
- `--resume-from` - Resume from domain/problem (e.g., "rover/p01")
- `--force` - Force re-run even if results exist
- `--dry-run` - Show what would run without executing
- `--use-object-subset` - Enable object subset selection (default: enabled)
- `--no-object-subset` - Disable object subset selection for full grounding

### 2. Post-Process Information Gain Results
Extract precision/recall metrics from experiment checkpoints.

```bash
python scripts/analyze_information_gain_metrics.py \
    --consolidated-dir results/paper/consolidated_experiments \
    --benchmarks-dir benchmarks/olam-compatible \
    --output-dir results/information_gain_metrics \
    --domain blocksworld
```

**Options:**
- `--min-observations N` - Exclude actions with fewer than N observations from metrics (default: 0)

**Output:**
- `{output-dir}/{domain}/p*/metrics_per_iteration.json` - Per-problem metrics
- `{output-dir}/{domain}/domain_metrics.json` - Aggregated domain metrics
- `{output-dir}/checkpoint_metrics.csv` - Flat CSV for analysis

### 3. Process External OLAM Results
Process OLAM checkpoint exports from external execution into metrics files.
**Note:** OLAM must be run externally. This script processes its output for comparison.

```bash
python scripts/process_olam_results.py \
    --olam-results /path/to/olam_results/blocksworld \
    --ground-truth benchmarks/olam-compatible/blocksworld/domain.pddl \
    --output-dir results/olam-results/blocksworld
```

**Expected Input Structure:**
```
olam_results/<domain>/
├── 1_p00_<domain>_gen/
│   └── checkpoints/
│       ├── iter_1/
│       │   ├── operator_certain_positive_effects.json
│       │   └── ...
│       └── iter_N/
└── 2_p01_<domain>_gen/
```

**Output:**
- `{output-dir}/p*_safe_metrics.json` - Safe model metrics per problem
- `{output-dir}/p*_complete_metrics.json` - Complete model metrics per problem
- `{output-dir}/domain_safe_metrics.json` - Aggregated safe metrics
- `{output-dir}/domain_complete_metrics.json` - Aggregated complete metrics

---

## Visualization Scripts

### 4. Generate Comparison Plots
Create per-domain OLAM vs Information Gain comparison plots.

```bash
python scripts/compare_algorithms_plots.py \
    --olam-results results/olam-results \
    --infogain-results results/information_gain_metrics \
    --domain blocksworld
```

**Output:** `results/comparison_plots/{domain}/{domain}_*.png`

### 5. Generate Aggregated Plots
Create cross-domain aggregated comparison plots.

```bash
python scripts/plot_aggregated_comparison.py
```

### 6. Generate Publication-Ready Visualizations
Create charts and tables for paper analysis.

```bash
python scripts/visualize_paper_results.py
```

---

## Analysis Scripts

### 7. Generate Results Tables
Create LaTeX/CSV tables from experiment results.

```bash
python scripts/generate_results_tables.py
```

### 8. Dual-Axis Algorithm Comparison
Create dual-axis plots comparing metrics.

```bash
python scripts/plot_dual_axis_algorithm_comparison.py
```

### 9. Success/Metric Correlation Analysis
Analyze correlation between execution success and model precision/recall.

```bash
python scripts/analyze_success_metric_correlation.py
```

---

## Pipeline Script

### run_domain_pipeline.py
Orchestrates the complete post-processing workflow for a single domain:
1. Process raw OLAM exports (optional, for external OLAM results)
2. Extract Information Gain metrics
3. Generate comparison plots

```bash
python scripts/run_domain_pipeline.py <domain> [options]

Options:
    --raw-olam-results PATH    Path to raw OLAM exports (external)
    --consolidated-dir DIR     Parent directory for all results
    --infogain-results DIR     InfoGain experiment results directory
    --benchmarks-dir DIR       Ground truth PDDL domains (default: benchmarks/olam-compatible)
    --skip-olam-processing     Skip OLAM processing step
    --skip-metrics             Skip InfoGain metrics extraction
    --skip-plots               Skip comparison plot generation
```

---

## Output Directory Structure

```
results/
├── paper/
│   └── comparison_YYYYMMDD_HHMMSS/
│       └── information_gain/
│           └── blocksworld/
│               └── p00/
│                   ├── config.yaml
│                   ├── experiment_summary.json
│                   └── models/
├── information_gain_metrics/
│   └── blocksworld/
│       ├── domain_metrics.json
│       └── p00/metrics_per_iteration.json
├── olam-results/           # External OLAM results (processed)
│   └── blocksworld/
│       └── domain_safe_metrics.json
└── comparison_plots/
    └── blocksworld/
        ├── blocksworld_preconditions_comparison.png
        └── blocksworld_effects_comparison.png
```
