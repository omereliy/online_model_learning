# Scripts Directory

## Quick Start: Run Information Gain Experiments

```bash
# Activate virtual environment
source venv/bin/activate

# Run quick test (5 domains, 100 iterations)
python3 scripts/run_full_experiments.py --mode quick

# Run standard benchmarks (8 domains, 500 iterations)
python3 scripts/run_full_experiments.py --mode standard

# Run custom domains
python3 scripts/run_full_experiments.py --domains blocksworld depots --iterations 400
```

---

## Core Scripts

### 1. Run Information Gain Experiments
Run the Information Gain algorithm on PDDL benchmark domains.

```bash
# Predefined modes
python3 scripts/run_full_experiments.py --mode quick      # 5 domains, 100 iterations
python3 scripts/run_full_experiments.py --mode standard   # 8 domains, 500 iterations
python3 scripts/run_full_experiments.py --mode full       # 22 domains, 500 iterations

# Custom configuration
python3 scripts/run_full_experiments.py --domains blocksworld hanoi --iterations 200

# Resume from failure
python3 scripts/run_full_experiments.py --mode standard --resume-from "rover/p01"

# Dry run (show what would run)
python3 scripts/run_full_experiments.py --mode standard --dry-run
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
python3 scripts/analyze_information_gain_metrics.py \
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

---

## Visualization Scripts

### 3. Generate Publication-Ready Visualizations
Create charts and tables for paper analysis.

```bash
python3 scripts/visualize_paper_results.py
```

### 4. Generate Results Tables
Create LaTeX/CSV tables from experiment results.

```bash
python3 scripts/generate_results_tables.py
```

### 5. Success/Metric Correlation Analysis
Analyze correlation between execution success and model precision/recall.

```bash
python3 scripts/analyze_success_metric_correlation.py
```

---

## Pipeline Script

### run_domain_pipeline.py
Orchestrates the complete post-processing workflow for a single domain:
1. Extract Information Gain metrics
2. Generate plots

```bash
python3 scripts/run_domain_pipeline.py <domain> [options]

Options:
    --consolidated-dir DIR     Parent directory for all results
    --infogain-results DIR     InfoGain experiment results directory
    --benchmarks-dir DIR       Ground truth PDDL domains (default: benchmarks/olam-compatible)
    --skip-metrics             Skip InfoGain metrics extraction
    --skip-plots               Skip plot generation
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
└── information_gain_metrics/
    └── blocksworld/
        ├── domain_metrics.json
        └── p00/metrics_per_iteration.json
```
