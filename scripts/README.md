# Scripts Directory

## Quick Start: Run Complete Pipeline for One Domain

```bash
# Activate virtual environment
source venv/bin/activate

# Run full pipeline with raw OLAM exports
python scripts/run_domain_pipeline.py blocksworld \
    --raw-olam-results /path/to/olam_results/blocksworld \
    --consolidated-dir results/consolidated_results101225 \
    --benchmarks-dir benchmarks/olam-compatible

# Run pipeline with already processed results
python scripts/run_domain_pipeline.py blocksworld \
    --consolidated-dir results/consolidated_results101225 \
    --skip-olam-processing

# Skip metrics extraction if already done
python scripts/run_domain_pipeline.py blocksworld \
    --consolidated-dir results/consolidated_results101225 \
    --skip-olam-processing \
    --skip-metrics
```

---

## Core Scripts

### 1. Run Experiments
Generate experiment data from PDDL domains.

```bash
python scripts/run_full_experiments.py --config configs/paper/blocksworld.yaml
```

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
- `--min-observations N` - Exclude actions with fewer than N observations from metrics (default: 0, include all)

**Output:**
- `{output-dir}/{domain}/p*/metrics_per_iteration.json` - Per-problem metrics
- `{output-dir}/{domain}/domain_metrics.json` - Aggregated domain metrics
- `{output-dir}/checkpoint_metrics.csv` - Flat CSV for analysis

### 3. Process External OLAM Results
Process OLAM checkpoint exports from external execution into metrics files.
**Run this before `analyze_olam_results.py`.**

```bash
python scripts/process_olam_results.py \
    --olam-results /path/to/olam_results/blocksworld \
    --ground-truth benchmarks/olam-compatible/blocksworld/domain.pddl \
    --output-dir results/olam-results/blocksworld
```

**Options:**
- `--olam-results` - Path to OLAM results domain directory containing checkpoint exports
- `--ground-truth` - Path to ground truth domain PDDL for metrics computation
- `--benchmarks-dir` - Path to benchmarks directory (default: benchmarks/olam-compatible)
- `--output-dir` - Output directory for processed metrics

**Expected Input Structure:**
```
olam_results/<domain>/
├── 1_p00_<domain>_gen/
│   └── checkpoints/
│       ├── iter_1/
│       │   ├── operator_certain_positive_effects.json
│       │   ├── operator_certain_negative_effects.json
│       │   └── ...
│       └── iter_N/
└── 2_p01_<domain>_gen/
```

**Output:**
- `{output-dir}/p*_safe_metrics.json` - Safe model metrics per problem
- `{output-dir}/p*_complete_metrics.json` - Complete model metrics per problem
- `{output-dir}/p*_detailed_metrics.json` - Per-action detailed metrics
- `{output-dir}/domain_safe_metrics.json` - Aggregated safe metrics
- `{output-dir}/domain_complete_metrics.json` - Aggregated complete metrics

### 4. Analyze OLAM Results (Alternative)
Analyze OLAM execution traces and compute metrics by replaying traces.

```bash
python scripts/analyze_olam_results.py \
    --domain benchmarks/olam-compatible/blocksworld/domain.pddl \
    --problem benchmarks/olam-compatible/blocksworld/p01.pddl \
    --output-dir results/olam_analysis
```

**Options:**
- `--min-observations N` - Exclude actions with fewer than N observations from metrics (default: 0, include all)

---

## Visualization Scripts

### 5. Generate Comparison Plots
Create per-domain OLAM vs Information Gain comparison plots.

```bash
# Process all common domains with default paths
python scripts/compare_algorithms_plots.py

# Process single domain with custom paths (including OLAM trace data)
python scripts/compare_algorithms_plots.py \
    --olam-results results/consolidated_results101225/olam \
    --infogain-results results/consolidated_results101225/information_gain \
    --raw-olam-path /home/omer/projects/olam_results10122025 \
    --domain blocksworld
```

**Options:**
- `--olam-results` - Directory containing processed OLAM metrics (default: results/olam-results)
- `--infogain-results` - Directory containing InfoGain results (default: results/information_gain_metrics)
- `--output-dir` - Output directory for plots (default: results/comparison_plots)
- `--domain` - Process single domain (optional, processes all common domains if not specified)
- `--raw-olam-path` - Path to raw OLAM results for trace data (needed for cumulative success plots)

**Note:** The `--raw-olam-path` should point to the parent directory containing domain subdirectories with `trace.json` files. Without this, cumulative success plots will only show Information Gain data.

**Output:** `results/comparison_plots/{domain}/{domain}_*.png`

### 6. Generate Aggregated Plots
Create cross-domain aggregated comparison plots.

```bash
python scripts/plot_aggregated_comparison.py
```

### 7. Generate Publication-Ready Visualizations
Create charts and tables for paper analysis.

```bash
python scripts/visualize_paper_results.py
```

---

## Analysis Scripts

### 8. Generate Results Tables
Create LaTeX/CSV tables from experiment results.

```bash
python scripts/generate_results_tables.py
```

### 9. Dual-Axis Algorithm Comparison
Create dual-axis plots comparing OLAM vs InfoGain per metric.

```bash
python scripts/plot_dual_axis_algorithm_comparison.py
```

### 10. Success/Metric Correlation Analysis
Analyze correlation between execution success and model precision/recall.

```bash
python scripts/analyze_success_metric_correlation.py
```

---

## Pipeline Script

### run_domain_pipeline.py
Orchestrates the complete post-processing workflow for a single domain:
1. Process raw OLAM exports (optional)
2. Extract Information Gain metrics
3. Generate comparison plots

```bash
python scripts/run_domain_pipeline.py <domain> [options]

Options:
    --raw-olam-results PATH    Path to raw OLAM exports (e.g., /path/to/olam_results/blocksworld)
                               Also used for trace data in cumulative success plots
    --consolidated-dir DIR     Parent directory for all results (default: results/consolidated_results)
    --infogain-results DIR     InfoGain experiment results directory (overrides consolidated-dir)
    --benchmarks-dir DIR       Ground truth PDDL domains (default: benchmarks/olam-compatible)
    --skip-olam-processing     Skip OLAM processing step
    --skip-metrics             Skip InfoGain metrics extraction
    --skip-plots               Skip comparison plot generation
```

**Examples:**
```bash
# Full pipeline with raw OLAM exports
python scripts/run_domain_pipeline.py blocksworld \
    --raw-olam-results /home/omer/projects/olam_results10122025/blocksworld \
    --consolidated-dir results/consolidated_results101225

# Skip OLAM processing (already done)
python scripts/run_domain_pipeline.py blocksworld \
    --consolidated-dir results/consolidated_results101225 \
    --skip-olam-processing

# Only generate plots (metrics already extracted)
python scripts/run_domain_pipeline.py blocksworld \
    --consolidated-dir results/consolidated_results101225 \
    --skip-olam-processing \
    --skip-metrics
```

---

## Output Directory Structure

When using `--consolidated-dir`:
```
results/consolidated_results101225/
├── olam/
│   └── blocksworld/
│       ├── domain_safe_metrics.json
│       ├── domain_complete_metrics.json
│       ├── domain_metrics.json
│       └── p*_*.json
└── information_gain/
    └── blocksworld/
        ├── domain_metrics.json
        ├── p00/metrics_per_iteration.json
        ├── p01/metrics_per_iteration.json
        └── ...

results/comparison_plots/
└── blocksworld/
    ├── blocksworld_preconditions_comparison.png
    ├── blocksworld_effects_comparison.png
    ├── blocksworld_cumulative_success.png
    └── blocksworld_per_problem_success_failure.png
```

---

## Fair Algorithm Comparison

When comparing OLAM and Information Gain algorithms, be aware of semantic differences:

### Key Differences

| Aspect | Information Gain | OLAM |
|--------|------------------|------|
| **Safe Model (unexecuted)** | La (all possible literals) | Empty without domain_file |
| **Certain Preconditions** | From failure constraints | From success intersection |
| **Contradiction Handling** | Remove from both add/del | Keep in add, remove from del |

### Filtering Unexecuted Actions

Use `--min-observations 1` to exclude unexecuted actions from aggregate metrics:

```bash
python scripts/analyze_information_gain_metrics.py \
    --domain blocksworld \
    --min-observations 1
```

### Expected Metrics for Edge Cases

**Unexecuted Action (0 observations):**
- Safe: prec_precision=0, prec_recall=1, eff_precision=1, eff_recall=0
- Complete: prec_precision=1, prec_recall=0, eff_precision=1, eff_recall=0

See `docs/ALGORITHM_COMPARISON_SEMANTICS.md` for detailed documentation.
