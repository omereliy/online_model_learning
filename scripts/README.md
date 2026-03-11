# Scripts

## AMLGym Experiments (preferred)

Run the Information Gain algorithm via AMLGym benchmarks.

```bash
# Single domain
python3 scripts/run_amlgym_experiment.py --domain blocksworld

# Multiple domains
python3 scripts/run_amlgym_experiment.py --domain blocksworld gripper hanoi

# All AMLGym benchmark domains
python3 scripts/run_amlgym_experiment.py --all-domains

# With evaluation metrics
python3 scripts/run_amlgym_experiment.py --domain blocksworld --evaluate

# Custom settings
python3 scripts/run_amlgym_experiment.py --domain blocksworld --max-steps 300 --model-mode complete --seed 123
```

**Options:**
- `--domain` / `--all-domains` - Domain(s) to run
- `--max-steps` - Learning iterations (default: 500)
- `--model-mode {safe,complete}` - Model reconstruction mode (default: safe)
- `--seed` - Random seed (default: 42)
- `--output-dir` - Output directory (default: results/amlgym)
- `--evaluate` - Run AMLGym evaluation metrics after learning
- `--verbose` - Enable verbose logging

## Local Framework Experiments

Run the Information Gain algorithm on local PDDL benchmark domains.

```bash
# Predefined modes
python3 scripts/run_full_experiments.py --mode quick      # 5 domains, 100 iterations
python3 scripts/run_full_experiments.py --mode standard   # 8 domains, 500 iterations
python3 scripts/run_full_experiments.py --mode full       # 22 domains, 500 iterations

# Custom configuration
python3 scripts/run_full_experiments.py --domains blocksworld hanoi --iterations 200

# Resume from failure
python3 scripts/run_full_experiments.py --mode standard --resume-from "rover/p01"

# Dry run
python3 scripts/run_full_experiments.py --mode standard --dry-run
```

**Options:**
- `--mode {quick,standard,full}` - Predefined experiment configuration
- `--domains` - Custom list of domains
- `--problems` - Problems to run (default: p00-p09)
- `--all-problems` - Auto-detect all problems per domain
- `--iterations` - Iterations per experiment (default: 400)
- `--output-dir` - Output directory
- `--resume-from` - Resume from domain/problem
- `--force` - Re-run even if results exist
- `--dry-run` - Preview without executing
- `--use-object-subset` / `--no-object-subset` - Object subset selection toggle

## Post-Processing: Metrics Extraction

Extract precision/recall metrics from local experiment checkpoints.

```bash
python3 scripts/analyze_information_gain_metrics.py \
    --consolidated-dir results/paper/consolidated_experiments \
    --benchmarks-dir benchmarks/olam-compatible \
    --output-dir results/information_gain_metrics \
    --domain blocksworld
```

**Options:**
- `--consolidated-dir` - Parent directory of experiment results
- `--benchmarks-dir` - Ground truth PDDL domains (default: benchmarks/olam-compatible)
- `--output-dir` - Where to write metrics (default: results/information_gain_metrics)
- `--domain` / `--problem` - Process single domain or problem
- `--min-observations N` - Exclude actions with fewer than N observations (default: 0)

**Output:**
- `{output-dir}/{domain}/p*/metrics_per_iteration.json` - Per-problem metrics
- `{output-dir}/{domain}/domain_metrics.json` - Aggregated domain metrics
- `{output-dir}/checkpoint_metrics.csv` - Flat CSV for analysis
