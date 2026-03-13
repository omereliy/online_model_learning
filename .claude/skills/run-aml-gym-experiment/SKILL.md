---
name: run-aml-gym-experiment
description: "PROACTIVE: Run a quick AMLGym benchmark with evaluation metrics. Use when the user mentions: evaluating, metrics, precision, recall, predictive power, benchmarking against reference, comparing learned vs ground truth, or wants a fast single-domain test after code changes. Suggest after algorithm code changes: 'Want to verify with /run-aml-gym-experiment blocksworld?'"
argument-hint: [domain(s) or "all"] [options]
---

## Run AMLGym Benchmark

Run the InformationGainAgent against AMLGym reference domains with syntactic/predictive evaluation.

**When to use this vs `/run-experiment`:**
- This (`/run-amlgym`): Quick single-domain test, evaluation metrics, comparing against ground truth
- `/run-experiment`: Multi-problem runs, full benchmark suites, local framework experiments

### Parse arguments from: $ARGUMENTS

- Domain names → `--domain blocksworld gripper`
- "all" → `--all-domains`
- Step count → `--max-steps N` (default: 500)
- "safe"/"complete" → `--model-mode safe|complete`
- "evaluate"/"metrics" → `--evaluate`
- "verbose" → `--verbose`
- "no negative preconditions" / "positive only" → `--no-negative-preconditions`
- "no subset" / "all objects" → `--no-object-subset`
- Seed → `--seed N`
- Output dir → `--output-dir PATH`

### Default command

```bash
python3 scripts/run_amlgym_experiment.py --domain blocksworld --max-steps 200 --evaluate --verbose
```

### Execute

1. Build the command from parsed arguments
2. Run via Bash
3. Show the summary output
4. Highlight evaluation metrics (precision, recall, predictive power)
5. If asked to see the learned model, read `results/amlgym/<domain>/learned_domain_<mode>.pddl`
