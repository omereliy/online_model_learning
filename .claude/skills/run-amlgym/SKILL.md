---
name: run-amlgym
description: Run an AMLGym experiment with the InformationGainAgent. Use when the user wants to run, test, or benchmark the algorithm on AMLGym domains.
argument-hint: [domain(s) or "all"] [options]
---

## Run AMLGym Experiment

Run the InformationGainAgent on AMLGym benchmark domains using `scripts/run_amlgym_experiment.py`.

### Parse arguments from: $ARGUMENTS

Interpret the user's request and map to the appropriate flags:
- Domain names (e.g., "blocksworld", "gripper") → `--domain blocksworld gripper`
- "all" or "all domains" → `--all-domains`
- Step/iteration count → `--max-steps N`
- "safe" or "complete" model → `--model-mode safe|complete`
- "evaluate", "metrics", "compare" → `--evaluate`
- "verbose" or "debug" → `--verbose`
- Seed number → `--seed N`

### Default command

If no specific arguments, run:
```bash
python3 scripts/run_amlgym_experiment.py --domain blocksworld --max-steps 200 --evaluate --verbose
```

### Execute

1. Run the experiment command via Bash
2. Show the user the summary output
3. If `--evaluate` was used, highlight the evaluation metrics
4. If the user asks to see the learned model, read the output file from `results/amlgym/<domain>/learned_domain_<mode>.pddl`
