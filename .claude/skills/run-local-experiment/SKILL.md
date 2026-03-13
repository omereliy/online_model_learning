---
name: run-local-experiment
description: "PROACTIVE: Run full experiments via the local framework across multiple domains/problems. Use when the user mentions: running experiments, full benchmark, multiple problems, quick/standard/full mode, batch runs, paper results, large-scale testing, or iteration counts. Also suggest when the user wants to run across many problems or needs YAML configs."
argument-hint: [domains] [mode] [options]
---

## Run Local Framework Experiment

Run experiments using the local ExperimentRunner with YAML configs, multi-problem support, and resumability.

**When to use this vs `/run-amlgym`:**
- This (`/run-experiment`): Multi-problem runs, batch experiments, predefined modes (quick/standard/full), resume support
- `/run-amlgym`: Quick single-domain test with AMLGym evaluation metrics

### Parse arguments from: $ARGUMENTS

- Domain names → `--domains blocksworld depots`
- "quick"/"standard"/"full" → `--mode quick|standard|full`
- Iteration count → `--iterations N` (default: 400)
- Problem list → `--problems p00 p01 p02`
- "all problems" → `--all-problems`
- "no negative preconditions" / "positive only" → `--no-negative-preconditions`
- "no subset" / "all objects" → `--no-object-subset`
- "dry run" → `--dry-run`
- "force" / "re-run" → `--force`
- "resume from X" → `--resume-from domain/problem`
- Output dir → `--output-dir PATH`

### Predefined modes

| Mode | Domains | Problems | Iterations | Use case |
|------|---------|----------|------------|----------|
| `quick` | 5 (blocksworld, hanoi, ferry, miconic, depots) | p00 | 100 | Smoke test |
| `standard` | 8 domains | p00-p09 | 500 | Paper results |
| `full` | 22 domains | p00-p09 | 500 | Complete benchmark |

### Default command

```bash
python3 scripts/run_full_experiments.py --mode quick
```

### Execute

1. Build the command from parsed arguments
2. If `--dry-run`, show what would run and ask to confirm
3. Run via Bash (may take minutes to hours depending on scale)
4. Show the summary: successful/failed counts, total runtime
5. Results saved to `results/<output-dir>/information_gain/<domain>/<problem>/`
