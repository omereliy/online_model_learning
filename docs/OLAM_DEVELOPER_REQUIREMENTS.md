# OLAM Enhancement Request for Experiment Integration (Updated)

We need to add features to OLAM to support batch experiments with checkpoint exports. Our system will run OLAM experiments independently and post-process the results.

## ⚠️ IMPORTANT: Domain Compatibility Fixes

### Floortile Domain Fix Required
The floortile domain has action names that conflict with Python's unified_planning library. You need to apply the same fixes we applied:

**In floortile domain.pddl:**
1. Rename predicates:
   - `(up ?x ?y)` → `(tile-up ?x ?y)`
   - `(down ?x ?y)` → `(tile-down ?x ?y)`
   - `(left ?x ?y)` → `(tile-left ?x ?y)`
   - `(right ?x ?y)` → `(tile-right ?x ?y)`

2. Rename actions:
   - `:action up` → `:action move-up`
   - `:action down` → `:action move-down`
   - `:action left` → `:action move-left`
   - `:action right` → `:action move-right`

3. Update all problem files to use the new predicate names

### Domains to Exclude
- **elevators**: Has numeric fluent issues with unified_planning, exclude from experiments

## Required Features

### 1. Command-Line Interface
Add support for batch experiment execution:
```bash
python main.py \
  --domains blocksworld gripper depots floortile gold-miner \
  --output-dir /path/to/results \
  --max-iter 500 \
  --domain-timeout 3600 \
  --planner-timeout 360 \
  --checkpoints 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50 60 70 80 90 100 120 140 160 180 200 250 300 350 400 450 500
```

- When a domain is specified, run ALL problems in that domain automatically
- Support multiple domains in a single execution
- Export model snapshots at specified checkpoint iterations

### 2. Timeout Configuration
```python
# Per-domain timeout (in Configuration.py or command-line)
DOMAIN_TIMEOUT_SECONDS = 3600  # 1 hour per domain
PLANNER_TIME_LIMIT = 360  # 6 minutes per planning call (1/10 of domain timeout)

# Checkpoint schedule (dense at beginning, sparse at end)
DEFAULT_CHECKPOINTS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Every iteration for first 10
    12, 14, 16, 18, 20,              # Every 2 iterations
    25, 30, 35, 40, 45, 50,          # Every 5 iterations
    60, 70, 80, 90, 100,              # Every 10 iterations
    120, 140, 160, 180, 200,          # Every 20 iterations
    250, 300, 350, 400, 450, 500      # Every 50 iterations
]
```

### 3. Output File Structure
For each domain and problem, create:

```
<output_dir>/
├── blocksworld/
│   ├── trace_complete.json          # All problems concatenated
│   ├── 1_p00_blocksworld_gen/
│   │   ├── trace.json               # Execution trace (JSON lines format)
│   │   ├── checkpoints/
│   │   │   ├── iter_1/
│   │   │   │   ├── operator_certain_predicates.json
│   │   │   │   ├── operator_uncertain_predicates.json
│   │   │   │   ├── operator_negative_preconditions.json
│   │   │   │   ├── certain_positive_effects.json
│   │   │   │   ├── certain_negative_effects.json
│   │   │   │   └── domain_learned.pddl
│   │   │   ├── iter_2/
│   │   │   ├── iter_3/
│   │   │   └── ... (all checkpoint iterations)
│   │   └── final/
│   │       └── [same files as checkpoints]
│   ├── 2_p01_blocksworld_gen/
│   └── ...
└── floortile/
    └── [same structure]
```

### 4. Trace File Format (JSON Lines)
Each line is a JSON object representing one iteration:

```json
{"domain": "blocksworld", "problem": "1_p00_blocksworld_gen", "iter": 1, "action": "pick-up(a)", "success": true, "strategy": "exploration", "state_before": ["clear(a)", "ontable(a)", "handempty"], "state_after": ["holding(a)"], "timestamp": "2024-11-18T10:30:00"}
{"domain": "blocksworld", "problem": "1_p00_blocksworld_gen", "iter": 2, "action": "stack(a,b)", "success": false, "strategy": "planning", "state_before": ["holding(a)", "clear(b)"], "learning_update": {"action": "stack", "added_precondition": "handempty"}, "timestamp": "2024-11-18T10:30:05"}
```

Required fields per line:
- domain: Domain name
- problem: Problem file name (without .pddl)
- iter: Iteration number
- action: Executed action with parameters
- success: Boolean execution result
- strategy: "exploration" or "planning"
- state_before: List of true predicates before action (optional but helpful)
- state_after: List of true predicates after action (if success=true)
- learning_update: What was learned (if any)
- timestamp: ISO format timestamp

### 5. Model Export Files
At each checkpoint, export current model state:

**operator_certain_predicates.json:**
```json
{
  "pick-up": ["(clear ?x)", "(ontable ?x)", "(handempty)"],
  "put-down": ["(holding ?x)"],
  "stack": ["(holding ?x)", "(clear ?y)"],
  "unstack": ["(on ?x ?y)", "(clear ?x)", "(handempty)"]
}
```

**certain_positive_effects.json:**
```json
{
  "pick-up": ["(holding ?x)"],
  "put-down": ["(clear ?x)", "(ontable ?x)", "(handempty)"],
  "stack": ["(on ?x ?y)", "(clear ?x)", "(handempty)"],
  "unstack": ["(holding ?x)", "(clear ?y)"]
}
```

**certain_negative_effects.json:**
```json
{
  "pick-up": ["(clear ?x)", "(ontable ?x)", "(handempty)"],
  "put-down": ["(holding ?x)"],
  "stack": ["(holding ?x)", "(clear ?y)"],
  "unstack": ["(on ?x ?y)", "(clear ?x)", "(handempty)"]
}
```

### 6. Implementation Requirements

- **Python 3.10+ compatibility**: Ensure regex patterns work with Python 3.10
- **Exit codes**: Return 0 on success, non-zero on failure
- **Error handling**:
  - Log errors to stderr
  - Continue with next problem if one fails
  - Skip domain if timeout reached
- **Progress indication**: Print progress to stdout
  ```
  [2024-11-18 10:30:00] Starting domain: blocksworld
  [2024-11-18 10:30:00] Processing blocksworld problem 1/10: 1_p00_blocksworld_gen
  [2024-11-18 10:30:05] Checkpoint saved: iter_1
  [2024-11-18 10:30:10] Checkpoint saved: iter_5
  ...
  [2024-11-18 10:45:00] Completed blocksworld problem 1/10 (iterations: 500, success: true)
  ```

### 7. Problem Naming Convention
Keep your existing convention:
- `1_p00_blocksworld_gen.pddl`
- `2_p01_blocksworld_gen.pddl`
- etc.

Our system will adapt to match this naming.

## Testing Commands
To verify the implementation works:
```bash
# Test single domain with dense checkpoints
python main.py \
  --domains blocksworld \
  --max-iter 50 \
  --domain-timeout 600 \
  --planner-timeout 60 \
  --checkpoints 1 2 3 4 5 10 20 30 40 50 \
  --output-dir /tmp/olam_test

# Check outputs
ls /tmp/olam_test/blocksworld/1_p00_blocksworld_gen/checkpoints/
cat /tmp/olam_test/blocksworld/1_p00_blocksworld_gen/trace.json | head -5
python -m json.tool /tmp/olam_test/blocksworld/1_p00_blocksworld_gen/checkpoints/iter_10/operator_certain_predicates.json
```

## Summary
Key requirements:
1. Apply floortile domain fixes (action/predicate renaming)
2. Skip elevators domain (numeric fluent issues)
3. Dense checkpoints at beginning (1-10 every iteration, then gradually sparse)
4. Timeouts: 3600s per domain, 360s per planner call
5. Run all problems per domain automatically
6. Export model state at every checkpoint iteration
7. JSON lines trace format with timestamps

Let me know if you need clarification on any requirement!