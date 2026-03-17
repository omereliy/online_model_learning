# information-gain-aml

A CNF/SAT-based information-theoretic approach for online action model learning in PDDL planning domains.

The algorithm maintains uncertainty over preconditions and effects using CNF formulas, selects actions that maximize expected information gain, and converges toward the true action model through online interaction with the environment.

## Installation

```bash
pip install information-gain-aml
```

## Quick Start

```python
from information_gain_aml.algorithms.information_gain import InformationGainLearner

learner = InformationGainLearner(
    domain_file="path/to/domain.pddl",
    problem_file="path/to/problem.pddl",
)

# Select an action based on expected information gain
action_name, objects = learner.select_action(current_state)

# After observing the outcome, update the model
learner.update_model()
```

## Key Features

- **CNF-based uncertainty representation** -- precondition and effect knowledge encoded as SAT formulas
- **Information-theoretic action selection** -- picks actions that maximize expected information gain
- **Lifted learning** -- learns at the operator level, generalizing across object instances
- **Object subset selection** -- scales to large domains by focusing on relevant object subsets
- **Parallel gain computation** -- optional multiprocessing for large action spaces
- **MCTS-based action selection** -- lookahead and full UCT strategies for deeper exploration

## Action Selection Strategies

| Strategy | Description | Speed |
|----------|-------------|-------|
| `greedy` | Selects the action with the highest immediate information gain | Fast (default) |
| `lookahead` | Bounded depth-limited lookahead with discounted future gain | Moderate |
| `mcts` | Full UCT-based Monte Carlo Tree Search | Slow (see note below) |

> **Performance note:** The `mcts` strategy performs SAT solving during rollouts, which makes it significantly slower than other strategies. For large domains it may be impractical. Performance improvements are planned for a future release. Use `lookahead` for a balance between exploration depth and speed.

## Configuration

```python
learner = InformationGainLearner(
    domain_file="domain.pddl",
    problem_file="problem.pddl",
    max_iterations=1000,                  # max learning iterations
    use_object_subset=True,               # object subset selection (default: True)
    spare_objects_per_type=2,             # extra objects per type beyond minimum
    num_workers=None,                      # parallel workers (None=auto, 0=sequential)
    learn_negative_preconditions=True,    # include negative precondition candidates
    selection_strategy="greedy",          # "greedy", "lookahead", or "mcts"
    lookahead_depth=2,                    # depth for lookahead strategy
    mcts_iterations=50,                   # iterations for mcts strategy
    mcts_rollout_depth=5,                # rollout depth for mcts strategy
)
```

## Requirements

- Python >= 3.10
- [python-sat](https://pysathq.github.io/) -- SAT solver for CNF management
- [unified-planning](https://unified-planning.readthedocs.io/) -- PDDL parsing and domain representation

## License

MIT
