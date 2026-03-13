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
)
```

## Requirements

- Python >= 3.10
- [python-sat](https://pysathq.github.io/) -- SAT solver for CNF management
- [unified-planning](https://unified-planning.readthedocs.io/) -- PDDL parsing and domain representation

## License

MIT
