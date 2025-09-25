# OLAM (Online Learning of Action Models) Interface Documentation

## Repository Structure
- **Location**: `/home/omer/projects/OLAM/`
- **Main Module**: `OLAM/Learner.py`
- **Supporting Modules**:
  - `OLAM/Planner.py` - Planning interface
  - `Util/Simulator.py` - Domain simulation
  - `Util/PddlParser.py` - PDDL parsing utilities
  - `Configuration.py` - Configuration settings

## Main Class: `OLAM.Learner`

### Constructor
```python
def __init__(self, parser, action_list, eval_frequency=10):
    """
    Parameters:
    - parser: PDDL parser object
    - action_list: List of action names (strings)
    - eval_frequency: How often to evaluate the model (default=10)
    """
```

### Key Attributes
- `action_labels`: Sorted list of action labels
- `operator_certain_predicates`: Dict of learned certain preconditions
- `operator_uncertain_predicates`: Dict of uncertain preconditions
- `operator_negative_preconditions`: Dict of negative preconditions
- `current_plan`: Current plan being executed
- `model_convergence`: Boolean flag for convergence

### Core Methods

#### Action Selection
```python
def select_action(self):
    """
    Select next action to execute
    Returns: (action_index, strategy)
    - action_index: Integer index into action_labels
    - strategy: Strategy identifier from Configuration.STRATEGIES
    """
```

#### Main Learning Loop
```python
def learn(self, eval_frequency=None, simulator=None):
    """
    Main learning loop
    Parameters:
    - eval_frequency: Optional evaluation frequency override
    - simulator: Domain simulator object

    Iterates until:
    - MAX_ITER reached (Configuration.MAX_ITER)
    - TIME_LIMIT_SECONDS reached
    - Model converges
    """
```

#### Learning from Failed Actions
```python
def learn_failed_action_precondition(self, simulator):
    """
    Update model based on failed action execution
    Parameters:
    - simulator: Domain simulator with current state

    Updates:
    - operator_uncertain_predicates
    - operator_certain_predicates
    """
```

#### Learning from Successful Actions
```python
def add_operator_precondition(self, action_label):
    """
    Update preconditions based on successful action execution
    Parameters:
    - action_label: Successfully executed action
    Returns: Boolean indicating if preconditions changed
    """

def add_operator_effects(self, action_label, old_state, current_state):
    """
    Learn effects from successful action execution
    Parameters:
    - action_label: Successfully executed action
    - old_state: State before execution
    - current_state: State after execution

    Updates:
    - Positive effects (additions)
    - Negative effects (deletions)
    """
```

#### Executability Constraint Update
```python
def update_executability_constr(self, op, op_preconds):
    """
    Update constraints on action executability
    Parameters:
    - op: Operator name
    - op_preconds: Preconditions to update
    """
```

### Configuration Parameters (Configuration.py)
- `RANDOM_SEED`: Random seed for reproducibility
- `MAX_ITER`: Maximum learning iterations
- `TIME_LIMIT_SECONDS`: Time limit for learning
- `STRATEGIES`: List of action selection strategies
- `RANDOM_WALK`: Boolean for random walk mode

### Usage Pattern
```python
# Initialize
parser = PddlParser()
learner = Learner(parser, action_list)

# Learning loop (called by framework)
learner.learn(simulator=simulator)

# During learning:
action_idx, strategy = learner.select_action()
# Execute action in simulator
if action_failed:
    learner.learn_failed_action_precondition(simulator)
```

## Integration Notes

### Required Inputs
1. PDDL parser instance
2. Complete list of ground actions
3. Simulator that provides:
   - Current state observation
   - Action execution with success/failure feedback

### Output Format
- Learned model stored in:
  - `operator_certain_predicates`: Definite preconditions
  - `operator_uncertain_predicates`: Possible preconditions
  - `operator_negative_preconditions`: Negative preconditions

### Key Dependencies
- numpy for random selection
- pandas for evaluation metrics
- Custom PDDL parser and simulator interfaces