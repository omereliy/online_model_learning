# OLAM (Online Learning of Action Models) Interface Documentation

## Repository Structure
- **Location**: `/home/omer/projects/OLAM/`
- **Main Module**: `OLAM/Learner.py`
- **Supporting Modules**:
  - `OLAM/Planner.py` - Planning interface
  - `Util/Simulator.py` - Domain simulation
  - `Util/PddlParser.py` - PDDL parsing utilities
  - `Configuration.py` - Configuration settings
- **Required Directories**:
  - `PDDL/` - Domain and problem files
  - `Info/` - State and constraint tracking

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
- `action_labels`: Sorted list of action labels (grounded actions)
- `operator_certain_predicates`: Dict of learned certain preconditions
- `operator_uncertain_predicates`: Dict of uncertain preconditions
- `operator_negative_preconditions`: Dict of negative preconditions
- `operator_executability_constr`: Dict of executability constraints
- `current_plan`: Current plan being executed
- `model_convergence`: Boolean flag for convergence
- `not_executable_actions_index`: List of action indices that are not executable

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
- `RANDOM_SEED`: Random seed for reproducibility (default: 2021)
- `MAX_ITER`: Maximum learning iterations (default: 5000)
- `TIME_LIMIT_SECONDS`: Time limit for learning (default: 1800)
- `STRATEGIES`: List of action selection strategies
- `RANDOM_WALK`: Boolean for random walk mode
- `JAVA_BIN_PATH`: Path to Java binary (required for `compute_not_executable_actionsJAVA()`)
- `JAVA_DIR`: Directory containing Java installation

### Usage Pattern
```python
# Initialize
parser = PddlParser()
learner = Learner(parser, action_list, eval_frequency=10)

# Set required attributes for execution
learner.initial_timer = default_timer()
learner.max_time_limit = 3600  # seconds
if not hasattr(learner, 'current_plan'):
    learner.current_plan = []

# Action selection and learning
action_idx, strategy = learner.select_action()
action_label = learner.action_labels[action_idx]

# After execution:
if action_failed:
    learner.learn_failed_action_precondition(simulator)
else:
    learner.add_operator_precondition(action_label)
    learner.add_operator_effects(action_label, old_state, new_state)
```

## Integration Notes

### Required Inputs
1. **PDDL parser instance** from `Util.PddlParser`
2. **Complete list of ground actions** as strings (e.g., `["pick-up(a)", "stack(a,b)"]`)
3. **Simulator** that provides:
   - Current state observation (as PDDL strings)
   - Action execution with success/failure feedback
   - State updates

### External Dependencies
- **Java Runtime**: Required for `compute_not_executable_actionsJAVA()` method
  - Uses `compute_not_executable_actions.jar` for action filtering
  - Can be bypassed in adapter implementations if Java is unavailable

### Output Format
- Learned model stored in:
  - `operator_certain_predicates`: Definite preconditions
  - `operator_uncertain_predicates`: Possible preconditions
  - `operator_negative_preconditions`: Negative preconditions
  - `operator_executability_constr`: Action executability constraints
- Files generated:
  - `PDDL/domain_learned.pddl`: Learned domain model
  - `Info/action_list.txt`: Indexed action list
  - `Info/action_executability_constraints.json`: Constraint tracking

### Key Dependencies
- **numpy**: Random selection and array operations
- **pandas**: Evaluation metrics tracking
- **json**: Constraint serialization
- **subprocess**: Java process execution
- **Custom modules**: PDDL parser and simulator interfaces

### Important Implementation Notes
1. **Action Indexing**: Actions are referenced by integer indices into `action_labels`
2. **State Format**: States should be provided as lists of PDDL predicate strings
3. **Java Dependency**: The `compute_not_executable_actionsJAVA()` method requires Java; adapters should implement bypass using learned model only
4. **Directory Structure**: OLAM expects `PDDL/` and `Info/` directories to exist in the working directory
5. **Learning Principles**: OLAM must learn from experience only - never provide ground truth preconditions or effects
6. **Expected Learning Curve**: Initial success rate ~20%, improving to ~70-80% over time through trial and error