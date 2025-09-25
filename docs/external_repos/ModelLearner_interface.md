# ModelLearner (Optimistic Exploration) Interface Documentation

## Repository Structure
- **Location**: `/home/omer/projects/ModelLearner/`
- **Main Module**: `src/model_learner/ModelLearnerLifted.py`
- **Supporting Modules**:
  - `src/model_learner/ModelSimulator.py` - Model simulation
  - `src/model_learner/Utils.py` - Utility functions
  - `src/model_learner/parser.py` - PDDL parser
  - `src/model_learner/constants.py` - Configuration constants

## Main Class: `ModelLearnerLifted`

### Constructor
```python
def __init__(self, initial_model, domain_template, problem_template,
             lifted_dict, plan_count=DEFAULT_PLAN_COUNT):
    """
    Parameters:
    - initial_model: Dict with domain and problem information
    - domain_template: Path to domain template file
    - problem_template: Path to problem template file
    - lifted_dict: Path to lifted action dictionary YAML file
    - plan_count: Number of diverse plans to generate (default=10)
    """
```

### Key Attributes
- `simulator`: ModelSimulator instance
- `current_model`: Current learned model (optimistic)
- `starting_model`: Empty initial model
- `original_actions`: List of original action names
- `action_copies`: Dict of action copies for disjunctive preconditions
- `failed_precondition_dict`: Dict of failed preconditions per action
- `seen_precondition_dict`: Dict of observed preconditions
- `failure_upper_bound`: Upper bound on failures before giving up

### Core Methods

#### Main Learning Step
```python
def learning_step_all_actions_updated(self):
    """
    Execute one learning iteration
    Returns: (status, successful_plan)
    - status: Boolean indicating if goal reached
    - successful_plan: Plan that reached goal (if status=True)

    Process:
    1. Generate diverse plans using current model
    2. Execute plans in simulator
    3. Update model based on observations
    """
```

#### Precondition Updates
```python
def update_precondition_with_new_actions(self, observation, failing_action):
    """
    Update preconditions when action fails
    Parameters:
    - observation: Current state observation
    - failing_action: Action that failed to execute

    Creates new action copies with disjunctive preconditions
    """
```

#### Effect Updates
```python
def update_effects(self, observation_seq, plan):
    """
    Update action effects based on successful execution
    Parameters:
    - observation_seq: Sequence of state observations
    - plan: Successfully executed plan

    Updates positive and negative effects
    """
```

#### Action Testing
```python
def test_all_actions(self, observation_seq, plan):
    """
    Test which actions succeeded/failed during plan execution
    Parameters:
    - observation_seq: State observations during execution
    - plan: Executed plan

    Returns: Lists of successful and failed actions
    """
```

### Model Representation

#### Model Dictionary Structure
```python
{
    "domain": {
        "action_name": {
            "precondition": set(),  # Precondition literals
            "positive_effect": set(),  # Add effects
            "negative_effect": set()   # Delete effects
        }
    },
    "problem": {
        "init": set(),  # Initial state
        "goal": set()   # Goal condition
    }
}
```

### Utility Functions (Utils.py)

```python
def create_empty_model(initial_model):
    """Create empty model with no preconditions/effects"""

def create_optimistic_model(model_dict):
    """Create optimistic model (empty preconditions)"""

def call_diverse_planner(model_dict, domain_file, problem_file, ...):
    """Call external planner for diverse plans"""
```

### Constants (from constants.py)
- `DEFAULT_PLAN_COUNT`: Default number of diverse plans (10)
- `FAILURE_UPPERBOUND`: Max failures before giving up
- `DEBUG_LEVEL`: Debug output verbosity
- `DOMAIN`, `PRECONDITION`, `POSITIVE_EFFECT`, `NEGATIVE_EFFECT`: Dict keys
- `ACT_ARG_SEPARATOR`, `PROP_ARG_SEPARATOR`: String separators

### Usage Pattern
```python
# Parse initial model
parser = Parser()
model_dict = parser.parse_model(domain_file, problem_file)

# Initialize learner
learner = ModelLearnerLifted(
    model_dict,
    domain_template_file,
    problem_template_file,
    lifted_dict_file,
    plan_count=10
)

# Learning loop
for iteration in range(max_iterations):
    status, plan = learner.learning_step_all_actions_updated()
    if status:
        print(f"Goal reached with plan: {plan}")
        break
```

## Integration Notes

### Required Inputs
1. Initial model dictionary (can be empty)
2. Domain and problem template files
3. Lifted dictionary YAML file defining action schemas
4. Access to external planner for generating diverse plans

### Output Format
- Learned model in `current_model` attribute
- Model includes preconditions and effects for all actions
- Support for disjunctive preconditions via action copies

### Key Dependencies
- yaml for configuration loading
- External planner (expects specific output format)
- ModelSimulator for execution
- Custom parser for PDDL files

### Lifted Dictionary Format (YAML)
```yaml
lifted_action_keys:
  - move
  - pickup
  - putdown
# Maps lifted action names to schemas
```