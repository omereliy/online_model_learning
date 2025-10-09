# Algorithm Integration Guide

## BaseActionModelLearner Interface

All algorithms must implement this abstract base class:

```python
class BaseActionModelLearner(ABC):
    def select_action(self, state) -> Tuple[str, List[str]]
    def observe(self, state, action, objects, success, next_state) -> None
    def get_learned_model() -> Dict[str, Any]
    def has_converged() -> bool
```

## Integration Pattern

### 1. Create Adapter Class
Extend `BaseActionModelLearner` and wrap external algorithm.

### 2. Handle State/Action Conversion
Convert between unified format and external format.

### 3. Implement Required Methods
Map external methods to interface methods.

## Algorithm Specifics

### OLAM
- See [OLAM_interface.md](OLAM_interface.md)
- Requires injective bindings
- No negative preconditions

### ModelLearner
- See [ModelLearner_interface.md](ModelLearner_interface.md)
- Requires lifted_dict YAML
- Optimistic exploration

## Unified Interface Design

### BaseActionModelLearner Abstract Class
```python
class BaseActionModelLearner(ABC):
    @abstractmethod
    def select_action(self, state) -> Action:
        """Select next action to execute"""

    @abstractmethod
    def observe(self, state, action, next_state, success) -> None:
        """Observe transition result"""

    @abstractmethod
    def update_model(self) -> None:
        """Update internal model based on observations"""

    @abstractmethod
    def get_learned_model(self) -> PDDLModel:
        """Return current learned model"""

    @abstractmethod
    def has_converged(self) -> bool:
        """Check if learning has converged"""
```

## Adapter Implementation Strategy

### OLAM Adapter (Current Implementation)
```python
from src.algorithms.olam_adapter import OLAMAdapter
from src.core.pddl_io import parse_pddl
from src.core.lifted_domain import LiftedDomainKnowledge
from src.core.up_adapter import UPAdapter

class OLAMAdapter(BaseActionModelLearner):
    def __init__(self, domain_file, problem_file, bypass_java=True):
        # Load domain using new architecture
        up_problem, initial_state = parse_pddl(domain_file, problem_file)
        self.domain = LiftedDomainKnowledge.from_up_problem(up_problem, UPAdapter)

        # Initialize OLAM components (external library)
        from OLAM.Learner import Learner
        from Util.PddlParser import PddlParser

        self.parser = PddlParser()
        self.learner = Learner(self.parser, domain_file, problem_file)
        self.bypass_java = bypass_java

    def select_action(self, state):
        # Convert state to OLAM format (list of predicates)
        olam_state = list(state)  # Set[str] → List[str]

        # Update OLAM's simulator with current state
        self._update_simulator_state(olam_state)

        # Select action using OLAM
        action_idx, strategy = self.learner.select_action()

        # Convert action index to (name, objects)
        return self._idx_to_action(action_idx)

    def observe(self, state, action_name, objects, success, next_state):
        if not success:
            # Learn from failure
            olam_state = list(state)
            self._update_simulator_state(olam_state)
            self.learner.learn_failed_action_precondition(self.simulator)
        else:
            # Update model from success
            self.learner.learn()
```

### ModelLearner Adapter (Conceptual - Repository Unavailable)
```python
# NOTE: ModelLearner repository currently unavailable
# This is a conceptual example of how the adapter would be implemented

from src.core.pddl_io import parse_pddl
from src.core.lifted_domain import LiftedDomainKnowledge
from src.core.up_adapter import UPAdapter

class OptimisticAdapter(BaseActionModelLearner):
    def __init__(self, domain_file, problem_file, lifted_dict_file):
        # Load domain using new architecture
        up_problem, initial_state = parse_pddl(domain_file, problem_file)
        self.domain = LiftedDomainKnowledge.from_up_problem(up_problem, UPAdapter)

        # Initialize ModelLearner (when available)
        # from model_learner.ModelLearnerLifted import ModelLearnerLifted
        # self.learner = ModelLearnerLifted(model_dict, domain_file, problem_file, lifted_dict_file)

        self.current_plan = []
        self.plan_index = 0
        self.observations = []

    def select_action(self, state):
        # Get action from current plan or generate new plan
        if self.plan_index >= len(self.current_plan):
            self._generate_new_plan(state)
        action = self.current_plan[self.plan_index]
        self.plan_index += 1
        return action

    def observe(self, state, action_name, objects, success, next_state):
        # Collect observations for batch update
        self.observations.append((state, action_name, objects, success, next_state))
        if not success or self.plan_index >= len(self.current_plan):
            self.update_model()
```

## Common Challenges and Solutions

### 1. State Representation
**Challenge**: Different internal state formats
**Solution**: Use unified `Set[str]` of grounded fluents throughout

```python
from src.core.up_adapter import UPAdapter

# Standard state format: Set[str] of grounded fluents
state = {'clear_a', 'on_b_c', 'handempty'}

# Convert to/from UP format when needed
up_state_dict = UPAdapter.fluent_set_to_up_state(state, problem)
state_from_up = UPAdapter.up_state_to_fluent_set(up_state_dict, problem)

# For OLAM: convert to list
olam_state = list(state)

# For environment execution: use directly
env.execute(action_string)
```

### 2. Action Representation
**Challenge**: Different action naming conventions
**Solution**: Use grounded action strings throughout

```python
from src.core.grounding import ground_action, parse_grounded_action_string

# Standard format: 'action_name_obj1_obj2'
action_str = 'stack_a_b'

# Parse to GroundedAction when needed
grounded = parse_grounded_action_string(action_str, domain)
# → GroundedAction(action_name='stack', objects=['a', 'b'], ...)

# Create from lifted action + objects
lifted = domain.get_action('stack')
grounded = ground_action(lifted, ['a', 'b'])
action_str = grounded.to_string()  # 'stack_a_b'
```

### 3. Planner Integration
**Challenge**: Different planner interfaces
**Solution**: Use Unified Planning Framework consistently

```python
from unified_planning.shortcuts import OneshotPlanner

# UP handles planner selection automatically
planner = OneshotPlanner(problem_kind=problem.kind)

# Get plan
result = planner.solve(problem)

if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
    plan = result.plan.actions
    # Convert UP actions to our format
    action_strings = [action_to_string(act) for act in plan]
```

### 4. Model Format
**Challenge**: Different internal model representations
**Solution**: Use `LiftedDomainKnowledge` as standard format

```python
from src.core.lifted_domain import LiftedDomainKnowledge, LiftedAction

# Convert from any algorithm to standard format
def algorithm_model_to_domain(algo_model) -> LiftedDomainKnowledge:
    """Convert algorithm-specific model to LiftedDomainKnowledge"""
    domain = LiftedDomainKnowledge('learned')

    for action_name, action_data in algo_model.items():
        lifted_action = LiftedAction(
            name=action_name,
            parameters=action_data['parameters'],
            preconditions=action_data['preconditions'],
            add_effects=action_data['add_effects'],
            del_effects=action_data['del_effects']
        )
        domain.lifted_actions[action_name] = lifted_action

    return domain
```

## File Dependencies

### OLAM Dependencies
- Parser: `Util/PddlParser.py`
- Simulator: `Util/Simulator.py`
- Configuration: `Configuration.py`
- Planner: `OLAM/Planner.py`

### ModelLearner Dependencies
- Parser: `src/model_learner/parser.py`
- Simulator: `src/model_learner/ModelSimulator.py`
- Utils: `src/model_learner/Utils.py`
- Constants: `src/model_learner/constants.py`

## Testing Strategy

1. **Unit Tests**: Test each adapter method individually
2. **Integration Tests**: Test full learning loop on simple domains
3. **Comparison Tests**: Ensure both adapters produce similar results
4. **Performance Tests**: Measure sample complexity and runtime

## Next Steps

1. Create unified state/action representations
2. Implement adapter classes
3. Set up planner wrappers
4. Create test suite
5. Validate on benchmark domains