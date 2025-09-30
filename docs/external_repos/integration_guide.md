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

### OLAM Adapter
```python
class OLAMAdapter(BaseActionModelLearner):
    def __init__(self, domain_file, problem_file):
        # Initialize OLAM components
        self.parser = PddlParser()
        self.action_list = self._extract_actions(domain_file)
        self.learner = Learner(self.parser, self.action_list)
        self.last_action_idx = None

    def select_action(self, state):
        # Convert state to OLAM format
        self._update_state(state)
        action_idx, strategy = self.learner.select_action()
        self.last_action_idx = action_idx
        return self._idx_to_action(action_idx)

    def observe(self, state, action, next_state, success):
        if not success and self.last_action_idx is not None:
            # Update OLAM's internal simulator state
            self.learner.learn_failed_action_precondition(
                self._create_simulator_view(state)
            )
```

### ModelLearner Adapter
```python
class OptimisticAdapter(BaseActionModelLearner):
    def __init__(self, domain_file, problem_file, lifted_dict):
        # Initialize ModelLearner
        parser = Parser()
        model_dict = parser.parse_model(domain_file, problem_file)
        self.learner = ModelLearnerLifted(
            model_dict,
            domain_file,  # as template
            problem_file,  # as template
            lifted_dict
        )
        self.current_plan = []
        self.plan_index = 0

    def select_action(self, state):
        # Get action from current plan or generate new plan
        if self.plan_index >= len(self.current_plan):
            self._generate_new_plan(state)
        action = self.current_plan[self.plan_index]
        self.plan_index += 1
        return action

    def observe(self, state, action, next_state, success):
        # Collect observations for batch update
        self.observations.append((state, action, next_state, success))
        if not success or self.plan_index >= len(self.current_plan):
            self.update_model()
```

## Common Challenges and Solutions

### 1. State Representation
**Challenge**: OLAM and ModelLearner use different state formats
**Solution**: Create unified State class with conversion methods
```python
class UnifiedState:
    def to_olam_format(self) -> set:
        """Convert to OLAM's predicate set"""

    def to_modellearner_format(self) -> dict:
        """Convert to ModelLearner's dict format"""
```

### 2. Action Representation
**Challenge**: Different action naming conventions
**Solution**: Standardize action format
```python
def standardize_action(action_str: str) -> Action:
    """Convert various formats to unified Action object"""
    # Handle formats like:
    # - "move(a,b)" (OLAM)
    # - "move_a_b" (ModelLearner)
    # - Action(name="move", params=["a", "b"]) (Unified)
```

### 3. Planner Integration
**Challenge**: Both repos expect different planner interfaces
**Solution**: Create planner wrapper with multiple backends
```python
class UnifiedPlanner:
    def plan(self, model, initial_state, goal) -> List[Action]:
        """Generate plan using appropriate backend"""
        if self.backend == "fast-downward":
            return self._fd_plan(model, initial_state, goal)
        elif self.backend == "ff":
            return self._ff_plan(model, initial_state, goal)
```

### 4. Model Format
**Challenge**: Different internal model representations
**Solution**: Create converters
```python
def olam_model_to_pddl(olam_model) -> PDDLModel:
    """Convert OLAM's predicate dicts to PDDL"""

def modellearner_model_to_pddl(ml_model) -> PDDLModel:
    """Convert ModelLearner's dict to PDDL"""
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