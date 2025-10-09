"""
ActiveEnvironment: Minimal execution-only environment.

Handles ONLY grounded action execution with true domain knowledge.
Does NOT expose lifted representations or domain manipulation.

Key principles:
- Minimal interface: only what's needed for execution
- Grounded only: no lifted operations
- True domain: complete, correct action models
- State tracking: maintains current state
"""

import logging
import time
from typing import Set, Tuple, List
from pathlib import Path
from unified_planning.shortcuts import SequentialSimulator
from unified_planning.model import State
from unified_planning.plans import ActionInstance

from src.core.pddl_io import PDDLReader
from src.core.up_adapter import UPAdapter

logger = logging.getLogger(__name__)


class ActiveEnvironment:
    """
    Execution environment with true domain knowledge.

    ONLY deals with GROUNDED representations.
    Minimal interface for action execution and state queries.
    """

    def __init__(self, domain_file: str, problem_file: str):
        """
        Initialize environment from PDDL files.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file

        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If PDDL parsing fails
        """
        logger.info(f"Initializing ActiveEnvironment: {domain_file}, {problem_file}")

        # Validate files exist
        if not Path(domain_file).exists():
            raise FileNotFoundError(f"Domain file not found: {domain_file}")
        if not Path(problem_file).exists():
            raise FileNotFoundError(f"Problem file not found: {problem_file}")

        self.domain_file = domain_file
        self.problem_file = problem_file

        # Parse PDDL using PDDLReader (we need UP problem for simulator)
        reader = PDDLReader()
        self._domain, self._initial_fluents = reader.parse_domain_and_problem(
            domain_file, problem_file
        )
        self._up_problem = reader.get_up_problem()

        # Initialize simulator for execution
        self._simulator = SequentialSimulator(self._up_problem)
        self._current_state = self._simulator.get_initial_state()
        self._initial_state = self._simulator.get_initial_state()

        # Adapter for conversions
        self._adapter = UPAdapter()

        logger.info(f"ActiveEnvironment initialized: "
                   f"{len(self._domain.lifted_actions)} actions, "
                   f"{len(self._initial_fluents)} initial fluents")

    # ========== Public Interface ==========

    def get_state(self) -> Set[str]:
        """
        Get current state as set of grounded fluent strings.

        Returns:
            Set of grounded fluents (e.g., {'clear_a', 'on_b_c', 'handempty'})

        Example:
            state = env.get_state()
            if 'clear_a' in state:
                print("Block a is clear")
        """
        return self._adapter.up_state_to_fluent_set(
            self._current_state,
            self._up_problem
        )

    def execute(self, action_name: str, parameters: List[str]) -> Tuple[bool, float]:
        """
        Execute grounded action.

        Args:
            action_name: Action name (e.g., 'pick-up', 'stack')
            parameters: Object list (e.g., ['a'] or ['a', 'b'])

        Returns:
            Tuple of (success, runtime_seconds)
            - success: True if action was applicable and executed
            - runtime: Time taken for execution check/apply

        Example:
            success, runtime = env.execute('pick-up', ['a'])
            if success:
                print(f"Successfully picked up block a in {runtime:.3f}s")
        """
        start_time = time.perf_counter()

        try:
            # Find action schema
            action_schema = self._find_action_schema(action_name)
            if action_schema is None:
                logger.warning(f"Action {action_name} not found in domain")
                return False, time.perf_counter() - start_time

            # Validate parameter count
            if len(parameters) != len(action_schema.parameters):
                logger.warning(
                    f"Wrong parameter count for {action_name}: "
                    f"expected {len(action_schema.parameters)}, got {len(parameters)}"
                )
                return False, time.perf_counter() - start_time

            # Resolve parameter objects
            param_objects = self._resolve_parameters(parameters)
            if param_objects is None:
                return False, time.perf_counter() - start_time

            # Create action instance
            action_instance = ActionInstance(action_schema, tuple(param_objects))

            # Check applicability and execute
            if self._simulator.is_applicable(self._current_state, action_instance):
                self._current_state = self._simulator.apply(self._current_state, action_instance)
                logger.debug(f"Executed {action_name}({','.join(parameters)}) successfully")
                return True, time.perf_counter() - start_time
            else:
                logger.debug(f"Action {action_name}({','.join(parameters)}) not applicable")
                return False, time.perf_counter() - start_time

        except Exception as e:
            logger.error(f"Error executing {action_name}: {e}")
            return False, time.perf_counter() - start_time

    def execute_plan(self, plan: List[Tuple[str, List[str]]]) -> Tuple[bool, float]:
        """
        Execute sequence of grounded actions (a plan).

        Stops on first failure.

        Args:
            plan: List of (action_name, parameters) tuples

        Returns:
            Tuple of (all_success, total_runtime)
            - all_success: True if all actions succeeded
            - total_runtime: Total time for execution

        Example:
            plan = [
                ('unstack', ['c', 'b']),
                ('put-down', ['c']),
                ('unstack', ['b', 'a'])
            ]
            success, runtime = env.execute_plan(plan)
        """
        start_time = time.perf_counter()
        all_success = True

        for i, (action_name, parameters) in enumerate(plan):
            success, _ = self.execute(action_name, parameters)
            if not success:
                logger.warning(f"Plan failed at step {i+1}: {action_name}({','.join(parameters)})")
                all_success = False
                break

        total_runtime = time.perf_counter() - start_time
        logger.info(f"Plan execution: {'SUCCESS' if all_success else 'FAILED'} "
                   f"({len(plan)} steps, {total_runtime:.3f}s)")

        return all_success, total_runtime

    def reset(self) -> None:
        """
        Reset environment to initial state.

        Example:
            env.execute('pick-up', ['a'])
            env.reset()  # Back to initial state
        """
        self._current_state = self._simulator.get_initial_state()
        logger.debug("Environment reset to initial state")

    def is_goal_reached(self) -> bool:
        """
        Check if goal conditions are satisfied in current state.

        Returns:
            True if goal is reached

        Example:
            if env.is_goal_reached():
                print("Goal achieved!")
        """
        try:
            from unified_planning.model.walkers import StateEvaluator
            evaluator = StateEvaluator(self._up_problem)

            for goal in self._up_problem.goals:
                result = evaluator.evaluate(goal, self._current_state)
                if not result.is_true():
                    return False
            return True

        except Exception as e:
            logger.error(f"Error checking goal: {e}")
            return False

    # ========== Optional Query Methods ==========

    def get_applicable_actions(self) -> List[Tuple[str, List[str]]]:
        """
        Get list of applicable actions in current state.

        Returns:
            List of (action_name, parameters) tuples

        Note: This is a query method, not part of minimal interface.
              Useful for debugging and analysis.

        Example:
            applicable = env.get_applicable_actions()
            for action, params in applicable:
                print(f"Can execute: {action}({', '.join(params)})")
        """
        from src.core.grounding import ground_all_actions

        applicable = []

        # Get all grounded actions
        all_grounded = ground_all_actions(self._domain, require_injective=False)

        # Check each for applicability
        for grounded in all_grounded:
            # Find action schema
            action_schema = self._find_action_schema(grounded.action_name)
            if action_schema:
                # Resolve objects
                param_objects = self._resolve_parameters(grounded.objects)
                if param_objects:
                    # Create instance and check
                    action_instance = ActionInstance(action_schema, tuple(param_objects))
                    if self._simulator.is_applicable(self._current_state, action_instance):
                        applicable.append((grounded.action_name, grounded.objects))

        return applicable

    def get_all_grounded_actions(self) -> List[Tuple[str, List[str]]]:
        """
        Get all possible grounded actions (cached).

        Returns:
            List of all (action_name, parameters) tuples

        Note: This is a query method, not part of minimal interface.
              Returns all possible actions, not just applicable ones.
        """
        from src.core.grounding import ground_all_actions

        all_grounded = ground_all_actions(self._domain, require_injective=False)
        return [(g.action_name, g.objects) for g in all_grounded]

    # ========== Internal Helper Methods ==========

    def _find_action_schema(self, action_name: str):
        """Find UP action schema by name."""
        for action in self._up_problem.actions:
            if action.name == action_name:
                return action
        return None

    def _resolve_parameters(self, parameters: List[str]) -> List:
        """Resolve parameter names to UP object instances."""
        param_objects = []
        for param_name in parameters:
            obj = None
            for problem_obj in self._up_problem.all_objects:
                if problem_obj.name == param_name:
                    obj = problem_obj
                    break

            if obj is None:
                logger.warning(f"Object {param_name} not found in problem")
                return None

            param_objects.append(obj)

        return param_objects

    def get_state_UP_format(self) -> State:
        """
        Get current state in UP's native format.

        Returns:
            UP State object

        Note: Only for compatibility/debugging. Prefer get_state().
        """
        return self._current_state

    def log_state_transition(self, action: str, params: List[str],
                           old_state: Set[str], new_state: Set[str]) -> None:
        """
        Log a state transition for debugging.

        Args:
            action: Action name
            params: Action parameters
            old_state: State before action
            new_state: State after action
        """
        added = new_state - old_state
        removed = old_state - new_state

        logger.info(f"State transition: {action}({','.join(params)})")
        if added:
            logger.info(f"  Added: {added}")
        if removed:
            logger.info(f"  Removed: {removed}")

    def __str__(self) -> str:
        """String representation."""
        return f"ActiveEnvironment(domain={Path(self.domain_file).name})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"ActiveEnvironment(domain={self.domain_file}, problem={self.problem_file})"
