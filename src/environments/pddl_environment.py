"""
PDDL Environment implementation using Unified Planning Framework.
Provides real PDDL action execution with actual state transitions.
"""

import logging
import time
from typing import Set, Tuple, List, Optional, Any
from pathlib import Path

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, SequentialSimulator
from unified_planning.model import State
from unified_planning.plans import SequentialPlan, ActionInstance
import unified_planning as up

logger = logging.getLogger(__name__)


class PDDLEnvironment:
    """
    PDDL environment that uses Unified Planning Framework for real action execution.

    This environment:
    - Executes grounded PDDL actions with real precondition checking
    - Tracks actual state transitions
    - Returns real success/failure based on action applicability
    - Provides state in format compatible with learning algorithms
    """

    def __init__(self, domain_file: str, problem_file: str):
        """
        Initialize PDDL environment with domain and problem files.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
        """
        self.domain_file = domain_file
        self.problem_file = problem_file

        # Parse PDDL files using UP
        reader = PDDLReader()
        self.problem = reader.parse_problem(domain_file, problem_file)

        # Initialize simulator for action execution
        self.simulator = SequentialSimulator(self.problem)

        # Track current state using simulator's state
        self.current_state = self.simulator.get_initial_state()
        self.initial_state = self.simulator.get_initial_state()

        # Cache grounded actions
        self._grounded_actions_cache = None

        logger.info(f"Initialized PDDL environment with domain: {domain_file}")
        logger.info(f"Problem has {len(self.problem.fluents)} fluents and {len(self.problem.actions)} action schemas")

    def get_state(self) -> Set[str]:
        """
        Get current state as a set of fluent strings.

        Returns:
            Set of strings representing true fluents in format: predicate_param1_param2
        """
        state_fluents = set()
        import itertools

        # Get all fluents from the problem
        for fluent in self.problem.fluents:
            # Get parameters for this fluent
            param_types = [p.type for p in fluent.signature]

            if not param_types:
                # Fluent with no parameters
                value = self.current_state.get_value(fluent())
                if value.is_true():
                    fluent_str = fluent.name
                    state_fluents.add(fluent_str)
            else:
                # Get all objects for each parameter type
                param_objects = []
                for ptype in param_types:
                    matching_objs = [obj for obj in self.problem.all_objects
                                   if self._type_matches(obj.type, ptype)]
                    param_objects.append(matching_objs)

                # Generate all combinations
                for obj_combo in itertools.product(*param_objects):
                    grounded_fluent = fluent(*obj_combo)
                    value = self.current_state.get_value(grounded_fluent)
                    if value.is_true():
                        obj_names = [o.name for o in obj_combo]
                        fluent_str = f"{fluent.name}_{'_'.join(obj_names)}"
                        state_fluents.add(fluent_str)

        return state_fluents

    def execute(self, action_name: str, parameters: List[str]) -> Tuple[bool, float]:
        """
        Execute a grounded action in the current state.

        Args:
            action_name: Name of the action to execute
            parameters: List of parameter values (object names)

        Returns:
            Tuple of (success, runtime_seconds)
        """
        start_time = time.perf_counter()

        try:
            # Find the action schema
            action_schema = None
            for action in self.problem.actions:
                if action.name == action_name:
                    action_schema = action
                    break

            if action_schema is None:
                logger.warning(f"Action {action_name} not found in domain")
                runtime = time.perf_counter() - start_time
                return False, runtime

            # Check parameter count
            if len(parameters) != len(action_schema.parameters):
                logger.warning(f"Wrong parameter count for {action_name}: expected {len(action_schema.parameters)}, got {len(parameters)}")
                runtime = time.perf_counter() - start_time
                return False, runtime

            # Create action instance with parameters
            param_objects = []
            for param_name in parameters:
                # Find object in problem
                obj = None
                for problem_obj in self.problem.all_objects:
                    if problem_obj.name == param_name:
                        obj = problem_obj
                        break

                if obj is None:
                    logger.warning(f"Object {param_name} not found in problem")
                    runtime = time.perf_counter() - start_time
                    return False, runtime

                param_objects.append(obj)

            # Create action instance
            action_instance = ActionInstance(action_schema, tuple(param_objects))

            # Check if action is applicable in current state
            if self.simulator.is_applicable(self.current_state, action_instance):
                # Execute action
                self.current_state = self.simulator.apply(self.current_state, action_instance)

                runtime = time.perf_counter() - start_time
                logger.debug(f"Successfully executed {action_name}({','.join(parameters)})")
                return True, runtime
            else:
                runtime = time.perf_counter() - start_time
                logger.debug(f"Action {action_name}({','.join(parameters)}) not applicable in current state")
                return False, runtime

        except Exception as e:
            logger.error(f"Error executing action {action_name}: {e}")
            runtime = time.perf_counter() - start_time
            return False, runtime

    def reset(self) -> None:
        """Reset environment to initial state."""
        self.current_state = self.simulator.get_initial_state()
        logger.debug("Reset environment to initial state")

    def get_applicable_actions(self) -> List[Tuple[str, List[str]]]:
        """
        Get list of all applicable actions in current state.

        Returns:
            List of (action_name, parameters) tuples
        """
        applicable = []

        for action_schema in self.problem.actions:
            # Get all possible groundings for this action
            groundings = self._get_action_groundings(action_schema)

            for grounding in groundings:
                # Create action instance
                action_instance = ActionInstance(action_schema, grounding)

                # Check if applicable
                if self.simulator.is_applicable(self.current_state, action_instance):
                    # Convert to our format
                    param_names = [obj.name for obj in grounding]
                    applicable.append((action_schema.name, param_names))

        return applicable

    def get_all_grounded_actions(self) -> List[Tuple[str, List[str]]]:
        """
        Get all grounded actions for the domain (cached).

        Returns:
            List of all possible (action_name, parameters) tuples
        """
        if self._grounded_actions_cache is not None:
            return self._grounded_actions_cache

        all_actions = []

        for action_schema in self.problem.actions:
            # Get all possible groundings
            groundings = self._get_action_groundings(action_schema)

            for grounding in groundings:
                param_names = [obj.name for obj in grounding]
                all_actions.append((action_schema.name, param_names))

        self._grounded_actions_cache = all_actions
        return all_actions

    def is_goal_reached(self) -> bool:
        """
        Check if goal is satisfied in current state.

        Returns:
            True if goal is reached, False otherwise
        """
        # Check each goal condition
        from unified_planning.model.walkers import StateEvaluator
        evaluator = StateEvaluator(self.problem)

        for goal in self.problem.goals:
            result = evaluator.evaluate(goal, self.current_state)
            if not result.is_true():
                return False
        return True

    def _fluent_to_string(self, fluent) -> str:
        """
        Convert UP fluent to string format.

        Args:
            fluent: UP fluent expression

        Returns:
            String in format predicate_param1_param2
        """
        try:
            # Get fluent name and arguments
            if hasattr(fluent, 'fluent'):
                # It's a fluent expression
                name = fluent.fluent().name
                args = fluent.args

                if args:
                    # Join with underscores
                    arg_names = [str(arg) for arg in args]
                    return f"{name}_{'_'.join(arg_names)}"
                else:
                    return name
            else:
                # Try to convert to string directly
                return str(fluent).replace('(', '').replace(')', '').replace(', ', '_').replace(' ', '_')

        except Exception as e:
            logger.warning(f"Could not convert fluent to string: {fluent}, error: {e}")
            return ""

    def _get_action_groundings(self, action_schema):
        """
        Get all possible groundings for an action schema.

        Args:
            action_schema: UP action schema

        Returns:
            List of tuples of objects representing groundings
        """
        import itertools

        if not action_schema.parameters:
            return [()]  # Action with no parameters

        # Get all possible combinations of objects for parameters
        param_domains = []
        for param in action_schema.parameters:
            # Get objects of the right type
            matching_objects = []
            for obj in self.problem.all_objects:
                # Check if object type matches parameter type
                if self._type_matches(obj.type, param.type):
                    matching_objects.append(obj)
            param_domains.append(matching_objects)

        # Generate all combinations
        groundings = []
        for combination in itertools.product(*param_domains):
            # Check for duplicate objects if action doesn't allow it
            if len(set(combination)) == len(combination):  # No duplicates
                groundings.append(combination)

        return groundings

    def _type_matches(self, obj_type, param_type) -> bool:
        """
        Check if object type matches parameter type (including type hierarchy).

        Args:
            obj_type: Object's type
            param_type: Parameter's required type

        Returns:
            True if types match
        """
        # Direct match
        if obj_type == param_type:
            return True

        # Check if obj_type is subtype of param_type
        if hasattr(obj_type, 'is_subtype_of'):
            return obj_type.is_subtype_of(param_type)

        # For untyped domains or when types don't match exactly
        # We might need more sophisticated type checking here
        return str(obj_type) == str(param_type)

    def get_state_UP_format(self) -> State:
        """
        Get current state in UP's native format.

        Returns:
            UP State object
        """
        return self.current_state

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
            logger.info(f"  Added fluents: {added}")
        if removed:
            logger.info(f"  Removed fluents: {removed}")