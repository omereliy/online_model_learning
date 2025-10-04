"""
Base abstract class for action model learning algorithms.
Provides unified interface for different learning approaches.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Any, Set
import logging

logger = logging.getLogger(__name__)


class BaseActionModelLearner(ABC):
    """
    Abstract base class for online action model learning algorithms.

    All learning algorithms must implement this interface to ensure
    compatibility with the experiment framework.
    """

    def __init__(self, domain_file: str, problem_file: str, **kwargs):
        """
        Initialize the learner with PDDL domain and problem.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            **kwargs: Algorithm-specific parameters
        """
        self.domain_file = domain_file
        self.problem_file = problem_file
        self.iteration_count = 0
        self.observation_count = 0
        self._converged = False

    @abstractmethod
    def select_action(self, state: Any) -> Tuple[str, List[str]]:
        """
        Select the next action to execute based on current state.

        Args:
            state: Current state (format depends on implementation)

        Returns:
            Tuple of (action_name, objects) where:
            - action_name: Name of the action (e.g., "pick-up")
            - objects: List of object names (e.g., ["a"])
        """
        pass

    @abstractmethod
    def observe(self,
                state: Any,
                action: str,
                objects: List[str],
                success: bool,
                next_state: Optional[Any] = None) -> None:
        """
        Observe the result of executing an action.

        Args:
            state: State before action execution
            action: Action name that was executed
            objects: Objects involved in the action
            success: Whether the action succeeded
            next_state: State after execution (if successful)
        """
        pass

    @abstractmethod
    def get_learned_model(self) -> Dict[str, Any]:
        """
        Export the current learned model.

        Returns:
            Dictionary containing the learned model with at minimum:
            - 'actions': Dict of learned action schemas
            - 'predicates': Set of discovered predicates
            - 'statistics': Learning statistics
        """
        pass

    def has_converged(self) -> bool:
        """
        Check if the learning has converged.

        Default implementation returns False. Override for specific
        convergence criteria.

        Returns:
            True if model has converged, False otherwise
        """
        return self._converged

    def reset(self) -> None:
        """
        Reset the learner to initial state.

        Override this method if the learner needs specific reset logic.
        """
        self.iteration_count = 0
        self.observation_count = 0
        self._converged = False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get learning statistics.

        Returns:
            Dictionary with learning statistics
        """
        return {
            'iterations': self.iteration_count,
            'observations': self.observation_count,
            'converged': self._converged
        }

    # Helper methods for state/action conversion (can be overridden)

    def state_to_string(self, state: Any) -> str:
        """
        Convert state to string representation for logging.

        Args:
            state: State in algorithm-specific format

        Returns:
            String representation of the state
        """
        return str(state)

    def action_to_string(self, action: str, objects: List[str]) -> str:
        """
        Convert action and objects to string representation.

        Args:
            action: Action name
            objects: Object parameters

        Returns:
            String representation like "pick-up(a)"
        """
        if objects:
            return f"{action}({','.join(objects)})"
        return action

    @staticmethod
    def parse_action_string(action_str: str) -> Tuple[str, List[str]]:
        """
        Parse action string into name and objects.

        Args:
            action_str: Action string like "pick-up(a)" or "stack(a,b)"

        Returns:
            Tuple of (action_name, objects)
        """
        if '(' in action_str:
            name = action_str[:action_str.index('(')]
            params_str = action_str[action_str.index('(') + 1:-1]
            if params_str:
                objects = [p.strip() for p in params_str.split(',')]
            else:
                objects = []
        else:
            name = action_str
            objects = []
        return name, objects
