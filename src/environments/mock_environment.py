"""
Mock environment for testing experiment framework.

This is a temporary implementation for Phase 3 testing.
Phase 4 will provide a real PDDL environment using UP's SequentialSimulator.
"""

import random
from typing import Tuple, Set, List, Optional
import logging

logger = logging.getLogger(__name__)


class MockEnvironment:
    """
    Mock environment that simulates action execution for testing.

    This class provides a placeholder interface that will be replaced
    by a real PDDL environment in Phase 4.
    """

    def __init__(self, success_rate: float = 0.7, seed: Optional[int] = None):
        """
        Initialize the mock environment.

        Args:
            success_rate: Probability of action success (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.success_rate = success_rate
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        # Mock state for blocksworld-like domain
        self.current_state = self._generate_initial_state()

        logger.info(f"Initialized MockEnvironment with success_rate={success_rate}")

    def _generate_initial_state(self) -> Set[str]:
        """
        Generate an initial state for testing.

        Returns:
            Set of true fluents representing the state
        """
        # Mock blocksworld state with 3 blocks
        return {
            'clear_a',
            'on_a_b',
            'on_b_c',
            'ontable_c',
            'handempty'
        }

    def get_state(self) -> Set[str]:
        """
        Get the current state.

        Returns:
            Set of true fluents
        """
        return self.current_state.copy()

    def execute(self, action: str, objects: List[str]) -> Tuple[bool, float]:
        """
        Execute an action in the environment.

        Args:
            action: Action name
            objects: Objects involved in the action

        Returns:
            Tuple of (success, runtime) where:
            - success: Whether the action succeeded
            - runtime: Simulated execution time
        """
        # Simulate execution with random success/failure
        success = random.random() < self.success_rate

        # Simulate runtime between 0.01 and 0.1 seconds
        runtime = random.uniform(0.01, 0.1)

        # Update state if successful (simplified logic)
        if success:
            self._update_state(action, objects)

        action_str = f"{action}({','.join(objects)})" if objects else action
        logger.debug(f"Executed {action_str}: Success={success}, Runtime={runtime:.3f}s")

        return success, runtime

    def _update_state(self, action: str, objects: List[str]) -> None:
        """
        Update the state after successful action execution.

        This is a simplified mock implementation.

        Args:
            action: Action name
            objects: Objects involved
        """
        # Simple state updates for testing
        if action == 'pick-up' and objects:
            obj = objects[0]
            # Remove clear and add holding
            self.current_state.discard(f'clear_{obj}')
            self.current_state.discard(f'ontable_{obj}')
            self.current_state.discard('handempty')
            self.current_state.add(f'holding_{obj}')

        elif action == 'put-down' and objects:
            obj = objects[0]
            # Add clear and remove holding
            self.current_state.discard(f'holding_{obj}')
            self.current_state.add(f'clear_{obj}')
            self.current_state.add(f'ontable_{obj}')
            self.current_state.add('handempty')

        elif action == 'stack' and len(objects) >= 2:
            obj1, obj2 = objects[0], objects[1]
            # Stack obj1 on obj2
            self.current_state.discard(f'holding_{obj1}')
            self.current_state.discard(f'clear_{obj2}')
            self.current_state.add(f'on_{obj1}_{obj2}')
            self.current_state.add(f'clear_{obj1}')
            self.current_state.add('handempty')

        elif action == 'unstack' and len(objects) >= 2:
            obj1, obj2 = objects[0], objects[1]
            # Unstack obj1 from obj2
            self.current_state.discard(f'on_{obj1}_{obj2}')
            self.current_state.discard(f'clear_{obj1}')
            self.current_state.discard('handempty')
            self.current_state.add(f'holding_{obj1}')
            self.current_state.add(f'clear_{obj2}')

    def reset(self) -> None:
        """Reset the environment to initial state."""
        self.current_state = self._generate_initial_state()
        logger.debug("Reset environment to initial state")

    def get_applicable_actions(self) -> List[Tuple[str, List[str]]]:
        """
        Get list of currently applicable actions.

        Returns:
            List of (action_name, objects) tuples
        """
        applicable = []

        # Mock logic for applicable actions based on state
        blocks = ['a', 'b', 'c']

        # Check pick-up actions
        for block in blocks:
            if f'clear_{block}' in self.current_state and \
               f'ontable_{block}' in self.current_state and \
               'handempty' in self.current_state:
                applicable.append(('pick-up', [block]))

        # Check put-down actions
        for block in blocks:
            if f'holding_{block}' in self.current_state:
                applicable.append(('put-down', [block]))

        # Check stack actions
        for b1 in blocks:
            if f'holding_{b1}' in self.current_state:
                for b2 in blocks:
                    if b1 != b2 and f'clear_{b2}' in self.current_state:
                        applicable.append(('stack', [b1, b2]))

        # Check unstack actions
        for b1 in blocks:
            for b2 in blocks:
                if b1 != b2 and f'on_{b1}_{b2}' in self.current_state and \
                   f'clear_{b1}' in self.current_state and \
                   'handempty' in self.current_state:
                    applicable.append(('unstack', [b1, b2]))

        return applicable

    def is_goal_satisfied(self, goal: Set[str]) -> bool:
        """
        Check if a goal is satisfied in current state.

        Args:
            goal: Set of fluents that must be true

        Returns:
            True if goal is satisfied
        """
        return goal.issubset(self.current_state)

    def set_success_rate(self, rate: float) -> None:
        """
        Update the success rate for action execution.

        Args:
            rate: New success rate (0.0 to 1.0)
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Success rate must be between 0.0 and 1.0, got {rate}")

        self.success_rate = rate
        logger.info(f"Updated success rate to {rate}")

    def get_statistics(self) -> dict:
        """
        Get environment statistics.

        Returns:
            Dictionary with environment stats
        """
        return {
            'success_rate': self.success_rate,
            'state_size': len(self.current_state),
            'applicable_actions': len(self.get_applicable_actions())
        }