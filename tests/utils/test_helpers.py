"""
Helper functions and mocks for testing.
"""

from unittest.mock import MagicMock, patch
import random


def mock_olam_select_action():
    """
    Create a mock for OLAM's select_action that returns dummy actions
    without calling the actual planner.
    """
    actions = [
        ("pick-up", ["a"]),
        ("put-down", ["a"]),
        ("stack", ["a", "b"]),
        ("unstack", ["a", "b"]),
    ]

    def select_action_mock(self):
        # Return a random action index and strategy
        action_idx = random.randint(0, len(self.action_list) - 1)
        strategy = "mock_strategy"
        return action_idx, strategy

    return select_action_mock


def mock_olam_planner():
    """
    Create a mock for OLAM's planner calls that returns dummy plans.
    """
    def mock_fd_dummy():
        # Return a simple dummy plan
        plan = ["(pick-up a)", "(stack a b)", "(pick-up c)"]
        found = True
        return plan, found

    def mock_fd_test():
        # Return empty plan for test
        return [], False

    return {
        'FD_dummy': mock_fd_dummy,
        'FD_test': mock_fd_test,
        'FD_dummy_goal': mock_fd_dummy,
        'FD_dummy_real_goal': mock_fd_dummy,
    }


class MockOLAMAdapter:
    """
    Mock version of OLAMAdapter for testing without external dependencies.
    """

    def __init__(self, domain_file, problem_file, **kwargs):
        self.domain_file = domain_file
        self.problem_file = problem_file
        self.iteration_count = 0
        self.learned_model = {}
        self.converged = False

        # Mock action list
        self.action_list = [
            "(pick-up a)",
            "(put-down a)",
            "(stack a b)",
            "(unstack a b)",
        ]

    def select_action(self, state):
        """Select a random mock action."""
        self.iteration_count += 1
        actions = [
            ("pick-up", ["a"]),
            ("put-down", ["a"]),
            ("stack", ["a", "b"]),
            ("unstack", ["a", "b"]),
        ]
        return random.choice(actions)

    def observe(self, state, action, objects, success, next_state=None):
        """Mock observe method."""
        pass

    def update_model(self):
        """Mock update method."""
        pass

    def get_learned_model(self):
        """Return mock learned model."""
        return {
            "actions": {
                "pick-up": {
                    "preconditions": ["clear(?x)", "handempty"],
                    "positive_effects": ["holding(?x)"],
                    "negative_effects": ["clear(?x)", "handempty"]
                }
            }
        }

    def has_converged(self):
        """Check mock convergence."""
        if self.iteration_count > 50:
            self.converged = True
        return self.converged
