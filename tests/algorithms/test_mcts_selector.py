"""Tests for IGMCTSSelector and BoundedLookaheadSelector."""

from collections import defaultdict
from unittest.mock import MagicMock

import pytest

from information_gain_aml.algorithms.mcts_selector import (
    BoundedLookaheadSelector,
    IGMCTSSelector,
    simulate_action,
)


def _make_mock_learner(
    gains_by_state=None,
    applicability=1.0,
    eff_add=None,
    eff_del=None,
):
    """Create a mock learner with deterministic behavior for MCTS testing."""
    learner = MagicMock()

    # Default: empty effects
    learner.eff_add = defaultdict(set, eff_add or {})
    learner.eff_del = defaultdict(set, eff_del or {})

    # Applicability: always succeed by default
    learner._calculate_applicability_probability.return_value = applicability

    # bindP_inverse: identity (return input as-is for mock)
    learner.bindP_inverse.side_effect = lambda literals, objects: set(literals)

    # Gains per state: use provided map or return empty
    if gains_by_state:
        def _get_gains(state):
            key = frozenset(state)
            return list(gains_by_state.get(key, []))
        learner._calculate_all_action_gains.side_effect = _get_gains
    else:
        learner._calculate_all_action_gains.return_value = []

    return learner


class TestSimulateAction:
    def test_successful_action(self):
        learner = _make_mock_learner(
            eff_add={'move': {'on_a_b'}},
            eff_del={'move': {'clear_b'}},
        )
        state = {'clear_a', 'clear_b', 'on_a_table'}
        next_state, succeeded = simulate_action(learner, state, 'move', ['a', 'b'])
        assert succeeded
        assert 'on_a_b' in next_state
        assert 'clear_b' not in next_state

    def test_failed_action_zero_applicability(self):
        learner = _make_mock_learner(applicability=0.0)
        state = {'clear_a'}
        next_state, succeeded = simulate_action(learner, state, 'move', ['a', 'b'])
        assert not succeeded
        assert next_state == state


class TestIGMCTSSelector:
    def test_single_action(self):
        """Single available action is returned immediately."""
        learner = _make_mock_learner()
        selector = IGMCTSSelector(learner, iterations=10)
        action_gains = [('move', ['a', 'b'], 0.5)]

        result = selector.select_action({'clear_a'}, action_gains)
        assert result == ('move', ['a', 'b'], 0.5)

    def test_returns_valid_action(self):
        """Returned action must be from the input action_gains list."""
        state = frozenset({'clear_a', 'clear_b'})
        action_gains = [
            ('move', ['a', 'b'], 0.8),
            ('pick', ['a'], 0.3),
            ('put', ['b'], 0.1),
        ]
        learner = _make_mock_learner(
            gains_by_state={state: action_gains},
        )
        selector = IGMCTSSelector(learner, iterations=20, rollout_depth=2)

        result = selector.select_action(set(state), list(action_gains))
        assert result[0] in ['move', 'pick', 'put']
        # Must match one of the original entries
        assert result in action_gains

    def test_prefers_higher_gain(self):
        """With enough iterations, MCTS should prefer the higher-gain action."""
        state = frozenset({'s1'})
        action_gains = [
            ('good', ['a'], 10.0),
            ('bad', ['b'], 0.01),
        ]
        learner = _make_mock_learner(
            gains_by_state={state: action_gains},
        )
        selector = IGMCTSSelector(learner, iterations=30, rollout_depth=1)

        result = selector.select_action(set(state), list(action_gains))
        assert result[0] == 'good'

    def test_handles_no_actions_in_rollout(self):
        """If simulated states have no actions, rollout terminates gracefully."""
        state = frozenset({'s1'})
        action_gains = [
            ('move', ['a'], 0.5),
            ('pick', ['b'], 0.3),
        ]
        # Root state has actions, but deeper states have none (rollout terminates)
        learner = _make_mock_learner(
            gains_by_state={state: action_gains},
            # Other states default to empty gains
        )
        selector = IGMCTSSelector(learner, iterations=10, rollout_depth=5)

        result = selector.select_action(set(state), list(action_gains))
        assert result in action_gains

    def test_zero_iterations_returns_first(self):
        """With 0 iterations, falls back to first action (no tree built)."""
        learner = _make_mock_learner()
        selector = IGMCTSSelector(learner, iterations=0)
        action_gains = [
            ('move', ['a'], 0.5),
            ('pick', ['b'], 0.3),
        ]

        result = selector.select_action({'s1'}, action_gains)
        # No children created, falls back to action_gains[0]
        assert result == ('move', ['a'], 0.5)

    def test_zero_applicability_does_not_crash(self):
        """Actions that never succeed should not cause errors."""
        state = frozenset({'s1'})
        action_gains = [
            ('impossible', ['a'], 0.5),
        ]
        learner = _make_mock_learner(
            gains_by_state={state: action_gains},
            applicability=0.0,
        )
        selector = IGMCTSSelector(learner, iterations=10, rollout_depth=2)

        result = selector.select_action(set(state), list(action_gains))
        assert result[0] == 'impossible'


class TestBoundedLookaheadSelector:
    def test_single_action(self):
        learner = _make_mock_learner()
        selector = BoundedLookaheadSelector(learner, depth=2)
        action_gains = [('move', ['a'], 0.5)]

        result = selector.select_action({'s1'}, action_gains)
        assert result == ('move', ['a'], 0.5)

    def test_depth_one_is_greedy(self):
        learner = _make_mock_learner()
        selector = BoundedLookaheadSelector(learner, depth=1)
        action_gains = [
            ('best', ['a'], 0.9),
            ('worse', ['b'], 0.1),
        ]

        result = selector.select_action({'s1'}, action_gains)
        assert result == ('best', ['a'], 0.9)
