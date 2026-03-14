"""
MCTS-based action selection for Information Gain learning.

Phase 1: Bounded lookahead — evaluates top-k actions with depth-limited
recursive future gain estimation using the current partial model.

Future phases will add full MCTS with UCT, tree nodes, and rollouts.
"""

import logging
import random
from typing import TYPE_CHECKING, List, Set, Tuple

if TYPE_CHECKING:
    from information_gain_aml.algorithms.information_gain import InformationGainLearner

logger = logging.getLogger(__name__)


def simulate_action(
    learner: 'InformationGainLearner',
    state: Set[str],
    action: str,
    objects: List[str],
) -> tuple[Set[str], bool]:
    """
    Simulate action execution using the learner's partial model.

    Uses confirmed effects only (pessimistic simulation).
    Applicability is sampled stochastically using P(app).

    Args:
        learner: The InformationGainLearner (read-only access)
        state: Current world state as grounded fluent strings
        action: Action schema name
        objects: Object binding

    Returns:
        (next_state, succeeded) — new state and whether the action applied
    """
    prob_app = learner._calculate_applicability_probability(action, objects, state)

    if random.random() > prob_app:
        return state.copy(), False

    # Apply confirmed effects
    new_state = state.copy()
    grounded_adds = learner.bindP_inverse(learner.eff_add[action], objects)
    grounded_dels = learner.bindP_inverse(learner.eff_del[action], objects)
    new_state -= grounded_dels
    new_state |= grounded_adds

    return new_state, True


class BoundedLookaheadSelector:
    """
    Depth-limited lookahead action selector.

    Evaluates the top-k actions by greedy information gain, simulates
    one step forward using the partial model, then recursively estimates
    future gain. Returns the action with highest total discounted gain.

    At depth=1 this degrades to greedy selection.
    """

    def __init__(
        self,
        learner: 'InformationGainLearner',
        depth: int = 2,
        top_k: int = 5,
        discount: float = 0.9,
    ):
        self.learner = learner
        self.depth = depth
        self.top_k = top_k
        self.gamma = discount

    def select_action(
        self,
        state: Set[str],
        action_gains: list[tuple[str, list[str], float]],
    ) -> tuple[str, list[str], float]:
        """
        Select action using bounded lookahead over pre-computed gains.

        Args:
            state: Current world state
            action_gains: Pre-computed (action, objects, gain) sorted by gain desc

        Returns:
            Best (action, objects, gain) tuple considering future gains
        """
        if len(action_gains) <= 1 or self.depth <= 1:
            return action_gains[0]

        best_value = -1.0
        best = action_gains[0]

        for action, objects, gain in action_gains[:self.top_k]:
            next_state, _succeeded = simulate_action(
                self.learner, state, action, objects
            )
            future = self._evaluate_future(next_state, self.depth - 1)
            total = gain + self.gamma * future

            if total > best_value:
                best_value = total
                best = (action, objects, gain)

            logger.debug(
                f"Lookahead: {action}({','.join(objects)}) "
                f"immediate={gain:.3f} future={future:.3f} total={total:.3f}"
            )

        logger.debug(
            f"Lookahead selected: {best[0]}({','.join(best[1])}) "
            f"with total value {best_value:.3f}"
        )
        return best

    def _evaluate_future(self, state: Set[str], remaining_depth: int) -> float:
        """Recursively estimate future information gain from a simulated state."""
        if remaining_depth <= 0:
            return 0.0

        gains = self.learner._calculate_all_action_gains(state)
        if not gains:
            return 0.0

        best_gain = gains[0][2]
        if remaining_depth == 1:
            return best_gain

        # Simulate best action, recurse (greedy future assumption)
        action, objects, _ = gains[0]
        next_state, _ = simulate_action(self.learner, state, action, objects)
        return best_gain + self.gamma * self._evaluate_future(
            next_state, remaining_depth - 1
        )
