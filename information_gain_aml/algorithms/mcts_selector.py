"""
MCTS-based action selection for Information Gain learning.

Phase 1: Bounded lookahead — evaluates top-k actions with depth-limited
recursive future gain estimation using the current partial model.

Phase 2: Full UCT-based MCTS — builds a search tree with UCT selection,
expansion, greedy rollout, and backpropagation for deeper exploration.
"""

import logging
import math
import random
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

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


class IGMCTSSelector:
    """
    Full UCT-based MCTS action selector for information gain maximization.

    Uses Monte Carlo Tree Search with UCT (Upper Confidence bounds applied
    to Trees) to select actions that maximize expected future information gain.

    The four MCTS phases per iteration:
    1. Selection — traverse tree using UCT to find a promising leaf
    2. Expansion — add one new child node by trying an untried action
    3. Rollout — simulate to a fixed depth using greedy policy
    4. Backpropagation — update visit counts and rewards along the path
    """

    _EXPLORATION_C = math.sqrt(2)  # UCT exploration constant
    _DISCOUNT = 0.9                # future gain discount factor

    class _Node:
        """MCTS tree node with UCT statistics."""

        def __init__(
            self,
            state: Set[str],
            parent: Optional['IGMCTSSelector._Node'] = None,
            action: Optional[str] = None,
            objects: Optional[List[str]] = None,
        ):
            self.state = state
            self.parent = parent
            self.action = action        # edge from parent
            self.objects = objects
            self.children: list[IGMCTSSelector._Node] = []
            self.visit_count: int = 0
            self.total_reward: float = 0.0
            self.untried_actions: Optional[list[tuple[str, list[str], float]]] = None

        def uct_value(self, c: float) -> float:
            if self.visit_count == 0:
                return float('inf')
            assert self.parent is not None
            return (
                self.total_reward / self.visit_count
                + c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
            )

    def __init__(
        self,
        learner: 'InformationGainLearner',
        iterations: int = 50,
        rollout_depth: int = 5,
    ):
        self.learner = learner
        self.iterations = iterations
        self.rollout_depth = rollout_depth

    def select_action(
        self,
        state: Set[str],
        action_gains: list[tuple[str, list[str], float]],
    ) -> tuple[str, list[str], float]:
        """
        Select action using MCTS with UCT.

        Args:
            state: Current world state
            action_gains: Pre-computed (action, objects, gain) sorted by gain desc

        Returns:
            Best (action, objects, gain) tuple based on MCTS evaluation
        """
        if len(action_gains) <= 1:
            return action_gains[0]

        root = self._Node(state=state.copy())
        root.untried_actions = list(action_gains)

        for _ in range(self.iterations):
            node = self._select(root)
            child = self._expand(node)
            if child is None:
                child = node
            reward = self._rollout(child)
            self._backpropagate(child, reward)

        return self._best_child_by_visits(root, action_gains)

    def _select(self, node: _Node) -> _Node:
        """Traverse tree using UCT until reaching an expandable or terminal node."""
        while node.untried_actions is not None and len(node.untried_actions) == 0:
            if not node.children:
                break
            node = max(node.children, key=lambda c: c.uct_value(self._EXPLORATION_C))
        return node

    def _expand(self, node: _Node) -> Optional[_Node]:
        """Expand node by trying one untried action. Returns new child or None."""
        # Lazy init: compute available actions for non-root nodes
        if node.untried_actions is None:
            node.untried_actions = self.learner._calculate_all_action_gains(node.state)

        if not node.untried_actions:
            return None

        action, objects, _gain = node.untried_actions.pop(0)
        next_state, _succeeded = simulate_action(
            self.learner, node.state, action, objects
        )

        child = self._Node(
            state=next_state,
            parent=node,
            action=action,
            objects=objects,
        )
        node.children.append(child)
        return child

    def _rollout(self, node: _Node) -> float:
        """Simulate from node using greedy policy, accumulating discounted IG."""
        state = node.state.copy()
        total_reward = 0.0
        discount = 1.0

        for _ in range(self.rollout_depth):
            gains = self.learner._calculate_all_action_gains(state)
            if not gains:
                break

            action, objects, gain = gains[0]
            total_reward += discount * gain
            discount *= self._DISCOUNT

            state, _succeeded = simulate_action(
                self.learner, state, action, objects
            )

        return total_reward

    def _backpropagate(self, node: _Node, reward: float) -> None:
        """Propagate reward up the tree, updating visit counts and totals."""
        current: Optional[IGMCTSSelector._Node] = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            current = current.parent

    def _best_child_by_visits(
        self,
        root: _Node,
        action_gains: list[tuple[str, list[str], float]],
    ) -> tuple[str, list[str], float]:
        """Select root child with highest visit count."""
        if not root.children:
            return action_gains[0]

        best_child = max(root.children, key=lambda c: c.visit_count)

        logger.debug(
            f"MCTS: {best_child.action}({','.join(best_child.objects or [])}) "
            f"visits={best_child.visit_count} avg_reward="
            f"{best_child.total_reward / max(best_child.visit_count, 1):.3f}"
        )

        # Match back to action_gains for the original gain value
        for action, objects, gain in action_gains:
            if action == best_child.action and objects == best_child.objects:
                return action, objects, gain

        # Fallback: should not reach here since root children come from action_gains
        action = best_child.action or action_gains[0][0]
        objects = best_child.objects or action_gains[0][1]
        return action, objects, 0.0
