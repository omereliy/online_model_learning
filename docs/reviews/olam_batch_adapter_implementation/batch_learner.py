"""
Base class for batch-oriented action model learning algorithms.

Batch algorithms run their entire learning process in one execution,
rather than being controlled step-by-step by an external loop.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

from .base_learner import BaseActionModelLearner

logger = logging.getLogger(__name__)


class BatchAlgorithmAdapter(BaseActionModelLearner):
    """
    Base class for algorithms that run in batch mode.

    Batch algorithms execute their entire learning process autonomously,
    then return complete results. This contrasts with iterative algorithms
    that expose select_action() and observe() for per-iteration control.

    Examples of batch algorithms:
    - OLAM: Has its own internal learning loop
    - Offline learning algorithms that process recorded traces

    Examples of iterative algorithms:
    - Information Gain: Requires per-iteration action selection
    - Optimistic Exploration: Per-iteration model updates
    """

    def __init__(self, domain_file: str, problem_file: str, **kwargs):
        """
        Initialize batch algorithm adapter.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            **kwargs: Algorithm-specific parameters
        """
        super().__init__(domain_file, problem_file, **kwargs)
        logger.info(f"Initialized batch algorithm adapter: {self.__class__.__name__}")

    @abstractmethod
    def run_experiment(self, max_iterations: int, **kwargs) -> Dict[str, Any]:
        """
        Run the complete experiment and return all results.

        This method executes the algorithm's entire learning process,
        from initial state to convergence or max iterations.

        Args:
            max_iterations: Maximum number of learning iterations
            **kwargs: Additional experiment parameters

        Returns:
            Dictionary containing:
            - 'learned_model': The final learned action model
            - 'metrics': Per-iteration or aggregate metrics
            - 'final_iteration': Number of iterations completed
            - 'converged': Whether the algorithm converged
            - Algorithm-specific data
        """
        pass

    def select_action(self, state: Any):
        """
        Not supported for batch algorithms.

        Raises:
            NotImplementedError: Batch algorithms don't support per-iteration control
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is a batch algorithm and does not support "
            f"per-iteration action selection. Use run_experiment() instead."
        )

    def observe(self, state, action, objects, success, next_state=None):
        """
        Not supported for batch algorithms.

        Raises:
            NotImplementedError: Batch algorithms don't support per-iteration control
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is a batch algorithm and does not support "
            f"per-iteration observations. Use run_experiment() instead."
        )

    def get_learned_model(self) -> Dict[str, Any]:
        """
        Get the learned model after run_experiment() completes.

        This method should be called after run_experiment() to retrieve
        the final learned model. The model format follows BaseActionModelLearner
        conventions for compatibility with analysis tools.

        Returns:
            Dictionary containing the learned model
        """
        if not hasattr(self, '_learned_model') or self._learned_model is None:
            logger.warning(
                "get_learned_model() called before run_experiment() - "
                "returning empty model"
            )
            return {
                'actions': {},
                'predicates': set(),
                'statistics': self.get_statistics()
            }

        return self._learned_model
