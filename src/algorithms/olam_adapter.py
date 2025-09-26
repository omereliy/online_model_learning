"""
OLAM (Online Learning of Action Models) adapter implementation.
Integrates OLAM learner into the unified experiment framework.
"""

import sys
import os
import logging
from typing import Tuple, List, Dict, Optional, Any, Set
from pathlib import Path
import re
from collections import defaultdict

# Add OLAM to path
olam_path = '/home/omer/projects/OLAM'
if olam_path not in sys.path:
    sys.path.append(olam_path)

# Import OLAM components
try:
    from OLAM.Learner import Learner as OLAMLearner
    from Util.PddlParser import PddlParser
    from Util.Simulator import Simulator
    import Configuration
except ImportError as e:
    logging.error(f"Failed to import OLAM components: {e}")
    raise

from .base_learner import BaseActionModelLearner

logger = logging.getLogger(__name__)


class OLAMAdapter(BaseActionModelLearner):
    """
    Adapter for OLAM (Online Learning of Action Models) algorithm.

    This adapter wraps the OLAM Learner to conform to the BaseActionModelLearner
    interface, handling state and action format conversions between UP and OLAM.
    """

    def __init__(self,
                 domain_file: str,
                 problem_file: str,
                 max_iterations: int = 1000,
                 eval_frequency: int = 10,
                 **kwargs):
        """
        Initialize OLAM adapter.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            max_iterations: Maximum learning iterations
            eval_frequency: How often to evaluate the model
            **kwargs: Additional parameters
        """
        super().__init__(domain_file, problem_file, **kwargs)

        self.max_iterations = max_iterations
        self.eval_frequency = eval_frequency

        # Initialize OLAM components
        self._initialize_olam()

        # State tracking
        self.current_state = None
        self.last_action_idx = None
        self.last_action_str = None

        # Mappings
        self.action_idx_to_str = {}
        self.action_str_to_idx = {}
        self._build_action_mappings()

    def _initialize_olam(self):
        """Initialize OLAM learner and related components."""
        # Create temporary PDDL directory for OLAM (it expects files in PDDL/)
        self._setup_olam_pddl_directory()

        # Initialize OLAM parser
        self.parser = PddlParser()

        # Extract action list from domain
        self.action_list = self._extract_action_list()

        # Initialize OLAM learner
        self.learner = OLAMLearner(self.parser, self.action_list, self.eval_frequency)

        # Initialize required OLAM attributes
        from timeit import default_timer
        self.learner.initial_timer = default_timer()
        self.learner.max_time_limit = 3600  # 1 hour default time limit
        if not hasattr(self.learner, 'current_plan'):
            self.learner.current_plan = []

        # Create simulator for state tracking
        self.simulator = Simulator()

        logger.info(f"Initialized OLAM with {len(self.action_list)} actions")

    def _setup_olam_pddl_directory(self):
        """
        Setup PDDL directory structure expected by OLAM.
        OLAM expects files in a 'PDDL/' directory relative to working directory.
        """
        # Create PDDL directory if it doesn't exist
        pddl_dir = Path("PDDL")
        pddl_dir.mkdir(exist_ok=True)

        # Create Info directory for OLAM
        info_dir = Path("Info")
        info_dir.mkdir(exist_ok=True)

        # Copy domain and problem files
        import shutil
        shutil.copy(self.domain_file, pddl_dir / "domain.pddl")
        shutil.copy(self.problem_file, pddl_dir / "facts.pddl")

        # OLAM also expects a domain_learned.pddl file (starts empty)
        # Copy domain as initial learned domain
        shutil.copy(self.domain_file, pddl_dir / "domain_learned.pddl")

        logger.debug(f"Setup OLAM PDDL directory with domain and problem files")

    def _extract_action_list(self) -> List[str]:
        """
        Extract grounded action list from domain and problem files.

        Returns:
            List of grounded action strings in OLAM format
        """
        actions = []

        # Parse domain to get action schemas
        with open(self.domain_file, 'r') as f:
            domain_content = f.read().lower()

        # Extract action names
        action_names = re.findall(r':action\s+(\S+)', domain_content)

        # Parse problem to get objects
        with open(self.problem_file, 'r') as f:
            problem_content = f.read().lower()

        # Extract objects
        objects_section = re.search(r':objects(.*?)\)', problem_content, re.DOTALL)
        if objects_section:
            objects_text = objects_section.group(1)
            # Extract object names (ignore types)
            object_names = []
            for token in objects_text.split():
                if token != '-' and not token.startswith('-'):
                    if token not in ['block', 'object']:  # Skip type names
                        object_names.append(token)

        # For now, create simple grounded actions
        # In practice, OLAM will handle this more sophisticatedly
        for action in action_names:
            if action in ['pick-up', 'put-down']:
                # Single parameter actions
                for obj in object_names:
                    actions.append(f"{action}({obj})")
            elif action in ['stack', 'unstack']:
                # Two parameter actions
                for obj1 in object_names:
                    for obj2 in object_names:
                        if obj1 != obj2:  # Can't stack/unstack object on itself
                            actions.append(f"{action}({obj1},{obj2})")
            else:
                # Parameterless action (e.g., handempty)
                actions.append(f"{action}()")

        return sorted(actions)

    def _build_action_mappings(self):
        """Build mappings between action indices and string representations."""
        for idx, action_str in enumerate(self.learner.action_labels):
            self.action_idx_to_str[idx] = action_str
            self.action_str_to_idx[action_str] = idx

    def select_action(self, state: Any) -> Tuple[str, List[str]]:
        """
        Select next action using OLAM's action selection strategy.

        Args:
            state: Current state (UP format or set of fluent strings)

        Returns:
            Tuple of (action_name, objects)
        """
        self.iteration_count += 1

        # Convert state to OLAM format
        olam_state = self._up_state_to_olam(state)
        self._update_simulator_state(olam_state)

        # Use OLAM's action selection
        action_idx, strategy = self.learner.select_action()
        self.last_action_idx = action_idx

        # Convert to our format
        action_str = self.action_idx_to_str[action_idx]
        action_name, objects = self.parse_action_string(action_str)

        self.last_action_str = action_str
        self.current_state = olam_state

        logger.debug(f"Selected action: {action_name}({','.join(objects)}) using strategy {strategy}")

        return action_name, objects

    def observe(self,
                state: Any,
                action: str,
                objects: List[str],
                success: bool,
                next_state: Optional[Any] = None) -> None:
        """
        Observe action execution result and update OLAM model.

        Args:
            state: State before action execution
            action: Action name that was executed
            objects: Objects involved in the action
            success: Whether the action succeeded
            next_state: State after execution (if successful)
        """
        self.observation_count += 1

        # Convert to OLAM format
        action_str = self._up_action_to_olam(action, objects)

        if not success:
            # Learn from failed action
            if self.last_action_idx is not None:
                # Update simulator with current state for OLAM
                olam_state = self._up_state_to_olam(state)
                self._update_simulator_state(olam_state)
                self.learner.learn_failed_action_precondition(self.simulator)
                logger.debug(f"Learned from failed action: {action_str}")
        else:
            # Learn from successful action
            if next_state is not None:
                # Convert states
                olam_state = self._up_state_to_olam(state)
                olam_next_state = self._up_state_to_olam(next_state)

                # Update preconditions
                changed = self.learner.add_operator_precondition(action_str)
                if changed:
                    logger.debug(f"Updated preconditions for {action_str}")

                # Update effects
                self.learner.add_operator_effects(action_str, olam_state, olam_next_state)
                logger.debug(f"Updated effects for {action_str}")

                # Update simulator state
                self._update_simulator_state(olam_next_state)

        # Check convergence periodically
        if self.observation_count % self.eval_frequency == 0:
            self._check_convergence()

    def get_learned_model(self) -> Dict[str, Any]:
        """
        Export the current learned model from OLAM.

        Returns:
            Dictionary containing the learned model
        """
        model = {
            'actions': {},
            'predicates': set(),
            'statistics': self.get_statistics()
        }

        # Extract learned operators from OLAM
        for action_label in self.learner.action_labels:
            action_name, objects = self.parse_action_string(action_label)

            model['actions'][action_label] = {
                'name': action_name,
                'parameters': objects,
                'preconditions': {
                    'certain': self.learner.operator_certain_predicates.get(action_label, []),
                    'uncertain': self.learner.operator_uncertain_predicates.get(action_label, []),
                    'negative': self.learner.operator_negative_preconditions.get(action_label, [])
                },
                'effects': {
                    'add': [],  # OLAM tracks these internally
                    'delete': []
                }
            }

            # Collect all predicates
            for pred_list in model['actions'][action_label]['preconditions'].values():
                if pred_list:
                    for pred in pred_list:
                        if isinstance(pred, str):
                            # Extract predicate name
                            pred_name = pred.strip('()').split('(')[0].split()[0]
                            model['predicates'].add(pred_name)

        return model

    def has_converged(self) -> bool:
        """
        Check if OLAM has converged.

        Returns:
            True if model has converged, False otherwise
        """
        # OLAM sets model_convergence flag
        if hasattr(self.learner, 'model_convergence'):
            self._converged = self.learner.model_convergence

        # Also check iteration limit
        if self.iteration_count >= self.max_iterations:
            self._converged = True

        return self._converged

    def _check_convergence(self):
        """Check and update convergence status."""
        # OLAM has its own convergence detection
        # We can also add custom convergence criteria here
        if hasattr(self.learner, 'model_convergence'):
            self._converged = self.learner.model_convergence

    # ========== State and Action Conversion Methods ==========

    def _up_state_to_olam(self, state: Any) -> List[str]:
        """
        Convert UP state format to OLAM format.

        Args:
            state: UP state (set of fluent strings or similar)

        Returns:
            List of PDDL predicate strings for OLAM
        """
        olam_state = []

        if isinstance(state, set):
            # Assume state is a set of fluent strings like {'clear_a', 'on_a_b'}
            for fluent in sorted(state):
                # Convert underscore format to PDDL format
                olam_pred = self._fluent_to_pddl_string(fluent)
                if olam_pred:
                    olam_state.append(olam_pred)
        elif isinstance(state, list):
            # Already in OLAM format
            olam_state = state
        else:
            # Try to handle other formats
            logger.warning(f"Unexpected state format: {type(state)}")

        return sorted(olam_state)

    def _olam_state_to_up(self, olam_state: List[str]) -> Set[str]:
        """
        Convert OLAM state format to UP format.

        Args:
            olam_state: List of PDDL predicate strings

        Returns:
            Set of fluent strings in UP format
        """
        up_state = set()

        for pred in olam_state:
            # Convert "(pred obj1 obj2)" to "pred_obj1_obj2"
            fluent = self._pddl_string_to_fluent(pred)
            if fluent:
                up_state.add(fluent)

        return up_state

    def _fluent_to_pddl_string(self, fluent: str) -> str:
        """
        Convert fluent string to PDDL predicate string.

        Args:
            fluent: Fluent like "clear_a" or "on_a_b"

        Returns:
            PDDL string like "(clear a)" or "(on a b)"
        """
        parts = fluent.split('_')

        if len(parts) == 1:
            # Parameterless predicate like "handempty"
            return f"({parts[0]})"
        elif len(parts) == 2:
            # Single parameter like "clear_a"
            return f"({parts[0]} {parts[1]})"
        elif len(parts) == 3:
            # Two parameters like "on_a_b"
            return f"({parts[0]} {parts[1]} {parts[2]})"
        else:
            logger.warning(f"Unexpected fluent format: {fluent}")
            return ""

    def _pddl_string_to_fluent(self, pddl_str: str) -> str:
        """
        Convert PDDL predicate string to fluent string.

        Args:
            pddl_str: PDDL string like "(clear a)" or "(on a b)"

        Returns:
            Fluent like "clear_a" or "on_a_b"
        """
        # Remove parentheses and split
        content = pddl_str.strip('()')
        parts = content.split()

        if len(parts) == 1:
            # Parameterless
            return parts[0]
        else:
            # Join with underscores
            return '_'.join(parts)

    def _up_action_to_olam(self, action: str, objects: List[str]) -> str:
        """
        Convert UP action format to OLAM format.

        Args:
            action: Action name
            objects: List of object names

        Returns:
            OLAM action string like "pick-up(a)"
        """
        if objects:
            return f"{action}({','.join(objects)})"
        else:
            return f"{action}()"

    def _olam_action_to_up(self, olam_action: str) -> Tuple[str, List[str]]:
        """
        Convert OLAM action string to UP format.

        Args:
            olam_action: OLAM action string

        Returns:
            Tuple of (action_name, objects)
        """
        return self.parse_action_string(olam_action)

    def _update_simulator_state(self, olam_state: List[str]):
        """
        Update OLAM's simulator with current state.

        Args:
            olam_state: State in OLAM format
        """
        # OLAM's simulator expects state as a list of PDDL predicates
        self.simulator.state = olam_state

    def reset(self) -> None:
        """Reset the learner to initial state."""
        super().reset()

        # Reinitialize OLAM components
        self._initialize_olam()

        # Reset state tracking
        self.current_state = None
        self.last_action_idx = None
        self.last_action_str = None