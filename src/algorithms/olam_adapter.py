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
                 pddl_handler=None,
                 pddl_environment=None,
                 bypass_java: bool = False,
                 use_system_java: bool = False,
                 **kwargs):
        """
        Initialize OLAM adapter.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            max_iterations: Maximum learning iterations
            eval_frequency: How often to evaluate the model
            pddl_handler: Optional PDDLHandler instance for proper grounding
            pddl_environment: Optional PDDL environment for action applicability checking
            bypass_java: If True, bypass Java dependency for action filtering
            use_system_java: If True, try to use system Java instead of bundled
            **kwargs: Additional parameters
        """
        super().__init__(domain_file, problem_file, **kwargs)

        self.max_iterations = max_iterations
        self.eval_frequency = eval_frequency
        self.bypass_java = bypass_java
        self.use_system_java = use_system_java
        self.pddl_environment = pddl_environment

        # Initialize PDDLHandler for proper action grounding
        if pddl_handler is None:
            from src.core.pddl_handler import PDDLHandler
            self.pddl_handler = PDDLHandler()
            self.pddl_handler.parse_domain_and_problem(domain_file, problem_file)
        else:
            self.pddl_handler = pddl_handler

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
        # Configure Java settings before OLAM initialization
        self._configure_java_settings()

        # Create temporary PDDL directory for OLAM (it expects files in PDDL/)
        self._setup_olam_pddl_directory()

        # Initialize PDDLHandler for predicate/object information
        from src.core.pddl_handler import PDDLHandler
        self.pddl_handler = PDDLHandler()
        self.pddl_handler.parse_domain_and_problem(self.domain_file, self.problem_file)

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

        # If bypassing Java, monkey-patch the Java computation method
        if self.bypass_java:
            self._setup_java_bypass()

        # Create simulator for state tracking
        self.simulator = Simulator()

        logger.info(f"Initialized OLAM with {len(self.action_list)} actions")
        if self.bypass_java:
            logger.info("Java bypass mode enabled - using Python fallback for action filtering")

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
        Extract grounded action list using PDDLHandler for proper domain-agnostic grounding.

        Returns:
            List of grounded action strings in OLAM format
        """
        grounded_actions = []

        # Use PDDLHandler's proper grounding mechanism
        for action, binding in self.pddl_handler._grounded_actions:
            if not binding:
                # Parameterless action
                action_str = f"{action.name}()"
            else:
                # Extract parameter values in order
                # We need to maintain the parameter order from the action definition
                param_values = []
                for param in action.parameters:
                    if param.name in binding:
                        param_values.append(binding[param.name].name)

                # Build OLAM format: action_name(param1,param2,...)
                if param_values:
                    action_str = f"{action.name}({','.join(param_values)})"
                else:
                    action_str = f"{action.name}()"

            grounded_actions.append(action_str)

        return sorted(grounded_actions)

    def _build_action_mappings(self):
        """Build mappings between action indices and string representations."""
        for idx, action_str in enumerate(self.learner.action_labels):
            self.action_idx_to_str[idx] = action_str
            self.action_str_to_idx[action_str] = idx

    def _configure_java_settings(self):
        """Configure Java settings for OLAM."""
        if self.bypass_java:
            # Set empty path to avoid Java calls
            Configuration.JAVA_BIN_PATH = ""
            return

        if self.use_system_java:
            # Try to use system Java
            Configuration.JAVA_BIN_PATH = "java"
            logger.info("Configured to use system Java")
        else:
            # Default OLAM behavior - look for bundled Java
            import os
            java_dir = os.path.join(os.path.dirname(Configuration.__file__), Configuration.JAVA_DIR)
            if os.path.exists(java_dir):
                java_dirs = [d for d in os.listdir(java_dir) if os.path.isdir(os.path.join(java_dir, d))]
                if java_dirs:
                    Configuration.JAVA_BIN_PATH = os.path.join(java_dir, java_dirs[0], "bin", "java")
                else:
                    logger.warning("No Java installation found in OLAM/Java directory")
                    Configuration.JAVA_BIN_PATH = ""
            else:
                Configuration.JAVA_BIN_PATH = ""

    def _setup_java_bypass(self):
        """Setup bypass for Java-dependent methods."""
        # Store original method
        self.learner._original_compute_not_executable = self.learner.compute_not_executable_actionsJAVA

        # Replace with Python fallback
        def compute_not_executable_bypass():
            """Properly compute non-executable actions using PDDL environment or Python fallback."""

            if self.pddl_environment:
                # Use PDDL environment for accurate action filtering
                try:
                    # Get current state from PDDL environment
                    current_state = self.pddl_environment.get_state()

                    # Get applicable actions from environment
                    applicable = self.pddl_environment.get_applicable_actions()
                    applicable_strs = {f"{name}({','.join(objs)})" for name, objs in applicable}

                    # Find NON-executable action indices
                    non_executable_indices = []
                    for idx, action_str in enumerate(self.action_list):
                        if action_str not in applicable_strs:
                            non_executable_indices.append(idx)

                    logger.debug(f"PDDL environment filtering: {len(non_executable_indices)}/{len(self.action_list)} non-executable")
                    return non_executable_indices

                except Exception as e:
                    logger.warning(f"Failed to use PDDL environment for filtering: {e}")
                    # Fall through to Python fallback

            # Fallback: use OLAM's Python method to compute non-executable
            logger.debug("Using Python fallback for non-executable computation")
            return self._compute_non_executable_python()

        self.learner.compute_not_executable_actionsJAVA = compute_not_executable_bypass

    def _compute_non_executable_python(self) -> List[int]:
        """
        Python fallback to compute non-executable actions without Java or PDDL environment.

        Returns:
            List of action indices that are NOT executable in the current state
        """
        # If no learning has happened yet, consider all actions potentially executable
        if not hasattr(self.learner, 'operator_certain_predicates'):
            logger.debug("No learned model yet, considering all actions potentially executable")
            return []

        # Read current state from OLAM's state files
        try:
            import os
            import re

            # Try to read current state from PDDL facts file
            facts_file = "PDDL/facts.pddl"
            if os.path.exists(facts_file):
                with open(facts_file, "r") as f:
                    data = [el.strip() for el in f.read().split("\n")]
                    facts = re.findall(r":init.*\(:goal", "".join(data))[0]
                    current_state_pddl = set(re.findall(r"\([^()]*\)", facts))
            else:
                # No state file, assume all actions potentially executable
                logger.debug("No PDDL facts file found, returning empty non-executable list")
                return []

            non_executable_indices = []

            # Check each action for basic precondition satisfaction
            for idx, action_label in enumerate(self.action_list):
                operator = action_label.split("(")[0]
                params = [el.strip() for el in action_label.split("(")[1][:-1].split(",")]

                # Get learned preconditions for this operator
                if operator in self.learner.operator_certain_predicates:
                    certain_preconds = self.learner.operator_certain_predicates[operator]

                    # Ground the preconditions with action parameters
                    grounded_preconds = []
                    for pred in certain_preconds:
                        grounded_pred = pred
                        for k, param in enumerate(params):
                            grounded_pred = grounded_pred.replace(f"?param_{k+1})", f"{param})")
                            grounded_pred = grounded_pred.replace(f"?param_{k+1} ", f"{param} ")
                        grounded_preconds.append(grounded_pred)

                    # Check if all certain preconditions are satisfied
                    if grounded_preconds and not all(pred in current_state_pddl for pred in grounded_preconds):
                        non_executable_indices.append(idx)

            logger.debug(f"Python fallback: {len(non_executable_indices)}/{len(self.action_list)} non-executable")
            return non_executable_indices

        except Exception as e:
            logger.warning(f"Error in Python fallback computation: {e}")
            # On error, return empty list (all actions potentially executable)
            return []

    def _compute_executable_actions_python(self, state: Set[str]) -> List[str]:
        """
        Python fallback for computing executable actions without Java.

        Args:
            state: Current state as set of fluent strings

        Returns:
            List of potentially executable action strings
        """
        executable = []

        for action_str in self.action_list:
            # Simple heuristic: check if action might be applicable
            # This is a simplified version - OLAM's Java does more sophisticated checking

            # For now, return all actions as potentially executable
            # A more sophisticated implementation would check preconditions
            executable.append(action_str)

        return executable

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

        General approach that works for any domain:
        1. Use PDDLHandler's predicate information if available
        2. Use environment's predicate information as fallback
        3. Otherwise use intelligent heuristics

        Args:
            fluent: Fluent like "clear_a" or "on_a_b" or "at_rover0_waypoint0"

        Returns:
            PDDL string like "(clear a)" or "(on a b)" or "(at rover0 waypoint0)"
        """
        if not fluent:
            return ""

        # First try: Get predicate names from PDDLHandler
        predicate_names = self._get_predicate_names()

        if predicate_names:
            # Try to match the longest predicate name first (handles multi-word predicates)
            for pred_name in sorted(predicate_names, key=len, reverse=True):
                if fluent.startswith(pred_name + '_') or fluent == pred_name:
                    if fluent == pred_name:
                        # Parameterless predicate
                        return f"({pred_name})"
                    else:
                        # Extract parameters after predicate
                        params_str = fluent[len(pred_name) + 1:]
                        # Smart parameter splitting - handle objects with underscores
                        params = self._smart_split_parameters(params_str)
                        # Keep predicate name as-is (preserving underscores)
                        return f"({pred_name} {' '.join(params)})"

        # Fallback: Intelligent heuristic approach
        # Try to guess predicate boundaries using object names
        object_names = self._get_object_names()

        if object_names:
            # Find the first object in the fluent to determine predicate boundary
            parts = fluent.split('_')
            for i in range(1, len(parts) + 1):
                # Check if we've found an object name
                if i < len(parts) and parts[i] in object_names:
                    # Everything before this is the predicate
                    predicate = '_'.join(parts[:i])
                    params = parts[i:]
                    return f"({predicate} {' '.join(params)})"

        # Final fallback: Simple split approach
        parts = fluent.split('_')

        if len(parts) == 1:
            # Parameterless
            return f"({parts[0]})"
        else:
            # Assume first part is predicate, rest are parameters
            # This works for many classical planning domains
            return f"({parts[0]} {' '.join(parts[1:])})"

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

        if len(parts) == 0:
            return ""
        elif len(parts) == 1:
            # Parameterless
            return parts[0]
        else:
            # Join with underscores
            # First part is predicate, might have underscores
            predicate = parts[0]
            params = parts[1:]

            # Handle multi-word predicates - they already have underscores
            return '_'.join([predicate] + params)

    def _get_predicate_names(self) -> Set[str]:
        """
        Get all predicate names from the domain.

        Returns:
            Set of predicate names (with underscores for multi-word predicates)
        """
        predicate_names = set()

        # Try to get from PDDLHandler
        if hasattr(self, 'pddl_handler') and self.pddl_handler:
            if hasattr(self.pddl_handler, 'problem') and self.pddl_handler.problem:
                for fluent_obj in self.pddl_handler.problem.fluents:
                    predicate_names.add(fluent_obj.name)

        # Also try to get from environment if available
        if not predicate_names and hasattr(self, 'environment'):
            if hasattr(self.environment, 'handler'):
                pddl_env_handler = self.environment.handler
                if hasattr(pddl_env_handler, 'problem') and pddl_env_handler.problem:
                    for fluent_obj in pddl_env_handler.problem.fluents:
                        predicate_names.add(fluent_obj.name)

        return predicate_names

    def _get_object_names(self) -> Set[str]:
        """
        Get all object names from the problem.

        Returns:
            Set of object names
        """
        object_names = set()

        # Try to get from PDDLHandler
        if hasattr(self, 'pddl_handler') and self.pddl_handler:
            if hasattr(self.pddl_handler, 'problem') and self.pddl_handler.problem:
                for obj in self.pddl_handler.problem.all_objects:
                    object_names.add(obj.name)

        # Also try to get from environment if available
        if not object_names and hasattr(self, 'environment'):
            if hasattr(self.environment, 'handler'):
                pddl_env_handler = self.environment.handler
                if hasattr(pddl_env_handler, 'problem') and pddl_env_handler.problem:
                    for obj in pddl_env_handler.problem.all_objects:
                        object_names.add(obj.name)

        return object_names

    def _smart_split_parameters(self, params_str: str) -> List[str]:
        """
        Split parameter string intelligently, preserving object names with underscores.

        Args:
            params_str: String like "camera0_high_res" or "rover0_waypoint0"

        Returns:
            List of parameters like ["camera0", "high_res"] or ["rover0", "waypoint0"]
        """
        if not params_str:
            return []

        # Get all known object names
        object_names = self._get_object_names()

        if not object_names:
            # Fallback: simple split
            return params_str.split('_')

        # Try to match object names greedily from left to right
        parts = params_str.split('_')
        result = []
        i = 0

        while i < len(parts):
            # Try to match the longest possible object name starting at position i
            found = False
            for length in range(len(parts) - i, 0, -1):
                candidate = '_'.join(parts[i:i+length])
                if candidate in object_names:
                    result.append(candidate)
                    i += length
                    found = True
                    break

            if not found:
                # No object match, take single part
                result.append(parts[i])
                i += 1

        return result

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