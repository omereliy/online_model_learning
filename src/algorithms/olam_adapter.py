"""
OLAM (Online Learning of Action Models) adapter implementation.
Integrates OLAM learner into the unified experiment framework.
"""

from .base_learner import BaseActionModelLearner
import sys
import os
import logging
from typing import Tuple, List, Dict, Optional, Any, Set
from pathlib import Path
import re
from collections import defaultdict
from contextlib import contextmanager

# Import centralized path configuration
from src.configuration.paths import OLAM_DIR

# Add OLAM to path
if str(OLAM_DIR) not in sys.path:
    sys.path.append(str(OLAM_DIR))

# Import OLAM components
try:
    from OLAM.Learner import Learner as OLAMLearner
    from Util.PddlParser import PddlParser
    from Util.Simulator import Simulator
    import Configuration
except ImportError as e:
    logging.error(f"Failed to import OLAM components: {e}")
    raise


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
                 bypass_java: bool = False,
                 use_system_java: bool = False,
                 planner_time_limit: Optional[int] = None,
                 max_precs_length: Optional[int] = None,
                 neg_eff_assumption: Optional[bool] = None,
                 output_console: Optional[bool] = None,
                 random_seed: Optional[int] = None,
                 time_limit_seconds: Optional[int] = None,
                 **kwargs):
        """
        Initialize OLAM adapter.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            max_iterations: Maximum learning iterations
            eval_frequency: How often to evaluate the model
            pddl_handler: Optional PDDLHandler instance for proper grounding
            bypass_java: If True, bypass Java dependency for action filtering
            use_system_java: If True, try to use system Java instead of bundled
            planner_time_limit: OLAM planner subprocess timeout (seconds)
            max_precs_length: OLAM negative precondition search depth
            neg_eff_assumption: OLAM STRIPS negative effects assumption
            output_console: OLAM console vs file logging
            random_seed: OLAM numpy random seed
            time_limit_seconds: OLAM total experiment timeout (seconds)
            **kwargs: Additional parameters
        """
        logger.info(f"Initializing OLAM adapter with domain={domain_file}, problem={problem_file}")
        logger.debug(
            f"Configuration: max_iterations={max_iterations}, eval_frequency={eval_frequency}, "
            f"bypass_java={bypass_java}, use_system_java={use_system_java}")

        # Convert paths to absolute before any directory changes
        domain_file = str(Path(domain_file).resolve())
        problem_file = str(Path(problem_file).resolve())
        logger.debug(f"Resolved absolute paths: domain={domain_file}, problem={problem_file}")

        super().__init__(domain_file, problem_file, **kwargs)

        self.max_iterations = max_iterations
        self.eval_frequency = eval_frequency
        self.bypass_java = bypass_java
        self.use_system_java = use_system_java

        # Store OLAM configuration parameters
        self.planner_time_limit = planner_time_limit
        self.max_precs_length = max_precs_length
        self.neg_eff_assumption = neg_eff_assumption
        self.output_console = output_console
        self.random_seed = random_seed
        self.time_limit_seconds = time_limit_seconds

        # Initialize domain knowledge for proper action grounding
        # OLAM requires injective bindings (no repeated objects in action parameters)
        if pddl_handler is None:
            logger.debug("Parsing domain with new architecture (PDDLReader + LiftedDomainKnowledge)")
            from src.core.pddl_io import PDDLReader
            reader = PDDLReader()
            self.domain, self.initial_state = reader.parse_domain_and_problem(domain_file, problem_file)
            # For backward compatibility, store as attribute
            self.pddl_handler = None
        else:
            logger.debug("Using provided PDDLHandler instance (legacy)")
            self.pddl_handler = pddl_handler
            # Extract domain from legacy handler if available
            if hasattr(pddl_handler, 'problem'):
                from src.core.lifted_domain import LiftedDomainKnowledge
                from src.core.up_adapter import UPAdapter
                adapter = UPAdapter()
                self.domain = LiftedDomainKnowledge.from_up_problem(pddl_handler.problem, adapter)
            else:
                self.domain = None

        # Initialize OLAM components
        logger.debug("Initializing OLAM components")
        self._initialize_olam()

        # State tracking
        self.current_state = None
        self.last_action_idx = None
        self.last_action_str = None
        logger.debug("State tracking initialized")

        # Mappings
        self.action_idx_to_str = {}
        self.action_str_to_idx = {}
        self._build_action_mappings()
        logger.info(
            f"OLAM adapter initialization complete with {len(self.action_idx_to_str)} actions mapped")

    @contextmanager
    def _olam_context(self):
        """
        Context manager to run OLAM operations in OLAM's working directory.

        OLAM expects to find its planner and other resources relative to its base directory.
        This context manager ensures OLAM operations run from the correct directory.
        """
        original_dir = os.getcwd()
        try:
            os.chdir(OLAM_DIR)
            logger.debug(f"Changed to OLAM directory: {OLAM_DIR}")
            yield
        finally:
            os.chdir(original_dir)
            logger.debug(f"Restored working directory: {original_dir}")

    def _initialize_olam(self):
        """Initialize OLAM learner and related components."""
        # Configure Java settings before OLAM initialization (no dir change needed)
        self._configure_java_settings()

        # Create temporary PDDL directory for OLAM (uses absolute paths, no dir change needed)
        with self._olam_context():
            self._setup_olam_pddl_directory()

        # Domain knowledge already initialized in __init__
        # No need to re-parse here

        # Initialize OLAM parser and learner in OLAM context
        with self._olam_context():
            # Initialize OLAM parser
            self.parser = PddlParser()

            # Extract action list from domain
            self.action_list = self._extract_action_list()

            # Initialize OLAM learner
            self.learner = OLAMLearner(self.parser, self.action_list, self.eval_frequency)

        # Initialize required OLAM attributes (no dir change needed)
        from timeit import default_timer
        self.learner.initial_timer = default_timer()
        self.learner.now = default_timer()  # Initialize timing attribute
        self.learner.max_time_limit = 3600  # 1 hour default time limit
        if not hasattr(self.learner, 'current_plan'):
            self.learner.current_plan = []
        if not hasattr(self.learner, 'time_at_iter'):
            self.learner.time_at_iter = []
        if not hasattr(self.learner, 'iter'):
            self.learner.iter = 0

        # Disable eval_log which has path dependencies
        self.learner.eval_log = lambda: None

        # Create simulator for state tracking (needs OLAM context for file paths)
        with self._olam_context():
            self.simulator = Simulator()

        # If bypassing Java, monkey-patch the Java computation method
        if self.bypass_java:
            self._setup_java_bypass()

        logger.info(f"Initialized OLAM with {len(self.action_list)} actions")
        if self.bypass_java:
            logger.info("Java bypass mode enabled - using Python fallback for action filtering")

    def _setup_olam_pddl_directory(self):
        """
        Setup PDDL directory structure expected by OLAM.
        OLAM expects files in a 'PDDL/' directory relative to working directory.
        """
        logger.debug("Setting up OLAM PDDL directory structure")

        # Create PDDL directory if it doesn't exist
        pddl_dir = Path("PDDL")
        pddl_dir.mkdir(exist_ok=True)
        logger.debug(f"Created/verified PDDL directory: {pddl_dir.absolute()}")

        # Create Info directory for OLAM
        info_dir = Path("Info")
        info_dir.mkdir(exist_ok=True)
        logger.debug(f"Created/verified Info directory: {info_dir.absolute()}")

        # Copy domain and problem files
        import shutil
        shutil.copy(self.domain_file, pddl_dir / "domain.pddl")
        logger.debug(f"Copied domain file: {self.domain_file} -> {pddl_dir / 'domain.pddl'}")

        shutil.copy(self.problem_file, pddl_dir / "facts.pddl")
        logger.debug(f"Copied problem file: {self.problem_file} -> {pddl_dir / 'facts.pddl'}")

        # OLAM also expects a domain_learned.pddl file (starts empty)
        # Copy domain as initial learned domain
        shutil.copy(self.domain_file, pddl_dir / "domain_learned.pddl")
        logger.debug(f"Created initial learned domain: {pddl_dir / 'domain_learned.pddl'}")

        logger.info(f"OLAM PDDL directory setup complete")

    def _extract_action_list(self) -> List[str]:
        """
        Extract grounded action list using PDDLHandler for proper domain-agnostic grounding.

        Returns:
            List of grounded action strings in OLAM format
        """
        logger.debug("Extracting grounded action list using new grounding utilities")
        action_strings = []

        # Use new grounding module
        # Note: Use require_injective=False to match old PDDLHandler behavior
        # (grounds all combinations, domain preconditions filter invalid ones at runtime)
        from src.core import grounding
        grounded_actions = grounding.ground_all_actions(self.domain, require_injective=False)
        logger.debug(f"Processing {len(grounded_actions)} grounded actions from domain")

        for grounded_action in grounded_actions:
            # Get parameter values in order (objects is already a List[str])
            param_values = grounded_action.objects

            # Build OLAM format: action_name(param1,param2,...)
            if param_values:
                action_str = f"{grounded_action.action_name}({','.join(param_values)})"
            else:
                action_str = f"{grounded_action.action_name}()"

            logger.debug(f"Added grounded action: {action_str} (params: {param_values})")
            action_strings.append(action_str)

        sorted_actions = sorted(action_strings)
        logger.info(f"Extracted {len(sorted_actions)} grounded actions")
        return sorted_actions

    def _build_action_mappings(self):
        """Build mappings between action indices and string representations."""
        logger.debug("Building action index ↔ string mappings")
        for idx, action_str in enumerate(self.learner.action_labels):
            self.action_idx_to_str[idx] = action_str
            self.action_str_to_idx[action_str] = idx
        logger.info(f"Built bidirectional mappings for {len(self.action_idx_to_str)} actions")

    def _configure_java_settings(self):
        """Configure Java and additional OLAM settings."""
        # Configure Java path
        if self.bypass_java:
            # Set empty path to avoid Java calls
            Configuration.JAVA_BIN_PATH = ""
        elif self.use_system_java:
            # Try to use system Java
            Configuration.JAVA_BIN_PATH = "java"
            logger.info("Configured to use system Java")
        else:
            # Default OLAM behavior - look for bundled Java
            import os
            java_dir = os.path.join(os.path.dirname(Configuration.__file__), Configuration.JAVA_DIR)
            if os.path.exists(java_dir):
                java_dirs = [
                    d for d in os.listdir(java_dir) if os.path.isdir(
                        os.path.join(
                            java_dir, d))]
                if java_dirs:
                    Configuration.JAVA_BIN_PATH = os.path.join(
                        java_dir, java_dirs[0], "bin", "java")
                else:
                    logger.warning("No Java installation found in OLAM/Java directory")
                    Configuration.JAVA_BIN_PATH = ""
            else:
                Configuration.JAVA_BIN_PATH = ""

        # Apply additional OLAM configuration parameters if provided
        if self.planner_time_limit is not None:
            Configuration.PLANNER_TIME_LIMIT = self.planner_time_limit
            logger.debug(f"Set PLANNER_TIME_LIMIT = {self.planner_time_limit}")

        if self.max_precs_length is not None:
            Configuration.MAX_PRECS_LENGTH = self.max_precs_length
            logger.debug(f"Set MAX_PRECS_LENGTH = {self.max_precs_length}")

        if self.neg_eff_assumption is not None:
            Configuration.NEG_EFF_ASSUMPTION = self.neg_eff_assumption
            logger.debug(f"Set NEG_EFF_ASSUMPTION = {self.neg_eff_assumption}")

        if self.output_console is not None:
            Configuration.OUTPUT_CONSOLE = self.output_console
            logger.debug(f"Set OUTPUT_CONSOLE = {self.output_console}")

        if self.random_seed is not None:
            Configuration.RANDOM_SEED = self.random_seed
            logger.debug(f"Set RANDOM_SEED = {self.random_seed}")

        if self.time_limit_seconds is not None:
            Configuration.TIME_LIMIT_SECONDS = self.time_limit_seconds
            logger.debug(f"Set TIME_LIMIT_SECONDS = {self.time_limit_seconds}")

    def _setup_java_bypass(self):
        """Setup bypass for Java-dependent methods."""
        # Store original method
        self.learner._original_compute_not_executable = self.learner.compute_not_executable_actionsJAVA

        # Replace with Python fallback that uses LEARNED model only
        def compute_not_executable_bypass():
            """Compute non-executable actions based on LEARNED model, not ground truth."""
            return self._filter_by_learned_model()

        self.learner.compute_not_executable_actionsJAVA = compute_not_executable_bypass

    def _filter_by_learned_model(self) -> List[int]:
        """
        Filter non-executable actions based on OLAM's LEARNED model only.

        This method uses only what OLAM has learned through experience,
        not any ground truth from the environment.

        Returns:
            List of action indices that are NOT executable based on learned preconditions
        """
        # If no learning has happened yet, high uncertainty - filter very few
        if not hasattr(self.learner, 'operator_certain_predicates'):
            logger.debug("No learned model yet - high uncertainty, filtering no actions")
            return []

        try:
            # Get current state from OLAM's simulator (already updated)
            if hasattr(self.learner, 'simulator') and hasattr(self.learner.simulator, 'facts'):
                current_state_pddl = set(self.learner.simulator.facts)
                logger.debug(f"Using simulator state with {len(current_state_pddl)} facts")
            else:
                # Try to read from facts file as fallback
                import os
                import re
                facts_file = "PDDL/facts.pddl"
                if os.path.exists(facts_file):
                    with open(facts_file, "r") as f:
                        data = [el.strip() for el in f.read().split("\n")]
                        facts = re.findall(r":init.*\(:goal", "".join(data))[0]
                        current_state_pddl = set(re.findall(r"\([^()]*\)", facts))
                    logger.debug(f"Using facts file state with {len(current_state_pddl)} facts")
                else:
                    # No state available, return empty (high uncertainty)
                    logger.debug("No state available, cannot filter actions")
                    return []

            non_executable_indices = []

            # Filter based on what OLAM has LEARNED (both certain and uncertain)
            total_certain_ops = sum(
                1 for v in self.learner.operator_certain_predicates.values() if v)
            total_uncertain_ops = sum(
                1 for v in self.learner.operator_uncertain_predicates.values() if v)
            logger.debug(
                f"Learned model has {total_certain_ops} ops with certain, {total_uncertain_ops} with uncertain")

            for idx, action_label in enumerate(self.action_list):
                operator = action_label.split("(")[0]
                params = [el.strip() for el in action_label.split("(")[1][:-1].split(",")]

                # Check both CERTAIN and UNCERTAIN preconditions
                certain_preconds = self.learner.operator_certain_predicates.get(operator, [])
                uncertain_preconds = self.learner.operator_uncertain_predicates.get(operator, [])

                # Filter if we have ANY learned preconditions (even just 1)
                if certain_preconds:
                    # Ground the certain preconditions with action parameters
                    grounded_certain = []
                    for pred in certain_preconds:
                        grounded_pred = pred
                        for k, param in enumerate(params):
                            grounded_pred = grounded_pred.replace(f"?param_{k+1})", f"{param})")
                            grounded_pred = grounded_pred.replace(f"?param_{k+1} ", f"{param} ")
                        grounded_certain.append(grounded_pred)

                    # Filter if ANY certain precondition is violated
                    if grounded_certain and not all(
                            pred in current_state_pddl for pred in grounded_certain):
                        non_executable_indices.append(idx)
                        continue

                # Also consider uncertain preconditions (weaker filtering)
                if len(uncertain_preconds) >= 3:  # Need multiple uncertain to filter
                    # Ground the uncertain preconditions
                    grounded_uncertain = []
                    for pred in uncertain_preconds:
                        grounded_pred = pred
                        for k, param in enumerate(params):
                            grounded_pred = grounded_pred.replace(f"?param_{k+1})", f"{param})")
                            grounded_pred = grounded_pred.replace(f"?param_{k+1} ", f"{param} ")
                        grounded_uncertain.append(grounded_pred)

                    # Filter if MOST uncertain preconditions are violated (>75%)
                    satisfied = sum(1 for pred in grounded_uncertain if pred in current_state_pddl)
                    if satisfied < len(grounded_uncertain) * 0.25:  # Less than 25% satisfied
                        non_executable_indices.append(idx)

            logger.debug(
                f"Learned model filtering: {len(non_executable_indices)}/{len(self.action_list)} non-executable")
            return non_executable_indices

        except Exception as e:
            logger.warning(f"Error in learned model filtering: {e}")
            # On error, high uncertainty - filter nothing
            return []

    def select_action(self, state: Any) -> Tuple[str, List[str]]:
        """
        Select next action using OLAM's action selection strategy.

        Args:
            state: Current state (UP format or set of fluent strings)

        Returns:
            Tuple of (action_name, objects)
        """
        self.iteration_count += 1
        logger.info(f"=== Iteration {self.iteration_count}: Selecting action ===")

        # Convert state to OLAM format
        logger.debug("Converting current state to OLAM format")
        olam_state = self._up_state_to_olam(state)
        self._update_simulator_state(olam_state)

        # Use OLAM's action selection (runs in OLAM context for planner access)
        logger.debug("Calling OLAM learner.select_action()")
        with self._olam_context():
            action_idx, strategy = self.learner.select_action()
        self.last_action_idx = action_idx
        logger.info(f"OLAM selected action index {action_idx} using strategy: {strategy}")

        # Convert to our format
        action_str = self.action_idx_to_str[action_idx]
        logger.debug(f"Action index {action_idx} maps to: {action_str}")
        action_name, objects = self.parse_action_string(action_str)

        self.last_action_str = action_str
        self.current_state = olam_state

        logger.info(f"Selected action: {action_name}({','.join(objects)})")

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
        logger.info(
            f"=== Observation {self.observation_count}: {action}({','.join(objects)}) -> {'SUCCESS' if success else 'FAILURE'} ===")

        # Convert to OLAM format
        action_str = self._up_action_to_olam(action, objects)
        logger.debug(f"Action in OLAM format: {action_str}")

        if not success:
            # Learn from failed action
            logger.info(f"Processing FAILED action: {action_str}")
            if self.last_action_idx is not None:
                # Update simulator with current state for OLAM
                logger.debug("Converting state for failed action learning")
                olam_state = self._up_state_to_olam(state)
                self._update_simulator_state(olam_state)

                logger.debug("Calling OLAM learn_failed_action_precondition()")
                with self._olam_context():
                    self.learner.learn_failed_action_precondition(self.simulator)
                logger.info(f"Learned precondition constraints from failed action: {action_str}")
            else:
                logger.warning("No last_action_idx available for failed action learning")
        else:
            # Learn from successful action
            logger.info(f"Processing SUCCESSFUL action: {action_str}")
            if next_state is not None:
                # Convert states
                logger.debug("Converting states for successful action learning")
                olam_state = self._up_state_to_olam(state)
                olam_next_state = self._up_state_to_olam(next_state)

                # Update preconditions
                logger.debug(f"Calling OLAM add_operator_precondition() for {action_str}")
                with self._olam_context():
                    changed = self.learner.add_operator_precondition(action_str)
                if changed:
                    logger.info(f"Updated preconditions for {action_str}")
                else:
                    logger.debug(f"No precondition changes for {action_str}")

                # Update effects
                logger.debug(f"Calling OLAM add_operator_effects() for {action_str}")
                with self._olam_context():
                    self.learner.add_operator_effects(action_str, olam_state, olam_next_state)
                logger.info(f"Updated effects for {action_str}")

                # Update simulator state
                self._update_simulator_state(olam_next_state)
            else:
                logger.warning("Successful action but no next_state provided - skipping learning")

        # Check convergence periodically
        if self.observation_count % self.eval_frequency == 0:
            logger.debug(
                f"Checking convergence (observation {self.observation_count}, frequency {self.eval_frequency})")
            self._check_convergence()

    def get_learned_model(self) -> Dict[str, Any]:
        """
        Export the current learned model from OLAM.

        Returns:
            Dictionary containing the learned model
        """
        logger.info("Extracting learned model from OLAM")
        model = {
            'actions': {},
            'predicates': set(),
            'statistics': self.get_statistics()
        }
        logger.debug(f"Processing {len(self.action_list)} actions from OLAM learner")

        # Extract learned operators from OLAM
        # OLAM stores knowledge at the operator (schema) level, not grounded action level
        operators_processed = set()
        for action_label in self.action_list:
            action_name = action_label.split("(")[0]  # Get operator name

            # Get learned preconditions from operator-level storage
            certain_precs = self.learner.operator_certain_predicates.get(action_name, [])
            uncertain_precs = self.learner.operator_uncertain_predicates.get(action_name, [])
            neg_precs = self.learner.operator_negative_preconditions.get(action_name, [])

            # Get effects if available
            pos_effects = []
            neg_effects = []
            if hasattr(self.learner, 'operator_positive_effects'):
                pos_effects = self.learner.operator_positive_effects.get(action_name, [])
            if hasattr(self.learner, 'operator_negative_effects'):
                neg_effects = self.learner.operator_negative_effects.get(action_name, [])

            # Log details for each unique operator
            if action_name not in operators_processed:
                logger.debug(f"Operator '{action_name}': certain_precs={len(certain_precs)}, "
                             f"uncertain_precs={len(uncertain_precs)}, neg_precs={len(neg_precs)}, "
                             f"pos_effects={len(pos_effects)}, neg_effects={len(neg_effects)}")
                operators_processed.add(action_name)

            model['actions'][action_label] = {
                'name': action_name,
                'parameters': action_label.split("(")[1].rstrip(")").split(",") if "(" in action_label else [],
                'preconditions': {
                    'certain': certain_precs,
                    'uncertain': uncertain_precs,
                    'negative': neg_precs},
                'effects': {
                    'positive': pos_effects,
                    'negative': neg_effects}}

            # Collect all predicates
            for pred_list in model['actions'][action_label]['preconditions'].values():
                if pred_list:
                    for pred in pred_list:
                        if isinstance(pred, str):
                            # Extract predicate name
                            pred_name = pred.strip('()').split('(')[0].split()[0]
                            model['predicates'].add(pred_name)

        logger.info(
            f"Learned model extracted: {len(model['actions'])} grounded actions, "
            f"{len(operators_processed)} unique operators, {len(model['predicates'])} predicates")
        return model

    def has_converged(self) -> bool:
        """
        Check if OLAM has converged.

        Returns:
            True if model has converged, False otherwise
        """
        # OLAM sets model_convergence flag
        if hasattr(self.learner, 'model_convergence'):
            olam_converged = self.learner.model_convergence
            if olam_converged != self._converged:
                logger.info(
                    f"OLAM convergence status changed: {self._converged} -> {olam_converged}")
            self._converged = olam_converged

        # Also check iteration limit
        if self.iteration_count >= self.max_iterations:
            if not self._converged:
                logger.info(f"Reached max iterations ({self.max_iterations}), forcing convergence")
            self._converged = True

        logger.debug(
            f"Convergence check: converged={self._converged}, iterations={self.iteration_count}/{self.max_iterations}")
        return self._converged

    def _check_convergence(self):
        """Check and update convergence status."""
        logger.debug("Running periodic convergence check")
        # OLAM has its own convergence detection
        # We can also add custom convergence criteria here
        if hasattr(self.learner, 'model_convergence'):
            old_status = self._converged
            self._converged = self.learner.model_convergence
            if old_status != self._converged:
                logger.info(f"Convergence status updated: {old_status} -> {self._converged}")
            else:
                logger.debug(f"Convergence status unchanged: {self._converged}")

    # ========== State and Action Conversion Methods ==========

    def _up_state_to_olam(self, state: Any) -> List[str]:
        """
        Convert UP state format to OLAM format.

        Args:
            state: UP state (set of fluent strings or similar)

        Returns:
            List of PDDL predicate strings for OLAM
        """
        logger.debug(f"Converting UP state to OLAM format (input type: {type(state).__name__})")
        olam_state = []

        if isinstance(state, set):
            # Assume state is a set of fluent strings like {'clear_a', 'on_a_b'}
            logger.debug(f"Converting {len(state)} fluents from set format")
            for fluent in sorted(state):
                # Convert underscore format to PDDL format
                olam_pred = self._fluent_to_pddl_string(fluent)
                if olam_pred:
                    olam_state.append(olam_pred)
                    logger.debug(f"  Converted: {fluent} -> {olam_pred}")
                else:
                    logger.warning(f"  Failed to convert fluent: {fluent}")
        elif isinstance(state, list):
            # Already in OLAM format
            logger.debug(f"State already in OLAM list format with {len(state)} predicates")
            olam_state = state
        else:
            # Try to handle other formats
            logger.warning(f"Unexpected state format: {type(state)}, attempting to handle")

        result = sorted(olam_state)
        logger.debug(f"UP->OLAM conversion complete: {len(result)} predicates")
        return result

    def _olam_state_to_up(self, olam_state: List[str]) -> Set[str]:
        """
        Convert OLAM state format to UP format.

        Args:
            olam_state: List of PDDL predicate strings

        Returns:
            Set of fluent strings in UP format
        """
        logger.debug(f"Converting OLAM state to UP format ({len(olam_state)} predicates)")
        up_state = set()

        for pred in olam_state:
            # Convert "(pred obj1 obj2)" to "pred_obj1_obj2"
            fluent = self._pddl_string_to_fluent(pred)
            if fluent:
                up_state.add(fluent)
                logger.debug(f"  Converted: {pred} -> {fluent}")
            else:
                logger.warning(f"  Failed to convert predicate: {pred}")

        logger.debug(f"OLAM->UP conversion complete: {len(up_state)} fluents")
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
            logger.debug("Empty fluent provided to _fluent_to_pddl_string")
            return ""

        logger.debug(f"Converting fluent to PDDL: {fluent}")

        # First try: Get predicate names from PDDLHandler
        predicate_names = self._get_predicate_names()

        if predicate_names:
            # Try to match the longest predicate name first (handles multi-word predicates)
            for pred_name in sorted(predicate_names, key=len, reverse=True):
                if fluent.startswith(pred_name + '_') or fluent == pred_name:
                    if fluent == pred_name:
                        # Parameterless predicate
                        result = f"({pred_name})"
                        logger.debug(f"  Matched parameterless predicate: {result}")
                        return result
                    else:
                        # Extract parameters after predicate
                        params_str = fluent[len(pred_name) + 1:]
                        # Smart parameter splitting - handle objects with underscores
                        params = self._smart_split_parameters(params_str)
                        # Keep predicate name as-is (preserving underscores)
                        result = f"({pred_name} {' '.join(params)})"
                        logger.debug(
                            f"  Matched predicate '{pred_name}' with params {params}: {result}")
                        return result

        # Fallback: Intelligent heuristic approach
        logger.debug("  No predicate match, trying heuristic with object names")
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
                    result = f"({predicate} {' '.join(params)})"
                    logger.debug(f"  Heuristic match at object '{parts[i]}': {result}")
                    return result

        # Final fallback: Simple split approach
        logger.debug("  Using simple split fallback")
        parts = fluent.split('_')

        if len(parts) == 1:
            # Parameterless
            result = f"({parts[0]})"
        else:
            # Assume first part is predicate, rest are parameters
            # This works for many classical planning domains
            result = f"({parts[0]} {' '.join(parts[1:])})"

        logger.debug(f"  Fallback result: {result}")
        return result

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

        # Get from LiftedDomainKnowledge
        if hasattr(self, 'domain') and self.domain:
            for pred_name in self.domain.predicates.keys():
                predicate_names.add(pred_name)
            logger.debug(f"Retrieved {len(predicate_names)} predicate names from domain")
        # Fallback: Try legacy PDDLHandler
        elif hasattr(self, 'pddl_handler') and self.pddl_handler:
            if hasattr(self.pddl_handler, 'problem') and self.pddl_handler.problem:
                for fluent_obj in self.pddl_handler.problem.fluents:
                    predicate_names.add(fluent_obj.name)
                logger.debug(f"Retrieved {len(predicate_names)} predicate names from PDDLHandler (legacy)")
        # Also try to get from environment if available
        elif hasattr(self, 'environment'):
            if hasattr(self.environment, 'handler'):
                pddl_env_handler = self.environment.handler
                if hasattr(pddl_env_handler, 'problem') and pddl_env_handler.problem:
                    for fluent_obj in pddl_env_handler.problem.fluents:
                        predicate_names.add(fluent_obj.name)
                    logger.debug(
                        f"Retrieved {len(predicate_names)} predicate names from environment")

        if not predicate_names:
            logger.debug("No predicate names available")

        return predicate_names

    def _get_object_names(self) -> Set[str]:
        """
        Get all object names from the problem.

        Returns:
            Set of object names
        """
        object_names = set()

        # Get from LiftedDomainKnowledge
        if hasattr(self, 'domain') and self.domain:
            for obj_name in self.domain.objects.keys():
                object_names.add(obj_name)
            logger.debug(f"Retrieved {len(object_names)} object names from domain")
        # Fallback: Try legacy PDDLHandler
        elif hasattr(self, 'pddl_handler') and self.pddl_handler:
            if hasattr(self.pddl_handler, 'problem') and self.pddl_handler.problem:
                for obj in self.pddl_handler.problem.all_objects:
                    object_names.add(obj.name)
                logger.debug(f"Retrieved {len(object_names)} object names from PDDLHandler (legacy)")

        # Also try to get from environment if available
        if not object_names and hasattr(self, 'environment'):
            if hasattr(self.environment, 'handler'):
                pddl_env_handler = self.environment.handler
                if hasattr(pddl_env_handler, 'problem') and pddl_env_handler.problem:
                    for obj in pddl_env_handler.problem.all_objects:
                        object_names.add(obj.name)
                    logger.debug(f"Retrieved {len(object_names)} object names from environment")

        if not object_names:
            logger.debug("No object names available")

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
                candidate = '_'.join(parts[i:i + length])
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
            result = f"{action}({','.join(objects)})"
        else:
            result = f"{action}()"
        logger.debug(f"UP->OLAM action: {action}({objects}) -> {result}")
        return result

    def _olam_action_to_up(self, olam_action: str) -> Tuple[str, List[str]]:
        """
        Convert OLAM action string to UP format.

        Args:
            olam_action: OLAM action string

        Returns:
            Tuple of (action_name, objects)
        """
        result = self.parse_action_string(olam_action)
        logger.debug(f"OLAM->UP action: {olam_action} -> {result}")
        return result

    def _update_simulator_state(self, olam_state: List[str]):
        """
        Update OLAM's simulator and write state to PDDL/facts.pddl file.

        OLAM reads state from file, so we must keep the file synchronized.

        Args:
            olam_state: State in OLAM format
        """
        logger.debug(f"Updating OLAM simulator state with {len(olam_state)} predicates")

        # Update in-memory state
        self.simulator.state = olam_state

        # Write to PDDL/facts.pddl so OLAM's compute_executable_actions() reads correct state
        with self._olam_context():
            self._write_facts_file(olam_state)

        logger.debug(f"Simulator state and facts file updated successfully")

    def _write_facts_file(self, state: List[str]):
        """
        Write current state to PDDL/facts.pddl in the format OLAM expects.

        Args:
            state: Current state as list of PDDL predicates
        """
        # Read the original problem file to preserve goal and other sections
        facts_path = Path("PDDL/facts.pddl")

        with open(facts_path, "r") as f:
            lines = f.readlines()

        # Find :init section and replace it with current state
        new_lines = []
        in_init = False

        for line in lines:
            if "(:init" in line or "(: init" in line:
                in_init = True
                new_lines.append("  (:init\n")
                # Add all current state predicates
                for pred in sorted(state):
                    new_lines.append(f"    {pred}\n")
            elif in_init and ("(:goal" in line or "(: goal" in line):
                in_init = False
                new_lines.append("  )\n")  # Close :init
                new_lines.append(line)  # Add :goal line
            elif not in_init:
                new_lines.append(line)

        # Write back the updated file
        with open(facts_path, "w") as f:
            f.writelines(new_lines)

        logger.debug(f"Wrote {len(state)} predicates to {facts_path}")

    def reset(self) -> None:
        """Reset the learner to initial state."""
        logger.info("Resetting OLAM adapter to initial state")
        super().reset()

        # Reinitialize OLAM components
        logger.debug("Reinitializing OLAM components")
        self._initialize_olam()

        # Reset state tracking
        self.current_state = None
        self.last_action_idx = None
        self.last_action_str = None
        logger.info("OLAM adapter reset complete")
