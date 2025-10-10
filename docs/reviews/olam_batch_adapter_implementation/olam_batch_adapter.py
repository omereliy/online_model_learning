"""
Simplified OLAM adapter that runs OLAM in batch mode.

This adapter treats OLAM as it was designed to be used: as an autonomous
learning algorithm that runs its own internal loop. This is much simpler
and more reliable than the per-iteration adapter approach.
"""

import sys
import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from .batch_learner import BatchAlgorithmAdapter
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


class OLAMBatchAdapter(BatchAlgorithmAdapter):
    """
    Simplified OLAM adapter that runs OLAM in batch mode.

    Key simplifications compared to the iterative adapter:
    - No monkey-patching of OLAM internals
    - No per-iteration state synchronization
    - No complex format conversions per action
    - Runs OLAM's learn() method once, parses results

    This approach is more faithful to OLAM's design and much more maintainable.
    """

    def __init__(self,
                 domain_file: str,
                 problem_file: str,
                 max_iterations: int = 1000,
                 eval_frequency: int = 10,
                 bypass_java: bool = False,
                 planner_time_limit: Optional[int] = None,
                 max_precs_length: Optional[int] = None,
                 neg_eff_assumption: Optional[bool] = None,
                 output_console: Optional[bool] = None,
                 random_seed: Optional[int] = None,
                 time_limit_seconds: Optional[int] = None,
                 **kwargs):
        """
        Initialize OLAM batch adapter.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            max_iterations: Maximum learning iterations
            eval_frequency: How often OLAM evaluates the model
            bypass_java: If True, bypass Java dependency (testing only)
            planner_time_limit: OLAM planner subprocess timeout (seconds)
            max_precs_length: OLAM negative precondition search depth
            neg_eff_assumption: OLAM STRIPS negative effects assumption
            output_console: OLAM console vs file logging
            random_seed: OLAM numpy random seed
            time_limit_seconds: OLAM total experiment timeout (seconds)
            **kwargs: Additional parameters
        """
        logger.info(f"Initializing OLAM batch adapter with domain={domain_file}")

        # Convert to absolute paths
        domain_file = str(Path(domain_file).resolve())
        problem_file = str(Path(problem_file).resolve())

        super().__init__(domain_file, problem_file, **kwargs)

        self.max_iterations = max_iterations
        self.eval_frequency = eval_frequency
        self.bypass_java = bypass_java

        # Store OLAM configuration parameters
        self.planner_time_limit = planner_time_limit
        self.max_precs_length = max_precs_length
        self.neg_eff_assumption = neg_eff_assumption
        self.output_console = output_console
        self.random_seed = random_seed
        self.time_limit_seconds = time_limit_seconds

        # Will be initialized when running experiment
        self.parser = None
        self.simulator = None
        self.action_list = None
        self.learner = None

        logger.info("OLAM batch adapter initialized")

    @contextmanager
    def _olam_context(self):
        """Context manager to run operations in OLAM's working directory."""
        original_dir = os.getcwd()
        try:
            os.chdir(OLAM_DIR)
            logger.debug(f"Changed to OLAM directory: {OLAM_DIR}")
            yield
        finally:
            os.chdir(original_dir)
            logger.debug(f"Restored working directory: {original_dir}")

    def _configure_olam(self):
        """Configure OLAM settings before running."""
        # Configure Java
        if self.bypass_java:
            Configuration.JAVA_BIN_PATH = ""
            logger.info("Java bypass enabled - using Python fallback")
        else:
            # Use OLAM's bundled Java
            java_dir = os.path.join(os.path.dirname(Configuration.__file__), Configuration.JAVA_DIR)
            if os.path.exists(java_dir):
                java_dirs = [d for d in os.listdir(java_dir) if os.path.isdir(os.path.join(java_dir, d))]
                if java_dirs:
                    Configuration.JAVA_BIN_PATH = os.path.join(java_dir, java_dirs[0], "bin", "java")
                    logger.info(f"Using OLAM bundled Java: {Configuration.JAVA_BIN_PATH}")
                else:
                    logger.warning("No Java found in OLAM/Java directory")
                    Configuration.JAVA_BIN_PATH = ""
            else:
                Configuration.JAVA_BIN_PATH = ""

        # Apply additional configuration parameters
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

    def _setup_pddl_environment(self):
        """Setup PDDL directory structure expected by OLAM."""
        logger.debug("Setting up OLAM PDDL environment")

        with self._olam_context():
            # Create required directories
            pddl_dir = Path("PDDL")
            pddl_dir.mkdir(exist_ok=True)

            info_dir = Path("Info")
            info_dir.mkdir(exist_ok=True)

            # Copy PDDL files
            shutil.copy(self.domain_file, pddl_dir / "domain.pddl")
            shutil.copy(self.problem_file, pddl_dir / "facts.pddl")
            shutil.copy(self.domain_file, pddl_dir / "domain_learned.pddl")

            logger.info("PDDL environment setup complete")

    def _initialize_olam_components(self):
        """Initialize OLAM's internal components."""
        logger.debug("Initializing OLAM components")

        with self._olam_context():
            # Initialize parser
            self.parser = PddlParser()

            # Initialize simulator
            self.simulator = Simulator()

            # Get action list using OLAM's standard approach
            from src.core.pddl_io import PDDLReader
            reader = PDDLReader()
            domain, _ = reader.parse_domain_and_problem(self.domain_file, self.problem_file)

            # Ground all actions
            from src.core import grounding
            grounded_actions = grounding.ground_all_actions(domain, require_injective=False)

            # Convert to OLAM format
            self.action_list = []
            for ga in grounded_actions:
                if ga.objects:
                    action_str = f"{ga.action_name}({','.join(ga.objects)})"
                else:
                    action_str = f"{ga.action_name}()"
                self.action_list.append(action_str)

            self.action_list = sorted(self.action_list)

            # Initialize OLAM learner
            self.learner = OLAMLearner(self.parser, self.action_list, self.eval_frequency)

            # Setup timing attributes OLAM expects
            from timeit import default_timer
            self.learner.initial_timer = default_timer()
            self.learner.now = default_timer()
            self.learner.max_time_limit = self.time_limit_seconds or 3600

            # Disable eval logging methods which have path dependencies we don't need
            self.learner.eval_log = lambda: None
            self.learner.eval_log_with_uncertain_neg = lambda: None

            logger.info(f"OLAM initialized with {len(self.action_list)} actions")

    def run_experiment(self, max_iterations: int = None, **kwargs) -> Dict[str, Any]:
        """
        Run OLAM experiment in batch mode.

        Args:
            max_iterations: Maximum iterations (uses instance default if None)
            **kwargs: Additional experiment parameters

        Returns:
            Dictionary containing learned model and metrics
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        logger.info(f"Starting OLAM batch experiment (max_iterations={max_iterations})")

        # Configure OLAM
        self._configure_olam()

        # Setup environment
        self._setup_pddl_environment()

        # Initialize OLAM
        self._initialize_olam_components()

        # Run OLAM's learning loop
        logger.info("Running OLAM learning loop...")
        with self._olam_context():
            self.learner.learn(eval_frequency=self.eval_frequency, simulator=self.simulator)

        logger.info(f"OLAM learning complete after {self.learner.iter} iterations")

        # Parse results
        learned_model = self._parse_learned_model()
        metrics = self._extract_metrics()

        # Store for get_learned_model()
        self._learned_model = learned_model

        # Update base class tracking
        self.iteration_count = self.learner.iter
        self._converged = getattr(self.learner, 'model_convergence', False)

        return {
            'learned_model': learned_model,
            'metrics': metrics,
            'final_iteration': self.learner.iter,
            'converged': self._converged
        }

    def _parse_learned_model(self) -> Dict[str, Any]:
        """
        Parse OLAM's learned model from its internal state.

        Returns:
            Dictionary containing the learned model in standard format
        """
        logger.info("Parsing learned model from OLAM")

        model = {
            'actions': {},
            'predicates': set(),
            'statistics': {
                'iterations': self.learner.iter,
                'converged': getattr(self.learner, 'model_convergence', False)
            }
        }

        # Extract learned operators (at schema level)
        operators_processed = set()
        for action_label in self.action_list:
            action_name = action_label.split("(")[0]

            # Get learned knowledge from OLAM
            certain_precs = self.learner.operator_certain_predicates.get(action_name, [])
            uncertain_precs = self.learner.operator_uncertain_predicates.get(action_name, [])
            neg_precs = self.learner.operator_negative_preconditions.get(action_name, [])

            pos_effects = []
            neg_effects = []
            if hasattr(self.learner, 'operator_positive_effects'):
                pos_effects = self.learner.operator_positive_effects.get(action_name, [])
            if hasattr(self.learner, 'operator_negative_effects'):
                neg_effects = self.learner.operator_negative_effects.get(action_name, [])

            if action_name not in operators_processed:
                logger.debug(
                    f"Operator '{action_name}': {len(certain_precs)} certain precs, "
                    f"{len(uncertain_precs)} uncertain precs, {len(neg_precs)} neg precs"
                )
                operators_processed.add(action_name)

            model['actions'][action_label] = {
                'name': action_name,
                'parameters': action_label.split("(")[1].rstrip(")").split(",") if "(" in action_label else [],
                'preconditions': {
                    'certain': certain_precs,
                    'uncertain': uncertain_precs,
                    'negative': neg_precs
                },
                'effects': {
                    'positive': pos_effects,
                    'negative': neg_effects
                }
            }

            # Collect predicates
            for pred_list in model['actions'][action_label]['preconditions'].values():
                if pred_list:
                    for pred in pred_list:
                        if isinstance(pred, str):
                            pred_name = pred.strip('()').split('(')[0].split()[0]
                            model['predicates'].add(pred_name)

        logger.info(
            f"Parsed model: {len(model['actions'])} grounded actions, "
            f"{len(operators_processed)} operators, {len(model['predicates'])} predicates"
        )

        return model

    def _extract_metrics(self) -> Dict[str, Any]:
        """
        Extract metrics from OLAM's execution.

        Returns:
            Dictionary containing metrics data
        """
        logger.info("Extracting metrics from OLAM execution")

        metrics = {
            'total_iterations': self.learner.iter,
            'converged': getattr(self.learner, 'model_convergence', False),
            'total_actions': len(self.action_list),
        }

        # Extract iteration-level data if available
        if hasattr(self.learner, 'time_at_iter'):
            metrics['iteration_times'] = self.learner.time_at_iter

        logger.info(f"Extracted metrics: {metrics['total_iterations']} iterations")

        return metrics

    def has_converged(self) -> bool:
        """Check if OLAM has converged."""
        if self.learner and hasattr(self.learner, 'model_convergence'):
            return self.learner.model_convergence
        return self._converged

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        stats = super().get_statistics()
        if self.learner:
            stats['olam_iterations'] = self.learner.iter
        return stats
