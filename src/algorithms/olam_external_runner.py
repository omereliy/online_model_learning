"""
OLAM External Runner for Post-Processing Approach.

This module runs OLAM as an independent subprocess and collects its outputs
for offline analysis. This replaces the complex real-time integration approach
with a simpler batch processing model.

Author: OLAM Refactor Implementation
Date: 2025
"""

import subprocess
import shutil
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class OLAMRunResult:
    """Results from an OLAM experiment run."""
    success: bool
    trace_file: Optional[Path] = None
    exports_dir: Optional[Path] = None
    learned_domain: Optional[Path] = None
    final_model: Optional[Dict] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class OLAMExternalRunner:
    """
    Run OLAM as an independent subprocess.

    This runner executes OLAM's main.py script with appropriate configuration
    and collects the resulting trace files and model exports.
    """

    def __init__(self, olam_dir: Path = Path("/home/omer/projects/OLAM")):
        """
        Initialize the OLAM external runner.

        Args:
            olam_dir: Path to OLAM repository
        """
        self.olam_dir = Path(olam_dir)
        self.main_script = self.olam_dir / "main.py"

        # Verify OLAM installation
        if not self.olam_dir.exists():
            raise ValueError(f"OLAM directory not found: {self.olam_dir}")
        if not self.main_script.exists():
            raise ValueError(f"OLAM main.py not found: {self.main_script}")

        logger.info(f"Initialized OLAM runner with directory: {self.olam_dir}")

    def run_experiment(self,
                      domain_file: Path,
                      problem_file: Path,
                      config: Dict[str, Any],
                      output_dir: Optional[Path] = None) -> OLAMRunResult:
        """
        Run OLAM experiment and collect results.

        Args:
            domain_file: Path to domain PDDL file
            problem_file: Path to problem PDDL file
            config: Experiment configuration dictionary
            output_dir: Optional directory for outputs (temp if not specified)

        Returns:
            OLAMRunResult with execution details and output paths
        """
        start_time = time.time()
        logger.info(f"Starting OLAM experiment: {domain_file.stem}/{problem_file.stem}")

        # Validate inputs
        if not domain_file.exists():
            return OLAMRunResult(
                success=False,
                error_message=f"Domain file not found: {domain_file}"
            )
        if not problem_file.exists():
            return OLAMRunResult(
                success=False,
                error_message=f"Problem file not found: {problem_file}"
            )

        # Setup output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="olam_run_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Prepare OLAM configuration
            olam_config = self._prepare_olam_config(config, output_dir)

            # 2. Setup OLAM working directory
            self._setup_olam_workspace(domain_file, problem_file, olam_config)

            # 3. Run OLAM
            # Extract domain name from parent directory (e.g., "blocksworld" from "benchmarks/olam-compatible/blocksworld/domain.pddl")
            domain_name = domain_file.parent.name
            result = self._execute_olam(
                domain_name=domain_name,
                problem_name=problem_file.stem,
                config=olam_config
            )

            if not result['success']:
                return OLAMRunResult(
                    success=False,
                    error_message=result.get('error', 'OLAM execution failed'),
                    stdout=result.get('stdout'),
                    stderr=result.get('stderr'),
                    execution_time=time.time() - start_time
                )

            # 4. Collect outputs
            outputs = self._collect_outputs(
                domain_name=domain_name,
                problem_name=problem_file.stem
            )

            # 5. Copy outputs to output directory
            final_outputs = self._copy_outputs_to_dir(outputs, output_dir)

            return OLAMRunResult(
                success=True,
                trace_file=final_outputs.get('trace_file'),
                exports_dir=final_outputs.get('exports_dir'),
                learned_domain=final_outputs.get('learned_domain'),
                final_model=final_outputs.get('final_model'),
                execution_time=time.time() - start_time,
                stdout=result.get('stdout'),
                stderr=result.get('stderr')
            )

        except Exception as e:
            logger.error(f"OLAM experiment failed: {e}")
            return OLAMRunResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
        finally:
            # Restore original Configuration.py
            self._restore_configuration()

    def _prepare_olam_config(self, config: Dict, output_dir: Path) -> Dict:
        """
        Convert our configuration to OLAM's format.

        Args:
            config: Our experiment configuration
            output_dir: Directory for outputs

        Returns:
            OLAM-compatible configuration dictionary
        """
        olam_config = {
            'OUTPUT_DIR': str(output_dir),
            'MAX_ITER': config.get('max_iterations', 300),
            'TIME_LIMIT_SECONDS': config.get('time_limit_seconds', 14400),
            'PLANNER_TIME_LIMIT': config.get('planner_time_limit', 240),
            'MAX_PRECS_LENGTH': config.get('max_precs_length', 8),
            'NEG_EFF_ASSUMPTION': config.get('neg_eff_assumption', False),
            'OUTPUT_CONSOLE': config.get('output_console', False),
            'RANDOM_SEED': config.get('random_seed', 42)
        }

        # Add any algorithm-specific parameters
        if 'olam_params' in config:
            olam_config.update(config['olam_params'])

        return olam_config

    def _setup_olam_workspace(self, domain_file: Path, problem_file: Path,
                              config: Dict) -> None:
        """
        Setup OLAM's expected directory structure.

        OLAM expects:
        - PDDL/domain.pddl
        - PDDL/domain_input.pddl
        - PDDL/facts.pddl
        - Configuration.py with settings
        """
        # Create PDDL directory
        pddl_dir = self.olam_dir / "PDDL"
        pddl_dir.mkdir(exist_ok=True)

        # Create backup directory for existing files
        backup_dir = self.olam_dir / "PDDL" / "backup"
        backup_dir.mkdir(exist_ok=True)

        # Backup existing files if they exist
        for filename in ['domain.pddl', 'domain_input.pddl', 'facts.pddl']:
            source = pddl_dir / filename
            if source.exists():
                dest = backup_dir / f"{filename}.{int(time.time())}"
                shutil.move(str(source), str(dest))
                logger.debug(f"Backed up {filename} to {dest}")

        # Copy new files
        # NOTE: We only copy domain.pddl and facts.pddl, NOT domain_input.pddl
        # OLAM has a fragile regex-based parser for domain_input.pddl that often fails
        # Without domain_input.pddl, OLAM starts from scratch which works reliably

        # Normalize domain file (lowercase action names) before copying
        self._normalize_pddl(domain_file, pddl_dir / "domain.pddl")
        shutil.copy(str(problem_file), str(pddl_dir / "facts.pddl"))

        # OLAM will create domain_learned.pddl itself

        # Write configuration file
        self._write_configuration(config)

        logger.debug(f"Setup OLAM workspace with domain={domain_file.name}, problem={problem_file.name}")

    def _normalize_pddl(self, source_file: Path, dest_file: Path) -> None:
        """
        Normalize PDDL file for OLAM compatibility.

        OLAM's regex-based parser expects lowercase action names.
        This method converts action names like ":action Drive" to ":action drive"
        """
        with open(source_file, 'r') as f:
            content = f.read()

        # Normalize action names to lowercase
        # Pattern: (:action ACTIONNAME -> (:action actionname
        import re
        def lowercase_action(match):
            return f':action {match.group(1).lower()}'

        content = re.sub(r':action\s+(\w+)', lowercase_action, content, flags=re.IGNORECASE)

        with open(dest_file, 'w') as f:
            f.write(content)

        logger.debug(f"Normalized PDDL: {source_file} -> {dest_file}")

    def _write_configuration(self, config: Dict) -> None:
        """
        Patch OLAM Configuration.py file with our settings.

        Instead of replacing the whole file, we just modify specific values.
        """
        # Backup original Configuration.py
        config_orig = self.olam_dir / "Configuration.py"
        config_backup = self.olam_dir / "Configuration.py.backup"
        if config_orig.exists() and not config_backup.exists():
            shutil.copy(str(config_orig), str(config_backup))

        # Read original Configuration.py
        with open(config_orig, 'r') as f:
            lines = f.readlines()

        # Patch specific values
        new_lines = []
        for line in lines:
            # Force OUTPUT_CONSOLE=False so logs are written to files
            if line.strip().startswith('OUTPUT_CONSOLE'):
                new_lines.append('OUTPUT_CONSOLE = False\n')
            # Update MAX_ITER
            elif line.strip().startswith('MAX_ITER'):
                new_lines.append(f'MAX_ITER = {config.get("MAX_ITER", 300)}\n')
            # Update TIME_LIMIT_SECONDS
            elif line.strip().startswith('TIME_LIMIT_SECONDS'):
                new_lines.append(f'TIME_LIMIT_SECONDS = {config.get("TIME_LIMIT_SECONDS", 14400)}\n')
            # Update PLANNER_TIME_LIMIT
            elif line.strip().startswith('PLANNER_TIME_LIMIT'):
                new_lines.append(f'PLANNER_TIME_LIMIT = {config.get("PLANNER_TIME_LIMIT", 240)}\n')
            else:
                new_lines.append(line)

        # Write patched configuration
        with open(config_orig, 'w') as f:
            f.writelines(new_lines)

        logger.debug(f"Patched Configuration.py with experiment settings")

    def _restore_configuration(self) -> None:
        """Restore original Configuration.py from backup."""
        config_path = self.olam_dir / "Configuration.py"
        config_backup = self.olam_dir / "Configuration.py.backup"

        if config_backup.exists():
            shutil.copy(str(config_backup), str(config_path))
            config_backup.unlink()  # Remove backup
            logger.debug("Restored original Configuration.py")

    def _execute_olam(self, domain_name: str, problem_name: str,
                     config: Dict) -> Dict[str, Any]:
        """
        Execute OLAM main.py as subprocess.

        Args:
            domain_name: Name of domain
            problem_name: Name of problem
            config: OLAM configuration

        Returns:
            Dictionary with success status and outputs
        """
        # Prepare environment
        env = os.environ.copy()

        # Override Configuration.py with our temporary version
        env['PYTHONPATH'] = str(self.olam_dir)

        # Set configuration via environment variables
        for key, value in config.items():
            env[f'OLAM_{key}'] = str(value)

        # Build command - use system Python 3.10 for OLAM compatibility
        cmd = [
            "/usr/bin/python3",  # System Python 3.10.12 works with OLAM
            str(self.main_script),
            "-d", domain_name  # Run specific domain
        ]

        logger.info(f"Executing OLAM: {' '.join(cmd)}")

        try:
            # Run OLAM with timeout
            timeout = config.get('TIME_LIMIT_SECONDS', 14400) + 60  # Add buffer
            result = subprocess.run(
                cmd,
                cwd=str(self.olam_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            success = result.returncode == 0

            if not success:
                logger.warning(f"OLAM exited with code {result.returncode}")
                if result.stderr:
                    logger.warning(f"OLAM stderr: {result.stderr[:1000]}")

            return {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except subprocess.TimeoutExpired:
            logger.error(f"OLAM execution timed out after {timeout} seconds")
            return {
                'success': False,
                'error': f'Timeout after {timeout} seconds'
            }
        except Exception as e:
            logger.error(f"Failed to execute OLAM: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _collect_outputs(self, domain_name: str, problem_name: str) -> Dict[str, Path]:
        """
        Collect OLAM output files.

        OLAM outputs are stored in:
        - Analysis/run_X/Tests/<domain>/<problem>/
        - Analysis/run_X/Results_cert/
        - Analysis/run_X/Results_uncert_neg/

        Args:
            domain_name: Domain name
            problem_name: Problem name

        Returns:
            Dictionary of output file paths
        """
        outputs = {}

        # Find the most recent run directory
        analysis_dir = self.olam_dir / "Analysis"
        if not analysis_dir.exists():
            logger.warning("OLAM Analysis directory not found")
            return outputs

        run_dirs = [d for d in analysis_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not run_dirs:
            logger.warning("No OLAM run directories found")
            return outputs

        # Sort by modification time to get the most recent
        latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
        logger.info(f"Collecting outputs from: {latest_run}")

        # Find trace file
        # OLAM stores trace in Tests/<domain>/<problem>/<problem>_log
        test_dir = latest_run / "Tests" / domain_name
        if test_dir.exists():
            # Find problem directory (may have numbered prefix like "1_p00_domain_gen")
            problem_dirs = [d for d in test_dir.iterdir() if d.is_dir() and problem_name in d.name]
            if problem_dirs:
                problem_dir = problem_dirs[0]
                log_files = list(problem_dir.glob("*_log"))
                if log_files:
                    outputs['trace_file'] = log_files[0]
                    logger.debug(f"Found trace file: {outputs['trace_file']}")

                # Find learned domain
                learned_domains = list(problem_dir.glob("domain_learned*.pddl"))
                if learned_domains:
                    outputs['learned_domain'] = learned_domains[0]
                    logger.debug(f"Found learned domain: {outputs['learned_domain']}")

        # Find JSON exports (may be in different locations)
        json_patterns = {
            'certain_precs': '**/operator_certain_predicates.json',
            'uncertain_precs': '**/operator_uncertain_predicates.json',
            'neg_precs': '**/operator_negative_preconditions.json',
            'add_effects': '**/certain_positive_effects.json',
            'del_effects': '**/certain_negative_effects.json'
        }

        exports = {}
        for key, pattern in json_patterns.items():
            matches = list(latest_run.glob(pattern))
            if matches:
                exports[key] = matches[0]
                logger.debug(f"Found {key}: {exports[key]}")

        if exports:
            outputs['exports'] = exports

        # Store the run directory for additional files
        outputs['run_dir'] = latest_run

        return outputs

    def _copy_outputs_to_dir(self, outputs: Dict[str, Any],
                             output_dir: Path) -> Dict[str, Any]:
        """
        Copy OLAM outputs to specified directory.

        Args:
            outputs: Dictionary of output paths
            output_dir: Target directory

        Returns:
            Dictionary with new paths
        """
        final_outputs = {}

        # Copy trace file
        if 'trace_file' in outputs and outputs['trace_file'].exists():
            dest = output_dir / "trace.log"
            shutil.copy(str(outputs['trace_file']), str(dest))
            final_outputs['trace_file'] = dest
            logger.debug(f"Copied trace to {dest}")

        # Copy learned domain
        if 'learned_domain' in outputs and outputs['learned_domain'].exists():
            dest = output_dir / "domain_learned.pddl"
            shutil.copy(str(outputs['learned_domain']), str(dest))
            final_outputs['learned_domain'] = dest
            logger.debug(f"Copied learned domain to {dest}")

        # Copy JSON exports
        if 'exports' in outputs:
            exports_dir = output_dir / "exports"
            exports_dir.mkdir(exist_ok=True)
            final_outputs['exports_dir'] = exports_dir

            for key, source_path in outputs['exports'].items():
                if source_path.exists():
                    dest = exports_dir / source_path.name
                    shutil.copy(str(source_path), str(dest))
                    logger.debug(f"Copied {key} to {dest}")

            # Also load and combine exports into final model
            final_model = {}
            for key, source_path in outputs['exports'].items():
                if source_path.exists():
                    try:
                        with open(source_path, 'r') as f:
                            final_model[key] = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse {source_path}")

            if final_model:
                final_outputs['final_model'] = final_model

        # Copy complete run directory for reference
        if 'run_dir' in outputs:
            complete_dir = output_dir / "complete_run"
            if outputs['run_dir'].exists():
                shutil.copytree(str(outputs['run_dir']), str(complete_dir), dirs_exist_ok=True)
                logger.debug(f"Copied complete run to {complete_dir}")

        return final_outputs

    def cleanup_workspace(self) -> None:
        """
        Clean up OLAM workspace after experiments.

        Restores backed up files and removes temporary configurations.
        """
        pddl_dir = self.olam_dir / "PDDL"
        backup_dir = pddl_dir / "backup"

        if backup_dir.exists():
            # Find most recent backups
            for filename in ['domain.pddl', 'domain_input.pddl', 'facts.pddl']:
                backups = sorted(backup_dir.glob(f"{filename}.*"))
                if backups:
                    latest_backup = backups[-1]
                    dest = pddl_dir / filename
                    shutil.copy(str(latest_backup), str(dest))
                    logger.debug(f"Restored {filename} from backup")

        # Remove temporary configuration
        temp_config = self.olam_dir / "Configuration_temp.py"
        if temp_config.exists():
            temp_config.unlink()
            logger.debug("Removed temporary configuration")

    def run_batch_experiments(self,
                             experiments: List[Tuple[Path, Path, Dict]],
                             output_base_dir: Path) -> List[OLAMRunResult]:
        """
        Run multiple OLAM experiments in batch.

        Args:
            experiments: List of (domain_file, problem_file, config) tuples
            output_base_dir: Base directory for all outputs

        Returns:
            List of OLAMRunResult objects
        """
        results = []

        for i, (domain_file, problem_file, config) in enumerate(experiments):
            # Create output directory for this experiment
            exp_name = f"{domain_file.stem}_{problem_file.stem}"
            output_dir = output_base_dir / exp_name

            logger.info(f"Running experiment {i+1}/{len(experiments)}: {exp_name}")

            # Run experiment
            result = self.run_experiment(
                domain_file=domain_file,
                problem_file=problem_file,
                config=config,
                output_dir=output_dir
            )

            results.append(result)

            # Brief pause between experiments
            time.sleep(1)

        # Cleanup after batch
        self.cleanup_workspace()

        return results