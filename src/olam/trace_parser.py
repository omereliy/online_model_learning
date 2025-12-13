"""
OLAM Trace Parser for Post-Processing Analysis.

This module parses OLAM's native output files (execution logs and JSON exports)
to enable offline analysis and model reconstruction at checkpoints.

Author: OLAM Refactor Implementation
Date: 2025
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple
import re
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class OLAMTraceStep:
    """Single step in OLAM's execution trace."""
    iteration: int
    action: str  # e.g., "pick-up(a)" or "stack(a,b)"
    success: bool
    state_before: Set[str]  # PDDL predicates before action
    state_after: Optional[Set[str]] = None  # PDDL predicates after (if successful)
    strategy: Optional[str] = None  # OLAM's selection strategy (e.g., "exploration", "planning")


class OLAMTraceParser:
    """
    Parse OLAM's native output files.

    OLAM generates several output files during execution:
    1. <problem>_log: Execution trace with actions and states
    2. operator_*.json: Learned model components (preconditions, effects)
    3. domain_learned.pddl: Final learned PDDL domain
    """

    def parse_log_file(self, log_path: Path) -> List[OLAMTraceStep]:
        """
        Parse OLAM's execution log file.

        OLAM log format (varies by version):
        - JSON Lines format: One JSON object per line (current OLAM format)
        - Standard format: Iteration, action, success, states
        - Extended format: Includes strategy and learning updates

        Args:
            log_path: Path to OLAM's trace file (trace.json or <problem>_log)

        Returns:
            List of trace steps in chronological order
        """
        if not log_path.exists():
            raise FileNotFoundError(f"OLAM log file not found: {log_path}")

        logger.info(f"Parsing OLAM trace from: {log_path}")
        steps = []

        with open(log_path, 'r') as f:
            content = f.read()

        # Try different parsing strategies based on format detection
        # Check for JSON Lines first (each line is a JSON object)
        if content.strip() and content.strip()[0] == '{':
            # Likely JSON Lines format
            steps = self._parse_json_lines_format(content)
        elif re.search(r'Time\s+\|\s+Iter\s+\|.*Real_precs', content):
            # OLAM's native format with metrics tables
            steps = self._parse_olam_native_format(content)
        elif "Iteration:" in content:
            steps = self._parse_standard_format(content)
        elif "ITER:" in content:
            steps = self._parse_compact_format(content)
        else:
            # Try to parse line by line for custom formats
            steps = self._parse_line_format(content)

        logger.info(f"Parsed {len(steps)} trace steps from OLAM log")
        return steps

    def _parse_standard_format(self, content: str) -> List[OLAMTraceStep]:
        """
        Parse standard OLAM log format.

        Format:
            Iteration: 1
            Selected action: pick-up(a)
            Success: True
            State before: (clear a) (ontable a) (handempty)
            State after: (holding a)
            ---
        """
        steps = []

        # Split by separator
        blocks = content.split('---')

        for block in blocks:
            if not block.strip():
                continue

            # Extract iteration number
            iter_match = re.search(r'Iteration:\s*(\d+)', block)
            if not iter_match:
                continue

            iteration = int(iter_match.group(1))

            # Extract action
            action_match = re.search(r'(?:Selected action|Action|Executed):\s*(.+)', block)
            if not action_match:
                continue
            action = action_match.group(1).strip()

            # Extract success status
            success = "Success: True" in block or "successful" in block.lower()
            if "Success: False" in block or "failed" in block.lower():
                success = False

            # Extract states
            state_before = self._extract_state(block, r'State(?: before)?:\s*(.+?)(?:\n|State after|$)')
            state_after = None

            if success:
                state_after = self._extract_state(block, r'State after:\s*(.+?)(?:\n|---|$)')
                # If no explicit "State after", try to infer from effects
                if not state_after:
                    state_after = self._infer_state_from_effects(state_before, block)

            # Extract strategy if available
            strategy = None
            strategy_match = re.search(r'Strategy:\s*(\w+)', block)
            if strategy_match:
                strategy = strategy_match.group(1)

            steps.append(OLAMTraceStep(
                iteration=iteration,
                action=action,
                success=success,
                state_before=state_before,
                state_after=state_after,
                strategy=strategy
            ))

        return steps

    def _parse_compact_format(self, content: str) -> List[OLAMTraceStep]:
        """
        Parse compact OLAM log format.

        Format:
            ITER:1 ACTION:pick-up(a) RESULT:SUCCESS STATE:(clear a)...
        """
        steps = []
        lines = content.split('\n')

        for line in lines:
            if 'ITER:' not in line:
                continue

            # Extract components using regex
            iter_match = re.search(r'ITER:(\d+)', line)
            action_match = re.search(r'ACTION:([^\s]+)', line)
            result_match = re.search(r'RESULT:(\w+)', line)
            state_match = re.search(r'STATE:(.+?)(?:NEXT:|$)', line)
            next_match = re.search(r'NEXT:(.+?)$', line)

            if not (iter_match and action_match):
                continue

            iteration = int(iter_match.group(1))
            action = action_match.group(1)
            success = result_match and result_match.group(1).upper() == 'SUCCESS'

            state_before = set()
            if state_match:
                state_str = state_match.group(1)
                state_before = self._parse_predicates(state_str)

            state_after = None
            if success and next_match:
                next_str = next_match.group(1)
                state_after = self._parse_predicates(next_str)

            steps.append(OLAMTraceStep(
                iteration=iteration,
                action=action,
                success=success,
                state_before=state_before,
                state_after=state_after
            ))

        return steps

    def _parse_line_format(self, content: str) -> List[OLAMTraceStep]:
        """
        Parse line-by-line format (fallback).

        Attempts to extract action executions from unstructured log.
        """
        steps = []
        lines = content.split('\n')

        iteration = 0
        for line in lines:
            # Look for action execution patterns
            action_patterns = [
                r'Executing:\s*(.+)',
                r'Action:\s*(.+)',
                r'Selected:\s*(.+)',
                r'(\w+\([^)]*\))\s*(?:executed|selected|tried)'
            ]

            for pattern in action_patterns:
                match = re.search(pattern, line)
                if match:
                    action = match.group(1).strip()
                    # Simple heuristic for success
                    success = 'fail' not in line.lower() and 'error' not in line.lower()

                    steps.append(OLAMTraceStep(
                        iteration=iteration,
                        action=action,
                        success=success,
                        state_before=set(),  # Cannot extract from this format
                        state_after=None
                    ))
                    iteration += 1
                    break

        return steps

    def _parse_json_lines_format(self, content: str) -> List[OLAMTraceStep]:
        """
        Parse JSON Lines format (one JSON object per line).

        This is the current OLAM output format where each line is a complete JSON object.

        Format example:
            {"domain": "depots", "problem": "1_p00_depots_gen", "iter": 1,
             "action": "lift(hoist0,crate1,pallet1,distributor0)",
             "success": false, "strategy": "Random",
             "timestamp": "2025-11-19T13:35:26.021015"}

        Note: OLAM's JSON trace does NOT include state_before or state_after.
        These fields will be empty sets in the returned OLAMTraceStep objects.

        Args:
            content: File content with one JSON object per line

        Returns:
            List of trace steps parsed from JSON
        """
        steps = []
        lines = content.strip().split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Extract required fields
                iteration = data.get('iter')
                action = data.get('action')
                success = data.get('success')

                if iteration is None or action is None or success is None:
                    logger.warning(f"Line {line_num}: Missing required fields (iter, action, success)")
                    continue

                # Extract optional fields
                strategy = data.get('strategy')

                # Note: OLAM does not provide state information in trace.json
                # State reconstruction must be done from checkpoint exports
                steps.append(OLAMTraceStep(
                    iteration=iteration,
                    action=action,
                    success=success,
                    state_before=set(),  # Not available in OLAM's JSON trace
                    state_after=None,     # Not available in OLAM's JSON trace
                    strategy=strategy
                ))

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Failed to parse JSON: {e}")
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Unexpected error: {e}")
                continue

        return steps

    def _parse_olam_native_format(self, content: str) -> List[OLAMTraceStep]:
        """
        Parse OLAM's native log format with metrics tables and action execution.

        Format:
            Time  | Iter |Real_precs|Learn_precs|...
            0.00  |  0   |    7     |     8     |...

            Successfully executed action 40: sail(l4,l0)

            Time  | Iter |Real_precs|Learn_precs|...
            2.25  |  2   |    7     |     8     |...
        """
        steps = []
        lines = content.split('\n')

        # Regex patterns
        iter_pattern = r'\s*([\d.]+)\s+\|\s+(\d+)\s+\|'
        action_success_pattern = r'^Successfully executed action \d+:\s*(.+)'
        action_fail_pattern = r'^Not [Ss]uccessfully executed action \d+:\s*(.+)'

        current_iter = 0  # Start at 0, actions increment this

        for i, line in enumerate(lines):
            # Check if this is a metrics table row
            iter_match = re.search(iter_pattern, line)
            if iter_match:
                # Update iteration counter from table
                current_iter = int(iter_match.group(2))
                continue

            # Check for action execution lines (check fail pattern first to avoid substring match)
            fail_match = re.search(action_fail_pattern, line)
            success_match = re.search(action_success_pattern, line) if not fail_match else None

            if success_match or fail_match:
                # Actions occur AFTER the metrics table for current_iter
                # So the action belongs to iteration current_iter + 1
                action_iter = current_iter + 1

                action = success_match.group(1) if success_match else fail_match.group(1)
                success = bool(success_match)

                steps.append(OLAMTraceStep(
                    iteration=action_iter,
                    action=action.strip(),
                    success=success,
                    state_before=set(),  # Not available in this format
                    state_after=None
                ))

        return steps

    def _extract_state(self, text: str, pattern: str) -> Set[str]:
        """Extract state predicates using regex pattern."""
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return set()

        state_str = match.group(1)
        return self._parse_predicates(state_str)

    def _parse_predicates(self, pred_str: str) -> Set[str]:
        """
        Parse PDDL predicates from string.

        Handles formats:
        - (clear a) (on a b)
        - clear(a) on(a,b)
        - Mixed formats
        """
        predicates = set()

        # Pattern for standard PDDL format: (pred arg1 arg2 ...)
        pddl_pattern = r'\([^()]+\)'
        predicates.update(re.findall(pddl_pattern, pred_str))

        # Pattern for function-like format: pred(arg1,arg2,...)
        func_pattern = r'\b(\w+)\(([^)]*)\)'
        for match in re.finditer(func_pattern, pred_str):
            pred_name = match.group(1)
            args = match.group(2)
            if ',' in args:
                # Convert comma-separated to space-separated
                args = args.replace(',', ' ')
            if args:
                predicates.add(f"({pred_name} {args})")
            else:
                predicates.add(f"({pred_name})")

        return predicates

    def _infer_state_from_effects(self, state_before: Set[str], block: str) -> Set[str]:
        """
        Infer next state from effects if not explicitly provided.

        Looks for effect patterns like:
        - Effects: +(holding a) -(ontable a) -(handempty)
        - Add: (holding a), Delete: (ontable a) (handempty)
        """
        state_after = state_before.copy()

        # Look for add effects
        add_patterns = [
            r'\+\(([^)]+)\)',
            r'Add(?:ed)?:\s*\(([^)]+)\)',
            r'Positive effects?:\s*\(([^)]+)\)'
        ]

        for pattern in add_patterns:
            for match in re.finditer(pattern, block):
                pred = f"({match.group(1)})"
                state_after.add(pred)

        # Look for delete effects
        del_patterns = [
            r'-\(([^)]+)\)',
            r'Delete?d?:\s*\(([^)]+)\)',
            r'Negative effects?:\s*\(([^)]+)\)'
        ]

        for pattern in del_patterns:
            for match in re.finditer(pattern, block):
                pred = f"({match.group(1)})"
                state_after.discard(pred)

        # Only return if we found some effects
        if state_after != state_before:
            return state_after
        return None

    def parse_json_exports(self, run_dir: Path) -> Dict[str, Dict]:
        """
        Parse OLAM's JSON export files.

        OLAM exports 8 JSON files with learned model components:
        - operator_certain_predicates.json: Certain preconditions
        - operator_uncertain_precs.json: Uncertain preconditions
        - operator_certain_positive_effects.json: Add effects
        - operator_certain_negative_effects.json: Delete effects
        - operator_uncertain_positive_effects.json: Uncertain add effects
        - operator_uncertain_negative_effects.json: Uncertain delete effects
        - operator_useless_negated_precs.json: Useless negative preconditions
        - operator_useless_possible_precs.json: Useless possible preconditions

        Args:
            run_dir: Directory containing OLAM's output files (checkpoint dir or problem dir)

        Returns:
            Dictionary with parsed model components
        """
        exports = {}

        # Updated to match actual OLAM output file names
        json_files = {
            'certain_precs': 'operator_certain_predicates.json',
            'uncertain_precs': 'operator_uncertain_precs.json',  # Fixed: was operator_uncertain_predicates.json
            'add_effects': 'operator_certain_positive_effects.json',  # Fixed: was certain_positive_effects.json
            'del_effects': 'operator_certain_negative_effects.json',  # Fixed: was certain_negative_effects.json
            'uncertain_add_effects': 'operator_uncertain_positive_effects.json',  # New
            'uncertain_del_effects': 'operator_uncertain_negative_effects.json',  # New
            'useless_neg_precs': 'operator_useless_negated_precs.json',  # New
            'useless_pos_precs': 'operator_useless_possible_precs.json'  # New
        }

        for key, filename in json_files.items():
            # Search recursively as OLAM may nest files in subdirs
            matches = list(run_dir.rglob(filename))
            if matches:
                filepath = matches[0]
                try:
                    with open(filepath, 'r') as f:
                        exports[key] = json.load(f)
                    logger.debug(f"Loaded {key} from {filepath}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse {filepath}: {e}")
                    exports[key] = {}
            else:
                # Some files may not exist (e.g., useless predicates may be empty)
                logger.debug(f"File {filename} not found in {run_dir} (may be optional)")
                exports[key] = {}

        return exports

    def parse_learned_domain(self, domain_path: Path) -> Dict[str, Dict]:
        """
        Parse OLAM's learned PDDL domain file.

        Extracts action schemas with preconditions and effects.

        Args:
            domain_path: Path to domain_learned.pddl

        Returns:
            Dictionary of action schemas
        """
        if not domain_path.exists():
            logger.warning(f"Learned domain not found: {domain_path}")
            return {}

        with open(domain_path, 'r') as f:
            content = f.read()

        actions = {}

        # Parse action definitions
        action_pattern = r':action\s+(\w+)(.*?)(?=:action|:predicates|\Z)'

        for match in re.finditer(action_pattern, content, re.DOTALL):
            action_name = match.group(1)
            action_body = match.group(2)

            # Extract parameters
            params_match = re.search(r':parameters\s*\(([^)]*)\)', action_body)
            parameters = []
            if params_match:
                params_str = params_match.group(1)
                # Parse parameter list (e.g., "?x - object ?y - object")
                param_pattern = r'\?(\w+)(?:\s+-\s+\w+)?'
                parameters = re.findall(param_pattern, params_str)

            # Extract preconditions
            prec_match = re.search(r':precondition\s*\(and([^)]*)\)', action_body)
            preconditions = []
            if prec_match:
                prec_str = prec_match.group(1)
                preconditions = self._parse_predicates(prec_str)

            # Extract effects
            eff_match = re.search(r':effect\s*\(and([^)]*)\)', action_body)
            add_effects = []
            del_effects = []
            if eff_match:
                eff_str = eff_match.group(1)
                # Positive effects
                add_effects = re.findall(r'(?<!not\s)\(([^)]+)\)', eff_str)
                # Negative effects
                del_effects = re.findall(r'\(not\s+\(([^)]+)\)\)', eff_str)

            actions[action_name] = {
                'parameters': parameters,
                'preconditions': list(preconditions),
                'add_effects': add_effects,
                'del_effects': del_effects
            }

        logger.info(f"Parsed {len(actions)} action schemas from learned domain")
        return actions

    def extract_checkpoints(self, trace: List[OLAMTraceStep],
                           checkpoint_iterations: List[int]) -> Dict[int, List[OLAMTraceStep]]:
        """
        Extract trace segments up to each checkpoint.

        Args:
            trace: Complete execution trace
            checkpoint_iterations: List of iterations to checkpoint

        Returns:
            Dictionary mapping checkpoint iteration to trace segment
        """
        checkpoints = {}

        for checkpoint in checkpoint_iterations:
            # Get all steps up to and including this checkpoint
            segment = [step for step in trace if step.iteration <= checkpoint]
            if segment:
                checkpoints[checkpoint] = segment
                logger.debug(f"Checkpoint {checkpoint}: {len(segment)} steps")
            else:
                logger.warning(f"No trace steps found for checkpoint {checkpoint}")

        return checkpoints