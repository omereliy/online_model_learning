"""
PDDL I/O: Read and write PDDL files using Unified Planning.

Provides clean wrappers around UP's PDDLReader and PDDLWriter,
converting to/from project types using UPAdapter.

Key responsibilities:
- Parse PDDL files → LiftedDomainKnowledge + initial state
- Export LiftedDomainKnowledge → PDDL files
- Handle file I/O and validation
"""

import logging
from pathlib import Path
from typing import Tuple, Set, Optional
from unified_planning.io import PDDLReader as UPReader, PDDLWriter as UPWriter
from unified_planning.model import Problem

from src.core.up_adapter import UPAdapter
from src.core.lifted_domain import LiftedDomainKnowledge

logger = logging.getLogger(__name__)


class PDDLReader:
    """
    Read PDDL files and convert to project types.

    Uses UP's PDDLReader internally and UPAdapter for conversions.
    """

    def __init__(self):
        """Initialize PDDL reader."""
        self.up_reader = UPReader()
        self.adapter = UPAdapter()
        self._last_up_problem: Optional[Problem] = None

    def parse_domain_and_problem(self, domain_file: str, problem_file: str) -> Tuple[LiftedDomainKnowledge, Set[str]]:
        """
        Parse PDDL domain and problem files.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file

        Returns:
            Tuple of (LiftedDomainKnowledge, initial_state_fluents)

        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If PDDL parsing fails

        Example:
            reader = PDDLReader()
            domain, initial_state = reader.parse_domain_and_problem(
                'domain.pddl', 'problem.pddl'
            )
        """
        # Validate files exist
        domain_path = Path(domain_file)
        problem_path = Path(problem_file)

        if not domain_path.exists():
            raise FileNotFoundError(f"Domain file not found: {domain_file}")
        if not problem_path.exists():
            raise FileNotFoundError(f"Problem file not found: {problem_file}")

        logger.info(f"Parsing PDDL files: {domain_file}, {problem_file}")

        try:
            # Parse using UP
            up_problem = self.up_reader.parse_problem(domain_file, problem_file)
            self._last_up_problem = up_problem

            # Convert to project types
            domain = LiftedDomainKnowledge.from_up_problem(up_problem, self.adapter)
            initial_state = self.adapter.get_initial_state_as_fluent_set(up_problem)

            logger.info(f"Successfully parsed: {len(domain.lifted_actions)} actions, "
                       f"{len(initial_state)} initial fluents")

            return domain, initial_state

        except Exception as e:
            logger.error(f"Failed to parse PDDL files: {e}")
            raise ValueError(f"PDDL parsing failed: {e}") from e

    def get_up_problem(self) -> Optional[Problem]:
        """
        Get the last parsed UP Problem.

        Useful for accessing UP-specific features or for compatibility.

        Returns:
            Last parsed UP Problem or None
        """
        return self._last_up_problem


class PDDLWriter:
    """
    Write project types to PDDL files.

    Uses UP's PDDLWriter internally and UPAdapter for conversions.
    """

    def __init__(self):
        """Initialize PDDL writer."""
        self.adapter = UPAdapter()

    def export_domain(self, domain: LiftedDomainKnowledge, output_file: str) -> None:
        """
        Export domain to PDDL file.

        Args:
            domain: LiftedDomainKnowledge to export
            output_file: Output PDDL file path

        Raises:
            IOError: If writing fails

        Note:
            This requires converting back to UP Problem format.
            For now, this is a placeholder - full implementation needs
            reverse conversion from LiftedDomainKnowledge to UP Problem.
        """
        # TODO: Implement reverse conversion LiftedDomainKnowledge → UP Problem
        # This is complex and may not be needed initially if we only read PDDL
        raise NotImplementedError(
            "Export from LiftedDomainKnowledge to PDDL not yet implemented. "
            "Use export_up_problem() if you have a UP Problem object."
        )

    def export_up_problem(self, up_problem: Problem, domain_file: str, problem_file: str) -> None:
        """
        Export UP Problem directly to PDDL files.

        This is a passthrough to UP's PDDLWriter for cases where we
        have a UP Problem object (e.g., from PDDLReader).

        Args:
            up_problem: UP Problem to export
            domain_file: Output domain file path
            problem_file: Output problem file path

        Raises:
            IOError: If writing fails

        Example:
            reader = PDDLReader()
            domain, state = reader.parse_domain_and_problem('d.pddl', 'p.pddl')

            # Modify problem...

            writer = PDDLWriter()
            writer.export_up_problem(
                reader.get_up_problem(),
                'output_domain.pddl',
                'output_problem.pddl'
            )
        """
        logger.info(f"Exporting UP Problem to: {domain_file}, {problem_file}")

        try:
            writer = UPWriter(up_problem)
            writer.write_domain(domain_file)
            writer.write_problem(problem_file)

            logger.info("Successfully exported PDDL files")

        except Exception as e:
            logger.error(f"Failed to export PDDL files: {e}")
            raise IOError(f"PDDL export failed: {e}") from e


# ========== Convenience Functions ==========

def parse_pddl(domain_file: str, problem_file: str) -> Tuple[LiftedDomainKnowledge, Set[str]]:
    """
    Convenience function to parse PDDL files.

    Args:
        domain_file: Path to PDDL domain file
        problem_file: Path to PDDL problem file

    Returns:
        Tuple of (LiftedDomainKnowledge, initial_state_fluents)

    Example:
        domain, initial_state = parse_pddl('domain.pddl', 'problem.pddl')
    """
    reader = PDDLReader()
    return reader.parse_domain_and_problem(domain_file, problem_file)


def validate_pddl_files(domain_file: str, problem_file: str) -> bool:
    """
    Validate that PDDL files can be parsed successfully.

    Args:
        domain_file: Path to PDDL domain file
        problem_file: Path to PDDL problem file

    Returns:
        True if valid, False otherwise
    """
    try:
        parse_pddl(domain_file, problem_file)
        return True
    except Exception as e:
        logger.error(f"PDDL validation failed: {e}")
        return False
