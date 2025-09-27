"""
Path configuration for external tools and planners.

This module defines paths to external tools like planners and validators
that are used throughout the project.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROJECTS_DIR = Path("/home/omer/projects")

# Planner paths
FAST_DOWNWARD_DIR = PROJECTS_DIR / "fast-downward"
FAST_DOWNWARD_EXEC = FAST_DOWNWARD_DIR / "fast-downward.py"

# Alternative: Use the compiled binary directly
FAST_DOWNWARD_BINARY = FAST_DOWNWARD_DIR / "builds" / "release" / "bin" / "downward"

# Validator paths
VAL_DIR = PROJECTS_DIR / "VAL"
VAL_VALIDATE = VAL_DIR / "bin" / "Validate"

# OLAM paths
OLAM_DIR = PROJECTS_DIR / "OLAM"
OLAM_PLANNERS_DIR = OLAM_DIR / "Planners"
OLAM_PDDL_DIR = OLAM_DIR / "PDDL"

# Set environment variables for compatibility
os.environ['FAST_DOWNWARD_PATH'] = str(FAST_DOWNWARD_DIR)
os.environ['VAL_PATH'] = str(VAL_DIR / "bin")
os.environ['OLAM_PATH'] = str(OLAM_DIR)

# Planner command templates
FD_COMMAND_TEMPLATE = (
    "{exec_path} --overall-time-limit {time_limit} "
    "{domain_file} {problem_file} "
    "--evaluator \"hff=ff()\" "
    "--evaluator \"hcea=cea()\" "
    "--search \"lazy_greedy([hff,hcea],preferred=[hff,hcea])\""
)

def get_fast_downward_command(domain_file: str, problem_file: str, time_limit: int = 30) -> str:
    """
    Get the Fast Downward command with proper paths.

    Args:
        domain_file: Path to domain PDDL file
        problem_file: Path to problem PDDL file
        time_limit: Time limit in seconds

    Returns:
        Command string to execute Fast Downward
    """
    return FD_COMMAND_TEMPLATE.format(
        exec_path=FAST_DOWNWARD_EXEC,
        time_limit=time_limit,
        domain_file=domain_file,
        problem_file=problem_file
    )

def verify_paths():
    """Verify that all required external tools are available."""
    issues = []

    if not FAST_DOWNWARD_EXEC.exists():
        issues.append(f"Fast Downward not found at {FAST_DOWNWARD_EXEC}")

    if not VAL_VALIDATE.exists():
        issues.append(f"Val validator not found at {VAL_VALIDATE}")

    if not OLAM_DIR.exists():
        issues.append(f"OLAM directory not found at {OLAM_DIR}")

    if issues:
        print("Warning: Some external tools are missing:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    return True

# Verify paths on import (but don't fail if missing during tests)
if not verify_paths():
    import logging
    logging.warning("Some external tools are missing. Tests may use mocks instead.")