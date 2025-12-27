"""
Colored logging utilities for Online Model Learning experiments.

Provides ANSI-colored console output with category-based highlighting:
- CYAN: Subset selection operations
- MAGENTA: Parallel computing operations
- YELLOW: Approximation algorithms
"""

import logging
import re
import sys
from typing import Optional


class Colors:
    """ANSI escape codes for terminal colors."""
    RESET = '\033[0m'
    CYAN = '\033[36m'      # Subset selection
    MAGENTA = '\033[35m'   # Parallel computing
    YELLOW = '\033[33m'    # Approximation


# Category to color mapping
CATEGORY_COLORS = {
    'SUBSET': Colors.CYAN,
    'PARALLEL': Colors.MAGENTA,
    'APPROX': Colors.YELLOW,
}

# Regex pattern to detect category tags like [SUBSET], [PARALLEL], [APPROX]
CATEGORY_PATTERN = re.compile(r'\[(SUBSET|PARALLEL|APPROX)\]\s*')


class ColoredFormatter(logging.Formatter):
    """Formatter that applies ANSI colors to categorized log messages for console output."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)

        # Check for category tags and apply color
        match = CATEGORY_PATTERN.search(message)
        if match:
            category = match.group(1)
            color = CATEGORY_COLORS.get(category, '')
            if color:
                # Color the entire message, keep tag for clarity
                return f"{color}{message}{Colors.RESET}"

        return message


class PlainFormatter(logging.Formatter):
    """Formatter that strips category tags for clean file output."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        # Remove category tags for clean file output
        return CATEGORY_PATTERN.sub('', message)


def setup_colored_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with colored console output.

    Args:
        level: Logging level for console output (default: INFO)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add colored console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    root_logger.addHandler(console_handler)


def create_file_handler(log_path: str, level: int = logging.DEBUG) -> logging.FileHandler:
    """
    Create a file handler with plain formatting (no ANSI codes).

    Args:
        log_path: Path to the log file
        level: Logging level for file output (default: DEBUG)

    Returns:
        Configured FileHandler
    """
    handler = logging.FileHandler(log_path, mode='w')
    handler.setLevel(level)
    handler.setFormatter(PlainFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    return handler
