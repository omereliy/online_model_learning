"""
Experiment framework for online action model learning.
"""

from .metrics import MetricsCollector

__all__ = ['MetricsCollector']

# Import runner separately to avoid circular dependencies
try:
    from .runner import ExperimentRunner
    __all__.append('ExperimentRunner')
except ImportError:
    # ExperimentRunner has additional dependencies that might not be available in tests
    pass