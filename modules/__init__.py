"""
Modules package for modular training components.

This package contains:
- config: Hardware detection and environment setup
- datasets: Data loading and tokenization
- callbacks: Training callbacks (logging, plotting, early stopping)
- utils: Utility functions
"""

from .config import detect_hardware, MODEL_PATH
from .datasets import TextFileDataset, TokenizedDataset, SubsetDataset
from .callbacks import (
    StepLoggingCallback,
    SafeEarlyStoppingCallback,
    PlottingCallback,
    EnhancedEvalCallback
)
from .utils import get_latest_checkpoint, print_training_summary

__all__ = [
    'detect_hardware',
    'MODEL_PATH',
    'TextFileDataset',
    'TokenizedDataset',
    'SubsetDataset',
    'StepLoggingCallback',
    'SafeEarlyStoppingCallback',
    'PlottingCallback',
    'EnhancedEvalCallback',
    'get_latest_checkpoint',
    'print_training_summary',
]
