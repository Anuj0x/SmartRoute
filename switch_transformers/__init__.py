"""
Switch Transformers: Efficient Sparse Transformers via Mixture-of-Experts.

This package provides a modern, high-performance implementation of Switch Transformers
with support for both auxiliary loss and auxiliary loss-free load balancing techniques.
"""

from .switch_transformer import (
    Router,
    ExpertAllocation,
    SwitchLayer,
    SwitchTransformerBlock,
    SwitchTransformer,
)
from .utils import update_gating_biases

__version__ = "0.2.0"

__all__ = [
    "Router",
    "ExpertAllocation",
    "SwitchLayer",
    "SwitchTransformerBlock",
    "SwitchTransformer",
    "update_gating_biases",
]
