"""
Minimal utilities for Switch Transformer load balancing.

This module is kept for backward compatibility but most functionality
has been integrated directly into the main SwitchLayer class.
"""

import torch


def update_gating_biases(
    gating_biases: torch.Tensor,
    error_sign: torch.Tensor,
    bias_update_rate: float = 0.001,
) -> torch.Tensor:
    """Update gating biases for auxiliary loss-free load balancing.

    Args:
        gating_biases: Current bias values for each expert
        error_sign: Error signal indicating load imbalance
        bias_update_rate: Learning rate for bias updates

    Returns:
        Updated gating biases
    """
    return gating_biases + (error_sign * bias_update_rate)
