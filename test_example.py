#!/usr/bin/env python3
"""
Simple test script to verify Switch Transformer implementation.
"""

import torch
from switch_transformers import SwitchTransformer


def test_basic_functionality():
    """Test basic model instantiation and forward pass."""

    # Model parameters
    batch_size, seq_len = 2, 128
    inp_dim, num_experts, num_heads, vocab_size = 256, 4, 8, 1000

    # Create model
    model = SwitchTransformer(
        inp_dim=inp_dim,
        num_experts=num_experts,
        num_heads=num_heads,
        vocab_size=vocab_size,
        depth=2,  # Small model for testing
        use_aux_loss=True
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        logits, aux_loss = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(".4f")
    print("✓ Basic functionality test passed!")


def test_biased_gating():
    """Test auxiliary loss-free load balancing."""

    inp_dim, num_experts, num_heads, vocab_size = 256, 4, 8, 1000

    model = SwitchTransformer(
        inp_dim=inp_dim,
        num_experts=num_experts,
        num_heads=num_heads,
        vocab_size=vocab_size,
        depth=2,
        use_biased_gating=True
    )

    input_ids = torch.randint(0, vocab_size, (2, 64))

    with torch.no_grad():
        logits, aux_loss = model(input_ids)

    print(f"Biased gating - Logits shape: {logits.shape}")
    print(".4f")
    print("✓ Biased gating test passed!")


if __name__ == "__main__":
    print("Testing Switch Transformer implementation...")
    test_basic_functionality()
    print()
    test_biased_gating()
    print("\n✅ All tests passed successfully!")
