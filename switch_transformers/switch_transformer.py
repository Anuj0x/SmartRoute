"""
Efficient Switch Transformer implementation with modern PyTorch.

This module implements the Switch Transformer architecture from the paper
https://arxiv.org/abs/2101.03961 with auxiliary loss-free load balancing.

Key improvements:
- Vectorized expert operations for better performance
- Modern Python type hints and dataclasses
- Efficient routing and capacity management
- Comprehensive documentation
"""

from __future__ import annotations

from typing import Optional
import torch
from torch import nn

from .utils import update_gating_biases


class Router(nn.Module):
    """Neural router for expert selection.

    Routes input tokens to experts using a learned linear projection
    followed by softmax normalization.
    """

    def __init__(self, inp_dim: int, num_experts: int) -> None:
        super().__init__()
        self.inp_dim = inp_dim
        self.num_experts = num_experts
        self.layer = nn.Linear(inp_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route inputs to experts.

        Args:
            x: Input tensor of shape (..., inp_dim)

        Returns:
            Routing probabilities of shape (..., num_experts)
        """
        return torch.softmax(self.layer(x), dim=-1)


class ExpertAllocation(nn.Module):
    """Manages expert allocation with capacity constraints and auxiliary loss.

    This class handles the routing logic, capacity enforcement, and
    optional auxiliary loss computation for load balancing.
    """

    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        use_aux_loss: bool = True,
        alpha: float = 0.01,
    ) -> None:
        super().__init__()
        self.inp_dim = inp_dim
        self.num_experts = num_experts
        self.router = Router(inp_dim, num_experts)
        self.capacity_factor = capacity_factor
        self.alpha = alpha
        self.use_aux_loss = use_aux_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate tokens to experts with capacity constraints.

        Args:
            x: Input tensor of shape (batch_size, seq_len, inp_dim)

        Returns:
            Tuple of (routed_expert_probs, aux_loss)
        """
        batch_size, seq_len = x.shape[:2]
        total_tokens = batch_size * seq_len
        expert_capacity = int((total_tokens / self.num_experts) * self.capacity_factor)

        # Get routing probabilities
        expert_probs = self.router(x)  # (batch, seq, experts)

        # Select top-1 expert for each token
        top_probs, top_indices = expert_probs.max(dim=-1, keepdim=True)
        routed_experts = torch.zeros_like(expert_probs).scatter_(
            dim=-1, index=top_indices, src=top_probs
        )

        # Compute auxiliary loss for load balancing
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.use_aux_loss:
            # Importance-weighted load balancing loss
            f_i = routed_experts.sum((0, 1)) / total_tokens  # Expert utilization
            P_i = expert_probs.sum((0, 1)) / total_tokens   # Routing probability mass
            aux_loss = self.alpha * self.num_experts * (f_i * P_i).sum()

        # Enforce capacity constraints at the sequence level
        flat_routed = routed_experts.view(-1, self.num_experts)

        # Sort tokens by routing probability (descending)
        sorted_probs, sort_indices = flat_routed.topk(total_tokens, dim=0)
        cumulative_allocation = torch.cumsum(sorted_probs, dim=0)

        # Create capacity mask
        capacity_mask = (cumulative_allocation <= expert_capacity).float()
        flat_routed = capacity_mask * sorted_probs

        # Restore original order using inverse permutation
        routed_experts = torch.zeros_like(flat_routed).scatter_(
            dim=0,
            index=sort_indices,
            src=flat_routed
        ).view(expert_probs.shape)

        # Route only assigned tokens
        assigned_tokens = routed_experts.sum(dim=-1) > 0
        routed_expert_probs = expert_probs * assigned_tokens.unsqueeze(-1)

        return routed_expert_probs, aux_loss


class SwitchLayer(nn.Module):
    """Switch layer with Mixture-of-Experts computation.

    Efficiently routes tokens through expert networks using vectorized operations.
    Supports both auxiliary loss and auxiliary loss-free load balancing.
    """

    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        use_aux_loss: bool = True,
        use_biased_gating: bool = False,
        alpha: float = 0.01,
    ) -> None:
        super().__init__()
        self.inp_dim = inp_dim
        self.num_experts = num_experts
        self.use_biased_gating = use_biased_gating

        self.expert_allocation = ExpertAllocation(
            inp_dim, num_experts, capacity_factor, use_aux_loss, alpha
        )

        # Initialize experts
        self.experts = nn.ModuleList([
            nn.Linear(inp_dim, inp_dim) for _ in range(num_experts)
        ])

        # Gating biases for auxiliary loss-free balancing
        if use_biased_gating:
            self.register_buffer('gating_biases', torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input through switch layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, inp_dim)

        Returns:
            Tuple of (output, aux_loss)
        """
        batch_size, seq_len, _ = x.shape
        flat_x = x.view(-1, self.inp_dim)  # (batch*seq, dim)

        # Get routing and capacity assignment
        routed_probs, aux_loss = self.expert_allocation(x)

        # Apply gating biases if using auxiliary loss-free balancing
        if self.use_biased_gating:
            routed_probs += self.gating_biases

        # Get final expert assignments and probabilities
        expert_probs, expert_indices = routed_probs.max(dim=-1)
        active_tokens = (routed_probs.sum(dim=-1) > 0).view(-1)

        # Update gating biases based on load
        if self.use_biased_gating and self.training:
            with torch.no_grad():
                # Compute token counts per expert
                active_expert_indices = expert_indices.view(-1)[active_tokens]
                token_counts = torch.bincount(
                    active_expert_indices, minlength=self.num_experts
                ).float()

                # Ideal load per expert
                ideal_load = (batch_size * seq_len) / self.num_experts
                load_error = ideal_load - token_counts

                # Update biases
                self.gating_biases.copy_(update_gating_biases(
                    self.gating_biases, load_error
                ))

        # Process tokens through assigned experts
        active_expert_indices = expert_indices.view(-1)[active_tokens]
        active_probs = expert_probs.view(-1)[active_tokens]
        active_inputs = flat_x[active_tokens]

        # Vectorized expert computation
        expert_outputs = torch.zeros_like(active_inputs)
        for expert_idx in range(self.num_experts):
            mask = active_expert_indices == expert_idx
            if mask.any():
                expert_output = self.experts[expert_idx](active_inputs[mask])
                expert_outputs[mask] = expert_output

        # Weight outputs by routing probability
        expert_outputs *= active_probs.unsqueeze(-1)

        # Assemble final output
        output = torch.zeros_like(flat_x)
        output[active_tokens] = expert_outputs
        output = output.view(batch_size, seq_len, self.inp_dim)

        return output, aux_loss


class SwitchTransformerBlock(nn.Module):
    """Single Switch Transformer block with attention and MoE layers."""

    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
        num_heads: int,
        capacity_factor: float = 1.0,
        use_aux_loss: bool = True,
        use_biased_gating: bool = False,
        alpha: float = 0.01,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(inp_dim)
        self.norm2 = nn.LayerNorm(inp_dim)
        self.dropout = nn.Dropout(dropout)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=inp_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Switch layer (MoE)
        self.switch_layer = SwitchLayer(
            inp_dim, num_experts, capacity_factor,
            use_aux_loss, use_biased_gating, alpha
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input through transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, inp_dim)

        Returns:
            Tuple of (output, aux_loss)
        """
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.dropout(attn_out) + residual

        # Pre-norm switch layer
        residual = x
        x = self.norm2(x)
        moe_out, aux_loss = self.switch_layer(x)
        x = self.dropout(moe_out) + residual

        return x, aux_loss


class SwitchTransformer(nn.Module):
    """Complete Switch Transformer model.

    A transformer architecture that uses sparse expert layers instead of
    dense feed-forward networks for improved parameter efficiency.
    """

    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
        num_heads: int,
        vocab_size: int,
        depth: int = 12,
        capacity_factor: float = 1.0,
        use_aux_loss: Optional[bool] = True,
        use_biased_gating: Optional[bool] = False,
        alpha: float = 0.01,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        # Validate load balancing options
        if use_aux_loss and use_biased_gating:
            raise ValueError(
                "Cannot use both auxiliary loss and biased gating simultaneously. "
                "Choose one load balancing method."
            )
        elif not (use_aux_loss or use_biased_gating):
            # Default to auxiliary loss if neither specified
            use_aux_loss = True

        # Model components
        self.embedding = nn.Embedding(vocab_size, inp_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, inp_dim)

        self.layers = nn.ModuleList([
            SwitchTransformerBlock(
                inp_dim=inp_dim,
                num_experts=num_experts,
                num_heads=num_heads,
                capacity_factor=capacity_factor,
                use_aux_loss=use_aux_loss,
                use_biased_gating=use_biased_gating,
                alpha=alpha,
                dropout=dropout,
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(inp_dim)
        self.output_head = nn.Linear(inp_dim, vocab_size)

        # Store config
        self.vocab_size = vocab_size
        self.inp_dim = inp_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Switch Transformer.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask (currently unused)

        Returns:
            Tuple of (logits, total_aux_loss)
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids)  # (batch, seq, dim)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x += self.pos_embedding(positions)

        # Process through layers
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss

        # Final layer norm and output projection
        x = self.norm(x)
        logits = self.output_head(x)

        return logits, total_aux_loss
