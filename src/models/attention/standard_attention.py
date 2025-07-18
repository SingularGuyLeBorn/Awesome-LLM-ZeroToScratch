# FILE: src/models/attention/standard_attention.py
"""
Bedrock Protocol: Standard Scaled Dot-Product Attention implementation.

This module provides a robust and clear implementation of the attention mechanism,
serving as a foundational component for our Transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StandardAttention(nn.Module):
    """
    Implements standard multi-head self-attention.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        # Mandate of Intentionality: Ensure hidden_size is divisible by num_attention_heads.
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.scaling = self.head_dim ** -0.5

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits the input tensor into multiple heads.
        Input shape: (batch_size, seq_len, hidden_size)
        Output shape: (batch_size, num_attention_heads, seq_len, head_dim)
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)  # Transpose to (batch_size, num_heads, seq_len, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merges attention heads back to the original hidden size.
        Input shape: (batch_size, num_attention_heads, seq_len, head_dim)
        Output shape: (batch_size, seq_len, hidden_size)
        """
        x = x.permute(0, 2, 1, 3).contiguous()  # Transpose back and make contiguous
        new_shape = x.size()[:-2] + (self.hidden_size,)
        return x.view(new_shape)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None  # Padding mask: (batch_size, 1, 1, seq_len)
    ) -> torch.Tensor:
        """
        Forward pass for standard multi-head self-attention.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional padding mask tensor of shape (batch_size, 1, 1, seq_len).
                            This mask will be combined with the causal mask internally.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Project query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Split into heads
        query_states = self._split_heads(query_states)  # (bs, num_heads, seq_len, head_dim)
        key_states = self._split_heads(key_states)  # (bs, num_heads, seq_len, head_dim)
        value_states = self._split_heads(value_states)  # (bs, num_heads, seq_len, head_dim)

        # Apply RoPE (Rotary Positional Embeddings) - conceptual placeholder
        # For a full implementation, RoPE would be applied here to query_states and key_states.
        # This tutorial focuses on the core architecture rather than advanced position embeddings.

        # Compute attention scores
        # (bs, num_heads, seq_len, head_dim) @ (bs, num_heads, head_dim, seq_len)
        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling

        # Apply attention mask: Combine causal mask and padding mask
        # We use F.scaled_dot_product_attention which handles both efficiently.
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,  # Pass padding mask here
            is_causal=True  # Explicitly apply causal mask
        )

        # Merge heads and project back to hidden_size
        attn_output = self._merge_heads(attn_output)  # (bs, seq_len, hidden_size)
        output = self.o_proj(attn_output)

        return output

# END OF FILE: src/models/attention/standard_attention.py