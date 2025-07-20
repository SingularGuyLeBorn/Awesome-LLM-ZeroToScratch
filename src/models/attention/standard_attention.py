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
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits the input tensor into multiple heads.
        Input shape: (batch_size, seq_len, hidden_size)
        Output shape: (batch_size, num_attention_heads, seq_len, head_dim)
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merges attention heads back to the original hidden size.
        Input shape: (batch_size, num_attention_heads, seq_len, head_dim)
        Output shape: (batch_size, seq_len, hidden_size)
        """
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.hidden_size,)
        return x.view(new_shape)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for standard multi-head self-attention.
        """
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        extended_attention_mask = None
        if attention_mask is not None:
            # +++ START OF THE FIX +++
            # Ensure the mask has the same data type as the query tensor.
            extended_attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * torch.finfo(query_states.dtype).min
            extended_attention_mask = extended_attention_mask.to(query_states.dtype)
            # +++ END OF THE FIX +++

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=extended_attention_mask,
            is_causal=True
        )

        attn_output = self._merge_heads(attn_output)
        output = self.o_proj(attn_output)

        return output