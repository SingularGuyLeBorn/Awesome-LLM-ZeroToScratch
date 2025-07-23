# FILE: src/models/attention/standard_attention.py
"""
Bedrock Protocol: Standard Scaled Dot-Product Attention implementation.

This module provides a robust and clear implementation of the attention mechanism,
serving as a foundational component for our Transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StandardAttention(nn.Module):
    """
    Implements standard multi-head self-attention, supporting KV caching and GQA/MQA.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, num_key_value_heads: Optional[int] = None):
        super().__init__()
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads  # Default to MHA

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        if hidden_size % num_key_value_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_key_value_heads ({num_key_value_heads})"
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        Splits the input tensor into multiple heads.
        Input shape: (batch_size, seq_len, hidden_size) or (batch_size, seq_len, num_kv_heads * head_dim)
        Output shape: (batch_size, num_heads, seq_len, head_dim)
        """
        new_shape = x.size()[:-1] + (num_heads, self.head_dim)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merges the multi-head attention output back into a single tensor.
        Input shape: (batch_size, num_heads, seq_len, head_dim)
        Output shape: (batch_size, seq_len, hidden_size)
        """
        # Permute back to (batch_size, seq_len, num_heads, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        # Reshape to (batch_size, seq_len, hidden_size)
        new_shape = x.size()[:-2] + (self.hidden_size,)
        return x.view(new_shape)

    def _repeat_kv(self, hidden_states: torch.Tensor, num_kv_groups: int) -> torch.Tensor:
        """
        Repeats K/V heads for Grouped-Query Attention (GQA).
        Input shape: (batch_size, num_key_value_heads, seq_len, head_dim)
        Output shape: (batch_size, num_attention_heads, seq_len, head_dim)
        """
        if num_kv_groups == 1:
            return hidden_states
        batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_kv_heads, num_kv_groups, seq_len,
                                                               head_dim)
        return hidden_states.reshape(batch_size, num_kv_heads * num_kv_groups, seq_len, head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,  # (batch_size, seq_len, hidden_size)
            attention_mask: Optional[torch.Tensor] = None,
            # (batch_size, current_total_seq_len) from transformers.generate()
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (key_past, value_past)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # (attn_output, (key_present, value_present))
        """
        Forward pass for standard multi-head self-attention with KV caching.
        """
        batch_size, q_seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states, self.num_attention_heads)  # (B, H_attn, S_q, D)
        key_states = self._split_heads(key_states, self.num_key_value_heads)  # (B, H_kv, S_kv, D)
        value_states = self._split_heads(value_states, self.num_key_value_heads)  # (B, H_kv, S_kv, D)

        if past_key_value is not None:
            # past_key_value: (key_past, value_past) where key_past.shape = (B, H_kv, S_past, D)
            key_states = torch.cat((past_key_value[0], key_states), dim=2)
            value_states = torch.cat((past_key_value[1], value_states), dim=2)

        # Apply GQA/MQA repetition for keys and values
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)  # (B, H_attn, S_total, D)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)  # (B, H_attn, S_total, D)

        # present_key_value for the next step of generation
        present_key_value = (key_states, value_states)

        _attn_mask_to_pass = None
        if attention_mask is not None:
            # Check if there is actual padding (any 0s in the mask)
            if torch.min(attention_mask) == 0:
                # If there's padding, construct the additive mask.
                _attn_mask_to_pass = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * torch.finfo(query_states.dtype).min
                _attn_mask_to_pass = _attn_mask_to_pass.to(query_states.dtype)


        attn_output_sdpa = F.scaled_dot_product_attention(
            # +++ START OF FIX +++
            query_states,  # <-- 直接使用 query_states
            key_states,    # <-- 直接使用 key_states
            value_states,  # <-- 直接使用 value_states
            # +++ END OF FIX +++
            attn_mask=_attn_mask_to_pass,
            is_causal=True,
        )

        attn_output = self._merge_heads(attn_output_sdpa)
        output = self.o_proj(attn_output)

        return output, present_key_value

# END OF FILE: src/models/attention/standard_attention.py