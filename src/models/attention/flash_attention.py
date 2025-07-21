# FILE: src/models/attention/flash_attention.py
"""
Bedrock Protocol: FlashAttention wrapper.

This module provides an abstraction layer for FlashAttention. It conditionally
uses the optimized FlashAttention implementation if available and compatible,
otherwise falls back to a standard PyTorch SDPA (Scaled Dot-Product Attention).
This embodies the "Mandate of Proactive Defense" by providing a robust fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Attempt to import FlashAttention. This will only succeed if `flash-attn` is installed
# and the CUDA environment is set up correctly for its compilation.
try:
    from flash_attn import flash_attn_func

    _has_flash_attn = True
except ImportError:
    _has_flash_attn = False
    print("Warning: FlashAttention not found or cannot be imported. Falling back to PyTorch SDPA.")


class FlashAttention(nn.Module):
    """
    Wrapper for FlashAttention, with a fallback to PyTorch's native SDPA.
    Supports KV caching and GQA/MQA.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, num_key_value_heads: Optional[int] = None):
        super().__init__()
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads  # Default to MHA

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        if hidden_size % num_key_value_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_key_value_heads ({num_key_value_heads})"
            )

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Check for FlashAttention availability and CUDA capability (Ampere+ for optimal performance)
        self.has_flash_attn = _has_flash_attn and torch.cuda.is_available() and \
                              torch.cuda.get_device_capability()[0] >= 8  # Check major compute capability >= 8 (Ampere)

        if self.has_flash_attn:
            print(f"Info: FlashAttention will be used on this device ({torch.cuda.get_device_name()}).")
        else:
            print(
                f"Info: Falling back to PyTorch SDPA for attention (FlashAttention not available/compatible or GPU < Ampere).")

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        Splits the input tensor into multiple heads.
        Input shape: (batch_size, seq_len, hidden_size) or (batch_size, seq_len, num_kv_heads * head_dim)
        Output shape: (batch_size, seq_len, num_heads, head_dim) # This is how flash_attn_func expects it.
        """
        new_shape = x.size()[:-1] + (num_heads, self.head_dim)
        x = x.view(new_shape)
        return x  # (batch_size, seq_len, num_heads, head_dim)

    def _repeat_kv(self, hidden_states: torch.Tensor, num_kv_groups: int) -> torch.Tensor:
        """
        Repeats K/V heads for Grouped-Query Attention (GQA).
        Input shape: (batch_size, seq_len, num_key_value_heads, head_dim)
        Output shape: (batch_size, seq_len, num_attention_heads, head_dim)
        """
        if num_kv_groups == 1:
            return hidden_states
        batch_size, seq_len, num_kv_heads, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, seq_len, num_kv_heads, num_kv_groups,
                                                               head_dim)
        return hidden_states.reshape(batch_size, seq_len, num_kv_heads * num_kv_groups, head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,  # (batch_size, q_seq_len, hidden_size)
            attention_mask: Optional[torch.Tensor] = None,
            # (batch_size, current_total_seq_len) from transformers.generate()
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (key_past, value_past) for each layer
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # (attn_output, (key_present, value_present))
        """
        Forward pass for FlashAttention (or fallback to PyTorch SDPA) with KV caching.

        Args:
            hidden_states: Input tensor of shape (batch_size, q_seq_len, hidden_size).
                           q_seq_len is typically 1 during incremental decoding.
            attention_mask: Optional padding mask (batch_size, current_total_seq_len).
                            0 for padded tokens, 1 for valid tokens.
            past_key_value: Optional tuple of (key_states, value_states) from previous steps.

        Returns:
            Output tensor of shape (batch_size, q_seq_len, hidden_size), and present_key_value.
        """
        batch_size, q_seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Split heads: (B, S, D_proj) -> (B, S, H, D_head)
        query_states = self._split_heads(query_states, self.num_attention_heads)
        key_states = self._split_heads(key_states, self.num_key_value_heads)
        value_states = self._split_heads(value_states, self.num_key_value_heads)

        # Handle KV caching: concatenate past and current K/V states
        if past_key_value is not None:
            # past_key_value: (key_past, value_past) where key_past.shape = (B, S_past, H_kv, D_head)
            key_states = torch.cat((past_key_value[0], key_states), dim=1)  # Concat along sequence dimension
            value_states = torch.cat((past_key_value[1], value_states), dim=1)

        # Apply GQA/MQA repetition for keys and values
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)  # (B, S_total, H_attn, D_head)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)  # (B, S_total, H_attn, D_head)

        # present_key_value for the next step of generation
        present_key_value = (key_states, value_states)

        if self.has_flash_attn:
            # FlashAttention requires QKV to be in (B, S, H, D) format
            # `causal=True` directly enforces causal masking within FlashAttention.
            # key_padding_mask is for (B, S_total) where True means padded.
            fa_key_padding_mask = None
            if attention_mask is not None:
                fa_key_padding_mask = (attention_mask == 0)  # attention_mask: 1 is valid, 0 is padded.
                # fa_key_padding_mask: True is padded.

            attn_output = flash_attn_func(
                query_states,  # (B, S_q, H, D)
                key_states,  # (B, S_total, H, D)
                value_states,  # (B, S_total, H, D)
                dropout_p=0.0,
                causal=True,
                key_padding_mask=fa_key_padding_mask
            )  # (B, S_q, H, D)
        else:
            # Fallback to PyTorch's native Scaled Dot-Product Attention (SDPA)
            # SDPA expects Q, K, V in (B, H, S, D)
            # The _split_heads for standard attention permutes to (B, H, S, D),
            # but for flash attention it keeps (B, S, H, D).
            # So, need to permute for SDPA here.
            query_sdpa = query_states.permute(0, 2, 1, 3)  # (B, H, S_q, D)
            key_sdpa = key_states.permute(0, 2, 1, 3)  # (B, H, S_total, D)
            value_sdpa = value_states.permute(0, 2, 1, 3)  # (B, H, S_total, D)

            additive_attention_mask = None
            if attention_mask is not None:
                # (batch_size, 1, 1, current_total_seq_len) to broadcast.
                # attention_mask: 0 for padded tokens, 1 for valid tokens.
                additive_attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * torch.finfo(
                    query_sdpa.dtype).min
                additive_attention_mask = additive_attention_mask.to(query_sdpa.dtype)

            attn_output_sdpa = F.scaled_dot_product_attention(
                query_sdpa,
                key_sdpa,
                value_sdpa,
                attn_mask=additive_attention_mask,
                is_causal=True,
            )  # (B, H, S_q, D)
            attn_output = attn_output_sdpa.permute(0, 2, 1, 3)  # Permute back to (B, S_q, H, D) for merging heads

        # Merge heads
        # attn_output is (batch_size, q_seq_len, num_heads, head_dim)
        attn_output = attn_output.contiguous().view(batch_size, q_seq_len, self.hidden_size)

        # Output projection
        output = self.o_proj(attn_output)
        return output, present_key_value

# END OF FILE src/models/attention/flash_attention.py