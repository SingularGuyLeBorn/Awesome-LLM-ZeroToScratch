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
from typing import Optional

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

    This module expects query, key, and value to be in the format
    (batch_size, seq_len, hidden_size), similar to standard attention modules,
    and handles the necessary transformations for FlashAttention if used.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Check for FlashAttention availability and CUDA capability (Ampere+ for optimal performance)
        # Note: FlashAttention can technically be compiled for older architectures but Ampere+ is the common target.
        self.has_flash_attn = _has_flash_attn and torch.cuda.is_available() and \
                              torch.cuda.get_device_capability() >= 8

        if self.has_flash_attn:
            print(f"Info: FlashAttention will be used on this device ({torch.cuda.get_device_name()}).")
        else:
            print(
                f"Info: Falling back to PyTorch SDPA for attention (FlashAttention not available/compatible or GPU < Ampere).")

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None # 修正: 接收 2D padding mask (batch_size, seq_len)
    ) -> torch.Tensor:
        """
        Forward pass for FlashAttention (or fallback to PyTorch SDPA).

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional padding mask (batch_size, seq_len).
                            FlashAttention internally handles causality. For PyTorch SDPA, `is_causal=True` handles causality,
                            and this `attention_mask` handles padding.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = hidden_states.shape # 修正：获取 batch_size 和 seq_len

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head processing for both FlashAttention and PyTorch SDPA
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, num_heads, head_dim)
        # 修正：view函数参数应为元组
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)

        if self.has_flash_attn:
            # FlashAttention requires QKV to be in (B, S, H, D) format
            # `causal=True` directly enforces causal masking within FlashAttention.
            # 修正：将 2D attention_mask 转换为 key_padding_mask (True for padded)
            key_padding_mask = None
            if attention_mask is not None:
                # FlashAttention 的 key_padding_mask 通常是布尔型，True 表示要 mask 的位置 (即填充)
                key_padding_mask = (attention_mask == 0) # 原始 attention_mask 1 为有效，0 为填充

            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=0.0,  # Dropout is typically handled by FlashAttention internally.
                causal=True,  # For language modeling, we always need a causal mask.
                key_padding_mask=key_padding_mask # 传入 key_padding_mask
            )
        else:
            # Fallback to PyTorch's native Scaled Dot-Product Attention (SDPA)
            # PyTorch 2.0+ has a highly optimized native SDPA implementation.
            # `is_causal=True` ensures causal masking. The `attn_mask` argument is for padding.
            # Ensure query, key, value are (B, H, S, D) for SDPA.
            # 修正：将 2D attention_mask 转换为 4D bias mask
            extended_attention_mask = None
            if attention_mask is not None:
                extended_attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * torch.finfo(
                    query_states.dtype).min

            attn_output = F.scaled_dot_product_attention(
                query_states.permute(0, 2, 1, 3),  # (B, H, S, D)
                key_states.permute(0, 2, 1, 3),  # (B, H, S, D)
                value_states.permute(0, 2, 1, 3),  # (B, H, S, D)
                attn_mask=extended_attention_mask,  # Pass padding mask here
                is_causal=True,  # Explicitly tell SDPA to apply a causal mask.
            ).permute(0, 2, 1, 3)  # Permute back to (B, S, H, D)

        # Merge heads
        # attn_output is (batch_size, seq_len, num_heads, head_dim)
        # 修正：view函数参数应为元组
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.o_proj(attn_output)
        return output

# END OF FILE: src/models/attention/flash_attention.py