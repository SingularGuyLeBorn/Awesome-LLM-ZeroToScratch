# FILE: src/models/ffn.py
"""
Bedrock Protocol: Feed-Forward Network (FFN) implementations.

This module provides standard FFN layers, including support for various activation
functions, notably SwiGLU, which is common in modern LLMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """
    Implements a standard Feed-Forward Network layer.
    Supports various activation functions including SwiGLU.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Mandate of Intentionality: Dynamically select activation function.
        if hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif hidden_act == "relu":
            self.act_fn = nn.ReLU()
        elif hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif hidden_act == "swiglu":
            # For SwiGLU, the gate_proj and up_proj are effectively part of the
            # same block. We'll handle it inside forward.
            self.act_fn = nn.SiLU()  # SiLU is part of SwiGLU
            # The structure for SwiGLU is (input -> gate_proj * act_fn(up_proj) -> down_proj)
            # In Llama, gate_proj and up_proj are parallel, then multiplied.
            # We already defined gate_proj and up_proj as separate layers above.
        else:
            raise ValueError(f"Unsupported activation function: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FFN layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        if self.hidden_act == "swiglu":
            # SwiGLU implementation: (input * SiLU(input)) @ W_down
            # Llama uses this structure: gate_proj(x) * act_fn(up_proj(x))
            # Where act_fn is SiLU.
            hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        else:
            # Standard FFN: act_fn(gate_proj(x)) @ W_down
            hidden_states = self.act_fn(self.gate_proj(x))

        return self.down_proj(hidden_states)

# END OF FILE: src/models/ffn.py