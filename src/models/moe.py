# FILE: src/models/moe.py
"""
Bedrock Protocol: Mixture-of-Experts (MoE) Layer implementation.

This module provides a basic Sparse MoE layer, including a router and
multiple expert FFNs. It's a key component for building scalable and
efficient large language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from src.models.ffn import FFN  # MoE layers typically contain FFNs as experts


class MoE(nn.Module):
    """
    Implements a basic Mixture-of-Experts (MoE) layer.

    This layer routes tokens to a subset of experts (Feed-Forward Networks).
    It includes a router for sparse activation and an optional auxiliary loss
    for load balancing.
    """

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            num_experts: int,
            num_experts_per_tok: int,
            router_aux_loss_coef: float = 0.001
    ):
        super().__init__()
        # Mandate of Intentionality: All MoE-specific parameters are clearly defined.
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef

        # Router: Maps input tokens to expert probabilities.
        # This is a linear layer that outputs num_experts scores for each token.
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts: A list of independent FFN modules.
        self.experts = nn.ModuleList([
            FFN(hidden_size, intermediate_size, hidden_act)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized and efficient forward pass for the MoE layer.
        """
        batch_size, seq_len, hidden_size = x.size()
        flat_x = x.view(-1, hidden_size)
        num_tokens = flat_x.shape[0]

        router_logits = self.gate(flat_x)
        router_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(x.dtype)
        top_k_weights, top_k_indices = torch.topk(router_weights, self.num_experts_per_tok, dim=-1)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

        load_balancing_loss = self._calculate_load_balancing_loss(router_weights, top_k_indices)

        final_output = torch.zeros_like(flat_x)
        
        # Create a flat list of tokens that are dispatched to experts
        flat_top_k_indices = top_k_indices.flatten()
        
        # Create a mask for combining expert outputs
        expert_mask = torch.nn.functional.one_hot(top_k_indices, num_classes=self.num_experts).permute(2, 0, 1)

        # Loop over experts (this loop is small and acceptable)
        for expert_idx, expert_layer in enumerate(self.experts):
            # Find which tokens are routed to this expert
            # token_indices: a boolean tensor of shape (num_tokens, num_experts_per_tok)
            token_indices = (top_k_indices == expert_idx)
            
            # Get the indices of tokens that go to this expert
            # selected_tokens: a 1D tensor of indices
            selected_tokens = token_indices.any(dim=1).nonzero(as_tuple=True)[0]

            if selected_tokens.numel() > 0:
                # Get the corresponding input tokens
                expert_input = flat_x[selected_tokens]
                
                # Pass through the expert
                expert_output = expert_layer(expert_input)

                # Get the weights for these tokens
                weights = top_k_weights[token_indices]
                
                # Weighted sum of expert outputs
                final_output.index_add_(0, selected_tokens, (expert_output.T * weights).T)

        return final_output.view(batch_size, seq_len, hidden_size), {
            "router_aux_loss": load_balancing_loss * self.router_aux_loss_coef
        }

    def _calculate_load_balancing_loss(self, router_weights: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculates a simplified load balancing loss for the router.
        Encourages experts to receive roughly equal traffic.
        """
        # router_weights: (num_tokens, num_experts) after softmax
        # top_k_indices: (num_tokens, num_experts_per_tok)

        # Probability of a token being routed to an expert
        expert_load = torch.zeros(self.num_experts, device=router_weights.device, dtype=router_weights.dtype)
        for i in range(router_weights.size(0)):  # Iterate through tokens
            for k_idx in range(self.num_experts_per_tok):
                expert_idx = top_k_indices[i, k_idx].item()
                expert_load[expert_idx] += router_weights[i, expert_idx]

        # P_expert: Average probability of routing to an expert
        P_expert = expert_load / router_weights.size(0)

        # P_token_expert: Proportion of tokens routed to each expert
        # Count how many tokens were actually routed to each expert
        tokens_per_expert = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts)
        P_token_expert = tokens_per_expert.float() / top_k_indices.numel()  # num_tokens * num_experts_per_tok

        # Load balancing loss: variance of expert loads
        # This is a simplified version. More complex losses (e.g., in Switch Transformers) exist.
        # It encourages P_expert and P_token_expert to be proportional.
        # This term tries to minimize the product of P_expert and P_token_expert
        # when they are not well-balanced.
        loss = (P_expert * P_token_expert).sum() * self.num_experts

        return loss

# END OF FILE: src/models/moe.py