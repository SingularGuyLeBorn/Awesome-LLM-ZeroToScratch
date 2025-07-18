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
        Forward pass for the MoE layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
            Also returns a dictionary of auxiliary losses, including router load balancing loss.
        """
        batch_size, seq_len, hidden_size = x.size()

        # Step 1: Compute expert probabilities using the router
        # Flatten the input to (batch_size * seq_len, hidden_size)
        flat_x = x.view(-1, hidden_size)

        # Router logits: (batch_size * seq_len, num_experts)
        router_logits = self.gate(flat_x)

        # Normalize probabilities across experts for each token
        router_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(x.dtype)

        # Get top-k experts for each token
        # top_k_weights: (batch_size * seq_len, num_experts_per_tok)
        # top_k_indices: (batch_size * seq_len, num_experts_per_tok)
        top_k_weights, top_k_indices = torch.topk(router_weights, self.num_experts_per_tok, dim=-1)

        # Normalize top-k weights to sum to 1
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

        # Step 2: Route tokens to experts and compute expert outputs
        # Create an empty tensor for the output, which will aggregate expert outputs.
        output = torch.zeros_like(flat_x)

        # Auxiliary loss for load balancing (Mandate of Proactive Defense for training stability)
        # We calculate this here, but it's typically added to the main loss in the trainer.
        # This is a simplified version; real implementations might use more complex functions.
        load_balancing_loss = self._calculate_load_balancing_loss(router_weights, top_k_indices)

        # Prepare to gather expert inputs
        # expert_inputs will be a list where expert_inputs[i] contains inputs for expert i
        expert_inputs = [[] for _ in range(self.num_experts)]
        # expert_indices will store the global indices of tokens assigned to each expert
        expert_indices = [[] for _ in range(self.num_experts)]

        # Iterate over each token and assign it to its top-k experts
        for i, token_x in enumerate(flat_x):
            for k_idx in range(self.num_experts_per_tok):
                expert_idx = top_k_indices[i, k_idx].item()
                expert_inputs[expert_idx].append(token_x)
                expert_indices[expert_idx].append(i)

        # Process each expert in parallel
        for expert_idx in range(self.num_experts):
            if expert_inputs[expert_idx]:
                expert_input_batch = torch.stack(expert_inputs[expert_idx])
                expert_output = self.experts[expert_idx](expert_input_batch)

                # Scatter expert_output back to the original flattened output tensor
                # Need to multiply by the corresponding top-k weight.
                for j, global_idx in enumerate(expert_indices[expert_idx]):
                    # Find the weight for this specific expert and token
                    # This requires finding where expert_idx is in top_k_indices[global_idx]
                    local_k_idx = (top_k_indices[global_idx] == expert_idx).nonzero(as_tuple=True)[0]
                    if local_k_idx.numel() > 0:  # Ensure the expert was indeed one of the top-k
                        weight = top_k_weights[global_idx, local_k_idx].squeeze()
                        output[global_idx] += expert_output[j] * weight

        return output.view(batch_size, seq_len, hidden_size), {
            "router_aux_loss": load_balancing_loss * self.router_aux_loss_coef}

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