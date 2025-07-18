# FILE: src/models/language_model.py
"""
Bedrock Protocol: Core Language Model (LLM/VLM) construction.

This module provides the `BaseLLM` class, which serves as the central factory
for assembling various architectural components (Attention, FFN, MoE) into
a complete Transformer-based language model. It's designed to be highly
configurable via a dictionary, adhering to the "industrial-grade" principle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from src.models.attention.standard_attention import StandardAttention
from src.models.attention.flash_attention import FlashAttention
from src.models.ffn import FFN
from src.models.moe import MoE  # Only if MoE is used


# Placeholder for Hugging Face integration for VLM vision encoder
# from transformers import AutoModel, AutoConfig

class VisionEncoderDummy(nn.Module):
    """
    A conceptual dummy Vision Encoder for VLM.
    In a real VLM, this would be a pre-trained model like CLIP's ViT.
    It simulates processing an image (C, H, W) into a sequence of features.
    """

    def __init__(self, output_dim: int = 768, num_image_tokens: int = 256):
        super().__init__()
        # Simulate a patch embedding / feature extraction: (C, H, W) -> (num_tokens, feature_dim)
        # For simplicity, we use a Conv2D and then flatten.
        self.conv = nn.Conv2d(3, output_dim // 4, kernel_size=16, stride=16)  # Roughly matches ViT patches
        self.pool = nn.AdaptiveAvgPool2d((1, num_image_tokens))  # Reduce to num_image_tokens
        self.linear = nn.Linear(output_dim // 4, output_dim)  # Project to final feature dim
        print(
            f"--- Bedrock: Initialized Conceptual Vision Encoder. Output feature dim: {output_dim}, Num image tokens: {num_image_tokens} ---")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Input image tensor of shape (batch_size, channels, H, W).
        Returns:
            Image features of shape (batch_size, num_image_tokens, output_dim).
        """
        # Ensure input is 3 channels for conv.
        if pixel_values.shape == 4:  # Handle RGBA
            pixel_values = pixel_values[:, :3, :, :]

        features = self.conv(pixel_values)  # (bs, output_dim//4, H/16, W/16)
        features = self.pool(features)  # (bs, output_dim//4, 1, num_image_tokens)
        features = features.squeeze(2).permute(0, 2, 1)  # (bs, num_image_tokens, output_dim//4)
        features = self.linear(features)  # (bs, num_image_tokens, output_dim)
        return features


class TransformerBlock(nn.Module):
    """
    A single Transformer block, composed of self-attention and a Feed-Forward Network (FFN)
    or a Mixture-of-Experts (MoE) layer.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config['hidden_size']

        # Layer Normalization before attention (Pre-normalization)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)  # Common epsilon for LLMs

        # Attention layer selection
        attention_type = config.get("attention_type", "standard")
        if attention_type == "flash":
            self.self_attn = FlashAttention(
                self.hidden_size, config['num_attention_heads']
            )
        elif attention_type == "standard":
            self.self_attn = StandardAttention(
                self.hidden_size, config['num_attention_heads']
            )
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        # Layer Normalization before FFN/MoE
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)

        # FFN or MoE layer selection
        if config['model_type'] == "MoELLM":
            self.mlp = MoE(
                hidden_size=self.hidden_size,
                intermediate_size=config['intermediate_size'],
                hidden_act=config['hidden_act'],
                num_experts=config['num_experts'],
                num_experts_per_tok=config['num_experts_per_tok'],
                router_aux_loss_coef=config['router_aux_loss_coef']
            )
        else:  # DenseLLM
            self.mlp = FFN(
                hidden_size=self.hidden_size,
                intermediate_size=config['intermediate_size'],
                hidden_act=config['hidden_act']
            )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None  # Padding mask (batch_size, 1, 1, seq_len)
    ) -> torch.Tensor:
        """
        Forward pass through a single Transformer block.
        """
        # Pre-normalization applies LayerNorm before the sub-layers.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-Attention
        # The attention module itself handles combining the padding mask with causal masking.
        attn_output = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + attn_output  # Add residual connection

        # Feed-Forward Network / MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_output_or_tuple = self.mlp(hidden_states)

        # MoE returns a tuple (output, aux_loss_dict), FFN returns just output
        if isinstance(mlp_output_or_tuple, tuple):
            mlp_output, aux_loss_dict = mlp_output_or_tuple
        else:
            mlp_output = mlp_output_or_tuple
            aux_loss_dict = {}  # No aux loss for standard FFN

        hidden_states = residual + mlp_output  # Add residual connection

        return hidden_states, aux_loss_dict


class BaseLLM(nn.Module):
    """
    Base Language Model (LLM) or Vision Language Model (VLM) for from-scratch pre-training.
    This class orchestrates the entire model architecture based on a configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.max_position_embeddings = config['max_position_embeddings']
        self.num_image_tokens = config.get("num_image_tokens", 256)  # Default for VLM

        # 1. Token Embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # 2. Vision Encoder (Optional, for VLM)
        # This is a conceptual placeholder. In a real VLM, you'd load a pre-trained
        # vision model (e.g., CLIP) here and freeze its weights during training.
        self.vision_encoder = None
        self.vision_projector = None
        if config.get("is_vlm", False):  # A flag in config to indicate VLM
            print("--- Bedrock: Initializing Vision Encoder (Conceptual) ---")
            # In a real VLM, you would do:
            # from transformers import CLIPVisionModel, CLIPImageProcessor
            # self.vision_encoder = CLIPVisionModel.from_pretrained(config["vision_encoder_model_name_or_path"])
            # self.vision_encoder.requires_grad_(False) # Freeze vision encoder
            # vision_feature_dim = self.vision_encoder.config.hidden_size # e.g., 768 for CLIP-L/14

            # For this tutorial, we use a dummy encoder to represent the feature extraction
            vision_feature_dim = config.get("vision_encoder_output_dim", 768)  # e.g., 768 for CLIP
            self.vision_encoder = VisionEncoderDummy(output_dim=vision_feature_dim,
                                                     num_image_tokens=self.num_image_tokens)

            # The projector maps vision features to the language model's hidden_size
            self.vision_projector = nn.Sequential(
                nn.Linear(vision_feature_dim, config.get("vision_projector_hidden_size", self.hidden_size * 2)),
                nn.SiLU(),
                nn.Linear(config.get("vision_projector_hidden_size", self.hidden_size * 2),
                          self.hidden_size * self.num_image_tokens)
                # Project to match total tokens in text embed dimension
            )
            print("--- Bedrock: Vision Encoder Initialized (Conceptual) ---")

        # 3. Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config['num_hidden_layers'])
        ])

        # 4. Final Layer Normalization
        self.norm = nn.LayerNorm(self.hidden_size, eps=1e-5)

        # 5. Language Model Head (for token prediction)
        # Often tied to the embedding weights for efficiency and better performance.
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        # Mandate of Empirical Proof: Weight tying is a standard optimization.
        self.lm_head.weight = self.embed_tokens.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Custom weight initialization (e.g., Xavier, Kaiming, or specific to LLMs)."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,  # Padding mask (batch_size, seq_len)
            pixel_values: Optional[torch.Tensor] = None  # For VLM, shape (batch_size, channels, H, W)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the language model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Padding mask of shape (batch_size, seq_len) where 1 indicates
                            actual tokens and 0 indicates padding.
            pixel_values: Optional image pixel values for VLM, shape (batch_size, channels, H, W).

        Returns:
            A dictionary containing:
            - "logits": Raw output logits for token prediction (batch_size, seq_len, vocab_size).
            - "aux_losses": Dictionary of auxiliary losses (e.g., router loss for MoE).
        """
        batch_size, seq_len = input_ids.size()
        aux_losses = {}  # Collect auxiliary losses from MoE layers

        # 1. Token Embeddings
        # (batch_size, seq_len, hidden_size)
        text_hidden_states = self.embed_tokens(input_ids)

        combined_hidden_states = text_hidden_states
        current_seq_len = seq_len  # Track current sequence length as we might prepend image tokens

        # 2. Process Vision Input (if VLM)
        if self.vision_encoder is not None and pixel_values is not None:
            # Mandate of Proactive Defense: Ensure vision data is correctly processed.
            # In a real VLM, `pixel_values` would be processed by the actual vision encoder.
            image_features = self.vision_encoder(pixel_values)  # (bs, num_image_tokens, vision_feature_dim)
            projected_image_features = self.vision_projector(
                image_features.flatten(1, 2))  # (bs, num_image_tokens * hidden_size)
            projected_image_features = projected_image_features.view(batch_size, self.num_image_tokens,
                                                                     self.hidden_size)  # (bs, num_image_tokens, hidden_size)

            # Simple prepend of vision features to token embeddings.
            # Real VLMs like LLaVA prepend special image tokens (e.g., <image> <image_token_features> <image>).
            # For this tutorial, we simply prepend the projected features.
            combined_hidden_states = torch.cat((projected_image_features, text_hidden_states), dim=1)
            current_seq_len += self.num_image_tokens
            print(f"--- Bedrock: VLM - Image features prepended. New seq_len: {current_seq_len} ---")

            # Adjust attention_mask to account for prepended image tokens
            if attention_mask is not None:
                image_attention_mask = torch.ones(
                    batch_size, self.num_image_tokens,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        # 3. Positional Embeddings (Implicitly handled by model design, e.g., RoPE in Attention)
        # For this base model, we assume relative positional encoding or absolute embeddings
        # are handled within the TransformerBlock if needed (e.g., RoPE in Attention).

        # 4. Attention Mask for causal language modeling and padding
        # The attention_mask passed to TransformerBlock is now a padding mask
        # of shape (batch_size, seq_len). TransformerBlock will convert it to
        # (batch_size, 1, 1, seq_len) and combine with causal masking.
        expanded_attention_mask_for_attn = None
        if attention_mask is not None:
            # Expand the padding mask from (batch_size, current_seq_len) to (batch_size, 1, 1, current_seq_len)
            # This is the format expected by F.scaled_dot_product_attention's attn_mask argument.
            expanded_attention_mask_for_attn = attention_mask[:, None, None,
                                               :].float()  # Ensure float for multiplication
            # Convert to attention bias style: -inf for padded positions, 0 for non-padded.
            expanded_attention_mask_for_attn = (1.0 - expanded_attention_mask_for_attn) * torch.finfo(
                combined_hidden_states.dtype).min

        # 5. Transformer Blocks
        hidden_states_in_blocks = combined_hidden_states
        for i, layer in enumerate(self.layers):
            layer_output, layer_aux_losses = layer(hidden_states_in_blocks, expanded_attention_mask_for_attn)
            hidden_states_in_blocks = layer_output
            aux_losses.update({f"layer_{i}_{k}": v for k, v in layer_aux_losses.items()})

        # 6. Final Layer Normalization
        final_hidden_states = self.norm(hidden_states_in_blocks)

        # 7. Language Model Head
        # If VLM, we only want logits for the text tokens, not image tokens.
        # So, we slice out the image tokens from the start of the sequence.
        if self.vision_encoder is not None and pixel_values is not None:
            # (batch_size, original_seq_len, vocab_size)
            logits = self.lm_head(final_hidden_states[:, self.num_image_tokens:, :])
        else:
            # (batch_size, seq_len, vocab_size)
            logits = self.lm_head(final_hidden_states)

        return {"logits": logits, "aux_losses": aux_losses}

# END OF FILE: src/models/language_model.py