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

from transformers import PreTrainedModel, PretrainedConfig

from src.models.attention.standard_attention import StandardAttention
from src.models.attention.flash_attention import FlashAttention
from src.models.ffn import FFN
from src.models.moe import MoE


class BaseLLMConfig(PretrainedConfig):
    model_type = "BaseLLM"

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=4,
            num_attention_heads=16,
            hidden_act="silu",
            max_position_embeddings=2048,
            attention_type="standard",
            model_type_llm="DenseLLM",
            tie_word_embeddings=True,
            # +++ START OF CRITICAL FIX FOR GENERATE() COMPATIBILITY +++
            # 明确声明这是一个解码器模型，且不是编码器-解码器模型
            is_decoder=True,
            is_encoder_decoder=False,
            # +++ END OF CRITICAL FIX FOR GENERATE() COMPATIBILITY +++
            # MoE Params
            num_experts=8,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.001,
            # VLM Params
            is_vlm=False,
            num_image_tokens=256,
            vision_encoder_output_dim=768,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.attention_type = attention_type
        self.model_type_llm = model_type_llm
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef
        self.is_vlm = is_vlm
        self.num_image_tokens = num_image_tokens
        self.vision_encoder_output_dim = vision_encoder_output_dim

        # 将 is_decoder 和 is_encoder_decoder 传递给基类
        super().__init__(tie_word_embeddings=tie_word_embeddings, is_decoder=is_decoder, is_encoder_decoder=is_encoder_decoder, **kwargs)


class VisionEncoderDummy(nn.Module):
    def __init__(self, output_dim: int = 768, num_image_tokens: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.num_image_tokens = num_image_tokens
        self.conv = nn.Conv2d(3, output_dim // 4, kernel_size=16, stride=16)
        self.pool = nn.AdaptiveAvgPool2d((1, num_image_tokens))
        self.linear = nn.Linear(output_dim // 4, output_dim)
        print(f"--- Bedrock: Initialized Conceptual Vision Encoder...")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() != 4 or pixel_values.shape[1] != 3:
            return torch.zeros(pixel_values.shape[0], self.num_image_tokens, self.output_dim,
                               device=pixel_values.device, dtype=pixel_values.dtype)
        features = self.conv(pixel_values)
        features = self.pool(features)
        features = features.squeeze(2).permute(0, 2, 1)
        features = self.linear(features)
        return features


class TransformerBlock(nn.Module):
    def __init__(self, config: BaseLLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_type = config.attention_type
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)

        if self.attention_type == "flash":
            self.self_attn = FlashAttention(self.hidden_size, config.num_attention_heads)
        else:
            self.self_attn = StandardAttention(self.hidden_size, config.num_attention_heads)

        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)

        if config.model_type_llm == "MoELLM":
            self.mlp = MoE(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size,
                           hidden_act=config.hidden_act, num_experts=config.num_experts,
                           num_experts_per_tok=config.num_experts_per_tok,
                           router_aux_loss_coef=config.router_aux_loss_coef)
        else:
            self.mlp = FFN(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size,
                           hidden_act=config.hidden_act)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output_or_tuple = self.mlp(hidden_states)

        if isinstance(mlp_output_or_tuple, tuple):
            mlp_output, aux_loss_dict = mlp_output_or_tuple
        else:
            mlp_output = mlp_output_or_tuple
            aux_loss_dict = {}

        hidden_states = residual + mlp_output
        return hidden_states, aux_loss_dict


class BaseLLM(PreTrainedModel):
    config_class = BaseLLMConfig

    def __init__(self, config: BaseLLMConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.vision_encoder = None
        self.vision_projector = None

        if config.is_vlm:
            self.vision_encoder = VisionEncoderDummy(output_dim=config.vision_encoder_output_dim,
                                                     num_image_tokens=config.num_image_tokens)
            self.vision_projector = nn.Sequential(nn.Linear(config.vision_encoder_output_dim, self.hidden_size),
                                                  nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size))

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.size()
        aux_losses = {}
        text_hidden_states = self.embed_tokens(input_ids)
        combined_hidden_states = text_hidden_states
        current_seq_len = seq_len

        if self.vision_encoder is not None and pixel_values is not None:
            image_features = self.vision_encoder(pixel_values)
            projected_image_features = self.vision_projector(image_features)
            combined_hidden_states = torch.cat((projected_image_features, text_hidden_states), dim=1)
            current_seq_len += self.config.num_image_tokens
            if attention_mask is not None:
                image_attention_mask = torch.ones(batch_size, self.config.num_image_tokens, dtype=attention_mask.dtype,
                                                  device=attention_mask.device)
                attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)
            else:
                attention_mask = torch.ones(batch_size, current_seq_len, dtype=torch.long, device=input_ids.device)

        hidden_states_in_blocks = combined_hidden_states
        for i, layer in enumerate(self.layers):
            layer_output, layer_aux_losses = layer(hidden_states_in_blocks, attention_mask)
            hidden_states_in_blocks = layer_output
            aux_losses.update({f"layer_{i}_{k}": v for k, v in layer_aux_losses.items()})

        final_hidden_states = self.norm(hidden_states_in_blocks)

        if self.vision_encoder is not None and pixel_values is not None:
            text_only_hidden_states = final_hidden_states[:, self.config.num_image_tokens:, :]
            logits = self.lm_head(text_only_hidden_states)
        else:
            logits = self.lm_head(final_hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1),
                                   ignore_index=getattr(self.config, 'pad_token_id', -100))

        if aux_losses:
            total_aux_loss = sum(aux_losses.values())
            if loss is not None:
                loss += total_aux_loss
            else:
                loss = total_aux_loss

        return {"loss": loss, "logits": logits, "aux_losses": aux_losses}