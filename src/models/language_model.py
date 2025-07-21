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
from typing import Dict, Any, Optional, Tuple, List

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin

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
            num_key_value_heads=16,  # Added num_key_value_heads for GQA/MQA support
            hidden_act="silu",
            max_position_embeddings=2048,
            attention_type="standard",
            model_type_llm="DenseLLM",
            tie_word_embeddings=True,
            is_decoder=True,
            is_encoder_decoder=False,
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
        self.num_key_value_heads = num_key_value_heads  # Store in config
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

        super().__init__(tie_word_embeddings=tie_word_embeddings, is_decoder=is_decoder,
                         is_encoder_decoder=is_encoder_decoder, **kwargs)


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
            # Return dummy output if input is not valid, matching expected shape
            batch_size = pixel_values.shape[0] if pixel_values.dim() >= 1 else 1
            return torch.zeros(batch_size, self.num_image_tokens, self.output_dim,
                               device=pixel_values.device, dtype=pixel_values.dtype)
        features = self.conv(pixel_values)
        features = self.pool(features)
        features = features.squeeze(2).permute(0, 2, 1)  # Permute to (B, S, D)
        features = self.linear(features)
        return features


class TransformerBlock(nn.Module):
    def __init__(self, config: BaseLLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_type = config.attention_type
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)

        if self.attention_type == "flash":
            self.self_attn = FlashAttention(self.hidden_size, config.num_attention_heads, config.num_key_value_heads)
        else:
            self.self_attn = StandardAttention(self.hidden_size, config.num_attention_heads, config.num_key_value_heads)

        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)

        if config.model_type_llm == "MoELLM":
            self.mlp = MoE(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size,
                           hidden_act=config.hidden_act, num_experts=config.num_experts,
                           num_experts_per_tok=config.num_experts_per_tok,
                           router_aux_loss_coef=config.router_aux_loss_coef)
        else:
            self.mlp = FFN(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size,
                           hidden_act=config.hidden_act)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # Added for KV caching
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[
        torch.Tensor, torch.Tensor]]:  # Added present_key_value to return tuple

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Pass past_key_value and receive present_key_value from attention
        attn_output, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )

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
        return hidden_states, aux_loss_dict, present_key_value  # Return present_key_value


class BaseLLM(PreTrainedModel, GenerationMixin):
    config_class = BaseLLMConfig
    _no_split_modules = ["TransformerBlock"]
    main_input_name = "input_ids"

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

        # Initialize weights and apply tie_word_embeddings
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        """
        Prepares model inputs for the next generation step. This is a crucial method
        for compatibility with the .generate() function.
        """
        # If past_key_values are provided, it means we are in incremental decoding.
        # input_ids will usually only contain the last generated token.
        # We need to ensure that the attention_mask matches the current total sequence length.

        # If pixel_values were previously passed and are now implicitly part of past_key_values,
        # they should not be passed again.
        # For simplicity, assume pixel_values are ONLY passed at the first forward call.

        model_inputs = {"input_ids": input_ids}

        # The `generate` method passes the `attention_mask` inside `kwargs`.
        # This attention_mask is already correctly padded and expanded by the `generate` method
        # to match the combined length of past tokens and current input_ids.
        model_inputs["attention_mask"] = kwargs.get("attention_mask", None)

        # Pass past_key_values to the forward method
        model_inputs["past_key_values"] = kwargs.get("past_key_values", None)

        # Ensure use_cache is passed for generation.
        model_inputs["use_cache"] = kwargs.get("use_cache", self.config.use_cache)

        # For VLM, pixel_values are typically only for the first token.
        # If past_key_values exist, it means we are continuing from a previous step,
        # so pixel_values should not be re-processed.
        if "pixel_values" in kwargs and kwargs["pixel_values"] is not None and kwargs.get("past_key_values") is None:
            model_inputs["pixel_values"] = kwargs["pixel_values"]

        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,  # Make optional for VLM-only first pass
            attention_mask: Optional[torch.Tensor] = None,  # Mask for the full current sequence
            pixel_values: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            # List of (key, value) for each layer
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,  # If True, past_key_values will be returned
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and pixel_values is None:
            raise ValueError("You must specify either input_ids or pixel_values (or both).")

        batch_size = input_ids.shape[0] if input_ids is not None else pixel_values.shape[0]
        # Current sequence length of text tokens only (excluding image tokens if VLM)
        current_text_seq_len = input_ids.shape[1] if input_ids is not None else 0

        aux_losses = {}

        # 1. Embed tokens
        if input_ids is not None:
            text_hidden_states = self.embed_tokens(input_ids)
        else:
            text_hidden_states = None  # No text input

        # 2. Process Vision Features (if VLM and first pass)
        # Vision features are prepended only on the very first forward pass.
        # Subsequent calls during generation will have past_key_values containing them.
        vision_features_prepended = False
        if self.config.is_vlm and self.vision_encoder is not None and pixel_values is not None and past_key_values is None:
            image_features = self.vision_encoder(pixel_values)
            projected_image_features = self.vision_projector(image_features)

            if text_hidden_states is not None:
                combined_hidden_states = torch.cat((projected_image_features, text_hidden_states), dim=1)
            else:
                combined_hidden_states = projected_image_features  # VLM-only input

            vision_features_prepended = True

            # Create or extend attention mask to include image tokens for the FIRST pass
            if attention_mask is not None:
                # current attention_mask is for input_ids. Need to prepend image mask.
                image_attention_mask = torch.ones(batch_size, self.config.num_image_tokens,
                                                  dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)
            else:
                # If no text input and no initial mask, create full mask including image tokens.
                attention_mask = torch.ones(batch_size, self.config.num_image_tokens + current_text_seq_len,
                                            dtype=torch.long, device=self.device)
        elif text_hidden_states is not None:
            combined_hidden_states = text_hidden_states
        else:  # Neither text nor initial image input
            raise ValueError("No input (input_ids or pixel_values) provided for model forward pass.")

        # 3. Transformer Layers with KV Caching
        present_key_values = [] if use_cache else None

        # `position_ids` usually handled by `transformers.modeling_utils._get_position_ids` or similar for `generate`.
        # For our custom model, we assume the `attention_mask` is correctly extended by `generate()`
        # or manually constructed here for the full sequence at each step.

        hidden_states_in_blocks = combined_hidden_states
        for i, layer in enumerate(self.layers):
            layer_past_key_value = past_key_values[i] if past_key_values else None

            layer_output, layer_aux_losses, current_layer_present_kv = layer(
                hidden_states_in_blocks,
                attention_mask=attention_mask,
                past_key_value=layer_past_key_value
            )
            hidden_states_in_blocks = layer_output
            aux_losses.update({f"layer_{i}_{k}": v for k, v in layer_aux_losses.items()})
            if use_cache:
                present_key_values.append(current_layer_present_kv)

        final_hidden_states = self.norm(hidden_states_in_blocks)

        # 4. Language Model Head
        # If VLM, the text logits correspond only to the text portion of the sequence.
        if self.config.is_vlm and vision_features_prepended:
            # If vision features were just prepended, the logits should only be calculated
            # for the *text* part of the sequence for causal LM loss.
            logits = self.lm_head(final_hidden_states[:, self.config.num_image_tokens:, :])
        elif self.config.is_vlm and past_key_values is not None:
            # If VLM and using cache (i.e., not first pass), input_ids is likely 1 token.
            # The `final_hidden_states` will be `(B, 1, D)`.
            # The VLM prefix is implicitly handled by the KV cache.
            logits = self.lm_head(final_hidden_states)
        else:  # Pure LLM or VLM's subsequent generation steps without new image input
            logits = self.lm_head(final_hidden_states)

        # 5. Calculate Loss
        loss = None
        if labels is not None:
            # Shift predictions and labels for causal language modeling
            # If VLM and vision_features_prepended, labels should also align to text part.
            if self.config.is_vlm and vision_features_prepended:
                # Labels correspond to text_hidden_states, so need to shift them too.
                # The labels should already be for the text sequence only.
                # Example: labels for [IMG_TOKENS, TEXT_TOKENS] would be [IGNORE, IGNORE, ..., TEXT_LABELS]
                # We expect labels to already be prepared for the text part.
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()  # labels should start from first text token
            else:
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

        # 6. Return Output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,  # Return present_key_values for caching
            hidden_states=None,  # For simplicity, not returning these for now
            attentions=None,  # For simplicity, not returning these for now
        )

# END OF /src/models/language_model.py