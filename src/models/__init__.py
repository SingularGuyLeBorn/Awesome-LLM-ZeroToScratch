# FILE: src/models/__init__.py
# Bedrock: This file makes the directory a Python package.

# +++ START OF FINAL ELEGANT FIX +++
# **终极解决方案**: 向 Transformers 框架注册我们的自定义模型。
# 这是让 `AutoModelForCausalLM.from_pretrained()` 能够识别和加载
# 本地自定义 `BaseLLM` 类的最规范、最稳健的方法。
from transformers import AutoConfig, AutoModelForCausalLM
from .language_model import BaseLLM, BaseLLMConfig

# 注册配置类
AutoConfig.register("BaseLLM", BaseLLMConfig)
# 注册模型类
AutoModelForCausalLM.register(BaseLLMConfig, BaseLLM)
# +++ END OF FINAL ELEGANT FIX +++


# Mandate of Zero Ambiguity: Expose key components for direct import.
from .attention.standard_attention import StandardAttention
from .attention.flash_attention import FlashAttention
from .ffn import FFN
from .moe import MoE
from .language_model import BaseLLM