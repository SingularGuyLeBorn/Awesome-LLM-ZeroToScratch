# Track B: 微调开源 LLM / VLM (Fine-tuning Open-Source LLMs/VLMs)

## 目标 (Goal)

这是教程中最具性价比和实用价值的路线。我们将学习如何利用现有的、强大的开源模型（如 Llama 3, Qwen），通过**参数高效微调（PEFT）** 技术，在**单张消费级/专业级GPU（如RTX 3090/4090）** 上，让它学会新的知识或遵循特定的对话风格。

---

### 1. 微调的核心思想 (The Core Idea of Fine-tuning)

预训练（Pre-training）让模型学习通用的世界知识，但它并不知道如何与人“好好说话”。后训练（Post-training），即微调，就是教会模型如何应用知识来遵循指令、进行对话。

本教程覆盖两个核心微调阶段：

1.  **有监督微调 (SFT - Supervised Fine-tuning):**
    *   **目标:** 让模型学会“模仿”。
    *   **数据:** 大量的“指令-回答”对。
    *   **过程:** 模型看到指令，尝试生成回答，我们用标准答案去纠正它（计算损失、反向传播）。经过成千上万次模仿，模型就掌握了遵循指令的模式。

2.  **对齐微调 (Alignment Tuning):**
    *   **目标:** 让模型学会人类的“偏好”。
    *   **问题:** 很多时候，一个问题没有唯一的标准答案，但有好坏之分。例如，一个回答可能有用但啰嗦，另一个可能简洁但有毒。
    *   **传统方法 (RLHF-PPO):** 过程极其复杂。需要训练一个奖励模型（RM）来给答案打分，然后用强化学习（PPO）去优化语言模型，让它生成能得高分的答案。工程上不稳定且耗费资源。
    *   **现代方法 (DPO):** **我们采用的方法。** DPO（直接偏好优化）巧妙地将“偏好”问题转化成一个简单的分类问题，**不再需要训练独立的奖励模型**。它直接使用“（问题，好的回答，坏的回答）”这样的偏好数据，一步到位地调整模型，效果媲美PPO，但过程更简单、稳定、高效。

---

### 2. 参数高效微调 (PEFT) 与 LoRA

**问题:** 微调一个7B（70亿参数）的模型，即便是SFT，也需要更新所有70亿个参数，这需要巨大的显存（数百GB）。

**解决方案:** 参数高效微调（PEFT）。我们**冻结**原始模型的绝大部分参数，只在模型的关键部分（如Attention层）旁边挂上一些小小的、可训练的“适配器”模块。

**LoRA (Low-Rank Adaptation)** 是最成功的一种PEFT方法。它假设模型微调时的参数改动是“低秩”的，可以用两个小得多的矩阵来模拟。

*   **效果:** 我们需要训练的参数从70亿骤降到几百万，降幅超过99.9%。
*   **结果:** **使得在单张24GB显存的GPU上微调7B甚至更大的模型成为可能。**
*   **如何选择 `lora_target_modules`?** 通常是Attention机制中的查询（`q_proj`）、键（`k_proj`）、值（`v_proj`）投影层，以及输出投影层（`o_proj`）。对于不同的模型架构（如 Mistral, Falcon），这些模块的名称可能不同。您可以通过 `model.named_modules()` 方法来查看模型中所有模块的名称，并选择进行LoRA微调的模块。

---

### 3. [执行] 有监督微调 (SFT)

这是微调的第一步。我们将使用 `TinyLlama-1.1B-Chat` 模型和 `mlabonne/guanaco-llama2-1k` 数据集进行演示。

1.  **检查配置**: 打开 `configs/training/finetune_sft.yaml`。
    *   `model_name_or_path`: 确认基础模型。
    *   `dataset_name`: 确认数据集。
    *   `lora_*` 参数: 这些是LoRA的核心配置。
    *   `output_dir`: 训练好的LoRA适配器将保存在这里。

2.  **启动训练**: 在项目根目录下，运行一键启动脚本。
    ```bash
    bash scripts/run_sft.sh
    ```
    这个脚本会调用 `accelerate launch` 来启动 `src/trainers/sft_trainer.py`。训练过程中的指标会默认上传到 `wandb`，方便您监控。

3.  **产出物**: 训练结束后，在 `checkpoints/sft-tinyllama-guanaco/final_model` 目录下，你会看到 `adapter_model.safetensors`, `adapter_config.json` 和 `tokenizer.json` (以及其他tokenizer文件)。这就是我们训练得到的、轻量级的LoRA适配器和配套的分词器。

---

### 4. [执行] 直接偏好优化 (DPO)

这是对齐的第二步，它建立在SFT模型之上。

1.  **检查配置**: 打开 `configs/training/finetune_dpo.yaml`。
    *   **`model_name_or_path`**: **至关重要！** 请确保此路径指向您上一步SFT训练产出的模型目录，即 `checkpoints/sft-tinyllama-guanaco/final_model`。
    *   `dataset_name`: DPO需要偏好数据集，我们使用一个标准数据集 `trl-internal-testing/hh-rlhf-trl-style`。
    *   `beta`: DPO损失函数的核心参数。

2.  **启动训练**:
    ```bash
    bash scripts/run_dpo.sh
    ```
    训练启动后，模型将在SFT的基础上，根据偏好数据进行进一步的“品味”提升。

3.  **产出物**: 训练结束后，在 `checkpoints/dpo-tinyllama-guanaco/final_model` 目录下，你会得到一个新的、经过DPO对齐的LoRA适配器和配套的分词器。这个模型在遵循指令的同时，其回答会更符合人类的偏好。

#### 部署微调后的模型 (Merging LoRA for Deployment)

为了部署通过 LoRA 微调后的模型，您通常需要将 LoRA 适配器的权重**合并**回原始的基础模型中。这样做的好处是：

*   **简化推理**: 不需要额外的 PEFT 库依赖，模型更接近一个“普通”的 Hugging Face 模型。
*   **潜在性能提升**: 避免了 LoRA 适配器带来的额外计算开销（尽管通常很小）。
*   **方便量化**: 可以对合并后的模型进行进一步的量化（如 AWQ/GPTQ）以节省显存。

我们的推理脚本 (`src/inference/inference.py`) 已经内置了自动合并 LoRA 权重的逻辑（通过 `model = model.merge_and_unload()`），所以您无需手动执行此步骤即可进行推理。但如果您希望保存一个完全合并后的模型用于其他部署工具，可以运行以下代码片段：

```python
# 概念代码: 合并 LoRA 适配器并保存为完整的模型
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 假设你的SFT或DPO训练输出目录
adapter_path = "./checkpoints/sft-tinyllama-guanaco/final_model" 
# 或者 "./checkpoints/dpo-tinyllama-guanaco/final_model"

# 假设的基础模型路径（与训练时一致）
# 从adapter_config.json中获取更鲁棒
try:
    from peft import PeftConfig
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_id = peft_config.base_model_name_or_path
except Exception:
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Fallback if config not found

# 1. 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16, # 或 torch.bfloat16
    device_map="auto"
)

# 2. 加载LoRA适配器
model = PeftModel.from_pretrained(model, adapter_path)

# 3. 合并LoRA权重到基础模型
merged_model = model.merge_and_unload()

# 4. 加载并保存分词器
tokenizer = AutoTokenizer.from_pretrained(adapter_path) # 从适配器目录加载分词器

# 5. 保存合并后的模型和分词器
merged_model_output_dir = "./checkpoints/sft-tinyllama-guanaco/merged_model"
# 或者 "./checkpoints/dpo-tinyllama-guanaco/merged_model"
merged_model.save_pretrained(merged_model_output_dir)
tokenizer.save_pretrained(merged_model_output_dir)

print(f"Merged model and tokenizer saved to {merged_model_output_dir}")
```

---

### 5. 资源估算 (微调7B模型)

*   **GPU**:
    *   **LoRA SFT/DPO**: 单张 **RTX 3090/4090 (24GB)** 即可轻松完成。
    *   **Full SFT**: 需要 1-2 x A100/A800 (80G)。(本教程不采用此路线)
*   **时间成本**:
    *   **LoRA SFT/DPO**: 在标准数据集上微调，通常需要 **2-10 小时**。
*   **金钱成本**:
    *   **LoRA**: 使用消费级GPU成本极低。若租用AutoDL的3090（约¥2/小时），完成SFT+DPO全程可能只需花费 **几十元人民币**。

---
**您已完成Track B。您不仅学会了如何微调模型，更掌握了从SFT到DPO这一业界领先的对齐技术。接下来，您可以移步至 [docs/05_deployment_and_evaluation.md](./05_deployment_and_evaluation.md)，学习如何评估您训练好的模型并将其部署为服务。**

# END OF FILE: docs/04_track_b_finetuning.md