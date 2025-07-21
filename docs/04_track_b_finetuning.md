# FILE: docs/04_track_b_finetuning.md

# Track B: 微调开源 LLM (Fine-tuning Open-Source LLMs)

## 目标 (Goal)

这是教程中最具性价比和实用价值的路线。我们将学习如何利用现有的、强大的开源模型（如 Llama 3, Qwen），通过**参数高效微调（PEFT）** 技术，在**单张消费级/专业级GPU（如RTX 3090/4090）** 上，甚至在**没有GPU的普通电脑（CPU）上**，让它学会新的知识或遵循特定的对话风格。

---

### 1. 微调的核心思想 (The Core Idea of Fine-tuning)

预训练让模型学习了通用的世界知识，但它并不知道如何与人“好好说话”。后训练（Post-training），即微调，就是教会模型如何应用知识来遵循指令、进行对话。本教程覆盖从SFT到DPO的完整对齐流程。

1.  **有监督微调 (SFT - Supervised Fine-tuning):**
    *   **目标:** 让模型学会“模仿”。
    *   **数据:** 大量的“指令-回答”对。
    *   **过程:** 模型看到指令，尝试生成回答，我们用标准答案去纠正它（计算损失、反向传播）。经过成千上万次模仿，模型就掌握了遵循指令的模式。

2.  **直接偏好优化 (DPO - Direct Preference Optimization):**
    *   **目标:** 让模型学会人类的“偏好”，做出更好的判断。
    *   **背景:** 传统方法（RLHF-PPO）过程复杂，需要训练一个独立的奖励模型。
    *   **DPO的革新:** DPO巧妙地将“偏好”问题转化成一个简单的分类问题，**不再需要训练独立的奖励模型**。它直接使用“（问题，好的回答，坏的回答）”这样的偏好数据，一步到位地调整模型，效果媲美PPO，但过程更简单、稳定、高效。

---

### 2. 参数高效微调 (PEFT) 与 LoRA

**问题:** 即便微调，更新一个7B（70亿参数）模型的所有参数也需要巨大的显存。

**解决方案: LoRA (Low-Rank Adaptation)**。我们**冻结**原始模型的绝大部分参数，只在模型的关键部分（如Attention层）旁边挂上一些小小的、可训练的“适配器”矩阵。

*   **效果:** 训练参数量骤降超过99.9%，使得在单张24GB显存的GPU上微调7B模型成为可能，甚至在CPU上进行（虽然慢）也变得可行。
*   **`lora_target_modules`**: 这些是我们要挂载“适配器”的目标模块，通常是Attention中的`q_proj`, `k_proj`, `v_proj`, `o_proj`。

---

### 3. 从零到一：CPU环境下的踩坑与通关实录

在没有GPU的环境下跑通流程，是对代码鲁棒性和问题解决能力的终极考验。以下是我们在此过程中遇到的真实问题及其解决方案，这些经验对所有开发者都极具价值。

#### **第一道坎：`RuntimeError: No GPU found. A GPU is needed for quantization.`**
*   **问题描述**: 首次运行`accelerate launch`，程序因找不到GPU而崩溃。
*   **根本原因**: 默认配置中使用了`bitsandbytes`进行4-bit量化，这是一个**GPU专属**的功能。代码请求了GPU功能，但环境只有CPU。
*   **解决方案**:
    1.  **修改配置文件 (`.yaml`)**:
        *   将优化器`optim`从`paged_adamw_8bit`（GPU专属）改为`adamw_torch`（通用）。
        *   将`bf16`和`fp16`（混合精度，GPU专属）设置为`false`。
    2.  **修改训练脚本 (`.py`)**:
        *   增加逻辑判断：在加载模型时，检查`torch.cuda.is_available()`。如果为`False`，则不创建`BitsAndBytesConfig`，并强制模型在CPU上以`float32`精度加载。

#### **第二道坎：`OSError: 页面文件太小，无法完成操作。`**
*   **问题描述**: 解决了GPU依赖后，在加载模型权重时程序再次崩溃。
*   **根本原因**: 这是Windows系统下的经典内存问题。加载一个2.2GB的模型需要大量内存。当物理内存(RAM)不足时，系统会使用硬盘上的虚拟内存（页面文件）。如果虚拟内存大小也不足，操作就会失败。
*   **解决方案**: **手动扩大Windows虚拟内存**。
    *   进入“高级系统设置” -> “性能设置” -> “高级” -> “虚拟内存”。
    *   取消“自动管理”，选择一个空间充足的硬盘（如C盘），设置一个较大的“自定义大小”（例如，初始大小16GB，最大值32GB）。
    *   **必须重启电脑**使设置生效。

#### **第三道坎：`TypeError: '<=' not supported between instances of 'float' and 'str'`**
*   **问题描述**: 解决了内存问题后，训练终于要开始了，却在创建优化器时报错。
*   **根本原因**: `pyyaml`库在解析配置文件时，可能会将科学记数法（如`2e-4`）当作字符串`'2e-4'`，而不是浮点数`0.0002`。当PyTorch的优化器接收到这个字符串类型的学习率时，无法进行数值比较，从而引发类型错误。
*   **解决方案**: **在代码中进行强制类型转换**。
    *   在所有训练脚本中，当从`config`字典创建`TrainingArguments`或`PPOConfig`时，对所有数值型参数（如`learning_rate`, `weight_decay`, `lora_dropout`等）使用`float()`或`int()`进行显式转换。

#### **第四道坎：`78.91s/it`，慢到怀疑人生**
*   **问题描述**: 所有错误都解决后，训练终于跑起来了，但迭代速度极慢，一个step需要一分多钟。
*   **根本原因**: 这不是错误，而是**CPU训练的正常速度**。CPU缺乏GPU那样的大规模并行计算单元（Tensor Cores），执行矩阵运算效率极低。
*   **解决方案**: **我们的目标是验证流程，而非产出模型**。
    1.  **修改配置文件 (`.yaml`)**:
        *   设置`max_steps: 5`，让训练在5步后就停止。
        *   增加一个自定义参数`dataset_subset_size_cpu: 16`，并修改训练脚本读取它，从而只用16个样本进行训练。
    2.  通过这种方式，我们可以在几分钟内跑完整个流程，验证代码的正确性。

---

### 4. [执行] 微调流程

现在，您可以使用我们经过血泪教训淬炼出的、CPU友好的配置来安全地运行所有流程。

1.  **启动SFT训练 (CPU快速验证版)**:
    ```bash
    accelerate launch src/trainers/sft_trainer.py configs/training/finetune_sft.yaml
    ```
    *   **配置文件**: `configs/training/finetune_sft.yaml`
    *   **预期**: 训练会在几分钟内完成5个step，并在`./checkpoints/sft-tinyllama-guanaco-cpu/final_model`下生成LoRA适配器。

2.  **启动DPO训练 (CPU快速验证版)**:
    ```bash
    accelerate launch src/trainers/dpo_trainer.py configs/training/finetune_dpo.yaml
    ```
    *   **前提**: SFT已成功运行。`finetune_dpo.yaml`中的`model_name_or_path`已指向SFT的CPU输出目录。
    *   **预期**: DPO训练也会在几分钟内完成，并在`./checkpoints/dpo-tinyllama-guanaco-cpu/final_model`下生成新的LoRA适配器。

3.  **启动PPO训练 (CPU快速验证版)**:
    ```bash
    accelerate launch src/trainers/ppo_trainer.py configs/training/finetune_ppo.yaml
    ```
    *   **前提**: SFT已成功运行。
    *   **预期**: PPO概念性训练也会在几分钟内完成。

---
**您已完成Track B。您不仅学会了如何在GPU和CPU上微调模型，更重要的是，您亲历了在受限环境中解决复杂工程问题的全过程。**

**接下来，请移步至 [docs/05_deployment_and_evaluation.md](./05_deployment_and_evaluation.md)，学习如何与您训练好的模型交互并评估它。**

# END OF FILE: docs/04_track_b_finetuning.md