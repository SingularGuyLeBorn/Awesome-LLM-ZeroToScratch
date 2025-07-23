# **Awesome-LLM-ZeroToScratch 命令行操作手册 (Dense Text LLM 专用)**

## **项目概述**

本手册专注于 **Dense Text LLM (稠密文本大语言模型)** 的端到端构建和优化流程。它将引导你通过命令行，完成从数据准备、模型预训练到多阶段微调和最终评估的所有关键步骤。

## **目录**

1.  **重要提示与前置条件**
    *   1.1 当前工作目录
    *   1.2 虚拟环境激活
    *   1.3 CPU 环境下的常见警告
    *   1.4 配置文件 (`configs/`)
    *   1.5 输出目录 (`checkpoints/`, `data/`)

2.  **环境设置与准备**
    *   2.1 基础依赖安装
    *   2.2 `accelerate` 配置

3.  **流程 A: 从零预训练 (Pre-training From Scratch)**
    *   3.1 阶段 1: 数据准备 (文本预训练数据)
    *   3.2 阶段 2: 执行预训练 (`0.5B_dense` 模型)
    *   3.3 阶段 3: 推理/评估预训练模型

4.  **流程 B: 微调与对齐 (Fine-tuning & Alignment)**
    *   4.1 阶段 1: 监督微调 (SFT)
    *   4.2 阶段 2 (路径 1): 直接偏好优化 (DPO)
    *   4.3 阶段 2 (路径 2): 近端策略优化 (PPO)
    *   4.4 微调与对齐工作流的并行性

5.  **模型推理与评估 (Inference & Evaluation)**
    *   5.1 交互式推理 (Inference CLI)
    *   5.2 模型评估 (Evaluation)

6.  **工作流顺序总结**

7.  **常见问题与故障排除**

---

## **1. 重要提示与前置条件**

### 1.1 当前工作目录

**所有命令都必须在项目的根目录 (`Awesome-LLM-ZeroToScratch/`) 下执行**。

```bash
# 示例：进入项目根目录
cd D:\ALL IN AI\Awesome-LLM-ZeroToScratch
```

### 1.2 虚拟环境激活

在运行任何 Python 命令之前，**务必激活你的虚拟环境**。

```bash
# Windows PowerShell
.\.venv\Scripts\activate

# Linux / macOS (或 Git Bash)
source .venv/bin/activate

# Conda 环境 (如果使用conda)
conda activate Awesome-LLM-ZeroToScratch
```

### 1.3 CPU 环境下的常见警告

以下警告在 CPU 环境下是正常的，不影响功能：

*   `The installed version of bitsandbytes was compiled without GPU support...`
*   `Warning: FlashAttention not found or cannot be imported. Falling back to PyTorch SDPA.`
*   `The following values were not passed to accelerate launch...`

### 1.4 配置文件 (`configs/`)

所有训练和数据处理的超参数都通过 YAML 文件定义在 `configs/` 目录下。

### 1.5 输出目录 (`checkpoints/`, `data/`)

训练好的模型权重将保存在 `checkpoints/` 目录下。
处理后的数据集和分词器将保存在 `data/` 目录下。

---

## **2. 环境设置与准备**

### 2.1 基础依赖安装 (仅需一次)

```bash
bash setup.sh
```

### 2.2 `accelerate` 配置 (推荐)

```bash
accelerate config
```
*   选择 CPU、单机、多进程等。
*   选择不使用 DeepSpeed。
*   选择 `no` 混合精度。

---

## **3. 流程 A: 从零预训练 (Pre-training From Scratch)**

此流程将从零开始构建并训练一个基础语言模型。

### 3.1 阶段 1: 数据准备 (文本预训练数据)

为从零开始预训练LLM准备文本数据。

```bash
python -m data_processing.download_and_reproduce text
```

*   **功能**: 下载 `wikitext` 数据集，执行文本清洗、去重、训练 SentencePiece 分词器，并保存处理后的数据和分词器。
*   **配置文件**: `configs/data/text_pretrain.yaml`。
    *   `dataset_subset_size`: 默认 `1000`，用于快速测试。设为 `0` 使用全量数据。
*   **关键产物**:
    *   `./data/processed/wikitext-2-raw-v1/`
    *   `./data/tokenizers/wikitext_spm_hf/`

### 3.2 阶段 2: 执行预训练 (`0.5B_dense` 模型)

使用你的自定义 `BaseLLM` 模型进行预训练。

```bash
accelerate launch src/trainers/pretrain_trainer.py configs/training/pretrain_llm.yaml
```

*   **功能**: 初始化 `BaseLLM` 模型，加载处理后的文本数据和分词器，开始训练循环。
*   **配置文件**: `configs/training/pretrain_llm.yaml`。
    *   `model_config_path`: 默认已设置为 `configs/model/0.5B_dense.yaml`。
    *   **重要**: 运行前，请**手动修改** `configs/training/pretrain_llm.yaml` 中的 `dataset_dir` 路径为：`./data/processed/wikitext-2-raw-v1`。
    *   `max_steps`: 默认 `10`，用于快速验证。

*   **关键产物**:
    *   `./checkpoints/pretrain_llm_demo/final_model/` (最终预训练模型)

### 3.3 阶段 3: 推理/评估预训练模型

加载你刚刚预训练的模型，并与其进行交互。

```bash
python -m src.inference.inference ./checkpoints/pretrain_llm_demo/final_model
```

*   **功能**: 启动一个交互式命令行界面。
*   **预期行为**: 由于模型只预训练了少量步骤，它将生成无意义的重复字符。这是正常的，表示模型功能已打通。

---

## **4. 流程 B: 微调与对齐 (Fine-tuning & Alignment)**

此流程以一个现有的预训练模型（如 `TinyLlama`）为起点，对其进行微调和对齐。

### 4.1 阶段 1: 监督微调 (SFT)

让模型学会遵循指令。

```bash
accelerate launch src/trainers/sft_trainer.py configs/training/finetune_sft.yaml
```

*   **功能**: 使用 `mlabonne/guanaco-llama2-1k` 数据集对 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 模型进行 LoRA 微调。
*   **配置文件**: `configs/training/finetune_sft.yaml`。
    *   `model_name_or_path`: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
    *   `dataset_subset_size_cpu`: 默认 `16`。
    *   `max_steps`: 默认 `5`。
*   **关键产物**:
    *   `./checkpoints/sft-tinyllama-guanaco-cpu/final_model/` (SFT 训练后的 LoRA 适配器)

### 4.2 阶段 2 (路径 1): 直接偏好优化 (DPO)

使用偏好数据对SFT模型进行对齐。

```bash
accelerate launch src/trainers/dpo_trainer.py configs/training/finetune_dpo.yaml
```

*   **功能**: 基于SFT模型，使用 `trl-internal-testing/hh-rlhf-trl-style` 数据集进行 DPO 训练。
*   **配置文件**: `configs/training/finetune_dpo.yaml`。
    *   `model_name_or_path`: 默认指向 `./checkpoints/sft-tinyllama-guanaco-cpu/final_model`。
    *   `dataset_subset_size`: 默认 `10`。
    *   `max_steps`: 默认 `2`。
*   **关键产物**:
    *   `./checkpoints/dpo-tinyllama-guanaco-cpu/final_model/` (DPO 训练后的 LoRA 适配器)

### 4.3 阶段 2 (路径 2): 近端策略优化 (PPO)

使用奖励模型（概念性）对SFT模型进行对齐。

```bash
accelerate launch src/trainers/ppo_trainer.py configs/training/finetune_ppo.yaml
```

*   **功能**: 基于SFT模型，使用 `imdb` 数据集（概念性奖励模型）进行 PPO 训练。
*   **配置文件**: `configs/training/finetune_ppo.yaml`。
    *   `model_name_or_path`: 默认指向 `./checkpoints/sft-tinyllama-guanaco-cpu/final_model`。
    *   `dataset_subset_size`: 默认 `20`。
    *   `ppo_steps`: 默认 `4`。
*   **关键产物**:
    *   `./checkpoints/ppo-tinyllama-guanaco-cpu/final_ppo_model/` (PPO 训练后的 LoRA 适配器)

### 4.4 微调与对齐工作流的并行性

*   **SFT 必须首先完成**。
*   **DPO 和 PPO 可以在 SFT 完成后并行执行**。可以在不同的终端窗口中同时运行各自的 `accelerate launch` 命令。

---

## **5. 模型推理与评估 (Inference & Evaluation)**

这些命令用于测试和验证训练好的模型的功能和性能。

### 5.1 交互式推理 (Inference CLI)

与你选择的模型进行实时交互。

```bash
# 示例：与 SFT 模型聊天
python -m src.inference.inference ./checkpoints/sft-tinyllama-guanaco-cpu/final_model [max_new_tokens]

# 示例：与 DPO 模型聊天 (max_new_tokens 默认为 20)
python -m src.inference.inference ./checkpoints/dpo-tinyllama-guanaco-cpu/final_model 100

# 示例：与 PPO 模型聊天
python -m src.inference.inference ./checkpoints/ppo-tinyllama-guanaco-cpu/final_ppo_model
```

*   **参数**:
    *   `<model_path>`: 必填，指向你希望交互的模型路径。
    *   `[max_new_tokens]`: 可选，模型生成文本的最大词元数。
*   **交互**: 输入文本后按回车，模型将流式输出其响应。输入 `exit` 退出。

### 5.2 模型评估 (Evaluation)

对模型进行定性评估，并提供定量评估的指导。

```bash
# 评估 SFT 模型
python -m src.evaluation.evaluate_llm ./checkpoints/sft-tinyllama-guanaco-cpu/final_model

# 评估 DPO 模型
python -m src.evaluation.evaluate_llm ./checkpoints/dpo-tinyllama-guanaco-cpu/final_model

# 评估 PPO 模型
python -m src.evaluation.evaluate_llm ./checkpoints/ppo-tinyllama-guanaco-cpu/final_ppo_model
```

*   **功能**: 加载指定模型，生成对一组固定提示的响应（定性评估），并打印关于如何使用 `lm-evaluation-harness` 进行定量基准测试的指导。

---

## **6. 工作流顺序总结**

以下是建议的端到端执行顺序：

1.  **预训练路径**:
    `数据准备 (text)` **→** `预训练` **→** `推理/评估 (预训练模型)`

2.  **微调与对齐路径**:
    `SFT` **→** (`DPO` **或** `PPO`，可并行) **→** `评估/推理 (SFT/DPO/PPO 模型)`

---

## **7. 常见问题与故障排除**

*   **`ModuleNotFoundError: No module named 'src'` 或 `No module named 'data_processing'`**:
    *   **原因**: Python 找不到内部包。
    *   **解决方案**: 确保你在项目根目录 (`Awesome-LLM-ZeroToScratch/`) 下运行命令，并且**使用了 `python -m <module.path>` 的正确格式**。

*   **预训练时出现 PyTorch SDPA 错误 (`Explicit attn_mask should not be set when is_causal=True` 或 `name 'query_sdpa' is not defined`)**:
    *   **原因**: `StandardAttention` 模块在回退到 PyTorch SDPA 时参数传递问题。
    *   **解决方案**: 确保 `src/models/attention/standard_attention.py` 文件已更新到最新版本。

*   **模型保存错误 (`RuntimeError: The weights trying to be saved contained shared tensors...`)**:
    *   **原因**: 模型权重共享与 `safetensors` 格式的严格检查冲突。
    *   **解决方案**: 确保 `src/trainers/pretrain_trainer.py` 文件中 `save_pretrained` 调用已添加 `safe_serialization=False` 参数。

*   **预训练模型输出无意义的重复字符**:
    *   **原因**: 正常现象，模型在少量训练后仍处于随机状态。
    *   **解决方案**: 增加 `max_steps` 并使用更大的数据集进行更长时间的预训练。
