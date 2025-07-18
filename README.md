# Awesome-LLM-ZeroToScratch

[](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SingularGuyLeBorn/Awesome-LLM-ZeroToScratch)

一个端到端、代码驱动的教程，用于从零开始预训练和微调大语言模型（LLMs）和视觉语言模型（VLMs）。

本仓库遵循 **Bedrock Protocol v: Absolute Provenance Edition** 构建，确保了理论与实践的鲁棒性、清晰度和可复现性。

## 核心理念

本项目采用双重理念设计，旨在同时满足不同用户的需求：

  * **保姆级 (Nanny-Level):** 提供极其详尽的、手把手的指南，包含可以直接运行的代码和命令，特别针对 AutoDL 等云平台进行了优化。它为具备理论知识但缺乏大规模模型训练实践经验的用户量身打造。
  * **工业级 (Industrial-Grade):** 代码库高度模块化、配置驱动（使用YAML），并为可扩展性而构建。它清晰地展示了将模型从0.5B参数扩展到70B+巨型模型所需的关键架构和策略调整（例如，并行策略、内存优化）。

## 主要特性

  * **端到端完整流程:** 覆盖从数据处理、分词器训练，到模型预训练、微调、推理和评估的全过程。
  * **双技术路线:**
      * **从零预训练 (Track A):** 学习如何构建并预训练一个自定义的Transformer模型，支持密集型（Dense）和专家混合（MoE）架构。
      * **高效微调 (Track B):** 学习如何使用LoRA技术对现有开源模型（如Llama, TinyLlama）进行有监督微调（SFT）和直接偏好优化（DPO）。
  * **模块化与可配置:** 所有实验均由YAML文件驱动，可以轻松切换模型架构、数据集和训练策略。
  * **先进技术集成:**
      * **分布式训练:** 通过 `Hugging Face Accelerate` 无缝支持 `DeepSpeed` (ZeRO-2, ZeRO-3) 和 `FSDP`。
      * **高效训练与推理:** 集成 `FlashAttention`、`bitsandbytes` 4位量化 和 `torch.compile` 等最新优化技术。
      * **实验追踪:** 内置 `Weights & Biases` 集成，方便监控训练过程。
  * **详尽的文档:** 提供保姆级的系列教程文档，解释每一步背后的原理和操作细节。

## 项目结构 (完整版)

仓库被组织成一个清晰、模块化的结构，以下是**完整、准确**的文件和目录说明：

```
/Awesome-LLM-ZeroToScratch
|-- .gitignore                  # 定义了Git应忽略的文件和目录
|-- README.md                   # 您正在阅读的文件
|-- LICENSE                     # MIT 许可证
|-- requirements.txt            # Python依赖清单，确保环境可复现
|-- setup.sh                    # 一键式环境安装脚本，专为稳定高效构建而设计
|-- create_project_structure.py # 用于自动生成项目骨架的脚本
|
|-- configs/                    # 所有YAML配置文件
|   |-- data/
|   |   |-- text_pretrain.yaml  # 文本预训练的数据处理配置
|   |   `-- vlm_pretrain.yaml   # VLM预训练的数据处理配置
|   |-- deepspeed/
|   |   |-- zero_stage2_config.json # DeepSpeed ZeRO Stage 2 配置文件
|   |   `-- zero_stage3_config.json # DeepSpeed ZeRO Stage 3 配置文件
|   |-- model/
|   |   |-- 0.5B_dense.yaml     # 0.5B参数密集型模型架构配置
|   |   |-- 0.8B_moe.yaml       # 0.8B参数专家混合(MoE)模型架构配置
|   |   `-- llama2-7b.yaml      # Llama-2 7B 模型的配置模板
|   `-- training/
|       |-- pretrain_llm.yaml   # 从零预训练的训练流程配置
|       |-- finetune_sft.yaml   # SFT微调的训练流程配置
|       `-- finetune_dpo.yaml   # DPO微调的训练流程配置
|
|-- data_processing/            # 数据下载、处理、分词器训练脚本
|   |-- __init__.py             # 使该目录成为一个Python包
|   |-- download_and_reproduce.py # 数据处理主入口，自动化下载、清洗和保存
|   |-- process_text.py       # 文本数据清洗逻辑
|   |-- process_vlm.py        # VLM(图文)数据处理逻辑
|   `-- build_tokenizer.py    # SentencePiece分词器训练脚本
|
|-- docs/                       # 详细的Markdown教程文档
|   |-- 01_environment_setup.md # Part 1: 环境搭建指南
|   |-- 02_data_pipeline.md     # Part 2: 数据流水线指南
|   |-- 03_track_a_pretraining.md # Part 3A: 从零预训练路线
|   |-- 04_track_b_finetuning.md  # Part 3B: 微调路线
|   |-- 05_deployment_and_evaluation.md # Part 4: 部署与评估
|   `-- SET_UP_GUIDE.md         # (补充)环境搭建指南的另一版本
|
|-- scripts/                    # 用于一键启动各项任务的Shell脚本
|   |-- run_pretrain.sh         # 启动预训练
|   |-- run_sft.sh              # 启动SFT微调
|   |-- run_dpo.sh              # 启动DPO微调
|   |-- run_inference.sh        # 启动命令行推理
|   `-- run_evaluation.sh       # (概念性)启动模型评估
|
`-- src/                        # 核心模块化源代码
    |-- __init__.py             # 暴露顶层组件
    |-- models/                 # 模型定义
    |   |-- __init__.py         # 暴露模型核心组件
    |   |-- attention/          # 注意力机制实现
    |   |   |-- __init__.py
    |   |   |-- standard_attention.py # 标准多头注意力
    |   |   `-- flash_attention.py    # FlashAttention的封装
    |   |-- ffn.py              # 前馈网络(FFN)实现，支持SwiGLU
    |   |-- moe.py              # 专家混合(MoE)层实现
    |   `-- language_model.py   # 核心模型构建器，组装所有模块
    |-- trainers/               # 训练器
    |   |-- __init__.py         # 暴露训练器函数
    |   |-- pretrain_trainer.py # 从零预训练的训练循环
    |   |-- sft_trainer.py      # SFT微调训练器(使用TRL)
    |   `-- dpo_trainer.py      # DPO微调训练器(使用TRL)
    |-- inference/              # 推理逻辑
    |   |-- init.py             # (注意: 文件名为init.py)
    |   `-- inference.py        # 命令行交互式推理脚本
    |-- evaluation/             # 评估逻辑
    |   |-- init.py             # (注意: 文件名为init.py)
    |   `-- evaluate_llm.py     # (概念性)模型评估脚本
    `-- utils/                  # 工具函数
        `-- __init__.py
```

## 快速开始

### 1\. 环境搭建

我们提供了一个一键安装脚本，可以自动处理所有依赖，包括 `PyTorch`, `DeepSpeed` 和 `flash-attn` 等。

详细的环境配置指南，请**务必首先阅读**：
**➡️ [docs/01\_environment\_setup.md](https://www.google.com/search?q=./docs/01_environment_setup.md)**

简要步骤如下：

```bash
# 推荐创建一个独立的 Conda 环境
conda create -n awesome-llm-env python=3.10 -y
conda activate awesome-llm-env

# 进入项目根目录
cd /path/to/Awesome-LLM-ZeroToScratch

# 运行一键安装脚本
bash setup.sh
```

### 2\. 数据准备

在开始训练之前，您需要准备数据集和分词器。

详细的数据流水线说明，请参考：
**➡️ [docs/02\_data\_pipeline.md](https://www.google.com/search?q=./docs/02_data_pipeline.md)**

运行以下命令来自动完成文本数据的处理和分词器训练：

```bash
# 执行文本数据流水线 (下载、清洗、保存处理后数据)
python data_processing/download_and_reproduce.py text

# 训练分词器 (参数来自configs/data/text_pretrain.yaml)
python data_processing/build_tokenizer.py \
    --output_path_prefix ./data/tokenizers/wikitext_spm \
    --corpus_path ./data/processed/wikitext/corpus.txt \
    --vocab_size 8000 \
    --model_type unigram
```

## 使用指南

### Track A: 从零预训练

此路线将引导您训练一个全新的语言模型。

详细的预训练指南，请参考：
**➡️ [docs/03\_track\_a\_pretraining.md](https://www.google.com/search?q=./docs/03_track_a_pretraining.md)**

1.  **配置:**
      * 打开 `configs/training/pretrain_llm.yaml`。
      * 通过修改 `model_config_path` 来选择模型架构，例如 `"configs/model/0.5B_dense.yaml"` 或 `"configs/model/0.8B_moe.yaml"`。
      * 确保 `dataset_dir` 指向您处理好的数据目录。
2.  **启动训练:**
    ```bash
    bash scripts/run_pretrain.sh
    ```

### Track B: 微调开源模型

此路线将教您如何高效地微调现有的强大开源模型。

详细的微调指南，请参考：
**➡️ [docs/04\_track\_b\_finetuning.md](https://www.google.com/search?q=./docs/04_track_b_finetuning.md)**

#### 1\. 有监督微调 (SFT)

1.  **配置:**
      * 打开 `configs/training/finetune_sft.yaml`。
      * 确认 `model_name_or_path` 指向您想微调的基础模型（例如 `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`）。
      * 确认 `dataset_name` 指向SFT数据集。
2.  **启动训练:**
    ```bash
    bash scripts/run_sft.sh
    ```

#### 2\. 直接偏好优化 (DPO)

DPO在SFT模型的基础上进行，以更好地对齐人类偏好。

1.  **配置:**
      * 打开 `configs/training/finetune_dpo.yaml`。
      * **至关重要:** 确保 `model_name_or_path` 指向上一步SFT训练产出的**适配器(adapter)目录**（例如 `"./checkpoints/sft-tinyllama-guanaco/final_model"`）。
2.  **启动训练:**
    ```bash
    bash scripts/run_dpo.sh
    ```

### 模型推理

使用我们提供的命令行界面（CLI）与您训练好的模型进行交互。

详细的部署与推理指南，请参考：
**➡️ [docs/05\_deployment\_and\_evaluation.md](https://www.google.com/search?q=./docs/05_deployment_and_evaluation.md)**

```bash
# 运行推理，将 <path_to_your_model_or_adapter> 替换为您的模型路径
bash scripts/run_inference.sh <path_to_your_model_or_adapter>
```

例如，要与SFT微调后的模型交互：

```bash
bash scripts/run_inference.sh ./checkpoints/sft-tinyllama-guanaco/final_model
```

### 模型评估

我们提供了一个概念性的脚本来演示如何使用 `lm-evaluation-harness` 等工具进行评估。

```bash
# 运行评估脚本
bash scripts/run_evaluation.sh <path_to_your_model_or_adapter>
```

## 许可证

本项目采用 MIT 许可证。详情请见 `LICENSE` 文件。